"""Iter-based pure-PyTorch training runner for SparseDrive on NAVSIM."""

from __future__ import annotations

import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import os
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

if __package__ in {None, ""}:
    _THIS_FILE = Path(__file__).resolve()
    _REPO_ROOT = _THIS_FILE.parents[2]
    _SPARSEDRIVE_ROOT = _REPO_ROOT / "sparsedrive_model"
    for _path in (str(_REPO_ROOT), str(_SPARSEDRIVE_ROOT)):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from sparsedrive_model.navsim_train.amp import Fp16Wrapper
from sparsedrive_model.configs.sparsedrive_hyperparams import get_stage_hyperparams
from sparsedrive_model.navsim_train.eval_hook import run_eval
from sparsedrive_model.navsim_train.optim import CosineWithLinearWarmup, build_optimizer, clip_grad_norm
from sparsedrive_model.navsim_train.scene_filter_loader import load_log_names, load_scene_filter_fields
from sparsedrive_model.sparsedrive import SparseDrive
from sparsedrive_model.sparsedrive import nn_utils as _sparsedrive_nn_utils


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _choose_device(device_config):
    if device_config != "auto":
        return torch.device(device_config)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patch_cuda_calls_for_local_device(device):
    if device.type == "cuda" or getattr(torch.Tensor, "_navsim_cuda_patched", False):
        return

    def _cuda_to_local(self, *args, **kwargs):
        non_blocking = bool(kwargs.get("non_blocking", False))
        return self.to(device, non_blocking=non_blocking)

    torch.Tensor.cuda = _cuda_to_local
    torch.Tensor._navsim_cuda_patched = True


def _patch_loss_weight_broadcasting():
    if getattr(_sparsedrive_nn_utils, "_navsim_weight_broadcast_patched", False):
        return

    def _weighted_loss(loss, weight=None, reduction="mean", avg_factor=None):
        if weight is not None:
            while weight.ndim < loss.ndim:
                weight = weight.unsqueeze(-1)
            loss = loss * weight
        if avg_factor is not None:
            return loss.sum() / max(float(avg_factor), 1.0)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"unsupported reduction: {reduction}")

    _sparsedrive_nn_utils.weighted_loss = _weighted_loss
    _sparsedrive_nn_utils._navsim_weight_broadcast_patched = True


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _load_data_api():
    try:
        from sparsedrive_model.navsim_train.data import NavSimSparseDriveDataset, build_dataloader, collate_fn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sparsedrive_model.navsim_train.data is required for full training. "
            "It is owned by the data-targets task and must provide "
            "NavSimSparseDriveDataset, build_dataloader, and collate_fn."
        ) from exc
    return NavSimSparseDriveDataset, build_dataloader, collate_fn


def _resolve_split_config(split_config, repo_root=None):
    """Resolve a split config entry to (directory_name, log_names_or_None, tokens_or_None).

    Accepts:
    - str: directory name only, no filtering applied.
    - dict with keys:
        - "dir": directory name under navsim_logs/ (required)
        - "log_names_yaml": YAML path for log names (optional, relative to repo_root or absolute)
        - "log_names_key": key in log_names_yaml; defaults to "log_names" (optional)
        - "tokens_yaml": YAML path for scene tokens (optional, may differ from log_names_yaml)
        - "tokens_key": key in tokens_yaml; defaults to "tokens" (optional)
      Omitting or setting a yaml field to None disables that filter.
    """
    if isinstance(split_config, str):
        return split_config, None, None

    dir_name = split_config["dir"]

    def _resolve_path(yaml_path_str):
        if yaml_path_str is None:
            return None
        p = Path(yaml_path_str)
        if not p.is_absolute() and repo_root is not None:
            p = Path(repo_root) / p
        return p

    log_names = None
    log_names_yaml = _resolve_path(split_config.get("log_names_yaml"))
    if log_names_yaml is not None:
        log_names_key = split_config.get("log_names_key", "log_names")
        log_names, _ = load_scene_filter_fields(log_names_yaml, log_names_key=log_names_key, tokens_key=None)

    tokens = None
    tokens_yaml = _resolve_path(split_config.get("tokens_yaml"))
    if tokens_yaml is not None:
        tokens_key = split_config.get("tokens_key", "tokens")
        _, tokens = load_scene_filter_fields(tokens_yaml, log_names_key=None, tokens_key=tokens_key)

    return dir_name, log_names, tokens


def _build_dataset(dataset_cls, *, split, config, test_mode, max_scenes, log_names=None, tokens=None):
    kwargs = {
        "split": split,
        "openscene_data_root": config["openscene_data_root"],
        "nuplan_maps_root": config["nuplan_maps_root"],
        "image_hw": (256, 704),
        "test_mode": test_mode,
    }
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    if log_names is not None:
        kwargs["log_names"] = log_names
    if tokens is not None:
        kwargs["tokens"] = tokens
    try:
        return dataset_cls(**kwargs)
    except TypeError:
        kwargs["mode"] = test_mode
        return dataset_cls(**kwargs)


def _build_loader(build_dataloader_fn, dataset, *, batch_size, num_workers, shuffle, collate_fn, sampler=None):
    effective_shuffle = shuffle if sampler is None else False
    try:
        return build_dataloader_fn(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=effective_shuffle,
            collate_fn=collate_fn,
            sampler=sampler,
        )
    except TypeError:
        # Fallback for older build_dataloader signatures that don't accept sampler
        return build_dataloader_fn(dataset, batch_size, num_workers, effective_shuffle, collate_fn)


def _prepare_hyperparams_for_local_build(hyperparams):
    prepared = dict(hyperparams)
    pretrained = prepared.get("backbone_pretrained")
    if pretrained and not Path(pretrained).exists():
        prepared["backbone_pretrained"] = None
    return prepared


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _load_weights(model, path, device, label):
    checkpoint = torch.load(path, map_location=device)
    incompatible = model.load_state_dict(_extract_model_state(checkpoint), strict=False)
    print(f"Loaded {label} from {path}: missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")


def _save_checkpoint(path, model, optimizer, scheduler, scaler, iteration, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "iter": iteration,
            "config": dict(config),
        },
        path,
    )


def _checkpoint_if_needed(output_dir, model, optimizer, scheduler, scaler, iteration, config):
    ckpt_dir = Path(output_dir) / "ckpt"
    iter_path = ckpt_dir / f"iter_{iteration}.pth"
    last_path = ckpt_dir / "last.pth"
    _save_checkpoint(iter_path, model, optimizer, scheduler, scaler, iteration, config)
    shutil.copyfile(iter_path, last_path)
    print(f"Saved checkpoint {iter_path}")


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------


def _init_distributed():
    """Initialize NCCL process group when launched via torchrun.

    Reads RANK / LOCAL_RANK / WORLD_SIZE from environment variables.
    Returns (local_rank, world_size) or (0, 1) for non-distributed runs.
    """
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0, 1  # non-DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, world_size


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _build_sampler(dataset, *, shuffle: bool):
    """Return a DistributedSampler when DDP is active, else None."""
    if _get_world_size() > 1:
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def _reset_temporal_state(model):
    """Reset all InstanceBank caches to break NaN propagation across iterations."""
    for module in model.modules():
        if hasattr(module, "reset") and callable(module.reset):
            try:
                module.reset()
            except Exception:
                pass


def run(config: dict):
    local_rank, world_size = _init_distributed()
    _set_seed(int(config.get("seed", 0)))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _choose_device(config.get("device", "auto"))
    _patch_cuda_calls_for_local_device(device)
    _patch_loss_weight_broadcasting()
    stage = config.get("stage", "stage1")
    hyperparams, recipe = get_stage_hyperparams(stage, num_cams_override=8)

    for key in ("total_batch_size", "num_epochs", "ckpt_epoch_interval", "eval_epoch_interval", "log_interval", "fp16_loss_scale"):
        if config.get(key) is not None:
            recipe[key] = config[key]
    if config.get("quick_smoke"):
        recipe.update(
            {
                "total_batch_size": 1,
                "num_epochs": 1,
                "warmup_iters": 2,
                "log_interval": 1,
                "ckpt_epoch_interval": 1,
                "eval_epoch_interval": 1,
            }
        )

    total_batch_size = int(recipe["total_batch_size"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    NavSimSparseDriveDataset, build_dataloader_fn, collate_fn_fn = _load_data_api()
    max_scenes = 2 if config.get("quick_smoke") else None
    splits = config["splits"]
    _REPO_ROOT_RUNNER = Path(__file__).resolve().parents[2]

    train_split_dir, train_log_names, train_tokens = _resolve_split_config(splits["train"], _REPO_ROOT_RUNNER)
    val_split_dir, val_log_names, val_tokens = _resolve_split_config(splits["val"], _REPO_ROOT_RUNNER)
    test_split_dir, test_log_names, test_tokens = _resolve_split_config(splits["test"], _REPO_ROOT_RUNNER)

    train_dataset = _build_dataset(
        NavSimSparseDriveDataset,
        split=train_split_dir,
        config=config,
        test_mode=False,
        max_scenes=max_scenes,
        log_names=train_log_names,
        tokens=train_tokens,
    )
    val_dataset = _build_dataset(
        NavSimSparseDriveDataset,
        split=val_split_dir,
        config=config,
        test_mode=True,
        max_scenes=max_scenes,
        log_names=val_log_names,
        tokens=val_tokens,
    )
    test_dataset = _build_dataset(
        NavSimSparseDriveDataset,
        split=test_split_dir,
        config=config,
        test_mode=True,
        max_scenes=max_scenes,
        log_names=test_log_names,
        tokens=test_tokens,
    )
    train_sampler = _build_sampler(train_dataset, shuffle=True)

    train_loader = _build_loader(
        build_dataloader_fn,
        train_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
        sampler=train_sampler,
    )
    val_loader = _build_loader(
        build_dataloader_fn,
        val_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
        sampler=None,
    )
    test_loader = _build_loader(
        build_dataloader_fn,
        test_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
        sampler=None,
    )

    model = SparseDrive(_prepare_hyperparams_for_local_build(hyperparams))
    model.init_weights()
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer = build_optimizer(raw_model, recipe["lr"], recipe["weight_decay"], recipe["backbone_lr_mult"])
    num_iters_per_epoch = max(1, len(train_dataset) // total_batch_size)
    max_iters = num_iters_per_epoch * int(recipe["num_epochs"])
    scheduler = CosineWithLinearWarmup(
        optimizer,
        max_iters=max_iters,
        warmup_iters=int(recipe["warmup_iters"]),
        warmup_ratio=recipe["warmup_ratio"],
        min_lr_ratio=recipe["min_lr_ratio"],
    )
    scaler = Fp16Wrapper(enabled=(device.type == "cuda"), init_scale=recipe["fp16_loss_scale"])

    start_iter = 0
    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], map_location=device)
        raw_model.load_state_dict(_extract_model_state(checkpoint), strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_iter = int(checkpoint.get("iter", 0))
        print(f"Resumed full training state from {config['resume_from']} at iter {start_iter}")
    elif config.get("load_from"):
        _load_weights(raw_model, config["load_from"], device, "model weights")

    writer = None
    if _is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=str(output_dir / "tb" / f"{stage}_{timestamp}"))
    eval_cadence = max(1, num_iters_per_epoch * int(recipe["eval_epoch_interval"]))
    ckpt_cadence = max(1, num_iters_per_epoch * int(recipe["ckpt_epoch_interval"]))

    iteration = start_iter
    pbar = tqdm(
        total=max_iters,
        initial=start_iter,
        desc="Training",
        unit="iter",
        dynamic_ncols=True,
        disable=not _is_main_process(),
    )
    try:
        for _epoch in range(int(recipe["num_epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(_epoch)
            for raw_batch in train_loader:
                if iteration >= max_iters:
                    break
                model.train()
                if not isinstance(raw_batch, dict) or "img" not in raw_batch:
                    raise KeyError("Training batches must be dicts containing an 'img' tensor.")
                batch = _move_to_device(raw_batch, device)
                img = batch.pop("img")
                optimizer.zero_grad(set_to_none=True)
                with scaler.autocast():
                    loss_dict = model(img=img, **batch)
                    loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
                if not torch.isfinite(loss):
                    tqdm.write(f"iter={iteration + 1}: non-finite loss={float(loss):.4f}, skipping backward")
                    _reset_temporal_state(raw_model)
                    optimizer.zero_grad(set_to_none=True)
                    iteration += 1
                    pbar.update(1)
                    scheduler.step(iteration)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                raw_model = model.module if isinstance(model, DDP) else model
                grad_norm = clip_grad_norm(raw_model, recipe["grad_clip_max_norm"], recipe["grad_clip_norm_type"])
                if not torch.isfinite(grad_norm):
                    tqdm.write(f"iter={iteration + 1}: NaN/inf gradients detected (scale={scaler.scaler.get_scale() if scaler.enabled else 'N/A'}), resetting temporal state")
                    _reset_temporal_state(raw_model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(iteration + 1)
                global_iter = iteration + 1

                pbar.update(1)
                if global_iter % int(recipe["log_interval"]) == 0:
                    losses_for_log = {k: float(v.detach().cpu()) for k, v in loss_dict.items() if torch.is_tensor(v)}
                    lr = optimizer.param_groups[-1]["lr"]
                    pbar.set_description(f"Epoch [{_epoch + 1}/{int(recipe['num_epochs'])}]")
                    pbar.set_postfix(
                        loss=f"{float(loss.detach().cpu()):.4f}",
                        lr=f"{lr:.2e}",
                        grad_norm=f"{float(grad_norm):.3f}",
                    )
                    if _is_main_process():
                        writer.add_scalar("train/loss_total", float(loss.detach().cpu()), global_iter)
                        writer.add_scalar("train/lr", lr, global_iter)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_iter)
                        for key, value in losses_for_log.items():
                            writer.add_scalar(f"train/{key}", value, global_iter)
                        tqdm.write(
                            f"iter={global_iter}/{max_iters} loss={float(loss.detach().cpu()):.6f} "
                            f"lr={lr:.8f} grad_norm={float(grad_norm):.4f}"
                        )

                if global_iter % eval_cadence == 0 and _is_main_process():
                    summary = run_eval(model, val_loader, device, output_dir, writer, global_iter, recipe["eval_mode"], "val")
                    print(f"val@{global_iter}: {summary}")
                if global_iter % ckpt_cadence == 0 and _is_main_process():
                    _checkpoint_if_needed(output_dir, model, optimizer, scheduler, scaler, global_iter, config)

                iteration += 1
            if iteration >= max_iters:
                break

        final_iter = max_iters
        if world_size > 1:
            dist.barrier()
        if _is_main_process():
            print(f"final_val@{final_iter}: {run_eval(model, val_loader, device, output_dir, writer, final_iter, recipe['eval_mode'], 'val_final')}")
            print(f"final_test@{final_iter}: {run_eval(model, test_loader, device, output_dir, writer, final_iter, recipe['eval_mode'], 'test_final')}")
    finally:
        pbar.close()
        _cleanup_distributed()
        if writer is not None:
            writer.flush()
            writer.close()


def _standalone_smoke():
    hyperparams, recipe = get_stage_hyperparams("stage1", num_cams_override=8)
    model = SparseDrive(_prepare_hyperparams_for_local_build(hyperparams))
    model.init_weights()
    optimizer = build_optimizer(model, recipe["lr"], recipe["weight_decay"], recipe["backbone_lr_mult"])
    scheduler = CosineWithLinearWarmup(
        optimizer,
        max_iters=10,
        warmup_iters=2,
        warmup_ratio=recipe["warmup_ratio"],
        min_lr_ratio=recipe["min_lr_ratio"],
    )
    print("lr_schedule:")
    for iteration in (0, 1, 2, 5, 9):
        lrs = scheduler.step(iteration)
        print(f"iter {iteration}: " + ", ".join(f"{lr:.10f}" for lr in lrs))
    amp = Fp16Wrapper(enabled=torch.cuda.is_available(), init_scale=recipe["fp16_loss_scale"])
    tensor = torch.tensor(1.0, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    with amp.autocast():
        loss = tensor * 2.0
    amp.scale(loss).backward()
    print(f"fp16_enabled={amp.enabled} dummy_grad={float(tensor.grad.detach().cpu()):.1f}")


if __name__ == "__main__":
    _standalone_smoke()
