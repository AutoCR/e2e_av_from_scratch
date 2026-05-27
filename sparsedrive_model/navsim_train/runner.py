"""Iter-based pure-PyTorch training runner for SparseDrive on NAVSIM."""

from __future__ import annotations

import itertools
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

if __package__ in {None, ""}:
    _THIS_FILE = Path(__file__).resolve()
    _REPO_ROOT = _THIS_FILE.parents[2]
    _SPARSEDRIVE_ROOT = _REPO_ROOT / "sparsedrive_model"
    for _path in (str(_REPO_ROOT), str(_SPARSEDRIVE_ROOT)):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from sparsedrive_model.navsim_train.amp import Fp16Wrapper
from sparsedrive_model.navsim_train.config import get_stage_hyperparams
from sparsedrive_model.navsim_train.eval_hook import run_eval
from sparsedrive_model.navsim_train.optim import CosineWithLinearWarmup, build_optimizer, clip_grad_norm
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


def _build_dataset(dataset_cls, *, split, config, test_mode, max_scenes):
    kwargs = {
        "split": split,
        "openscene_data_root": config["openscene_data_root"],
        "nuplan_maps_root": config["nuplan_maps_root"],
        "image_hw": (256, 704),
        "test_mode": test_mode,
    }
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    try:
        return dataset_cls(**kwargs)
    except TypeError:
        kwargs["mode"] = test_mode
        return dataset_cls(**kwargs)


def _build_loader(build_dataloader_fn, dataset, *, batch_size, num_workers, shuffle, collate_fn):
    try:
        return build_dataloader_fn(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    except TypeError:
        return build_dataloader_fn(dataset, batch_size, num_workers, shuffle, collate_fn)


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
    torch.save(
        {
            "model": model.state_dict(),
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


def run(config: dict):
    _set_seed(int(config.get("seed", 0)))
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
                "total_batch_size": 2,
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
    train_dataset = _build_dataset(NavSimSparseDriveDataset, split=splits["train"], config=config, test_mode=False, max_scenes=max_scenes)
    val_dataset = _build_dataset(NavSimSparseDriveDataset, split=splits["val"], config=config, test_mode=True, max_scenes=max_scenes)
    test_dataset = _build_dataset(NavSimSparseDriveDataset, split=splits["test"], config=config, test_mode=True, max_scenes=max_scenes)

    train_loader = _build_loader(
        build_dataloader_fn,
        train_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=True,
        collate_fn=collate_fn_fn,
    )
    val_loader = _build_loader(
        build_dataloader_fn,
        val_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
    )
    test_loader = _build_loader(
        build_dataloader_fn,
        test_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
    )

    model = SparseDrive(_prepare_hyperparams_for_local_build(hyperparams))
    model.init_weights()
    model.to(device)

    optimizer = build_optimizer(model, recipe["lr"], recipe["weight_decay"], recipe["backbone_lr_mult"])
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
        model.load_state_dict(_extract_model_state(checkpoint), strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_iter = int(checkpoint.get("iter", 0))
        print(f"Resumed full training state from {config['resume_from']} at iter {start_iter}")
    elif config.get("load_from"):
        _load_weights(model, config["load_from"], device, "model weights")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(output_dir / "tb" / f"{stage}_{timestamp}"))
    train_iter = itertools.cycle(train_loader)
    eval_cadence = max(1, num_iters_per_epoch * int(recipe["eval_epoch_interval"]))
    ckpt_cadence = max(1, num_iters_per_epoch * int(recipe["ckpt_epoch_interval"]))

    try:
        for iteration in range(start_iter, max_iters):
            model.train()
            raw_batch = next(train_iter)
            if not isinstance(raw_batch, dict) or "img" not in raw_batch:
                raise KeyError("Training batches must be dicts containing an 'img' tensor.")
            batch = _move_to_device(raw_batch, device)
            img = batch.pop("img")
            optimizer.zero_grad(set_to_none=True)
            with scaler.autocast():
                loss_dict = model(img=img, **batch)
                loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm(model, recipe["grad_clip_max_norm"], recipe["grad_clip_norm_type"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step(iteration + 1)
            global_iter = iteration + 1

            if global_iter % int(recipe["log_interval"]) == 0:
                losses_for_log = {k: float(v.detach().cpu()) for k, v in loss_dict.items() if torch.is_tensor(v)}
                writer.add_scalar("train/loss_total", float(loss.detach().cpu()), global_iter)
                writer.add_scalar("train/lr", optimizer.param_groups[-1]["lr"], global_iter)
                writer.add_scalar("train/grad_norm", float(grad_norm), global_iter)
                for key, value in losses_for_log.items():
                    writer.add_scalar(f"train/{key}", value, global_iter)
                print(
                    f"iter={global_iter}/{max_iters} loss={float(loss.detach().cpu()):.6f} "
                    f"lr={optimizer.param_groups[-1]['lr']:.8f} grad_norm={float(grad_norm):.4f}"
                )

            if global_iter % eval_cadence == 0:
                summary = run_eval(model, val_loader, device, output_dir, writer, global_iter, recipe["eval_mode"], "val")
                print(f"val@{global_iter}: {summary}")
            if global_iter % ckpt_cadence == 0:
                _checkpoint_if_needed(output_dir, model, optimizer, scheduler, scaler, global_iter, config)

        final_iter = max_iters
        print(f"final_val@{final_iter}: {run_eval(model, val_loader, device, output_dir, writer, final_iter, recipe['eval_mode'], 'val_final')}")
        print(f"final_test@{final_iter}: {run_eval(model, test_loader, device, output_dir, writer, final_iter, recipe['eval_mode'], 'test_final')}")
    finally:
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
