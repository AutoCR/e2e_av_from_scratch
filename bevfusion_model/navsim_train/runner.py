"""Iter-based training runner for BEVFusion on NAVSIM.

Simplified detection-only training (5 NAVSIM classes, no map/motion eval, from scratch).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bevfusion_model.navsim_train.runner_utils import (
    set_seed,
    choose_device,
    patch_cuda_calls_for_local_device,
    move_to_device,
    resolve_split_config,
    init_distributed,
    is_main_process,
    install_console_log,
    cleanup_distributed,
)

# BEVFusion modules
from bevfusion_model.configs.bevfusion_hyperparams import (
    get_training_hyperparams,
    get_training_recipe,
)
from bevfusion_model.navsim_dataset import NavSimBEVFusionDataset, collate_fn, build_dataloader
from bevfusion_model.navsim_train.optim import (
    build_optimizer,
    CosineWithLinearWarmup,
    clip_grad_norm,
)
from bevfusion_model.navsim_train.amp import Fp16Wrapper
from bevfusion_model.bevfusion import BEVFusion


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _save_ckpt(path, raw_model, optimizer, scheduler, scaler, iteration, samples_per_iter, config):
    """Save training checkpoint.

    samples_seen is the portable progress measure: iterations are only
    meaningful for a fixed batch size × world size, so resuming under a
    different one converts through samples.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "iter": iteration,
            "samples_seen": int(iteration) * int(samples_per_iter),
            "config": dict(config),
        },
        path,
    )


def _release_cuda_memory_for_eval(optimizer, device):
    """Return cached GPU memory to the driver before an eval pass.

    spconv's implicit_gemm allocates its tuning workspace with raw cudaMalloc
    (cumm TensorStorage), outside PyTorch's caching allocator. After a long
    training stretch the allocator has reserved nearly the whole GPU, so that
    raw allocation OOMs even though the memory is "free" inside the cache.
    Dropping the already-applied grads and emptying the cache (cudaFree of
    every fully-free cached segment) returns that memory to the driver.
    """
    if device.type != "cuda":
        return
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()


def _try_loss_eval(raw_model, loader, device, scaler, max_batches, optimizer, tag):
    """Run _run_loss_eval, but never let an eval-time CUDA OOM kill training.

    Eval is an auxiliary signal; on OOM we free the cache, restore train mode
    (the normal restore in _run_loss_eval is skipped when it raises mid-loop)
    and return {} so the caller logs nothing for this round.
    """
    _release_cuda_memory_for_eval(optimizer, device)
    try:
        return _run_loss_eval(raw_model, loader, device, scaler, max_batches)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        raw_model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        tqdm.write(f"{tag}: eval skipped (CUDA OOM during eval): {exc}")
        return {}


@torch.no_grad()
def _run_loss_eval(raw_model, loader, device, scaler, max_batches):
    """Average detection-loss components over (a prefix of) an eval split.

    Runs the loss path (``_forward_train``) explicitly with the model in eval
    mode, on the unwrapped model so no DDP collectives are involved — must be
    called on rank 0 only. No mAP/NDS metrics exist in this port, so val/test
    loss is the evaluation signal.
    """
    was_training = raw_model.training
    raw_model.eval()
    sums: dict = {}
    count = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = move_to_device(raw_batch, device)
        img = batch.pop("img")
        with scaler.autocast():
            loss_dict = raw_model._forward_train(img=img, **batch)
        total = sum(v for v in loss_dict.values() if torch.is_tensor(v))
        sums["loss_total"] = sums.get("loss_total", 0.0) + float(total)
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
    if was_training:
        raw_model.train()
    if count == 0:
        return {}
    return {key: value / count for key, value in sums.items()}


def run(config: dict):
    """Train BEVFusion on NAVSIM (detection-only, 5 classes, from scratch).

    Args:
        config: runtime configuration dict with keys:
            - splits: {"train": split_config, ...}
            - openscene_data_root, nuplan_maps_root
            - output_dir
            - seed, num_workers, device, quick_smoke
            - resume_from (optional)
            - camera_order, image_hw (optional)
    """
    # Distributed setup
    local_rank, world_size = init_distributed()
    set_seed(int(config.get("seed", 0)))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = choose_device(config.get("device", "auto"))
    patch_cuda_calls_for_local_device(device)

    # Hyperparameters and recipe
    recipe = get_training_recipe()
    # Override recipe from config if present
    for key in ("total_batch_size", "num_epochs", "ckpt_epoch_interval", "eval_epoch_interval", "eval_max_batches", "log_interval", "fp16_loss_scale", "num_workers"):
        if config.get(key) is not None:
            recipe[key] = config[key]

    # Quick smoke test mode
    quick_smoke = bool(config.get("quick_smoke", False))
    if quick_smoke:
        recipe.update(
            {
                "total_batch_size": 1,
                "num_epochs": 1,
                "warmup_iters": 2,
                "log_interval": 1,
                "ckpt_epoch_interval": 1,
                "eval_epoch_interval": 1,
                "eval_max_batches": 1,
            }
        )
        max_scenes = 2
    else:
        max_scenes = None

    hyperparams = get_training_hyperparams()

    # Output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Mirror rank-0 stdout/stderr to output_dir/console_<timestamp>.log so a bare
    # `torchrun ...` always produces a log without a manual `| tee`.
    install_console_log(output_dir)

    # Resolve splits
    splits = config["splits"]
    train_dir, train_logs, train_tokens = resolve_split_config(splits["train"], _REPO_ROOT)
    val_dir, val_logs, val_tokens = resolve_split_config(splits["val"], _REPO_ROOT)
    test_dir, test_logs, test_tokens = resolve_split_config(splits["test"], _REPO_ROOT)

    # Build datasets (test_mode=False everywhere: eval computes loss, so GT is needed)
    def _build_split_dataset(split_dir, log_names, tokens, name):
        try:
            return NavSimBEVFusionDataset(
                split=split_dir,
                openscene_data_root=config["openscene_data_root"],
                nuplan_maps_root=config["nuplan_maps_root"],
                camera_order=config.get("camera_order"),
                image_hw=tuple(config.get("image_hw", (256, 704))),
                test_mode=False,
                max_scenes=max_scenes,
                log_names=log_names,
                tokens=tokens,
            )
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to build NAVSIM {name} dataset: {e}\n"
                "NAVSIM sensor data not found at configured paths. "
                "Please run training on the server with NAVSIM/OpenScene data available."
            ) from e

    train_dataset = _build_split_dataset(train_dir, train_logs, train_tokens, "train")
    val_dataset = _build_split_dataset(val_dir, val_logs, val_tokens, "val")
    test_dataset = _build_split_dataset(test_dir, test_logs, test_tokens, "test")

    # Dataloader with DDP sampler if multi-GPU
    total_batch_size = int(recipe["total_batch_size"])
    num_workers = int(config.get("num_workers", recipe["num_workers"]))

    if world_size > 1:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = build_dataloader(
            train_dataset,
            batch_size=total_batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn_override=collate_fn,
            sampler=sampler,
        )
    else:
        sampler = None
        train_loader = build_dataloader(
            train_dataset,
            batch_size=total_batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn_override=collate_fn,
            sampler=None,
        )

    # Eval loaders: rank-0-only loss eval, no sampler, no shuffle
    val_loader = build_dataloader(
        val_dataset,
        batch_size=total_batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn_override=collate_fn,
        sampler=None,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=total_batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn_override=collate_fn,
        sampler=None,
    )

    # Model
    model = BEVFusion(hyperparams)
    model.to(device)
    if world_size > 1:
        # gradient_as_bucket_view: grads live inside the reducer buckets
        # instead of being duplicated — saves a full gradient copy per GPU.
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    raw_model = model.module if isinstance(model, DDP) else model

    # Optimizer, scheduler, scaler
    optimizer = build_optimizer(
        raw_model, recipe["lr"], recipe["weight_decay"], recipe.get("backbone_lr_mult", 1.0)
    )
    # Each iteration consumes total_batch_size samples PER RANK (the
    # DistributedSampler shards the dataset), so epoch/schedule accounting
    # must include world_size or multi-GPU runs get a 6-8x too-long cosine
    # and ckpt/eval cadences that never fire.
    samples_per_iter = total_batch_size * world_size
    num_iters_per_epoch = max(1, len(train_dataset) // samples_per_iter)
    max_iters = num_iters_per_epoch * int(recipe["num_epochs"])
    scheduler = CosineWithLinearWarmup(
        optimizer,
        max_iters=max_iters,
        warmup_iters=int(recipe["warmup_iters"]),
        warmup_ratio=recipe["warmup_ratio"],
        min_lr_ratio=recipe["min_lr_ratio"],
    )
    scaler = Fp16Wrapper(enabled=(device.type == "cuda"), init_scale=recipe["fp16_loss_scale"])

    # Resume from checkpoint if provided
    start_iter = 0
    if config.get("resume_from"):
        # Load to CPU: map_location=device would pin a second full copy of the
        # model weights + AdamW moments on the GPU for the lifetime of run()
        # (the dict stays referenced), which is enough to OOM the first
        # backward after resume. load_state_dict moves state to the params'
        # device on its own.
        checkpoint = torch.load(config["resume_from"], map_location="cpu")
        raw_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        # Progress is tracked in samples so a checkpoint can be resumed under
        # a different batch size or GPU count. Precedence: explicit config
        # override > samples_seen stored in the checkpoint > legacy fallback
        # (old checkpoints stored only "iter"; assume they came from a
        # single-process run at the current batch size — set
        # resume_samples_seen in the config when that guess is wrong).
        samples_seen = config.get("resume_samples_seen")
        if samples_seen is None:
            samples_seen = checkpoint.get("samples_seen")
        if samples_seen is None:
            samples_seen = int(checkpoint.get("iter", 0)) * total_batch_size
        start_iter = int(samples_seen) // samples_per_iter
        # The scheduler state is deliberately NOT loaded: lr is a pure
        # function of (absolute iteration, max_iters), and both are in the
        # current run's units; a stored max_iters from a different batch
        # size / world size would mis-pace the cosine.
        scheduler.step(start_iter)
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if is_main_process():
            print(
                f"Resumed from {config['resume_from']}: samples_seen={int(samples_seen)} "
                f"-> start iter {start_iter}/{max_iters} "
                f"(batch {total_batch_size} x {world_size} ranks)"
            )

    # TensorBoard writer (main process only)
    writer = None
    if is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=str(output_dir / "tb" / timestamp))

    # Training loop
    # Intervals may be fractional (e.g. 0.25 = 4 checkpoints per epoch) so a
    # crash never costs more than a fraction of an epoch of training.
    ckpt_cadence = max(1, int(num_iters_per_epoch * float(recipe["ckpt_epoch_interval"])))
    eval_cadence = max(1, int(num_iters_per_epoch * float(recipe["eval_epoch_interval"])))
    eval_max_batches = recipe.get("eval_max_batches", 100)
    iteration = start_iter
    pbar = tqdm(
        total=max_iters,
        initial=start_iter,
        disable=not is_main_process(),
        desc="BEVFusion Training",
        unit="iter",
    )

    try:
        for epoch in range(int(recipe["num_epochs"])):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for raw_batch in train_loader:
                if iteration >= max_iters:
                    break

                model.train()
                batch = move_to_device(raw_batch, device)
                img = batch.pop("img")

                optimizer.zero_grad(set_to_none=True)
                with scaler.autocast():
                    loss_dict = model(img=img, **batch)
                    loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))

                # Always run backward: skipping it on a subset of ranks desyncs
                # the DDP collectives (buffer broadcasts pair with other ranks'
                # grad allreduces), which corrupts reducer state and crashes
                # with device-side index asserts. On non-finite loss the grads
                # come out non-finite and scaler.step() skips the update on
                # every rank consistently (found_inf is recorded at unscale_).
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm(raw_model, recipe["grad_clip_max_norm"], recipe["grad_clip_norm_type"])

                if not bool(torch.isfinite(loss.detach())) and is_main_process():
                    components = {
                        k: float(v) for k, v in loss_dict.items() if torch.is_tensor(v)
                    }
                    tqdm.write(
                        f"iter {iteration + 1}: non-finite loss {components}, update skipped"
                    )

                if scaler.enabled:
                    scaler.step(optimizer)
                    scaler.update()
                elif torch.isfinite(grad_norm):
                    optimizer.step()
                scheduler.step(iteration + 1)

                iteration += 1
                pbar.update(1)

                if is_main_process() and iteration % int(recipe["log_interval"]) == 0:
                    lr = optimizer.param_groups[-1]["lr"]
                    writer.add_scalar("train/loss_total", float(loss), iteration)
                    for k, v in loss_dict.items():
                        if torch.is_tensor(v):
                            writer.add_scalar(f"train/{k}", float(v), iteration)
                    writer.add_scalar("train/lr", lr, iteration)
                    writer.add_scalar("train/grad_norm", float(grad_norm), iteration)
                    pbar.set_postfix(loss=f"{float(loss):.3f}", lr=f"{lr:.2e}")

                # Checkpoint + val loss eval at epoch intervals (rank 0 only)
                ran_ckpt = iteration % ckpt_cadence == 0
                ran_eval = iteration % eval_cadence == 0
                if ran_ckpt and is_main_process():
                    _save_ckpt(output_dir / f"iter_{iteration}.pth", raw_model, optimizer, scheduler, scaler, iteration, samples_per_iter, config)
                    _save_ckpt(output_dir / "last.pth", raw_model, optimizer, scheduler, scaler, iteration, samples_per_iter, config)
                if ran_eval and is_main_process():
                    val_metrics = _try_loss_eval(raw_model, val_loader, device, scaler, eval_max_batches, optimizer, f"val@{iteration}")
                    for key, value in val_metrics.items():
                        writer.add_scalar(f"val/{key}", value, iteration)
                    if val_metrics:
                        tqdm.write(f"val@{iteration}: " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                # Hold every rank until rank 0 finishes eval/checkpoint; otherwise the
                # other ranks race into the next DDP collective and NCCL watchdogs fire.
                if world_size > 1 and (ran_ckpt or ran_eval):
                    dist.barrier()

            if iteration >= max_iters:
                break

        pbar.close()
        if world_size > 1:
            dist.barrier()

        # Final val/test loss eval (rank 0 only; raw_model avoids DDP collectives)
        if is_main_process():
            for tag, loader in (("val_final", val_loader), ("test_final", test_loader)):
                metrics = _try_loss_eval(raw_model, loader, device, scaler, eval_max_batches, optimizer, f"{tag}@{iteration}")
                for key, value in metrics.items():
                    writer.add_scalar(f"{tag}/{key}", value, iteration)
                if metrics:
                    print(f"{tag}@{iteration}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        cleanup_distributed()
