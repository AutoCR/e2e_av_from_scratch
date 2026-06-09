"""Iter-based training runner for BEVFusion on NAVSIM.

Simplified detection-only training (5 NAVSIM classes, no map/motion eval, from scratch).
Reuses distributed and device helpers from sparsedrive runner.
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

# Reuse helpers from sparsedrive runner (pure generic utilities, no sparsedrive model deps)
from sparsedrive_model.navsim_train.runner import (
    _set_seed,
    _choose_device,
    _patch_cuda_calls_for_local_device,
    _move_to_device,
    _resolve_split_config,
    _init_distributed,
    _get_rank,
    _get_world_size,
    _is_main_process,
    _cleanup_distributed,
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


def _save_ckpt(path, raw_model, optimizer, scheduler, scaler, iteration, config):
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
    local_rank, world_size = _init_distributed()
    _set_seed(int(config.get("seed", 0)))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _choose_device(config.get("device", "auto"))
    _patch_cuda_calls_for_local_device(device)

    # Hyperparameters and recipe
    recipe = get_training_recipe()
    # Override recipe from config if present
    for key in ("total_batch_size", "num_epochs", "ckpt_epoch_interval", "log_interval", "fp16_loss_scale", "num_workers"):
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
            }
        )
        max_scenes = 2
    else:
        max_scenes = None

    hyperparams = get_training_hyperparams()

    # Output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve train split
    splits = config["splits"]
    train_dir, train_logs, train_tokens = _resolve_split_config(splits["train"], _REPO_ROOT)

    # Build dataset
    try:
        train_dataset = NavSimBEVFusionDataset(
            split=train_dir,
            openscene_data_root=config["openscene_data_root"],
            nuplan_maps_root=config["nuplan_maps_root"],
            camera_order=config.get("camera_order"),
            image_hw=tuple(config.get("image_hw", (256, 704))),
            test_mode=False,
            max_scenes=max_scenes,
            log_names=train_logs,
            tokens=train_tokens,
        )
    except (FileNotFoundError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to build NAVSIM dataset: {e}\n"
            "NAVSIM sensor data not found at configured paths. "
            "Please run training on the server with NAVSIM/OpenScene data available."
        ) from e

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

    # Model
    model = BEVFusion(hyperparams)
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if isinstance(model, DDP) else model

    # Optimizer, scheduler, scaler
    optimizer = build_optimizer(
        raw_model, recipe["lr"], recipe["weight_decay"], recipe.get("backbone_lr_mult", 1.0)
    )
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

    # Resume from checkpoint if provided
    start_iter = 0
    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], map_location=device)
        raw_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_iter = int(checkpoint.get("iter", 0))
        if _is_main_process():
            print(f"Resumed full training state from {config['resume_from']} at iter {start_iter}")

    # TensorBoard writer (main process only)
    writer = None
    if _is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=str(output_dir / "tb" / timestamp))

    # Training loop
    iteration = start_iter
    pbar = tqdm(
        total=max_iters,
        initial=start_iter,
        disable=not _is_main_process(),
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
                batch = _move_to_device(raw_batch, device)
                img = batch.pop("img")

                optimizer.zero_grad(set_to_none=True)
                with scaler.autocast():
                    loss_dict = model(img=img, **batch)
                    loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))

                if not torch.isfinite(loss):
                    tqdm.write(f"iter {iteration + 1}: non-finite loss, skipping")
                    optimizer.zero_grad(set_to_none=True)
                    iteration += 1
                    pbar.update(1)
                    scheduler.step(iteration)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm(raw_model, recipe["grad_clip_max_norm"], recipe["grad_clip_norm_type"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(iteration + 1)

                iteration += 1
                pbar.update(1)

                if _is_main_process() and iteration % int(recipe["log_interval"]) == 0:
                    lr = optimizer.param_groups[-1]["lr"]
                    writer.add_scalar("train/loss_total", float(loss), iteration)
                    for k, v in loss_dict.items():
                        if torch.is_tensor(v):
                            writer.add_scalar(f"train/{k}", float(v), iteration)
                    writer.add_scalar("train/lr", lr, iteration)
                    writer.add_scalar("train/grad_norm", float(grad_norm), iteration)
                    pbar.set_postfix(loss=f"{float(loss):.3f}", lr=f"{lr:.2e}")

                # Checkpoint at epoch intervals
                if _is_main_process() and iteration % (num_iters_per_epoch * int(recipe["ckpt_epoch_interval"])) == 0:
                    _save_ckpt(output_dir / f"iter_{iteration}.pth", raw_model, optimizer, scheduler, scaler, iteration, config)
                    _save_ckpt(output_dir / "last.pth", raw_model, optimizer, scheduler, scaler, iteration, config)

            if iteration >= max_iters:
                break

        pbar.close()
        if world_size > 1:
            dist.barrier()

    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        _cleanup_distributed()
