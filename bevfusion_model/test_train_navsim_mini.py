"""Smoke test the BEVFusion-on-NAVSIM training pipeline on the mini split.

Configures train/val/test all on the NAVSIM ``mini`` split (located at
``/Users/chenran/Code/navsim/dataset``), each limited to 1-2 scenes, and runs a
few real training iterations end-to-end:

    forward -> loss -> backward -> grad-clip -> optimizer step -> scheduler step

This exercises the full pipeline with REAL sensor data (images + LiDAR +
3D-box GT) rather than fabricated tensors. It runs on CPU/macOS via the
``spconv_mac`` LiDAR fallback (slow but functional); the CUDA path activates
automatically on a GPU server.

Run:
    uv run python bevfusion_model/test_train_navsim_mini.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "bevfusion_model"), str(_REPO_ROOT / "sparsedrive_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_ROOT = "/Users/chenran/Code/navsim/dataset"
MAPS_ROOT = "/Users/chenran/Code/navsim/dataset/maps"


def _mini_split(max_scenes: int) -> dict:
    """A split config pointing at the mini directory with no token/log filtering.

    ``log_names``/``tokens`` are left unset so the navtrain/navtest YAML filters
    (which don't contain mini logs) are not applied; ``max_scenes`` bounds the
    sample count instead.
    """
    return {"dir": "mini", "max_scenes": max_scenes}


def main() -> None:
    import torch

    from bevfusion_model.configs.bevfusion_hyperparams import (
        get_training_hyperparams,
        get_training_recipe,
    )
    from bevfusion_model.navsim_dataset import (
        NavSimBEVFusionDataset,
        collate_fn,
        build_dataloader,
    )
    from bevfusion_model.navsim_train.optim import (
        build_optimizer,
        CosineWithLinearWarmup,
        clip_grad_norm,
    )
    from bevfusion_model.navsim_train.amp import Fp16Wrapper
    from bevfusion_model.bevfusion import BEVFusion

    torch.manual_seed(0)
    device = torch.device("cpu")

    # --- Build train/val/test datasets, all from the mini split (1-2 scenes each) ---
    print("[1/5] Building mini-split datasets (train=2, val=1, test=1) ...")
    train_ds = NavSimBEVFusionDataset(
        split="mini", openscene_data_root=DATA_ROOT, nuplan_maps_root=MAPS_ROOT,
        test_mode=False, max_scenes=2,
    )
    val_ds = NavSimBEVFusionDataset(
        split="mini", openscene_data_root=DATA_ROOT, nuplan_maps_root=MAPS_ROOT,
        test_mode=True, max_scenes=1,
    )
    test_ds = NavSimBEVFusionDataset(
        split="mini", openscene_data_root=DATA_ROOT, nuplan_maps_root=MAPS_ROOT,
        test_mode=True, max_scenes=1,
    )
    print(f"      train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} scenes")
    assert len(train_ds) >= 1 and len(val_ds) >= 1 and len(test_ds) >= 1

    train_loader = build_dataloader(
        train_ds, batch_size=1, num_workers=0, shuffle=True, collate_fn_override=collate_fn,
    )

    # --- Model (5-class NAVSIM head, from scratch) ---
    print("[2/5] Building BEVFusion (5-class head, from scratch) ...")
    recipe = get_training_recipe()
    model = BEVFusion(get_training_hyperparams())
    model.to(device)
    assert model.heads["object"].num_classes == 5

    optimizer = build_optimizer(model, recipe["lr"], recipe["weight_decay"])
    num_iters = 4
    scheduler = CosineWithLinearWarmup(
        optimizer, max_iters=num_iters, warmup_iters=2,
        warmup_ratio=recipe["warmup_ratio"], min_lr_ratio=recipe["min_lr_ratio"],
    )
    scaler = Fp16Wrapper(enabled=False, init_scale=recipe["fp16_loss_scale"])

    # --- Train loop on real data ---
    print(f"[3/5] Running {num_iters} train iterations on real mini-split data ...")
    model.train()
    losses = []
    it = 0
    while it < num_iters:
        for raw_batch in train_loader:
            if it >= num_iters:
                break
            img = raw_batch.pop("img").to(device)
            batch = {
                k: ([t.to(device) for t in v] if isinstance(v, list) and v and torch.is_tensor(v[0]) else v)
                for k, v in raw_batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with scaler.autocast():
                loss_dict = model(img=img, **batch)
                loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
            assert torch.isfinite(loss), loss_dict
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = clip_grad_norm(model, recipe["grad_clip_max_norm"], recipe["grad_clip_norm_type"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(it + 1)
            losses.append(float(loss))
            comps = {k: round(float(v), 2) for k, v in loss_dict.items() if torch.is_tensor(v)}
            print(
                f"      iter {it + 1}: loss={float(loss):.2f} {comps} "
                f"grad_norm={float(gn):.1f} lr={optimizer.param_groups[-1]['lr']:.2e}"
            )
            it += 1

    # --- Inference on val/test sample (eval branch) ---
    print("[4/5] Running inference on a val + test sample (eval branch) ...")
    model.eval()
    for name, ds in (("val", val_ds), ("test", test_ds)):
        s = ds[0]
        b = collate_fn([s])
        img = b.pop("img")
        out = model(img=img, **b)
        print(f"      {name}: {out[0]['boxes_3d'].shape[0]} boxes, "
              f"score_max={float(out[0]['scores_3d'].max()) if out[0]['scores_3d'].numel() else 0.0:.3f}")

    print("[5/5] DONE — full train + inference pipeline runs on the mini split.")
    print(f"      losses: {[round(l, 1) for l in losses]}")
    assert all(l == l for l in losses), "NaN loss encountered"


if __name__ == "__main__":
    main()
