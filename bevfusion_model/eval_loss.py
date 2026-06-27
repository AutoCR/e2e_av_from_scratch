"""Standalone validation-loss evaluator for a trained BEVFusion NAVSIM checkpoint.

What this computes
------------------
Loads a training checkpoint (the dict saved by ``navsim_train/runner.py``, whose
weights live under the ``"model"`` key), rebuilds the model with
``BEVFusion(get_training_hyperparams())`` exactly as the runner does, and runs
the detection loss path (``_forward_train``) over a prefix of the VAL split (and
optionally the TEST split). It averages the loss components returned by the head
-- ``loss_cls``, ``loss_bbox``, ``loss_heatmap`` -- plus a derived ``loss_total``
(sum of the tensor components), over the evaluated batches.

This mirrors the runner's ``_run_loss_eval`` / ``_try_loss_eval`` logic
(``model.eval()``, ``torch.no_grad()``, pop ``img``, call ``_forward_train``,
average the loss dict). No mAP/NDS is computed here -- see
``eval_detection_map.py`` for that. Loss is the same signal the training runner
logs for val/test.

Design notes
------------
- Single GPU, no DDP. Run with ``CUDA_VISIBLE_DEVICES=0``.
- The eval loop is REPLICATED inline (rather than imported from runner.py) so
  this script has no dependency on the runner's training machinery and can never
  trigger distributed init. The logic is byte-for-byte equivalent to
  ``runner._run_loss_eval``.
- OOM-robust: configurable (small) batch size, ``--max-batches`` cap, and
  ``torch.cuda.empty_cache()`` between batches. spconv's implicit_gemm tuning
  workspace is allocated outside PyTorch's caching allocator, so freeing the
  cache between batches keeps headroom for it.

Example (remote)
----------------
    cd ~/Code/e2e_av_from_scratch && \
      CUDA_VISIBLE_DEVICES=0 /home/pnc/.local/bin/uv run python \
      bevfusion_model/eval_loss.py --max-batches 20            # smoke

    cd ~/Code/e2e_av_from_scratch && \
      CUDA_VISIBLE_DEVICES=0 /home/pnc/.local/bin/uv run python \
      bevfusion_model/eval_loss.py --splits val test --max-batches 200  # full
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bevfusion_model.bevfusion import BEVFusion
from bevfusion_model.configs.bevfusion_hyperparams import (
    get_runtime_config,
    get_training_hyperparams,
)
from bevfusion_model.navsim_dataset import (
    NavSimBEVFusionDataset,
    build_dataloader,
    collate_fn,
)
from bevfusion_model.navsim_train.amp import Fp16Wrapper
from bevfusion_model.navsim_train.runner_utils import (
    choose_device,
    move_to_device,
    resolve_split_config,
    set_seed,
)


DEFAULT_CKPT = "bevfusion_model/outputs/train_navsim/iter_170208.pth"


def build_split_dataset(split_cfg, config, max_scenes):
    """Build a GT-bearing dataset for one split (test_mode=False, as the runner does for eval)."""
    split_dir, log_names, tokens = resolve_split_config(split_cfg, _REPO_ROOT)
    return NavSimBEVFusionDataset(
        split=split_dir,
        openscene_data_root=config["openscene_data_root"],
        nuplan_maps_root=config["nuplan_maps_root"],
        camera_order=config.get("camera_order"),
        image_hw=tuple(config.get("image_hw", (256, 704))),
        test_mode=False,  # GT needed: eval computes loss
        max_scenes=max_scenes,
        log_names=log_names,
        tokens=tokens,
        num_history_frames=int(config.get("num_history_frames", 1)),
        num_future_frames=int(config.get("num_future_frames", 0)),
    )


@torch.no_grad()
def run_loss_eval(raw_model, loader, device, scaler, max_batches):
    """Average detection-loss components over (a prefix of) an eval split.

    NaN-safe averaging
    ------------------
    Some val frames have zero positive matches; for those the head's cls/bbox
    losses divide by ``avg_factor = num_pos = 0`` and come back as NaN (the
    heatmap loss uses a different normalization -- peak count floored at 1 -- so
    it stays finite). A single NaN batch must NOT poison the running mean, so we
    accumulate, PER COMPONENT, the sum and count of only the FINITE (non-nan,
    non-inf) batch values. The reported average for a key is
    ``finite_sum / finite_count`` (or NaN if no batch contributed a finite value
    for that key).

    ``loss_total`` is defined as the SUM OF THE PER-COMPONENT FINITE MEANS (i.e.
    the sum of the individual component averages computed at the end), so it is
    always finite as long as at least one component (e.g. heatmap) is finite.
    It is intentionally NOT a per-batch sum, which would otherwise be NaN on any
    empty-GT batch. ``loss_total`` therefore has no finite-count of its own; its
    finite_count is reported as the max over the contributing components.

    Inline replica of ``runner._run_loss_eval`` with an added per-batch
    ``empty_cache`` for OOM headroom plus the NaN-safety above. Returns
    ``(avg_dict, finite_counts, n_batches)`` where ``finite_counts[key]`` is how
    many evaluated batches contributed a finite value for ``key``.
    """
    import math

    was_training = raw_model.training
    raw_model.eval()
    sums: dict = {}
    finite_counts: dict = {}
    count = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = move_to_device(raw_batch, device)
        img = batch.pop("img")
        try:
            with scaler.autocast():
                loss_dict = raw_model._forward_train(img=img, **batch)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  [batch {batch_index}] skipped (CUDA OOM): {exc}")
            continue
        # Accumulate per-component finite sums/counts only (skip nan/inf).
        for key, value in loss_dict.items():
            if not torch.is_tensor(value):
                continue
            fval = float(value)
            if math.isfinite(fval):
                sums[key] = sums.get(key, 0.0) + fval
                finite_counts[key] = finite_counts.get(key, 0) + 1
        count += 1
        del batch, img, loss_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if was_training:
        raw_model.train()
    if count == 0:
        return {}, {}, 0
    # Per-component finite mean (nan if no finite batch contributed).
    avg = {
        key: (sums[key] / finite_counts[key] if finite_counts.get(key) else float("nan"))
        for key in sums
    }
    # loss_total = sum of the per-component finite means (always finite if at
    # least one component is finite). Its finite_count = max component count.
    avg["loss_total"] = sum(v for v in avg.values() if v == v)  # skip nan
    finite_counts["loss_total"] = max(finite_counts.values()) if finite_counts else 0
    return avg, finite_counts, count


def main():
    parser = argparse.ArgumentParser(description="BEVFusion NAVSIM validation-loss evaluator")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint (.pth with 'model' key)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        choices=["val", "test"],
        help="Which split(s) to evaluate (default: val)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1, OOM-safe)")
    parser.add_argument("--max-batches", type=int, default=200, help="Max batches per split (default: 200)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap on scenes loaded per split")
    parser.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast (use fp32)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_runtime_config()
    device = choose_device(config.get("device", "auto"))
    print(f"Device: {device}")

    # Resolve checkpoint path (allow relative-to-repo).
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model exactly as the runner does, load weights (strict=False).
    print(f"Building BEVFusion(get_training_hyperparams()) ...")
    model = BEVFusion(get_training_hyperparams())
    model.to(device)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if isinstance(ckpt, dict):
        if "iter" in ckpt:
            print(f"  checkpoint iter={ckpt.get('iter')}, samples_seen={ckpt.get('samples_seen')}")
    del ckpt, state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scaler = Fp16Wrapper(enabled=(device.type == "cuda" and not args.no_amp), init_scale=512.0)

    results = {}
    for split_name in args.splits:
        print(f"\n=== Evaluating split: {split_name} ===")
        dataset = build_split_dataset(config["splits"][split_name], config, args.max_scenes)
        print(f"  dataset scenes: {len(dataset)}")
        loader = build_dataloader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn_override=collate_fn,
            sampler=None,
        )
        t0 = time.time()
        avg, finite_counts, n_batches = run_loss_eval(model, loader, device, scaler, args.max_batches)
        dt = time.time() - t0
        n_samples = n_batches * args.batch_size
        results[split_name] = (avg, finite_counts, n_batches, n_samples, dt)
        if not avg:
            print(f"  no batches evaluated for {split_name}")
        else:
            print(
                f"  {n_batches} batches / {n_samples} samples in {dt:.1f}s "
                f"({dt / max(1, n_batches):.2f}s/batch)"
            )

    # Final summary block.
    print("\n" + "=" * 64)
    print("BEVFUSION VALIDATION-LOSS SUMMARY")
    print("=" * 64)
    print(f"checkpoint : {ckpt_path}")
    print(f"batch_size : {args.batch_size}   max_batches: {args.max_batches}   amp: {not args.no_amp}")
    header = f"{'split':>6} | {'batches':>7} | {'samples':>7} | {'loss_total':>10} | {'loss_cls':>9} | {'loss_bbox':>9} | {'loss_heatmap':>12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for split_name in args.splits:
        avg, finite_counts, n_batches, n_samples, _ = results[split_name]
        if not avg:
            print(f"{split_name:>6} | {'--':>7} | {'--':>7} | {'(no batches)':>10}")
            continue
        print(
            f"{split_name:>6} | {n_batches:>7d} | {n_samples:>7d} | "
            f"{avg.get('loss_total', float('nan')):>10.4f} | "
            f"{avg.get('loss_cls', float('nan')):>9.4f} | "
            f"{avg.get('loss_bbox', float('nan')):>9.4f} | "
            f"{avg.get('loss_heatmap', float('nan')):>12.4f}"
        )
    print("=" * len(header))

    # Per-component finite-batch counts: how many evaluated batches yielded a
    # finite value for each component. A low finite_cls / finite_bbox count means
    # many frames had zero positive matches (empty-GT), so those losses were NaN
    # for those batches and were excluded from the mean above.
    print("\nFinite-batch counts (finite / evaluated) per component:")
    for split_name in args.splits:
        avg, finite_counts, n_batches, _, _ = results[split_name]
        if not avg:
            print(f"  {split_name:>6}: (no batches)")
            continue
        parts = " ".join(
            f"finite_{key.replace('loss_', '')}={finite_counts.get(key, 0)}/{n_batches}"
            for key in ("loss_cls", "loss_bbox", "loss_heatmap")
        )
        print(f"  {split_name:>6}: {parts}")


if __name__ == "__main__":
    main()
