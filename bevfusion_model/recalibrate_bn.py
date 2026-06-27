"""Standalone BatchNorm-recalibration for a BEVFusion NAVSIM checkpoint.

Why this exists
---------------
The training checkpoint ``bevfusion_model/outputs/train_navsim/iter_170208.pth``
(a dict whose weights live under the ``"model"`` key) has CORRUPT (NaN)
BatchNorm running statistics in all 6 prediction-head branches of the detection
head::

    heads.object.prediction_heads.0.{center,height,dim,rot,vel,heatmap}.0.bn.running_mean
    heads.object.prediction_heads.0.{center,height,dim,rot,vel,heatmap}.0.bn.running_var

All learnable PARAMETERS (weights/biases, including the BN ``weight``/``bias``)
are finite -- only these BN *buffers* (``running_mean`` / ``running_var``) are
NaN. The cause is fp16 BN-stat divergence during training. In ``eval()`` mode
BatchNorm normalizes with the (NaN) running stats, which poisons every head
prediction and yields 0 detections. The fix, proven empirically, is to RESET the
corrupt BN buffers and re-estimate them with a handful of forward passes over
real data in BN-train mode (NO backward / NO gradients).

What this script does
---------------------
1. Builds ``BEVFusion(get_training_hyperparams())`` exactly as the eval scripts
   (``eval_loss.py`` / ``eval_detection_map.py``) do, loads ``ckpt["model"]``
   with ``strict=False``, and moves it to CUDA.
2. Scans EVERY BatchNorm{1,2,3}d module and reports which ones have non-finite
   running stats. It then RECALIBRATES ALL BatchNorm modules (see "Design
   choices" below) by:
     - ``reset_running_stats()`` on each (running_mean=0, running_var=1,
       num_batches_tracked=0) -- this is mandatory for the NaN ones and harmless
       for the finite ones,
     - setting ``momentum = None`` so BatchNorm uses a CUMULATIVE moving average
       (true running mean over all batches seen) instead of an exponential one,
       giving a stable estimate over the recalibration set,
     - putting the whole model in ``.eval()`` and then flipping ONLY the BN
       modules back to ``.train()`` so they (and nothing else) update their
       running stats during the forward passes.
3. Runs ``--num-batches`` forward passes over the VAL split under
   ``torch.no_grad()`` using the model's INFERENCE path,
   ``model(img=img, **batch)`` (``BEVFusion._forward_test``). That path calls
   ``detection_head.forward_single`` -> ``self.prediction_heads[i](query_feat)``
   (detection_head.py:496) whose outputs ``get_bboxes`` consumes, so data
   provably flows through the prediction-head BNs and their stats accumulate.
   We use ``_forward_test`` (not ``_forward_train``) because it needs no GT and
   exercises exactly the BNs that are corrupt; it is also the path that breaks
   at eval time, so recalibrating under it is the most faithful.
4. Verifies that NO BN buffer is non-finite anymore (before/after NaN counts).
5. Saves a NEW, minimal checkpoint preserving the original top-level metadata
   (``iter`` / ``samples_seen`` / any other non-``model`` keys, EXCEPT a bulky
   ``optimizer`` state which is dropped) with the recalibrated ``model``
   state_dict and a ``bn_recalibrated=True`` flag.

Design choices (documented)
---------------------------
- RECALIBRATE ALL BN, not just the corrupt head BNs. A forward pass over a few
  hundred batches is cheap, and recomputing every BN's stats with a consistent
  cumulative average yields one clean, internally-consistent set of statistics
  rather than mixing freshly-estimated head stats with the (differently-trained)
  backbone stats. The corrupt head BNs MUST be reset; resetting the rest too is
  the conservative, fully-consistent choice. (If you want to recalibrate ONLY
  the NaN ones, pass ``--only-nan``.)
- ``momentum=None`` (cumulative average) only on the modules being recalibrated.
- NO autocast/fp16 here: BN stats are accumulated in fp32 to avoid reintroducing
  the very fp16 divergence that corrupted them. (Inputs still flow through the
  rest of the net in fp32; this is a stat-estimation pass, not a speed path.)

Example (remote)
----------------
    cd ~/Code/e2e_av_from_scratch && \
      CUDA_VISIBLE_DEVICES=0 /home/pnc/.local/bin/uv run python \
      bevfusion_model/recalibrate_bn.py --num-batches 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

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
from bevfusion_model.navsim_train.runner_utils import (
    choose_device,
    move_to_device,
    resolve_split_config,
    set_seed,
)

DEFAULT_CKPT = "bevfusion_model/outputs/train_navsim/iter_170208.pth"
DEFAULT_OUT = "bevfusion_model/outputs/train_navsim/iter_170208_bnfix.pth"

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def build_val_dataset(config, max_scenes):
    """Build the GT-bearing VAL dataset (test_mode=False -> real data), exactly
    as eval_detection_map.build_val_dataset does. GT is unused by the inference
    forward path but test_mode=False guarantees we run over real validation
    frames (not synthetic/test placeholders)."""
    split_dir, log_names, tokens = resolve_split_config(config["splits"]["val"], _REPO_ROOT)
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
        num_history_frames=int(config.get("num_history_frames", 1)),
        num_future_frames=int(config.get("num_future_frames", 0)),
    )


def _buffer_is_nonfinite(t):
    return (t is not None) and torch.is_tensor(t) and (not torch.isfinite(t).all().item())


def bn_module_is_nonfinite(m):
    """True if this BN module's running_mean or running_var has any non-finite value."""
    return _buffer_is_nonfinite(getattr(m, "running_mean", None)) or _buffer_is_nonfinite(
        getattr(m, "running_var", None)
    )


def scan_bn(model):
    """Return (all_bn, nan_bn) lists of (name, module) for every BN module and
    those with non-finite running stats."""
    all_bn, nan_bn = [], []
    for name, m in model.named_modules():
        if isinstance(m, _BN_TYPES):
            all_bn.append((name, m))
            if bn_module_is_nonfinite(m):
                nan_bn.append((name, m))
    return all_bn, nan_bn


def count_nonfinite_bn_buffers(model):
    """Count how many BN running-stat buffers (running_mean/running_var) are non-finite."""
    n = 0
    for _, m in scan_bn(model)[0]:
        if _buffer_is_nonfinite(getattr(m, "running_mean", None)):
            n += 1
        if _buffer_is_nonfinite(getattr(m, "running_var", None)):
            n += 1
    return n


@torch.no_grad()
def recalibrate(model, loader, device, num_batches):
    """Run forward passes to re-estimate BN running stats. Returns n_batches used."""
    n = 0
    t0 = time.time()
    for batch_index, raw_batch in enumerate(loader):
        if num_batches is not None and n >= int(num_batches):
            break
        batch = move_to_device(raw_batch, device)
        # GT keys are ignored by _forward_test's **kwargs; pop img like the eval scripts.
        img = batch.pop("img")
        try:
            # _forward_test path -> forward_single -> prediction_heads[i](...) -> BN updates.
            _ = model(img=img, **batch)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  [batch {batch_index}] skipped (CUDA OOM): {exc}")
            del batch, img
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        n += 1
        del batch, img
        if device.type == "cuda" and (n % 10 == 0):
            torch.cuda.empty_cache()
        if n % 25 == 0:
            print(f"  recalibrated over {n} batches ({time.time() - t0:.1f}s)")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Recalibrate (re-estimate) corrupt BatchNorm running stats of a BEVFusion checkpoint"
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Input checkpoint (.pth with 'model' key)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output checkpoint path")
    parser.add_argument("--num-batches", type=int, default=300, help="Forward passes over VAL (default: 300)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1, OOM-safe)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap on scenes loaded")
    parser.add_argument(
        "--only-nan",
        action="store_true",
        help="Recalibrate ONLY the BN modules with non-finite stats (default: recalibrate ALL BN)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_runtime_config()
    device = choose_device(config.get("device", "auto"))
    print(f"Device: {device}")

    # ---- resolve + load checkpoint --------------------------------------- #
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path

    print("Building BEVFusion(get_training_hyperparams()) ...")
    model = BEVFusion(get_training_hyperparams())
    model.to(device)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # Preserve original top-level metadata (everything except 'model' and a bulky
    # 'optimizer'); record original iter/samples_seen explicitly.
    orig_meta = {}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if k in ("model", "optimizer"):
                continue
            orig_meta[k] = v
    orig_iter = orig_meta.get("iter")
    orig_samples = orig_meta.get("samples_seen")
    print(f"  checkpoint iter={orig_iter}, samples_seen={orig_samples}")
    del ckpt, state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- scan BN BEFORE -------------------------------------------------- #
    all_bn, nan_bn = scan_bn(model)
    n_buf_nan_before = count_nonfinite_bn_buffers(model)
    print(f"\nBN scan (before): {len(all_bn)} BatchNorm modules total, "
          f"{len(nan_bn)} with NON-FINITE running stats ({n_buf_nan_before} corrupt buffers).")
    if nan_bn:
        print("  Corrupt BN modules:")
        for name, _ in nan_bn:
            print(f"    NaN  {name}")
    else:
        print("  (no corrupt BN modules found -- nothing strictly needs reset)")

    # ---- choose targets + reset + cumulative-average momentum ------------- #
    if args.only_nan:
        targets = nan_bn
        print(f"\nRecalibration target: ONLY the {len(targets)} corrupt BN modules (--only-nan).")
    else:
        targets = all_bn
        print(f"\nRecalibration target: ALL {len(targets)} BN modules "
              f"(clean, fully-consistent stats; corrupt ones are a subset).")

    target_ids = set()
    for name, m in targets:
        m.reset_running_stats()      # running_mean=0, running_var=1, num_batches_tracked=0
        m.momentum = None            # cumulative moving average over the recalibration set
        target_ids.add(id(m))

    # eval() everything (deterministic), then flip ONLY target BN modules to train()
    # so they -- and nothing else -- accumulate running stats during forward.
    model.eval()
    n_train_bn = 0
    for _, m in all_bn:
        if id(m) in target_ids:
            m.train()
            n_train_bn += 1
    print(f"model.eval(); {n_train_bn} target BN modules switched to train() to accumulate stats.")

    # ---- data loader ----------------------------------------------------- #
    dataset = build_val_dataset(config, args.max_scenes)
    print(f"VAL dataset scenes: {len(dataset)}  (running up to {args.num_batches} batches)")
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn_override=collate_fn,
        sampler=None,
    )

    # ---- recalibrate ----------------------------------------------------- #
    print("\nRecalibrating BN running stats via _forward_test (model(img=..., **batch)) ...")
    t0 = time.time()
    n_used = recalibrate(model, loader, device, args.num_batches)
    print(f"Recalibration forward passes: {n_used} batches in {time.time() - t0:.1f}s")

    # ---- verify AFTER ---------------------------------------------------- #
    model.eval()  # back to full eval mode for the verification scan
    _, nan_bn_after = scan_bn(model)
    n_buf_nan_after = count_nonfinite_bn_buffers(model)
    print(f"\nBN scan (after): {len(nan_bn_after)} modules still non-finite "
          f"({n_buf_nan_after} corrupt buffers remaining).")
    if nan_bn_after:
        print("  STILL NON-FINITE:")
        for name, _ in nan_bn_after:
            print(f"    NaN  {name}")
    assert n_buf_nan_after == 0, (
        f"Recalibration FAILED: {n_buf_nan_after} BN buffers are still non-finite. "
        "Likely some corrupt BN was never exercised by the forward path -- "
        "increase --num-batches or check that data flowed through it."
    )
    print("  OK: all BN running stats are finite.")

    # ---- save minimal recalibrated checkpoint ---------------------------- #
    new_ckpt = dict(orig_meta)  # carry iter/samples_seen + any other small metadata
    new_ckpt["model"] = model.state_dict()
    new_ckpt["iter"] = orig_iter
    new_ckpt["samples_seen"] = orig_samples
    new_ckpt["bn_recalibrated"] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, str(out_path))
    print(f"\nSaved recalibrated checkpoint -> {out_path}")

    # ---- final summary --------------------------------------------------- #
    print("\n" + "=" * 64)
    print("BN RECALIBRATION SUMMARY")
    print("=" * 64)
    print(f"input  ckpt        : {ckpt_path}")
    print(f"output ckpt        : {out_path}")
    print(f"batches used       : {n_used} (batch_size={args.batch_size})")
    print(f"BN modules total   : {len(all_bn)}")
    print(f"BN recalibrated    : {len(targets)} "
          f"({'only NaN' if args.only_nan else 'all'})")
    print(f"NaN BN before/after: {len(nan_bn)} / {len(nan_bn_after)} modules "
          f"({n_buf_nan_before} / {n_buf_nan_after} buffers)")
    print(f"iter / samples_seen: {orig_iter} / {orig_samples}")
    print("=" * 64)
    print(f"now run: bevfusion_model/eval_detection_map.py --ckpt {out_path}")


if __name__ == "__main__":
    main()
