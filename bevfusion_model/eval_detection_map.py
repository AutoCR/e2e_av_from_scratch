"""Standalone 3D-detection mAP / NDS evaluator for a trained BEVFusion checkpoint.

What this computes
------------------
Runs BEVFusion inference (the ``_forward_test`` path: ``model(img=..., **batch)``
with ``model.eval()``) over the NAVSIM VAL split, collects predicted boxes /
scores / labels and ground-truth boxes / labels (all in the LiDAR/BEV frame),
and computes a nuScenes-convention 3D detection score for the 5 NAVSIM classes:

    index -> name = {0: car, 1: barrier, 2: bicycle, 3: pedestrian, 4: traffic_cone}

(order taken from ``navsim_train/targets.py`` ``NAVSIM_OBJECT_CLASSES``).

Box layout (both pred and GT, LiDAR frame), 9 columns:
    [cx, cy, cz, w, l, h, yaw, vx, vy]
Only [cx, cy] (BEV center), [w, l, h] (size), and yaw are used for scoring.

Matching convention (nuScenes)
------------------------------
Matching is by 2D CENTER DISTANCE in the BEV plane (Euclidean over x,y), NOT IoU.
For each class and each distance threshold in {0.5, 1.0, 2.0, 4.0} m:
  - predictions of that class are sorted by descending score,
  - each prediction is greedily matched to the nearest still-unmatched GT of the
    same class whose center is within the threshold,
  - a precision-recall curve is accumulated, and AP is the average precision
    using the nuScenes interpolation: integrate the (recall, precision) curve
    over a uniform recall grid, after clipping recall < 0.1 and precision < 0.1
    to 0 and renormalizing by (1 - 0.1). (Use ``--voc-ap`` for plain VOC-style
    all-points AP without the 0.1 clipping.)
AP per class = mean AP over the 4 thresholds. mAP = mean AP over the 5 classes.

True-positive (TP) errors -- computed on matches at the 2.0 m threshold:
  - ATE: average BEV center L2 distance (meters) over matches.
  - ASE: average (1 - 3D IoU of size-aligned boxes). The boxes are aligned to a
    common center and common orientation (yaw removed), and IoU is computed from
    the size (w,l,h) overlap -- the standard nuScenes ASE. Implemented exactly:
    aligned 3D IoU = prod(min(dim_i))/(vol_a + vol_b - prod(min(dim_i))).
  - AOE: average yaw difference in radians, wrapped to [0, pi]. For the
    orientation-symmetric classes barrier / traffic_cone the error is taken mod
    pi (a barrier facing theta and theta+pi are equivalent).
We do NOT compute AVE (velocity) or AAE (attribute): NAVSIM is a 5-class
detection-only port with no attribute labels, so those nuScenes TP metrics are
omitted by design.

NDS (SIMPLIFIED)
----------------
True nuScenes NDS = (1/10) * (5*mAP + sum over 5 TP metrics of (1 - min(1, mTP))).
Here only 3 TP metrics exist (ATE, ASE, AOE -- no AVE/AAE), so we report:

    NDS_simplified = (1 / (5 + 3)) * (5*mAP + sum(1 - min(1, mTP)
                                              for mTP in [ATE, ASE, AOE]))

This is clearly labelled "NDS (simplified, 3 TP metrics)" in the output and is a
documented deviation from the official 5-TP-metric nuScenes NDS.

Pure torch/numpy: no nuscenes-devkit dependency.

Example (remote)
----------------
    cd ~/Code/e2e_av_from_scratch && \
      CUDA_VISIBLE_DEVICES=0 /home/pnc/.local/bin/uv run python \
      bevfusion_model/eval_detection_map.py --max-samples 100     # smoke

    cd ~/Code/e2e_av_from_scratch && \
      CUDA_VISIBLE_DEVICES=0 /home/pnc/.local/bin/uv run python \
      bevfusion_model/eval_detection_map.py --full                # full val split
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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
from bevfusion_model.navsim_train.runner_utils import (
    choose_device,
    move_to_device,
    resolve_split_config,
    set_seed,
)
from bevfusion_model.navsim_train.targets import NAVSIM_OBJECT_CLASSES

DEFAULT_CKPT = "bevfusion_model/outputs/train_navsim/iter_170208.pth"

CLASS_NAMES = list(NAVSIM_OBJECT_CLASSES)  # ["car","barrier","bicycle","pedestrian","traffic_cone"]
NUM_CLASSES = len(CLASS_NAMES)
DIST_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]
TP_DIST_THRESHOLD = 2.0  # nuScenes: TP errors measured at the 2.0 m match threshold

# Orientation-symmetric classes -> yaw error taken mod pi.
SYMMETRIC_CLASSES = {"barrier", "traffic_cone"}

# nuScenes AP curve clipping.
MIN_RECALL = 0.1
MIN_PRECISION = 0.1


def build_val_dataset(config, max_scenes):
    """Build the GT-bearing VAL dataset (test_mode=False so real GT is returned)."""
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


# --------------------------------------------------------------------------- #
# Geometry helpers (BEV-frame, numpy)
# --------------------------------------------------------------------------- #
def wrap_to_pi(angle):
    """Wrap radians to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def yaw_error(a, b, symmetric):
    """Absolute yaw difference in [0, pi] (or [0, pi/2] folded for symmetric classes)."""
    d = abs(wrap_to_pi(a - b))
    if symmetric:
        # theta and theta+pi are equivalent: fold the error into [0, pi/2].
        d = min(d, abs(np.pi - d))
    return float(d)


def aligned_3d_iou(dim_a, dim_b):
    """3D IoU of two boxes aligned to a common center and orientation (nuScenes ASE).

    With both boxes centered at the origin and axis-aligned, the intersection
    volume is prod(min(w), min(l), min(h)) and IoU follows directly.
    dim_* = (w, l, h).
    """
    wa, la, ha = float(dim_a[0]), float(dim_a[1]), float(dim_a[2])
    wb, lb, hb = float(dim_b[0]), float(dim_b[1]), float(dim_b[2])
    inter = max(0.0, min(wa, wb)) * max(0.0, min(la, lb)) * max(0.0, min(ha, hb))
    vol_a = max(0.0, wa) * max(0.0, la) * max(0.0, ha)
    vol_b = max(0.0, wb) * max(0.0, lb) * max(0.0, hb)
    denom = vol_a + vol_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


# --------------------------------------------------------------------------- #
# Per-class matching and AP
# --------------------------------------------------------------------------- #
def match_class(preds, gts, threshold):
    """Greedily match predictions to GT for one class at one distance threshold.

    Args:
        preds: list of dicts {center(2,), dim(3,), yaw, score} sorted by score desc.
        gts:   list of dicts {center(2,), dim(3,), yaw}; 'taken' tracked locally.
        threshold: center-distance threshold (m).
    Returns:
        tp (np.ndarray bool, len npred), fp (np.ndarray bool, len npred),
        scores (np.ndarray, len npred), match_idx (np.ndarray int, len npred,
        index into gts or -1).
    """
    npred = len(preds)
    tp = np.zeros(npred, dtype=bool)
    fp = np.zeros(npred, dtype=bool)
    scores = np.array([p["score"] for p in preds], dtype=np.float64)
    match_idx = -np.ones(npred, dtype=np.int64)
    taken = np.zeros(len(gts), dtype=bool)
    gt_centers = np.array([g["center"] for g in gts], dtype=np.float64) if gts else np.zeros((0, 2))
    for i, p in enumerate(preds):
        if len(gts) == 0:
            fp[i] = True
            continue
        d = np.linalg.norm(gt_centers - np.asarray(p["center"], dtype=np.float64)[None, :], axis=1)
        d_masked = np.where(taken, np.inf, d)
        j = int(np.argmin(d_masked))
        if d_masked[j] <= threshold:
            tp[i] = True
            taken[j] = True
            match_idx[i] = j
        else:
            fp[i] = True
    return tp, fp, scores, match_idx


def compute_ap(tp, fp, scores, n_gt, use_voc):
    """AP from per-prediction TP/FP, sorted by descending score.

    nuScenes style (default): interpolate precision over a uniform recall grid,
    zero out recall<MIN_RECALL and precision<MIN_PRECISION, renormalize.
    VOC style (--voc-ap): all-points area under the PR curve, no clipping.
    """
    if n_gt == 0:
        return float("nan")  # class not present in GT -> excluded from mAP mean
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp_c = np.cumsum(tp[order]).astype(np.float64)
    fp_c = np.cumsum(fp[order]).astype(np.float64)
    recall = tp_c / float(n_gt)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)

    if use_voc:
        # VOC all-points: envelope of precision, integrate over recall steps.
        mrec = np.concatenate(([0.0], recall, [recall[-1]]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    # nuScenes interpolation over a 101-point recall grid.
    rec_grid = np.linspace(0.0, 1.0, 101)
    # precision envelope (monotonic decreasing in recall, from the right)
    prec_env = np.copy(precision)
    for k in range(len(prec_env) - 2, -1, -1):
        prec_env[k] = max(prec_env[k], prec_env[k + 1])
    # interpolate precision at each recall grid point: precision of the first
    # prediction whose recall >= grid point.
    interp = np.zeros_like(rec_grid)
    for gi, r in enumerate(rec_grid):
        inds = np.where(recall >= r)[0]
        interp[gi] = prec_env[inds[0]] if len(inds) else 0.0
    # nuScenes clipping: drop the low-recall / low-precision corner, renormalize.
    interp = interp[rec_grid >= MIN_RECALL]
    interp = np.where(interp < MIN_PRECISION, 0.0, interp - MIN_PRECISION)
    ap = float(np.mean(interp)) / (1.0 - MIN_PRECISION)
    return ap


# --------------------------------------------------------------------------- #
# Inference + collection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(model, loader, device, max_samples):
    """Run inference; return per-class accumulators of preds and GT.

    preds_by_class[c]: list of dicts {center, dim, yaw, score}
    gts_by_class[c]:   list of dicts {center, dim, yaw, sample}   (sample = running idx)
    """
    model.eval()
    preds_by_class = [[] for _ in range(NUM_CLASSES)]
    gts_by_class = [[] for _ in range(NUM_CLASSES)]
    n_done = 0
    for batch in loader:
        if max_samples is not None and n_done >= max_samples:
            break
        gt_boxes_list = batch["gt_bboxes_3d"]
        gt_labels_list = batch["gt_labels_3d"]
        model_batch = move_to_device(batch, device)
        img = model_batch.pop("img")
        # _forward_test ignores gt_* via **kwargs; pass the full batch.
        try:
            outputs = model(img=img, **model_batch)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  [sample {n_done}] inference skipped (CUDA OOM): {exc}")
            n_done += len(gt_boxes_list)
            continue

        for bi, out in enumerate(outputs):
            sample_id = n_done + bi
            pb = out["boxes_3d"].cpu().numpy()
            ps = out["scores_3d"].cpu().numpy()
            pl = out["labels_3d"].cpu().numpy().astype(int)
            for k in range(pb.shape[0]):
                c = int(pl[k])
                if c < 0 or c >= NUM_CLASSES:
                    continue
                box = pb[k]
                preds_by_class[c].append(
                    {
                        "center": box[0:2].astype(np.float64),
                        "dim": box[3:6].astype(np.float64),  # w, l, h
                        "yaw": float(box[6]),
                        "score": float(ps[k]),
                        "sample": sample_id,
                    }
                )
            gb = gt_boxes_list[bi].cpu().numpy() if torch.is_tensor(gt_boxes_list[bi]) else np.asarray(gt_boxes_list[bi])
            gl = gt_labels_list[bi].cpu().numpy().astype(int) if torch.is_tensor(gt_labels_list[bi]) else np.asarray(gt_labels_list[bi]).astype(int)
            for k in range(gb.shape[0]):
                c = int(gl[k])
                if c < 0 or c >= NUM_CLASSES:
                    continue
                box = gb[k]
                gts_by_class[c].append(
                    {
                        "center": box[0:2].astype(np.float64),
                        "dim": box[3:6].astype(np.float64),
                        "yaw": float(box[6]),
                        "sample": sample_id,
                    }
                )
        n_done += len(outputs)
        del model_batch, img, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return preds_by_class, gts_by_class, n_done


def evaluate(preds_by_class, gts_by_class, use_voc):
    """Compute per-class/per-threshold AP, mAP, and the 3 TP errors.

    Matching is done PER SAMPLE (a prediction can only match GT from its own
    sample) to respect the nuScenes per-frame convention.
    """
    per_thresh_ap = np.full((NUM_CLASSES, len(DIST_THRESHOLDS)), np.nan)
    tp_records = {"ate": [], "ase": [], "aoe": []}  # accumulated over all classes

    for c in range(NUM_CLASSES):
        symmetric = CLASS_NAMES[c] in SYMMETRIC_CLASSES
        # group preds and gts by sample
        preds_by_sample = {}
        gts_by_sample = {}
        for p in preds_by_class[c]:
            preds_by_sample.setdefault(p["sample"], []).append(p)
        for g in gts_by_class[c]:
            gts_by_sample.setdefault(g["sample"], []).append(g)
        n_gt_total = len(gts_by_class[c])
        all_samples = set(preds_by_sample) | set(gts_by_sample)

        for ti, thr in enumerate(DIST_THRESHOLDS):
            tp_all, fp_all, score_all = [], [], []
            for s in all_samples:
                preds = sorted(preds_by_sample.get(s, []), key=lambda x: -x["score"])
                gts = gts_by_sample.get(s, [])
                tp, fp, scores, match_idx = match_class(preds, gts, thr)
                tp_all.append(tp)
                fp_all.append(fp)
                score_all.append(scores)
                # Collect TP errors only at the dedicated TP threshold.
                if abs(thr - TP_DIST_THRESHOLD) < 1e-9:
                    for pi, mi in enumerate(match_idx):
                        if mi < 0:
                            continue
                        pred = preds[pi]
                        g = gts[mi]
                        ate = float(np.linalg.norm(np.asarray(pred["center"]) - np.asarray(g["center"])))
                        ase = 1.0 - aligned_3d_iou(pred["dim"], g["dim"])
                        aoe = yaw_error(pred["yaw"], g["yaw"], symmetric)
                        tp_records["ate"].append(ate)
                        tp_records["ase"].append(ase)
                        tp_records["aoe"].append(aoe)
            tp_cat = np.concatenate(tp_all) if tp_all else np.zeros(0, dtype=bool)
            fp_cat = np.concatenate(fp_all) if fp_all else np.zeros(0, dtype=bool)
            score_cat = np.concatenate(score_all) if score_all else np.zeros(0)
            per_thresh_ap[c, ti] = compute_ap(tp_cat, fp_cat, score_cat, n_gt_total, use_voc)

    # Per-class AP = mean over thresholds (nan-safe). mAP = mean over present classes.
    per_class_ap = np.nanmean(per_thresh_ap, axis=1)
    mean_ap = float(np.nanmean(per_class_ap)) if np.any(~np.isnan(per_class_ap)) else 0.0

    mate = float(np.mean(tp_records["ate"])) if tp_records["ate"] else float("nan")
    mase = float(np.mean(tp_records["ase"])) if tp_records["ase"] else float("nan")
    maoe = float(np.mean(tp_records["aoe"])) if tp_records["aoe"] else float("nan")
    return per_thresh_ap, per_class_ap, mean_ap, (mate, mase, maoe), len(tp_records["ate"])


def simplified_nds(mean_ap, mate, mase, maoe):
    """NDS_simplified = (1/(5+3)) * (5*mAP + sum(1 - min(1, mTP) for 3 TP metrics)).

    NaN TP errors (no matches) contribute 0 to the TP sum.
    """
    tp_terms = 0.0
    for m in (mate, mase, maoe):
        if m != m:  # NaN
            tp_terms += 0.0
        else:
            tp_terms += 1.0 - min(1.0, m)
    return (5.0 * mean_ap + tp_terms) / (5.0 + 3.0)


def main():
    parser = argparse.ArgumentParser(description="BEVFusion NAVSIM 3D-detection mAP/NDS evaluator")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint (.pth with 'model' key)")
    parser.add_argument("--max-samples", type=int, default=500, help="Max VAL samples to evaluate (default: 500)")
    parser.add_argument("--full", action="store_true", help="Evaluate the entire VAL split (ignores --max-samples)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1, OOM-safe)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap on scenes loaded")
    parser.add_argument("--voc-ap", action="store_true", help="Use plain VOC all-points AP (no nuScenes 0.1 clipping)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_runtime_config()
    device = choose_device(config.get("device", "auto"))
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Building BEVFusion(get_training_hyperparams()) ...")
    model = BEVFusion(get_training_hyperparams())
    model.to(device)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if isinstance(ckpt, dict) and "iter" in ckpt:
        print(f"  checkpoint iter={ckpt.get('iter')}, samples_seen={ckpt.get('samples_seen')}")
    del ckpt, state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    max_samples = None if args.full else args.max_samples
    dataset = build_val_dataset(config, args.max_scenes)
    print(f"VAL dataset scenes: {len(dataset)}  (evaluating {'all' if max_samples is None else max_samples})")
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn_override=collate_fn,
        sampler=None,
    )

    t0 = time.time()
    preds_by_class, gts_by_class, n_done = collect_predictions(model, loader, device, max_samples)
    infer_dt = time.time() - t0
    per_thresh_ap, per_class_ap, mean_ap, (mate, mase, maoe), n_tp = evaluate(
        preds_by_class, gts_by_class, args.voc_ap
    )
    nds = simplified_nds(mean_ap, mate, mase, maoe)

    # ----------------------------- report ----------------------------------- #
    n_pred = sum(len(x) for x in preds_by_class)
    n_gt = sum(len(x) for x in gts_by_class)
    print("\n" + "=" * 72)
    print("BEVFUSION 3D-DETECTION EVALUATION (nuScenes center-distance convention)")
    print("=" * 72)
    print(f"checkpoint   : {ckpt_path}")
    print(f"samples eval : {n_done}   preds: {n_pred}   gts: {n_gt}   infer: {infer_dt:.1f}s")
    print(f"AP mode      : {'VOC all-points' if args.voc_ap else 'nuScenes (min_recall=0.1, min_precision=0.1)'}")
    print(f"dist thresh  : {DIST_THRESHOLDS} m    TP errors at {TP_DIST_THRESHOLD} m  (matched TPs: {n_tp})")

    # Per-class / per-threshold AP table.
    print("\nPer-class AP (rows=class, cols=distance threshold):")
    head = f"{'class':>13} | " + " | ".join(f"{thr:>6.1f}m" for thr in DIST_THRESHOLDS) + f" | {'AP':>6}"
    print(head)
    print("-" * len(head))
    for c in range(NUM_CLASSES):
        cells = " | ".join(
            ("  nan " if np.isnan(per_thresh_ap[c, ti]) else f"{per_thresh_ap[c, ti]:>6.3f}")
            for ti in range(len(DIST_THRESHOLDS))
        )
        ap_c = "  nan " if np.isnan(per_class_ap[c]) else f"{per_class_ap[c]:>6.3f}"
        print(f"{CLASS_NAMES[c]:>13} | {cells} | {ap_c}")
    print("-" * len(head))

    def fmt(x):
        return "nan" if (x != x) else f"{x:.4f}"

    print(f"\nmAP                         : {mean_ap:.4f}")
    print(f"mATE (avg trans err, m)     : {fmt(mate)}")
    print(f"mASE (avg 1 - aligned 3DIoU): {fmt(mase)}")
    print(f"mAOE (avg yaw err, rad)     : {fmt(maoe)}")
    print(f"NDS (simplified, 3 TP metrics): {nds:.4f}")
    print("=" * 72)
    print("NOTE: NDS here uses mAP + (ATE, ASE, AOE) only -- AVE/AAE omitted (NAVSIM")
    print("      5-class, no velocity/attribute eval). This is a documented")
    print("      simplification of the official nuScenes NDS (5 TP metrics).")


if __name__ == "__main__":
    main()
