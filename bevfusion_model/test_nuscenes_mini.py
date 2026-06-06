"""BEVFusion nuScenes mini inference test.

Loads bevfusion-det.pth checkpoint, runs inference on nuScenes mini dataset,
saves visualizations and detection summaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BEVFUSION_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, BEVFUSION_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bevfusion_model.visualization import visualize_sample

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NUSCENES_DATA_ROOT = "/Users/chenran/Code/nuscenes/nuscenes"
NUSCENES_VERSION = "v1.0-mini"
CHECKPOINT_PATH = "model_weights/bevfusion/bevfusion-det.pth"
LIMIT = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("bevfusion_model/outputs/nuscenes_mini_inference")

CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

OBJECT_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# ──────────────────────────────────────────────────────────────────────────────
# Model building and checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def _build_model() -> torch.nn.Module:
    from bevfusion_model.configs.bevfusion_hyperparams import get_fusion_hyperparams
    from bevfusion_model.bevfusion import BEVFusion
    hp = get_fusion_hyperparams()
    model = BEVFusion(hp)
    model.eval()
    return model


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(sd, strict=False)

    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)
    lidar_unexpected = [k for k in result.unexpected_keys if "lidar" in k]
    other_unexpected = [k for k in result.unexpected_keys if "lidar" not in k]

    print(f"  Checkpoint loaded: {n_missing} missing, {n_unexpected} unexpected keys")
    if n_missing > 0:
        print(f"  WARNING — Missing keys: {result.missing_keys[:5]}")
    if other_unexpected:
        print(f"  WARNING — Unexpected non-lidar keys: {other_unexpected[:5]}")
    if lidar_unexpected:
        print(f"  (Lidar backbone keys skipped: {len(lidar_unexpected)} — no spconv on this platform)")

    return {"missing": result.missing_keys, "unexpected": result.unexpected_keys}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_samples(limit: int) -> list[Any]:
    from bevfusion_model.nuscenes_adapter import load_bevfusion_samples
    print(f"  Loading nuScenes data from {NUSCENES_DATA_ROOT} ({NUSCENES_VERSION})")
    samples = load_bevfusion_samples(
        dataset_root=NUSCENES_DATA_ROOT,
        version=NUSCENES_VERSION,
        max_samples=limit,
        camera_order=CAMERA_ORDER,
        image_hw=(256, 704),
    )
    print(f"  Loaded {len(samples)} samples")
    return samples


def _collate_to_device(samples: list[Any], device: str) -> dict[str, Any]:
    from bevfusion_model.nuscenes_adapter import collate_bevfusion_samples, sample_to_device
    batch = collate_bevfusion_samples(samples)
    batch = sample_to_device(batch, device)
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def _run_inference(model: torch.nn.Module, batch: dict[str, Any]) -> list[dict[str, Any]]:
    with torch.no_grad():
        outputs = model(
            img=batch["img"],
            points=batch["points"],
            camera2ego=batch["camera2ego"],
            lidar2ego=batch["lidar2ego"],
            lidar2camera=batch["lidar2camera"],
            lidar2image=batch["lidar2image"],
            camera_intrinsics=batch["camera_intrinsics"],
            camera2lidar=batch["camera2lidar"],
            img_aug_matrix=batch["img_aug_matrix"],
            lidar_aug_matrix=batch["lidar_aug_matrix"],
            metas=batch["metas"],
        )
    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Result decoding and summarization
# ──────────────────────────────────────────────────────────────────────────────

def _decode_output(output: dict[str, Any]) -> dict[str, Any]:
    boxes = output["boxes_3d"].numpy()   # (N, 9+) [x, y, z, l, w, h, yaw, vx, vy]
    scores = output["scores_3d"].numpy()  # (N,)
    labels = output["labels_3d"].numpy()  # (N,)

    detections = []
    n_dets = len(scores) if len(scores) > 0 else 0
    for i in range(n_dets):
        det = {
            "score": float(scores[i]),
            "label": int(labels[i]),
            "class": OBJECT_CLASSES[int(labels[i])] if int(labels[i]) < len(OBJECT_CLASSES) else "unknown",
            "box": boxes[i].tolist() if boxes.size > 0 and boxes.ndim > 1 else [],
        }
        detections.append(det)

    detections.sort(key=lambda d: d["score"], reverse=True)
    return {
        "num_detections": len(detections),
        "detections": detections[:20],  # top-20 for readability
    }


def _summarize_sample(sample: Any, output: dict[str, Any]) -> dict[str, Any]:
    try:
        decoded = _decode_output(output)
    except Exception as e:
        decoded = {
            "num_detections": 0,
            "detections": [],
            "error": str(e),
        }
    return {
        "token": sample.token,
        "scene": sample.scene_name,
        "timestamp": sample.timestamp,
        **decoded,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEVFusion nuScenes Mini Inference Test")
    print("=" * 60)

    # 1. Build model
    print("\n[1/4] Building model...")
    model = _build_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    model = model.to(DEVICE)

    # 2. Load checkpoint
    print(f"\n[2/4] Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt_info = _load_checkpoint(model, CHECKPOINT_PATH)

    # 3. Load data
    print(f"\n[3/4] Loading nuScenes data (limit={LIMIT})...")
    samples = _load_samples(LIMIT)
    if not samples:
        print("  ERROR: No samples loaded!")
        return

    # 4. Run inference sample by sample
    print(f"\n[4/4] Running inference on {len(samples)} sample(s)...")
    summaries = []

    for idx, sample in enumerate(samples):
        print(f"\n  Sample {idx + 1}/{len(samples)}: {sample.token[:16]}...")
        print(f"    Scene: {sample.scene_name}")
        print(f"    Points: {sample.points.shape[0]:,}")
        print(f"    Images: {sample.img.shape}")

        # Collate single sample into batch
        batch = _collate_to_device([sample], DEVICE)

        try:
            outputs = _run_inference(model, batch)
            output = outputs[0]

            boxes = output["boxes_3d"]
            scores = output["scores_3d"]
            labels = output["labels_3d"]
            print(f"    Detections: {len(scores)} total")
            if len(scores) > 0:
                print(f"    Score range: [{scores.min():.3f}, {scores.max():.3f}]")
                high_conf = scores > 0.3
                print(f"    High-confidence (>0.3): {high_conf.sum()}")
                # Show top-5 detections
                top_idx = scores.argsort(descending=True)[:5]
                for rank, i in enumerate(top_idx):
                    cls = OBJECT_CLASSES[int(labels[i])] if int(labels[i]) < len(OBJECT_CLASSES) else "?"
                    print(f"      [{rank+1}] {cls}: score={scores[i]:.3f}")

            summary = _summarize_sample(sample, output)
            summaries.append(summary)

            visualize_sample(sample, output, OUTPUT_DIR, idx, camera_order=CAMERA_ORDER)

        except Exception as e:
            print(f"    ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
            summaries.append({"token": sample.token, "error": str(e)})

    # Save summaries to JSON
    summary_path = OUTPUT_DIR / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
