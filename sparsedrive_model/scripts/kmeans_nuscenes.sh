#!/bin/bash
set -euo pipefail

# Generate K-means anchors from local nuScenes.
# Override defaults via environment variables:
#   DET_K=4 MAP_K=20 bash scripts/kmeans_nuscenes.sh
DATA_PATH=${DATA_PATH:-/Users/chenran/Code/nuscenes/nuscenes}
VERSION=${VERSION:-v1.0-mini}
OUT_DIR=${OUT_DIR:-data/kmeans_nuscenes}

DET_K=${DET_K:-900}
DET_DIS_THRESH=${DET_DIS_THRESH:-55.0}
DET_WORKERS=${DET_WORKERS:-1}

MAP_K=${MAP_K:-100}
MAP_NUM_SAMPLE=${MAP_NUM_SAMPLE:-20}

MOTION_K=${MOTION_K:-6}
MOTION_DIS_THRESH=${MOTION_DIS_THRESH:-55.0}
MOTION_FUTURE_STEPS=${MOTION_FUTURE_STEPS:-12}
MOTION_SCARCITY=${MOTION_SCARCITY:-repeat}

PLAN_K=${PLAN_K:-6}
PLAN_FUTURE_STEPS=${PLAN_FUTURE_STEPS:-6}
PLAN_TURN_THRESH=${PLAN_TURN_THRESH:-2.0}

uv run --no-sync python tools/kmeans/kmeans_nuscenes_det.py --data_path "$DATA_PATH" --version "$VERSION" --out_dir "$OUT_DIR" --k "$DET_K" --dis_thresh "$DET_DIS_THRESH" --workers "$DET_WORKERS"
uv run --no-sync python tools/kmeans/kmeans_nuscenes_map.py --data_path "$DATA_PATH" --version "$VERSION" --out_dir "$OUT_DIR" --k "$MAP_K" --num_sample "$MAP_NUM_SAMPLE"
uv run --no-sync python tools/kmeans/kmeans_nuscenes_motion.py --data_path "$DATA_PATH" --version "$VERSION" --out_dir "$OUT_DIR" --k "$MOTION_K" --dis_thresh "$MOTION_DIS_THRESH" --future_steps "$MOTION_FUTURE_STEPS" --scarcity "$MOTION_SCARCITY"
uv run --no-sync python tools/kmeans/kmeans_nuscenes_plan.py --data_path "$DATA_PATH" --version "$VERSION" --out_dir "$OUT_DIR" --k "$PLAN_K" --future_steps "$PLAN_FUTURE_STEPS" --turn_thresh "$PLAN_TURN_THRESH"
