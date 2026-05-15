#!/bin/bash
# Generate K-means anchors from NavSim dataset.
# Override defaults via environment variables:
#   SPLIT=trainval DATA_PATH=/path/to/navsim/dataset bash scripts/kmeans.sh
SPLIT=${SPLIT:-mini}
DATA_PATH=${DATA_PATH:-/Users/chenran/Code/navsim/dataset}
OUT_DIR=${OUT_DIR:-data/kmeans}

python tools/kmeans/kmeans_det.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR"
python tools/kmeans/kmeans_map.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR"
python tools/kmeans/kmeans_motion.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR"
python tools/kmeans/kmeans_plan.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR"
