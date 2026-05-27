#!/bin/bash
# Generate K-means anchors from NavSim dataset.
# Override defaults via environment variables:
#   SPLIT=trainval DATA_PATH=/path/to/navsim/dataset bash scripts/kmeans.sh
SPLIT=${SPLIT:-mini}
DATA_PATH=${DATA_PATH:-/Users/chenran/Code/navsim/dataset}
OUT_DIR=${OUT_DIR:-data/kmeans}

# Prevent OpenBLAS from allocating per-process thread pools across the 128+
# multiprocessing workers, which exhausts the system's shared memory regions
# and causes a segfault. Data loading doesn't use BLAS; KMeans (K<=6) is
# trivially fast single-threaded.
export OPENBLAS_NUM_THREADS=1

python tools/kmeans/kmeans_det.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR"
python tools/kmeans/kmeans_map.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR" --maps_root "$DATA_PATH/maps"
python tools/kmeans/kmeans_motion.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR"
python tools/kmeans/kmeans_plan.py --data_path "$DATA_PATH" --split "$SPLIT" --out_dir "$OUT_DIR"
