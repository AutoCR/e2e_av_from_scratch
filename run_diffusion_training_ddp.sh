#!/usr/bin/env bash
# Launch phase-2 diffusion training with DDP on all visible GPUs.
#
# Usage:
#   ./run_diffusion_training_ddp.sh              # uses all 8 GPUs
#   NPROC_PER_NODE=4 ./run_diffusion_training_ddp.sh   # override GPU count

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Dataset / experiment paths consumed by diffusion_training.py and navsim imports.
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/prediction_database/nuplan/dataset/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/prediction_database/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SCRIPT_DIR}/exp}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "${SCRIPT_DIR}"

exec torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node "${NPROC_PER_NODE}" \
    "phase-2 model/diffusion_training.py"
