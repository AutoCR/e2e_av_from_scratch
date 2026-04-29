#!/usr/bin/env bash
# Run DiffusionPlanner two-stage PDM evaluation.
#
# Usage:
#   ./run_diffusion_pdm_eval.sh --ckpt /path/to/ckpt.pt
#   ./run_diffusion_pdm_eval.sh --ckpt /path/to/ckpt.pt --limit 10
#   LIMIT=5 ./run_diffusion_pdm_eval.sh --ckpt /path/to/ckpt.pt
#
# Any extra args are forwarded to diffusion_planner_pdm_eval.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/prediction_database/nuplan/dataset/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/prediction_database/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SCRIPT_DIR}/exp}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/diffusion_pdm_eval}"
SPLIT="${SPLIT:-navtest_two_stage}"

cd "${SCRIPT_DIR}"

exec uv run python "phase-2 model/diffusion_planner_pdm_eval.py" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
