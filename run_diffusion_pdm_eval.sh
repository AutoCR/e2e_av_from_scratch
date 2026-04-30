#!/usr/bin/env bash
# Run DiffusionPlanner two-stage PDM evaluation.
#
# Usage:
#   ./run_diffusion_pdm_eval.sh                                # use default CKPT below
#   CKPT=/path/to/ckpt.pth ./run_diffusion_pdm_eval.sh         # override checkpoint
#   ./run_diffusion_pdm_eval.sh --limit 3                      # extra args forwarded
#
# Any extra args are forwarded to diffusion_planner_pdm_eval.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/prediction_database/nuplan/dataset/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/prediction_database/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${SCRIPT_DIR}/exp}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CKPT="${CKPT:-${SCRIPT_DIR}/exp/diffusion_planner/2026-04-26_14-52-13/ckpt/model_epoch_500_trainloss_0.0276.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/diffusion_pdm_eval}"
SPLIT="${SPLIT:-navhard_two_stage}"

cd "${SCRIPT_DIR}"

exec uv run python "phase-2 model/diffusion_planner_pdm_eval.py" \
    --ckpt "${CKPT}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
