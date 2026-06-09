"""BEVFusion-on-NAVSIM training entry point.

All training parameters live in bevfusion_model/configs/bevfusion_hyperparams.py.

Single-GPU:
    uv run python bevfusion_model/train_navsim.py

Multi-GPU (e.g. 8 GPUs on one node):
    torchrun --nproc_per_node=8 bevfusion_model/train_navsim.py

Trains 3D detection (5 NAVSIM classes: car, barrier, bicycle, pedestrian, traffic_cone) from scratch.
No checkpoint warm-start; detection-only task.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "bevfusion_model"), str(_REPO_ROOT / "sparsedrive_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__":
    from bevfusion_model.configs.bevfusion_hyperparams import get_runtime_config
    from bevfusion_model.navsim_train.runner import run

    run(get_runtime_config())
