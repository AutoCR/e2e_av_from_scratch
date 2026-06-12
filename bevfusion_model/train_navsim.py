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

import os

# Reduce CUDA allocator fragmentation (must be set before torch initializes
# CUDA): the 8-camera LSS frustum allocates multi-GiB activations whose sizes
# vary per scene, which otherwise strands GiBs as reserved-but-unallocated.
# expandable_segments would be ideal, but torch 2.2.1+cu121 on the server
# rejects it ("not supported on this platform"); cap block splitting and let
# the allocator GC fully-free blocks instead.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:512,garbage_collection_threshold:0.8",
)

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "bevfusion_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__":
    from bevfusion_model.configs.bevfusion_hyperparams import get_runtime_config
    from bevfusion_model.navsim_train.runner import run

    run(get_runtime_config())
