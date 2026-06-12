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

# Allocator note: stay on the DEFAULT caching allocator. Both alternatives
# failed on the server (torch 2.2.1+cu121, 6-GPU DDP):
#   - expandable_segments: rejected ("not supported on this platform");
#   - backend:cudaMallocAsync: leaks under DDP — frees recorded on NCCL side
#     streams are deferred and the driver pool retains its high-watermark,
#     until the device is full with only ~8 GiB actually allocated by torch.
# Fragmentation of the default allocator only mattered at per-rank batch 4
# (~18.4 GiB steady + 3.2 GiB backward spikes on 24 GB); at batch 2-3 there
# is enough headroom that no allocator tuning is needed.

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
