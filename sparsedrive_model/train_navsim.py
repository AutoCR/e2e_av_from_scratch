"""SparseDrive-on-NAVSIM training entry point.

All training parameters are defined in one place:
    sparsedrive_model/configs/sparsedrive_hyperparams.py

Edit that file, then launch training with:

Single-GPU:
    uv run python sparsedrive_model/train_navsim.py

Multi-GPU DDP (e.g. 8 GPUs on one node):
    torchrun --nproc_per_node=8 sparsedrive_model/train_navsim.py

The runner auto-detects DDP from the RANK/LOCAL_RANK/WORLD_SIZE env vars
set by torchrun and falls back to single-device when they are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPARSEDRIVE_ROOT = _REPO_ROOT / "sparsedrive_model"
for _path in (str(_REPO_ROOT), str(_SPARSEDRIVE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if __name__ == "__main__":
    from sparsedrive_model.configs.sparsedrive_hyperparams import get_runtime_config
    from sparsedrive_model.navsim_train.runner import run

    run(get_runtime_config())
