"""SparseDrive-on-NAVSIM training entry point."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPARSEDRIVE_ROOT = _REPO_ROOT / "sparsedrive_model"
for _path in (str(_REPO_ROOT), str(_SPARSEDRIVE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CONFIG = {
    "stage": "stage1",
    "splits": {"train": "mini", "val": "mini", "test": "mini"},
    "openscene_data_root": "/Users/chenran/Code/navsim/dataset",
    "nuplan_maps_root": "/Users/chenran/Code/navsim/dataset/maps",
    "output_dir": "sparsedrive_model/outputs/train_navsim",
    "load_from": None,
    "resume_from": None,
    "seed": 0,
    "num_workers": 4,
    "device": "cpu",
    "quick_smoke": True,
    "total_batch_size": None,
    "num_epochs": None,
    "log_interval": 51,
    "ckpt_epoch_interval": None,
    "eval_epoch_interval": None,
    "fp16_loss_scale": 32.0,
}

if __name__ == "__main__":
    from sparsedrive_model.navsim_train.runner import run

    run(CONFIG)
