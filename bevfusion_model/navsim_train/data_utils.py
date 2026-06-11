from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def load_lidar_points(frame: Mapping[str, Any], sensor_root: Path) -> torch.Tensor:
    lidar_path = frame.get("lidar_path")
    if lidar_path is None:
        return torch.zeros((0, 5), dtype=torch.float32)
    full_path = sensor_root / lidar_path
    try:
        from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

        with open(full_path, "rb") as fp:
            import io

            pc = LidarPointCloud.from_buffer(io.BytesIO(fp.read()), "pcd").points
        arr = np.asarray(pc, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] in {3, 4, 5, 6}:
            arr = arr.T
        if arr.shape[1] < 5:
            arr = np.pad(arr, ((0, 0), (0, 5 - arr.shape[1])), mode="constant")
        return torch.from_numpy(arr[:, :5].astype(np.float32))
    except Exception:
        return torch.zeros((0, 5), dtype=torch.float32)
