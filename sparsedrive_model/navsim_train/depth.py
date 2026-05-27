from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class MultiScaleDepthMapGenerator:
    def __init__(self, downsample: Sequence[int] = (4, 8, 16), max_depth: float = 60.0) -> None:
        self.downsample = tuple(int(x) for x in downsample)
        self.max_depth = float(max_depth)

    def __call__(
        self,
        lidar_points: torch.Tensor | np.ndarray | None,
        projection_mats: torch.Tensor | np.ndarray,
        image_wh: torch.Tensor | np.ndarray | Sequence[int],
    ) -> list[torch.Tensor]:
        points = self._points_np(lidar_points)
        proj = self._as_np(projection_mats).astype(np.float32)
        if proj.ndim != 3 or proj.shape[1:] != (3, 4):
            raise ValueError(f"projection_mats must have shape (n_cams, 3, 4), got {proj.shape}")
        wh = self._image_wh_np(image_wh, proj.shape[0])
        outputs = [np.zeros((proj.shape[0], int(wh[0, 1] // ds), int(wh[0, 0] // ds)), dtype=np.float32) for ds in self.downsample]
        if points.size == 0:
            return [torch.from_numpy(x) for x in outputs]

        xyz1 = np.concatenate([points[:, :3].astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        for cam_idx, lidar2img in enumerate(proj):
            uvz = xyz1 @ lidar2img.T
            z = uvz[:, 2]
            valid_z = z >= 0.1
            if not np.any(valid_z):
                continue
            u = np.round(uvz[:, 0] / np.maximum(z, 1.0e-6)).astype(np.int64)
            v = np.round(uvz[:, 1] / np.maximum(z, 1.0e-6)).astype(np.int64)
            w, h = int(wh[cam_idx, 0]), int(wh[cam_idx, 1])
            valid = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(valid):
                continue
            u, v, depths = u[valid], v[valid], np.clip(z[valid], 0.1, self.max_depth)
            order = np.argsort(depths)[::-1]
            u, v, depths = u[order], v[order], depths[order]
            for out_idx, ds in enumerate(self.downsample):
                out = outputs[out_idx]
                hh, ww = out.shape[1], out.shape[2]
                uu = np.floor(u / ds).astype(np.int64)
                vv = np.floor(v / ds).astype(np.int64)
                keep = (uu >= 0) & (uu < ww) & (vv >= 0) & (vv < hh)
                out[cam_idx, vv[keep], uu[keep]] = depths[keep]
        return [torch.from_numpy(x) for x in outputs]

    @staticmethod
    def _as_np(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _points_np(self, points: torch.Tensor | np.ndarray | None) -> np.ndarray:
        if points is None:
            return np.zeros((0, 5), dtype=np.float32)
        arr = self._as_np(points).astype(np.float32)
        if arr.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] in {3, 4, 5, 6} and arr.shape[1] != 5:
            arr = arr.T
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"lidar_points must be (N, >=3) or channel-first point cloud, got {arr.shape}")
        if arr.shape[1] < 5:
            arr = np.pad(arr, ((0, 0), (0, 5 - arr.shape[1])), mode="constant")
        return arr[:, :5]

    @staticmethod
    def _image_wh_np(image_wh: torch.Tensor | np.ndarray | Sequence[int], n_cams: int) -> np.ndarray:
        wh = image_wh.detach().cpu().numpy() if torch.is_tensor(image_wh) else np.asarray(image_wh)
        wh = wh.astype(np.float32)
        if wh.ndim == 1:
            if wh.shape[0] != 2:
                raise ValueError(f"image_wh must have shape (2,) or (n_cams, 2), got {wh.shape}")
            wh = np.repeat(wh[None], n_cams, axis=0)
        if wh.shape != (n_cams, 2):
            raise ValueError(f"image_wh must have shape ({n_cams}, 2), got {wh.shape}")
        return wh
