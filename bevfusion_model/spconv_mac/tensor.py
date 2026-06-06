from __future__ import annotations

import torch


class SparseConvTensorLite:
    """Small sparse tensor container compatible with this BEVFusion fallback.

    Indices are stored as int tensors with columns ``(batch, x, y, z)`` and
    ``spatial_shape`` is ``(x_size, y_size, z_size)``.
    """

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: list[int] | tuple[int, int, int],
        batch_size: int,
    ) -> None:
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError(f"indices must have shape (N, 4), got {tuple(indices.shape)}")
        if features.ndim != 2:
            raise ValueError(f"features must have shape (N, C), got {tuple(features.shape)}")
        if features.shape[0] != indices.shape[0]:
            raise ValueError("features and indices must have the same first dimension")

        self.features = features
        self.indices = indices.to(device=features.device, dtype=torch.long)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.batch_size = int(batch_size)

    def replace_feature(self, features: torch.Tensor) -> "SparseConvTensorLite":
        return SparseConvTensorLite(features, self.indices, self.spatial_shape, self.batch_size)

    def dense(self) -> torch.Tensor:
        """Materialize as ``(B, C, X, Y, Z)``."""
        b, c = self.batch_size, self.features.shape[1]
        x_size, y_size, z_size = self.spatial_shape
        dense = self.features.new_zeros((b, c, x_size, y_size, z_size))
        if self.features.numel() == 0:
            return dense

        coords = self.indices
        dense[
            coords[:, 0],
            :,
            coords[:, 1],
            coords[:, 2],
            coords[:, 3],
        ] = self.features
        return dense


def coord_to_key(indices: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    """Pack ``(batch, x, y, z)`` coordinates into sortable int64 keys."""
    x_size, y_size, z_size = spatial_shape
    return (
        indices[:, 0].long() * (x_size * y_size * z_size)
        + indices[:, 1].long() * (y_size * z_size)
        + indices[:, 2].long() * z_size
        + indices[:, 3].long()
    )
