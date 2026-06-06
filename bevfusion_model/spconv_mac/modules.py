from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn

from .tensor import SparseConvTensorLite, coord_to_key


def _triple(value) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"expected int or length-3 tuple/list, got {value!r}")
    return tuple(int(v) for v in value)


def _kernel_offsets(kernel_size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    ranges = [torch.arange(k, device=device, dtype=torch.long) for k in kernel_size]
    mesh = torch.meshgrid(*ranges, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=1)


class SparseSequentialLite(nn.Sequential):
    """SparseSequential variant that applies dense ops to sparse features."""

    _is_sparse_lite = True

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            super().__init__(args[0])
        else:
            super().__init__(*args)

    def forward(self, input):
        for module in self:
            if isinstance(input, SparseConvTensorLite) and not getattr(module, "_is_sparse_lite", False):
                input = input.replace_feature(module(input.features))
            else:
                input = module(input)
        return input


class SubMConv3dLite(nn.Module):
    """Submanifold sparse 3D convolution using spconv weight layout."""

    _is_sparse_lite = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding=0,
        bias: bool = False,
        indice_key: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _triple(kernel_size)
        self.padding = _triple(padding)
        self.indice_key = indice_key
        self.weight = nn.Parameter(
            torch.empty(*self.kernel_size, self.in_channels, self.out_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight.view(-1, self.out_channels).T, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: SparseConvTensorLite) -> SparseConvTensorLite:
        if input.features.numel() == 0:
            out = input.features.new_zeros((0, self.out_channels))
            return input.replace_feature(out)

        in_keys = coord_to_key(input.indices, input.spatial_shape)
        sorted_keys, order = torch.sort(in_keys)
        sorted_features = input.features[order]
        out_features = input.features.new_zeros((input.features.shape[0], self.out_channels))

        offsets = _kernel_offsets(self.kernel_size, input.features.device)
        padding = torch.tensor(self.padding, device=input.features.device, dtype=torch.long)

        for flat_idx, offset in enumerate(offsets):
            in_coords = input.indices.clone()
            in_coords[:, 1:] = in_coords[:, 1:] + offset - padding
            valid = _coords_in_bounds(in_coords, input.spatial_shape)
            if not bool(valid.any()):
                continue

            lookup_keys = coord_to_key(in_coords[valid], input.spatial_shape)
            pos = torch.searchsorted(sorted_keys, lookup_keys)
            in_range = pos < sorted_keys.numel()
            matched = torch.zeros_like(in_range)
            if bool(in_range.any()):
                matched[in_range] = sorted_keys[pos[in_range]] == lookup_keys[in_range]
            if not bool(matched.any()):
                continue

            out_rows = torch.where(valid)[0][matched]
            src_features = sorted_features[pos[matched]]
            weight = self.weight.reshape(-1, self.in_channels, self.out_channels)[flat_idx]
            out_features[out_rows] += src_features @ weight

        if self.bias is not None:
            out_features += self.bias
        return input.replace_feature(out_features)


class SparseConv3dLite(nn.Module):
    """Sparse 3D convolution with stride/padding and spconv weight layout."""

    _is_sparse_lite = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = False,
        indice_key: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.indice_key = indice_key
        self.weight = nn.Parameter(
            torch.empty(*self.kernel_size, self.in_channels, self.out_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight.view(-1, self.out_channels).T, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: SparseConvTensorLite) -> SparseConvTensorLite:
        out_shape = _conv_output_shape(input.spatial_shape, self.kernel_size, self.stride, self.padding)
        if input.features.numel() == 0:
            out = input.features.new_zeros((0, self.out_channels))
            indices = input.indices.new_zeros((0, 4))
            return SparseConvTensorLite(out, indices, out_shape, input.batch_size)

        offsets = _kernel_offsets(self.kernel_size, input.features.device)
        stride = torch.tensor(self.stride, device=input.features.device, dtype=torch.long)
        padding = torch.tensor(self.padding, device=input.features.device, dtype=torch.long)

        all_keys = []
        all_features = []

        for flat_idx, offset in enumerate(offsets):
            numer = input.indices[:, 1:] + padding - offset
            divisible = (numer % stride == 0).all(dim=1)
            if not bool(divisible.any()):
                continue
            out_xyz = numer[divisible] // stride
            out_indices = torch.cat([input.indices[divisible, :1], out_xyz], dim=1)
            valid = _coords_in_bounds(out_indices, out_shape)
            if not bool(valid.any()):
                continue

            out_indices = out_indices[valid]
            src_features = input.features[divisible][valid]
            weight = self.weight.reshape(-1, self.in_channels, self.out_channels)[flat_idx]
            all_keys.append(coord_to_key(out_indices, out_shape))
            all_features.append(src_features @ weight)

        if not all_keys:
            out = input.features.new_zeros((0, self.out_channels))
            indices = input.indices.new_zeros((0, 4))
            return SparseConvTensorLite(out, indices, out_shape, input.batch_size)

        keys = torch.cat(all_keys, dim=0)
        values = torch.cat(all_features, dim=0)
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        out_features = values.new_zeros((unique_keys.shape[0], self.out_channels))
        out_features.index_add_(0, inverse, values)
        if self.bias is not None:
            out_features += self.bias

        out_indices = _key_to_coord(unique_keys, out_shape)
        return SparseConvTensorLite(out_features, out_indices, out_shape, input.batch_size)


def _coords_in_bounds(indices: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    x_size, y_size, z_size = spatial_shape
    return (
        (indices[:, 1] >= 0)
        & (indices[:, 1] < x_size)
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < y_size)
        & (indices[:, 3] >= 0)
        & (indices[:, 3] < z_size)
    )


def _conv_output_shape(
    spatial_shape: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(
        (spatial_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
        for i in range(3)
    )


def _key_to_coord(keys: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    x_size, y_size, z_size = spatial_shape
    batch_stride = x_size * y_size * z_size
    b = keys // batch_stride
    rem = keys % batch_stride
    x = rem // (y_size * z_size)
    rem = rem % (y_size * z_size)
    y = rem // z_size
    z = rem % z_size
    return torch.stack([b, x, y, z], dim=1).long()
