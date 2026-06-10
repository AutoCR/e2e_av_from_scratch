"""BEVFusion LiDAR encoder: sparse convolution backbone."""

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential, SubMConv3d, SparseConv3d, SparseModule
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False
    spconv = None
    SparseModule = nn.Module


class HardVoxelization(nn.Module):
    """Pure PyTorch hard voxelization (no CUDA kernels)."""

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=(90000, 120000)):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels  # (train_max, test_max)

        self.pc_range_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
        self.pc_range_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
        grid_size = torch.round((self.pc_range_max - self.pc_range_min) / self.voxel_size).long()
        self.grid_size = grid_size  # (nx, ny, nz)

    def forward(self, points):
        """Voxelize point cloud.

        Args:
            points (torch.Tensor): Points in shape (N, C) where C >= 3 (x, y, z, ...).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - voxels: (num_voxels, max_num_points, C)
                - coords: (num_voxels, 4) with columns (batch_idx, x, y, z)
                - num_points_per_voxel: (num_voxels,)
        """
        max_voxels = self.max_voxels[0] if self.training else self.max_voxels[1]

        voxel_size = self.voxel_size.to(points.device)
        pc_min = self.pc_range_min.to(points.device)
        pc_max = self.pc_range_max.to(points.device)
        grid_size = self.grid_size.to(points.device)

        # Filter points in range
        mask = (
            (points[:, 0] >= pc_min[0])
            & (points[:, 0] < pc_max[0])
            & (points[:, 1] >= pc_min[1])
            & (points[:, 1] < pc_max[1])
            & (points[:, 2] >= pc_min[2])
            & (points[:, 2] < pc_max[2])
        )
        points = points[mask]

        if points.shape[0] == 0:
            C = points.shape[1] if points.ndim > 1 else 4
            voxels = points.new_zeros((0, self.max_num_points, C))
            coords = points.new_zeros((0, 4), dtype=torch.int32)
            num_points_per_voxel = points.new_zeros((0,), dtype=torch.int32)
            return voxels, coords, num_points_per_voxel

        # Compute voxel coordinates (integer indices)
        coords_float = (points[:, :3] - pc_min.unsqueeze(0)) / voxel_size.unsqueeze(0)
        coords_int = coords_float.long()  # (N, 3) -> x_idx, y_idx, z_idx

        # Clamp to grid bounds
        coords_int[:, 0].clamp_(0, grid_size[0] - 1)
        coords_int[:, 1].clamp_(0, grid_size[1] - 1)
        coords_int[:, 2].clamp_(0, grid_size[2] - 1)

        # Create unique voxel keys for grouping
        nx = grid_size[0].item()
        ny = grid_size[1].item()
        keys = coords_int[:, 2] * nx * ny + coords_int[:, 1] * nx + coords_int[:, 0]

        # Sort by key for efficient grouping
        sorted_indices = torch.argsort(keys)
        keys_sorted = keys[sorted_indices]
        points_sorted = points[sorted_indices]
        coords_sorted = coords_int[sorted_indices]

        # Find unique voxels
        unique_keys, inverse_indices, counts = torch.unique_consecutive(
            keys_sorted, return_inverse=True, return_counts=True
        )

        num_voxels = unique_keys.shape[0]
        if num_voxels > max_voxels:
            num_voxels = max_voxels

        C = points.shape[1]
        voxels = torch.zeros(
            num_voxels, self.max_num_points, C, dtype=points.dtype, device=points.device
        )
        coords_out = torch.zeros(num_voxels, 4, dtype=torch.int32, device=points.device)
        num_points_per_voxel = torch.zeros(num_voxels, dtype=torch.int32, device=points.device)

        # Vectorized voxel fill. ``keys_sorted`` is sorted so points of the same
        # voxel are contiguous; ``inverse_indices`` (group id per sorted point)
        # and ``counts`` come from unique_consecutive above.
        # Per-point rank within its voxel = position - start_of_group.
        group_start = torch.zeros_like(counts)
        if counts.shape[0] > 1:
            group_start[1:] = torch.cumsum(counts, dim=0)[:-1]
        within_voxel_rank = (
            torch.arange(inverse_indices.shape[0], device=points.device)
            - group_start[inverse_indices]
        )

        # Keep at most max_num_points per voxel, and at most num_voxels voxels.
        keep = (within_voxel_rank < self.max_num_points) & (inverse_indices < num_voxels)
        dst_voxel = inverse_indices[keep]
        dst_slot = within_voxel_rank[keep]
        voxels[dst_voxel, dst_slot] = points_sorted[keep]

        # Output coords: (batch_idx, x, y, z), matching sparse_shape=(x, y, z).
        # The first point of each (capped) voxel carries its coordinate.
        capped_counts = counts[:num_voxels].clamp(max=self.max_num_points)
        coords_out[:, 1:] = coords_sorted[group_start[:num_voxels]].to(torch.int32)
        num_points_per_voxel[:] = capped_counts.to(torch.int32)

        return voxels, coords_out, num_points_per_voxel


class _SparseResBlock(SparseModule):
    """Minimal sparse residual block compatible with spconv v2."""

    def __init__(self, in_channels, out_channels, norm_eps=1e-3, norm_momentum=0.01, indice_key="subm"):
        super().__init__()
        self.conv1 = SubMConv3d(
            in_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = nn.BatchNorm1d(out_channels, eps=norm_eps, momentum=norm_momentum)
        self.conv2 = SubMConv3d(
            out_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = nn.BatchNorm1d(out_channels, eps=norm_eps, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = SparseSequential(
                SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key + "_sc"),
                nn.BatchNorm1d(out_channels, eps=norm_eps, momentum=norm_momentum),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        if self.shortcut is not None:
            identity = self.shortcut(x).features
        out = out.replace_feature(self.relu(out.features + identity))
        return out


class SparseEncoder(nn.Module):
    """Sparse encoder using spconv v2 for 3D convolutions."""

    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        norm_eps=1e-3,
        norm_momentum=0.01,
    ):
        super().__init__()
        assert SPCONV_AVAILABLE, "spconv is required. Install with: pip install spconv-cu120 or spconv-cpu"
        assert block_type in ["conv_module", "basicblock"]

        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.block_type = block_type

        def make_norm(c):
            return nn.BatchNorm1d(c, eps=norm_eps, momentum=norm_momentum)

        # conv_input: SubMConv3d
        self.conv_input = SparseSequential(
            SubMConv3d(in_channels, base_channels, 3, padding=1, bias=False, indice_key="subm1"),
            make_norm(base_channels),
            nn.ReLU(inplace=True),
        )

        # encoder_layers: build encoder stages
        self.encoder_layers = SparseSequential()
        encoder_out_channels = self._make_encoder_layers(make_norm, norm_eps, norm_momentum)

        # conv_out: SparseConv3d with downsampling
        self.conv_out = SparseSequential(
            SparseConv3d(
                encoder_out_channels,
                output_channels,
                kernel_size=(1, 1, 3),
                stride=(1, 1, 2),
                padding=0,
                bias=False,
                indice_key="spconv_down2",
            ),
            make_norm(output_channels),
            nn.ReLU(inplace=True),
        )

    def _make_encoder_layers(self, make_norm, norm_eps, norm_momentum):
        """Build encoder layers with sparse convolutions."""
        cur_channels = self.base_channels

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_ch in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]

                if i != 0 and j == 0 and self.block_type == "conv_module":
                    # Strided conv at start of stages 2+
                    blocks_list.append(
                        SparseSequential(
                            SparseConv3d(
                                cur_channels,
                                out_ch,
                                3,
                                stride=2,
                                padding=padding,
                                bias=False,
                                indice_key=f"spconv{i + 1}",
                            ),
                            make_norm(out_ch),
                            nn.ReLU(inplace=True),
                        )
                    )
                elif self.block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        # Downsampling conv at end of stage
                        blocks_list.append(
                            SparseSequential(
                                SparseConv3d(
                                    cur_channels,
                                    out_ch,
                                    3,
                                    stride=2,
                                    padding=padding,
                                    bias=False,
                                    indice_key=f"spconv{i + 1}",
                                ),
                                make_norm(out_ch),
                                nn.ReLU(inplace=True),
                            )
                        )
                    else:
                        # Residual block
                        blocks_list.append(
                            _SparseResBlock(
                                cur_channels,
                                out_ch,
                                norm_eps=norm_eps,
                                norm_momentum=norm_momentum,
                                indice_key=f"subm{i + 1}",
                            )
                        )
                else:
                    # Default: SubMConv3d module
                    blocks_list.append(
                        SparseSequential(
                            SubMConv3d(
                                cur_channels,
                                out_ch,
                                3,
                                padding=padding,
                                bias=False,
                                indice_key=f"subm{i + 1}",
                            ),
                            make_norm(out_ch),
                            nn.ReLU(inplace=True),
                        )
                    )
                cur_channels = out_ch

            stage_name = f"encoder_layer{i + 1}"
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)

        return cur_channels

    def forward(self, voxel_features, coors, batch_size):
        """Forward pass.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (num_voxels, C).
            coors (torch.Tensor): Coordinates in shape (num_voxels, 4) with columns
                (batch_idx, x_idx, y_idx, z_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: BEV features in shape (batch_size, C*D, H, W).
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        for encoder_layer in self.encoder_layers.children():
            x = encoder_layer(x)

        out = self.conv_out(x)
        spatial_features = out.dense()

        # Reshape from (N, C, H, W, D) to (N, C*D, H, W)
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features


class LidarEncoder(nn.Module):
    """LiDAR encoder: voxelization + sparse convolution backbone."""

    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        sparse_shape,
        in_channels=4,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        voxelize_reduce=True,
    ):
        super().__init__()
        self.voxelizer = HardVoxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )
        self.backbone = SparseEncoder(
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            block_type=block_type,
        )
        self.voxelize_reduce = voxelize_reduce

    def voxelize_batch(self, points_list):
        """Voxelize a batch of point clouds.

        Args:
            points_list (list[torch.Tensor]): List of point clouds, each (N_i, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - feats: concatenated voxel features
                - coords: concatenated voxel coordinates with columns (batch, x, y, z)
                - sizes: number of points per voxel
        """
        feats, coords, sizes = [], [], []
        for k, pts in enumerate(points_list):
            f, c, n = self.voxelizer(pts)
            feats.append(f)
            c = c.clone()
            c[:, 0] = k
            coords.append(c)
            sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        sizes = torch.cat(sizes, dim=0)

        if self.voxelize_reduce:
            feats = feats.sum(dim=1, keepdim=False) / sizes.float().view(-1, 1).clamp(min=1.0)
            feats = feats.contiguous()

        return feats, coords, sizes

    def forward(self, points_list):
        """Forward pass.

        Args:
            points_list (list[torch.Tensor]): Batch of point clouds.

        Returns:
            torch.Tensor: BEV features.
        """
        feats, coords, sizes = self.voxelize_batch(points_list)
        batch_size = coords[-1, 0].item() + 1 if coords.shape[0] > 0 else 1
        return self.backbone(feats, coords, batch_size)
