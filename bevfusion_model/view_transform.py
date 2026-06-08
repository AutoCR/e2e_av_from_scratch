"""BEVFusion view transforms (LSS-based) in pure PyTorch.

No mmdet/mmcv dependencies. This module provides:
  - LSSTransform: simple depth prediction + BEV pooling
  - DepthLSSTransform: depth map generation + dtransform + depthnet + BEV pooling
"""

from typing import Tuple
import torch
from torch import nn

from .bev_pool import bev_pool as bev_pool_cuda, BEV_POOL_CUDA_AVAILABLE


def gen_dx_bx(xbound, ybound, zbound):
    """Generate grid cell size, center offset, and grid dimensions.

    Args:
        xbound: (min, max, step)
        ybound: (min, max, step)
        zbound: (min, max, step)

    Returns:
        dx: cell size [dx, dy, dz]
        bx: center offset [bx, by, bz]
        nx: grid dimensions [nx, ny, nz]
    """
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
    """Base class for view transforms. Handles frustum backprojection to BEV."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size  # (H, W)
        self.feature_size = feature_size  # (fH, fW)
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self._create_frustum()
        self.D = self.frustum.shape[0]

    def _create_frustum(self):
        """Create frustum: (D, fH, fW, 3) with (x, y, depth) coordinates."""
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D = ds.shape[0]

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        extra_rots=None,
        extra_trans=None,
    ):
        """Backproject frustum to lidar frame.

        Returns:
            geom: (B, N, D, fH, fW, 3) 3D points in lidar frame
        """
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation (image augmentation)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        points = points.squeeze(-1)

        # un-project: multiply xy by depth z
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )

        # cam2lidar: combine rotation and inverse intrinsics
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = points.squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        # apply lidar augmentation if provided
        if extra_rots is not None:
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if extra_trans is not None:
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        """Process camera features. To be implemented by subclasses."""
        raise NotImplementedError

    def bev_pool_pure(self, geom_feats, x):
        """Pool camera features into BEV grid using pure PyTorch.

        Args:
            geom_feats: (B, N, D, fH, fW, 3) 3D points in lidar frame
            x: (B, N, D, fH, fW, C) camera features

        Returns:
            final: (B, C*D, Nx, Ny) BEV features
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten features
        x = x.reshape(Nprime, C)

        # convert 3D coords to grid indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)

        # add batch index
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out-of-bounds points
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        Nz = int(self.nx[2].item())
        Nx = int(self.nx[0].item())
        Ny = int(self.nx[1].item())

        if BEV_POOL_CUDA_AVAILABLE and x.is_cuda:
            # CUDA fast path: fused sort + scatter. geom_feats is (x, y, z, batch)
            # and the kernel returns (B, C, Nz, Nx, Ny), matching the pure path.
            bev = bev_pool_cuda(x, geom_feats, B, Nz, Nx, Ny)
        else:
            # Pure-PyTorch fallback: scatter into BEV grid (B, Nz, Nx, Ny, C).
            bev = torch.zeros(B, Nz, Nx, Ny, C, device=x.device, dtype=x.dtype)
            xi, yi, zi, bi = geom_feats[:, 0], geom_feats[:, 1], geom_feats[:, 2], geom_feats[:, 3]
            # scatter-add using index_put_
            bev.index_put_((bi, zi, xi, yi), x, accumulate=True)
            # permute to (B, C, Nz, Nx, Ny)
            bev = bev.permute(0, 4, 1, 2, 3)

        # collapse Z dimension → (B, C*Nz, Nx, Ny)
        final = torch.cat(bev.unbind(dim=2), 1)

        return final

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class LSSTransform(BaseTransform):
    """Lift, Splat, Shoot transform without explicit depth map.

    Uses a simple depthnet to predict depth and camera features directly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )

        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)

        if downsample > 1:
            assert downsample == 2
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        """Predict depth and multiply by camera features.

        Args:
            x: (B, N, C, fH, fW) camera features

        Returns:
            feats: (B, N, D, fH, fW, C) depth-weighted features
        """
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ):
        """Forward pass for LSS transform.

        Args:
            img: (B, N, C, H, W) camera images
            points: (B, P, 3+) lidar points
            camera2ego: (B, N, 4, 4) extrinsics
            lidar2ego: (B, 4, 4)
            lidar2camera: (B, N, 4, 4)
            lidar2image: (B, N, 4, 4)
            camera_intrinsics: (B, N, 3, 4)
            camera2lidar: (B, N, 4, 4)
            img_aug_matrix: (B, N, 4, 4)
            lidar_aug_matrix: (B, 4, 4)
            metas: metadata dict
            **kwargs: additional arguments

        Returns:
            x: (B, C*D, Nx, Ny) BEV features
        """
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        # compute geometry (backproject frustum to lidar frame)
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # get camera features with depth prediction
        x = self.get_cam_feats(img)

        # BEV pool
        x = self.bev_pool_pure(geom, x)

        # downsample if needed
        x = self.downsample(x)

        return x


class DepthLSSTransform(BaseTransform):
    """LSS transform with explicit depth map from LiDAR projection.

    Uses depth map generated from projected LiDAR points, passed through
    dtransform to produce depth features, combined with image features
    via depthnet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )

        # depth map processing
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # depth + features -> depth distribution + feature embedding
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )

        if downsample > 1:
            assert downsample == 2
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        """Process depth map and camera features.

        Args:
            x: (B, N, C, fH, fW) camera features
            d: (B, N, 1, H, W) depth map

        Returns:
            feats: (B, N, D, fH, fW, C) depth-weighted features
        """
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        # transform depth map to depth features
        d = self.dtransform(d)

        # upsample depth features back to feature size to match image features
        if d.shape[-2:] != (fH, fW):
            d = torch.nn.functional.interpolate(
                d, size=(fH, fW), mode='bilinear', align_corners=True
            )

        # combine depth features with image features
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        # extract depth distribution and feature embedding
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ):
        """Forward pass for Depth LSS transform.

        Args:
            img: (B, N, C, H, W) camera images
            points: (B, P, 3+) lidar points
            camera2ego: (B, N, 4, 4) extrinsics
            lidar2ego: (B, 4, 4)
            lidar2camera: (B, N, 4, 4)
            lidar2image: (B, N, 4, 4)
            camera_intrinsics: (B, N, 3, 4)
            camera2lidar: (B, N, 4, 4)
            img_aug_matrix: (B, N, 4, 4)
            lidar_aug_matrix: (B, 4, 4)
            metas: metadata dict
            **kwargs: additional arguments

        Returns:
            x: (B, C*D, Nx, Ny) BEV features
        """
        B = img.shape[0]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        # build depth map from LiDAR points projected to camera
        iH, iW = self.image_size
        N = img.shape[1]
        depth = torch.zeros(B, N, 1, iH, iW, device=img.device)

        for b in range(B):
            cur_points = points[b][:, :3]  # (P, 3)
            cur_lidar_aug = lidar_aug_matrix[b]  # (4, 4)
            cur_lidar2image = lidar2image[b]  # (N, 4, 4)
            cur_img_aug = img_aug_matrix[b]  # (N, 4, 4)

            # undo lidar augmentation
            cur_points = cur_points - cur_lidar_aug[:3, 3].unsqueeze(0)
            cur_points = torch.inverse(cur_lidar_aug[:3, :3]).matmul(
                cur_points.T
            )  # (3, P)

            # project to image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_points)  # (N, 3, P)
            cur_coords += cur_lidar2image[:, :3, 3:4]  # (N, 3, P)

            # get depth (z) and normalize pixel coords
            dist = cur_coords[:, 2, :]  # (N, P)
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # apply image augmentation
            cur_coords = cur_img_aug[:, :3, :3].matmul(cur_coords)  # (N, 3, P)
            cur_coords += cur_img_aug[:, :3, 3:4]  # (N, 3, P)
            cur_coords = cur_coords[:, :2, :].permute(0, 2, 1)  # (N, P, 2)
            cur_coords = cur_coords[..., [1, 0]]  # swap to (row, col)

            # filter in-image points
            on_img = (
                (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 0] < iH)
                & (cur_coords[..., 1] >= 0)
                & (cur_coords[..., 1] < iW)
            )

            for c in range(N):
                mask = on_img[c]
                if mask.sum() == 0:
                    continue
                coords_c = cur_coords[c, mask].long()  # (M, 2)
                depth_c = dist[c, mask]
                depth[b, c, 0, coords_c[:, 0], coords_c[:, 1]] = depth_c

        # compute geometry (backproject frustum to lidar frame)
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # get camera features with depth map
        x = self.get_cam_feats(img, depth)

        # BEV pool
        x = self.bev_pool_pure(geom, x)

        # downsample if needed
        x = self.downsample(x)

        return x
