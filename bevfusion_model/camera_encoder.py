"""BEVFusion Camera Encoder with SwinTransformer backbone and LSSFPN neck.

Pure PyTorch implementation. No mmdet/mmcv imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .swin_transformer import SwinTransformer


class _ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm2d + ReLU block.

    Mimics mmcv's ConvModule structure for checkpoint compatibility.
    Attributes: .conv, .bn, .activate
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class GeneralizedLSSFPN(nn.Module):
    """Generalized LSS FPN (Feature Pyramid Network).

    Takes multi-scale features from backbone and produces unified output
    through lateral connections and FPN refinement.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 1,
        start_level: int = 0,
        upsample_mode: str = "bilinear",
        upsample_align_corners: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.upsample_mode = upsample_mode
        self.upsample_align_corners = upsample_align_corners

        num_ins = len(in_channels)
        backbone_end_level = num_ins - 1

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(start_level, backbone_end_level):
            if i == backbone_end_level - 1:
                lateral_in_ch = in_channels[i] + in_channels[i + 1]
            else:
                lateral_in_ch = in_channels[i] + out_channels

            l_conv = _ConvBnRelu(lateral_in_ch, out_channels, kernel_size=1)
            fpn_conv = _ConvBnRelu(out_channels, out_channels, kernel_size=3, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: List[torch.Tensor]):
        """Forward pass to fuse multi-scale features.

        Args:
            inputs: list of feature maps from backbone in order
                    (e.g., [stride-8, stride-16, stride-32] for typical backbones)

        Returns:
            outs: fused feature map(s). If num_outs=1, returns single tensor;
                  otherwise returns list of feature maps.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        laterals = list(inputs[self.start_level:])

        num_ins = len(inputs)
        backbone_end_level = num_ins - 1
        used_levels = backbone_end_level - self.start_level

        for i in range(used_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.upsample_align_corners if self.upsample_mode != "nearest" else None,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        outs = [laterals[i] for i in range(used_levels)]

        while len(outs) < self.num_outs:
            outs.append(outs[-1])

        if self.num_outs == 1:
            return outs[0]
        return outs


class CameraEncoder(nn.Module):
    """Camera encoder combining SwinTransformer backbone, LSSFPN neck, and view transform."""

    def __init__(
        self,
        backbone: Optional[SwinTransformer] = None,
        neck: Optional[GeneralizedLSSFPN] = None,
        vtransform: Optional[nn.Module] = None,
        backbone_kwargs: Optional[dict] = None,
        neck_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if backbone is None:
            if backbone_kwargs is None:
                backbone_kwargs = {}
            backbone = SwinTransformer(**backbone_kwargs)
        self.backbone = backbone

        if neck is None:
            if neck_kwargs is None:
                neck_kwargs = {"in_channels": [512, 1024, 2048]}
            neck = GeneralizedLSSFPN(**neck_kwargs)
        self.neck = neck

        self.vtransform = vtransform

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
        """Forward pass for camera encoder.

        Args:
            img: (B, N, C, H, W) stacked camera images
            points: (B, P, 3+) lidar points
            camera2ego: (B, N, 4, 4) camera to ego extrinsics
            lidar2ego: (B, 4, 4) lidar to ego transform
            lidar2camera: (B, N, 4, 4) lidar to camera transform
            lidar2image: (B, N, 4, 4) lidar to image projection
            camera_intrinsics: (B, N, 3, 4) camera intrinsic matrices
            camera2lidar: (B, N, 4, 4) camera to lidar transform
            img_aug_matrix: (B, N, 4, 4) image augmentation transform
            lidar_aug_matrix: (B, 4, 4) lidar augmentation transform
            metas: metadata dict
            **kwargs: additional arguments

        Returns:
            x: (B, C, Nx, Ny) BEV features from view transform
        """
        B, N, C, H, W = img.shape

        x = img.view(B * N, C, H, W)

        feats = self.backbone(x)

        x = self.neck(feats)
        if isinstance(x, (list, tuple)):
            x = x[0]

        BN, C_out, fH, fW = x.shape
        x = x.view(B, N, C_out, fH, fW)

        if self.vtransform is not None:
            x = self.vtransform(
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas=metas,
            )

        return x
