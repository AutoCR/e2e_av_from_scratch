from __future__ import annotations

import torch
import torch.nn as nn

from .modules import SparseConv3dLite, SparseSequentialLite, SubMConv3dLite
from .tensor import SparseConvTensorLite


class SparseBasicBlockLite(nn.Module):
    """Official SparseBasicBlock-compatible residual block."""

    _is_sparse_lite = True
    expansion = 1

    def __init__(self, inplanes: int, planes: int, norm_eps=1e-3, norm_momentum=0.01) -> None:
        super().__init__()
        if inplanes != planes:
            raise ValueError("SparseBasicBlockLite expects inplanes == planes")
        self.conv1 = SubMConv3dLite(inplanes, planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes, eps=norm_eps, momentum=norm_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SubMConv3dLite(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes, eps=norm_eps, momentum=norm_momentum)

    def forward(self, x: SparseConvTensorLite) -> SparseConvTensorLite:
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out_features = self.bn2(out.features) + identity
        return out.replace_feature(self.relu(out_features))


class TorchSparseEncoder(nn.Module):
    """Pure PyTorch fallback matching BEVFusion's spconv SparseEncoder keys."""

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
        if block_type not in ["conv_module", "basicblock"]:
            raise ValueError(f"Unsupported block_type: {block_type}")
        if tuple(order) != ("conv", "norm", "act"):
            raise ValueError("TorchSparseEncoder currently supports order=('conv', 'norm', 'act')")

        self.sparse_shape = tuple(int(v) for v in sparse_shape)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.block_type = block_type

        def make_norm(c):
            return nn.BatchNorm1d(c, eps=norm_eps, momentum=norm_momentum)

        self.conv_input = SparseSequentialLite(
            SubMConv3dLite(in_channels, base_channels, 3, padding=1, bias=False, indice_key="subm1"),
            make_norm(base_channels),
            nn.ReLU(inplace=True),
        )

        self.encoder_layers = SparseSequentialLite()
        encoder_out_channels = self._make_encoder_layers(make_norm, norm_eps, norm_momentum)

        self.conv_out = SparseSequentialLite(
            SparseConv3dLite(
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
        cur_channels = self.base_channels

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_ch in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]

                if i != 0 and j == 0 and self.block_type == "conv_module":
                    blocks_list.append(
                        SparseSequentialLite(
                            SparseConv3dLite(
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
                        blocks_list.append(
                            SparseSequentialLite(
                                SparseConv3dLite(
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
                        blocks_list.append(
                            SparseBasicBlockLite(
                                cur_channels,
                                out_ch,
                                norm_eps=norm_eps,
                                norm_momentum=norm_momentum,
                            )
                        )
                else:
                    blocks_list.append(
                        SparseSequentialLite(
                            SubMConv3dLite(
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

            self.encoder_layers.add_module(f"encoder_layer{i + 1}", SparseSequentialLite(*blocks_list))

        return cur_channels

    def forward(self, voxel_features: torch.Tensor, coors: torch.Tensor, batch_size: int) -> torch.Tensor:
        if coors.ndim != 2 or coors.shape[1] != 4:
            raise ValueError(f"coors must have shape (N, 4), got {tuple(coors.shape)}")

        x = SparseConvTensorLite(voxel_features, coors.int(), self.sparse_shape, batch_size)
        x = self.conv_input(x)

        for encoder_layer in self.encoder_layers.children():
            x = encoder_layer(x)

        out = self.conv_out(x)
        spatial_features = out.dense()

        # (B, C, X, Y, Z) -> (B, C*Z, X, Y)
        b, c, x_size, y_size, z_size = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        return spatial_features.view(b, c * z_size, x_size, y_size)
