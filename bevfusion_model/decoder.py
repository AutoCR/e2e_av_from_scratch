"""BEVFusion decoder: SECOND backbone + SECONDFPN neck (pure PyTorch)."""

import torch
import torch.nn as nn


class SECONDBackbone(nn.Module):
    """SECOND backbone for BEV feature extraction.

    Args:
        in_channels (int): Input channels. Default: 128.
        out_channels (list[int]): Output channels for each block. Default: [128, 256].
        layer_nums (list[int]): Number of conv layers in each block. Default: [5, 5].
        layer_strides (list[int]): Stride of first conv in each block. Default: [1, 2].
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=(128, 256),
        layer_nums=(5, 5),
        layer_strides=(1, 2),
    ):
        super().__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels] + list(out_channels[:-1])
        blocks = []

        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1,
                        bias=False,
                    )
                )
                block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            blocks.append(nn.Sequential(*block))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale feature maps.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class SECONDNeck(nn.Module):
    """SECONDFPN neck for upsampling and concatenating multi-scale features.

    Args:
        in_channels (list[int]): Input channels for each scale. Default: [128, 256].
        out_channels (list[int]): Output channels for each deblock. Default: [256, 256].
        upsample_strides (list[int]): Upsampling stride for each scale. Default: [1, 2].
    """

    def __init__(
        self,
        in_channels=(128, 256),
        out_channels=(256, 256),
        upsample_strides=(1, 2),
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels) == len(upsample_strides)

        self.in_channels = in_channels
        self.out_channels = out_channels
        deblocks = []

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]

            if stride > 1:
                upsample_layer = nn.ConvTranspose2d(
                    in_channels[i], out_channel,
                    kernel_size=stride, stride=stride, bias=False,
                )
            else:
                # stride=1: use regular Conv2d (matches SECONDFPN use_conv_for_no_stride=True)
                upsample_layer = nn.Conv2d(
                    in_channels[i], out_channel,
                    kernel_size=1, stride=1, bias=False,
                )

            deblock = nn.Sequential(
                upsample_layer,
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)

        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tuple[torch.Tensor]): Multi-scale input features.

        Returns:
            list[torch.Tensor]: Concatenated upsampled features.
        """
        assert len(x) == len(self.in_channels)
        ups = [self.deblocks[i](x[i]) for i in range(len(self.deblocks))]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]


class BEVDecoder(nn.Module):
    """BEVFusion decoder combining SECOND backbone and SECONDFPN neck.

    Args:
        in_channels (int): Input channels to backbone. Default: 256.
        backbone_out_channels (tuple[int]): Backbone output channels. Default: (128, 256).
        backbone_layer_nums (tuple[int]): Layers per backbone block. Default: (5, 5).
        backbone_layer_strides (tuple[int]): Backbone block strides. Default: (1, 2).
        neck_out_channels (tuple[int]): Neck output channels. Default: (256, 256).
        neck_upsample_strides (tuple[int]): Neck upsample strides. Default: (1, 2).
    """

    def __init__(
        self,
        in_channels=256,
        backbone_out_channels=(128, 256),
        backbone_layer_nums=(5, 5),
        backbone_layer_strides=(1, 2),
        neck_out_channels=(256, 256),
        neck_upsample_strides=(1, 2),
    ):
        super().__init__()

        self.backbone = SECONDBackbone(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
            layer_nums=backbone_layer_nums,
            layer_strides=backbone_layer_strides,
        )

        self.neck = SECONDNeck(
            in_channels=backbone_out_channels,
            out_channels=neck_out_channels,
            upsample_strides=neck_upsample_strides,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Decoded and fused feature map.
        """
        x = self.backbone(x)
        x = self.neck(x)
        return x[0]
