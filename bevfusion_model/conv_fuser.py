"""BEVFusion fuser: multi-modal BEV feature fusion."""

import torch
import torch.nn as nn


class ConvFuser(nn.Sequential):
    """Fuses multi-modal BEV features by concatenation + conv.

    Args:
        in_channels (list[int]): List of input channel counts from each modality.
        out_channels (int): Output channel count.
    """

    def __init__(self, in_channels: list, out_channels: int):
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inputs: list) -> torch.Tensor:
        """Fuse multi-modal features.

        Args:
            inputs (list[torch.Tensor]): List of BEV features from each modality.

        Returns:
            torch.Tensor: Fused BEV features.
        """
        return super().forward(torch.cat(inputs, dim=1))
