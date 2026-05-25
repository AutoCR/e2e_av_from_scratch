from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
except Exception:  # pragma: no cover - import failure is surfaced at module construction.
    deform_conv2d = None


class DCNv2Pack(nn.Module):
    """Small ModulatedDeformConv2dPack-compatible module for UniAD's ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if deform_conv2d is None:
            raise RuntimeError(
                "torchvision.ops.deform_conv2d is required for UniAD DCNv2 support. "
                "Set build_uniad(use_dcn=False) for an approximate smoke test."
            )
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.conv_offset = nn.Conv2d(
            in_channels,
            deform_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.conv_offset(x)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask.sigmoid(),
        )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        with_dcn: bool = False,
        deform_groups: int = 1,
    ) -> None:
        super().__init__()
        if style not in {"pytorch", "caffe"}:
            raise ValueError(f"Unsupported ResNet style: {style}")
        conv1_stride = stride if style == "caffe" else 1
        conv2_stride = 1 if style == "caffe" else stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=conv1_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if with_dcn:
            self.conv2 = DCNv2Pack(
                planes,
                planes,
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation,
                deform_groups=deform_groups,
                bias=False,
            )
        else:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class ResNet(nn.Module):
    """Subset of MMDetection ResNet needed by the UniAD base e2e config."""

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
    }

    def __init__(
        self,
        depth: int = 101,
        num_stages: int = 4,
        out_indices: Sequence[int] = (1, 2, 3),
        frozen_stages: int = -1,
        norm_cfg: dict | None = None,
        norm_eval: bool = False,
        style: str = "pytorch",
        dcn: dict | None = None,
        stage_with_dcn: Sequence[bool] = (False, False, False, False),
        **_: object,
    ) -> None:
        super().__init__()
        if depth not in self.arch_settings:
            raise ValueError(f"Unsupported ResNet depth {depth}; supported: {sorted(self.arch_settings)}")
        if num_stages != 4:
            raise ValueError("This UniAD ResNet port supports exactly 4 stages.")

        block, stage_blocks = self.arch_settings[depth]
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.norm_requires_grad = bool((norm_cfg or {}).get("requires_grad", True))
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_planes = (64, 128, 256, 512)
        stage_strides = (1, 2, 2, 2)
        deform_groups = int((dcn or {}).get("deform_groups", 1))
        self.res_layers: list[str] = []
        for stage_idx, (planes, blocks, stride) in enumerate(zip(stage_planes, stage_blocks, stage_strides)):
            layer = self._make_layer(
                block=block,
                planes=planes,
                blocks=blocks,
                stride=stride,
                style=style,
                with_dcn=bool(dcn) and bool(stage_with_dcn[stage_idx]),
                deform_groups=deform_groups,
            )
            layer_name = f"layer{stage_idx + 1}"
            self.add_module(layer_name, layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

    def _make_layer(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int,
        style: str,
        with_dcn: bool,
        deform_groups: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                style=style,
                with_dcn=with_dcn,
                deform_groups=deform_groups,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    style=style,
                    with_dcn=with_dcn,
                    deform_groups=deform_groups,
                )
            )
        return nn.Sequential(*layers)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for module in (self.conv1, self.bn1):
                for param in module.parameters():
                    param.requires_grad = False
        for stage_idx in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{stage_idx}", None)
            if layer is None:
                continue
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
        if not self.norm_requires_grad:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True) -> "ResNet":
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        outs = []
        for idx, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if idx in self.out_indices:
                outs.append(x)
        return tuple(outs)


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FPN(nn.Module):
    """Subset of MMDetection FPN needed by UniAD."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        start_level: int = 0,
        add_extra_convs: str | bool = False,
        num_outs: int = 4,
        relu_before_extra_convs: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.start_level = start_level
        self.backbone_end_level = len(in_channels)
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in self.in_channels[start_level:]:
            self.lateral_convs.append(ConvModule(in_channel, out_channels, kernel_size=1))
            self.fpn_convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1))

        extra_levels = num_outs - len(self.fpn_convs)
        if extra_levels < 0:
            raise ValueError(f"num_outs={num_outs} is smaller than input levels={len(self.fpn_convs)}")
        if extra_levels and add_extra_convs:
            for level_idx in range(extra_levels):
                in_channel = self.in_channels[-1] if level_idx == 0 and add_extra_convs == "on_input" else out_channels
                self.fpn_convs.append(ConvModule(in_channel, out_channels, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        inputs = list(inputs)
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"FPN expected {len(self.in_channels)} inputs, got {len(inputs)}")

        laterals = [
            lateral_conv(inputs[idx + self.start_level])
            for idx, lateral_conv in enumerate(self.lateral_convs)
        ]
        for idx in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[idx - 1].shape[2:]
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(laterals[idx], size=prev_shape, mode="nearest")

        outs = [self.fpn_convs[idx](laterals[idx]) for idx in range(len(laterals))]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                while self.num_outs > len(outs):
                    outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
            else:
                extra_source = inputs[-1] if self.add_extra_convs == "on_input" else outs[-1]
                outs.append(self.fpn_convs[len(laterals)](extra_source))
                while self.num_outs > len(outs):
                    extra_source = F.relu(outs[-1]) if self.relu_before_extra_convs else outs[-1]
                    outs.append(self.fpn_convs[len(outs)](extra_source))
        return tuple(outs)
