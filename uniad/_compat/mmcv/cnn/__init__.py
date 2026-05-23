import torch.nn as nn

from uniad.core.init import bias_init_with_prob, constant_init, normal_init, xavier_init
from uniad.core.transformer import build_activation_layer, build_norm_layer

Linear = nn.Linear
Conv2d = nn.Conv2d


class ConvModule(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, conv_cfg=None, norm_cfg=None, act_cfg=dict(type="ReLU"), **kwargs):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
        if norm_cfg is not None:
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        if act_cfg is not None:
            layers.append(build_activation_layer(act_cfg))
        super().__init__(*layers)


def build_conv_layer(cfg, *args, **kwargs):
    return nn.Conv2d(*args, **kwargs)

__all__ = [
    "Linear",
    "Conv2d",
    "ConvModule",
    "bias_init_with_prob",
    "build_conv_layer",
    "constant_init",
    "normal_init",
    "xavier_init",
    "build_activation_layer",
    "build_norm_layer",
]
