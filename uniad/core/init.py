import math
import torch.nn as nn


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean=mean, std=std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    return float(-math.log((1 - prior_prob) / prior_prob))
