import math

import torch
import torch.nn as nn


def xavier_init(module, gain=1.0, bias=0.0, distribution="normal"):
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif distribution == "normal":
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        raise ValueError(f"unsupported xavier distribution: {distribution}")
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0.0):
    nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    return float(-math.log((1 - prior_prob) / prior_prob))


def build_norm_layer(name, num_features=None, **kwargs):
    if isinstance(name, dict):
        cfg = dict(name)
        layer_type = cfg.pop("type")
        num_features = cfg.pop("normalized_shape", num_features)
        kwargs.update(cfg)
        name = layer_type
    if name == "LN":
        return name, nn.LayerNorm(num_features, **kwargs)
    if name in {"BN", "BN2d"}:
        return name, nn.BatchNorm2d(num_features, **kwargs)
    if name == "BN1d":
        return name, nn.BatchNorm1d(num_features, **kwargs)
    if name == "GN":
        num_groups = kwargs.pop("num_groups", 32)
        return name, nn.GroupNorm(num_groups, num_features, **kwargs)
    raise ValueError(f"unsupported norm layer: {name}")


def build_activation_layer(name, **kwargs):
    if isinstance(name, dict):
        cfg = dict(name)
        layer_type = cfg.pop("type")
        kwargs.update(cfg)
        name = layer_type
    if name == "ReLU":
        return nn.ReLU(**kwargs)
    if name == "GELU":
        return nn.GELU(**kwargs)
    if name == "SiLU":
        return nn.SiLU(**kwargs)
    raise ValueError(f"unsupported activation layer: {name}")


def build_dropout(name=None, drop_prob=None):
    if name is None:
        return nn.Identity()
    if isinstance(name, dict):
        cfg = dict(name)
        layer_type = cfg.pop("type")
        drop_prob = cfg.pop("drop_prob", cfg.pop("p", drop_prob))
        name = layer_type
    drop_prob = 0.0 if drop_prob is None else drop_prob
    if name == "Dropout":
        return nn.Dropout(drop_prob)
    if name == "DropPath":
        from timm.layers import DropPath

        return DropPath(drop_prob)
    raise ValueError(f"unsupported dropout layer: {name}")


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


def weighted_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is not None:
        return loss.sum() / max(float(avg_factor), 1.0)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unsupported reduction: {reduction}")


class L1Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        loss = torch.abs(pred - target)
        return weighted_loss(loss, weight, reduction, avg_factor) * self.loss_weight


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=self.beta)
        return weighted_loss(loss, weight, reduction, avg_factor) * self.loss_weight


class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0):
        super().__init__()
        if not use_sigmoid:
            raise ValueError("Only sigmoid focal loss is implemented.")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        pred = torch.nan_to_num(
            pred.float(), nan=0.0, posinf=80.0, neginf=-80.0
        ).clamp_(-80.0, 80.0)
        num_classes = pred.shape[-1]
        valid = (target >= 0) & (target < num_classes)
        target_onehot = pred.new_zeros(pred.shape)
        target_onehot[valid, target[valid]] = 1
        pred_sigmoid = pred.sigmoid()
        pt = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
        focal_weight = (self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)) * (1 - pt).pow(self.gamma)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_onehot, reduction="none") * focal_weight
        return weighted_loss(loss, weight, reduction, avg_factor) * self.loss_weight


class CrossEntropyLoss(nn.Module):
    def __init__(self, use_sigmoid=False, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        if self.use_sigmoid:
            pred = torch.nan_to_num(
                pred.float(), nan=0.0, posinf=80.0, neginf=-80.0
            ).clamp_(-80.0, 80.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        else:
            loss = torch.nn.functional.cross_entropy(pred, target.long(), reduction="none")
        return weighted_loss(loss, weight, reduction, avg_factor) * self.loss_weight


class GaussianFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=4.0, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        # Sanitize: pred is expected to be a probability in (0, 1); NaN/Inf from
        # upstream sigmoid would survive .clamp(1e-6, 1-1e-6) (clamp(NaN)=NaN),
        # so replace non-finite values explicitly before clamping.
        pred = torch.nan_to_num(pred.float(), nan=0.5, posinf=1.0, neginf=0.0)
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred.log()) * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -((1 - pred).log()) * pred.pow(self.alpha) * neg_weights * (~pos_weights)
        return weighted_loss(pos_loss + neg_loss, weight, reduction, avg_factor) * self.loss_weight
