"""Optimizer, LR scheduler, and grad clipping for the NAVSIM SparseDrive runner."""

from __future__ import annotations

import math

import torch


def build_optimizer(model, lr, weight_decay, backbone_lr_mult):
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("img_backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay, "name": "img_backbone"}
        )
    if other_params:
        param_groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay, "name": "default"})
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


class CosineWithLinearWarmup:
    """Per-iteration LR schedule matching SparseDrive's MMCV cosine recipe."""

    def __init__(self, optimizer, max_iters, warmup_iters, warmup_ratio, min_lr_ratio, last_iter=0):
        self.optimizer = optimizer
        self.max_iters = int(max(1, max_iters))
        self.warmup_iters = int(max(0, warmup_iters))
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_iter = int(last_iter)
        self._apply_lr(self.last_iter)

    def get_lr(self, iteration=None):
        i = self.last_iter if iteration is None else int(iteration)
        if self.warmup_iters > 0 and i < self.warmup_iters:
            scale = self.warmup_ratio + (1.0 - self.warmup_ratio) * (i / self.warmup_iters)
            return [base_lr * scale for base_lr in self.base_lrs]

        cosine_iters = max(1, self.max_iters - self.warmup_iters)
        progress = min(max(i - self.warmup_iters, 0), cosine_iters) / cosine_iters
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine) for base_lr in self.base_lrs]

    def _apply_lr(self, iteration):
        lrs = self.get_lr(iteration)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        return lrs

    def step(self, iteration=None):
        if iteration is None:
            self.last_iter += 1
        else:
            self.last_iter = int(iteration)
        return self._apply_lr(self.last_iter)

    def state_dict(self):
        return {
            "max_iters": self.max_iters,
            "warmup_iters": self.warmup_iters,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "last_iter": self.last_iter,
        }

    def load_state_dict(self, state_dict):
        self.max_iters = int(state_dict["max_iters"])
        self.warmup_iters = int(state_dict["warmup_iters"])
        self.warmup_ratio = float(state_dict["warmup_ratio"])
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self.base_lrs = [float(lr) for lr in state_dict["base_lrs"]]
        self.last_iter = int(state_dict["last_iter"])
        self._apply_lr(self.last_iter)


def clip_grad_norm(model, max_norm, norm_type):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=norm_type)
