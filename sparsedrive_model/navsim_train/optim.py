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


def compute_grad_norm(parameters, norm_type=2.0):
    """Total gradient norm computed in fp64 to avoid fp32 reduction overflow.

    ``torch.nn.utils.clip_grad_norm_`` squares-and-sums every parameter's grad
    in the grads' own dtype. In pure fp32 a few large-but-finite gradients
    (O(1e18)) make the global sum-of-squares exceed the fp32 max (3.4e38) and
    saturate to ``inf`` even though *no individual grad element is non-finite*.
    The old code then misread that ``inf`` as "NaN/inf gradients" and skipped the
    step forever. Accumulating each per-parameter norm in float64 lifts the
    overflow ceiling to ~1.8e308, so a genuinely finite (if large) gradient
    yields a finite total norm and can be clipped normally.

    Returns a 0-dim float64 tensor. It is non-finite only when some grad element
    is actually NaN/Inf -- a true explosion -- not merely large.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros((), dtype=torch.float64)
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        return max(g.detach().abs().max().to(torch.float64) for g in grads)
    total = torch.zeros((), dtype=torch.float64)
    for g in grads:
        # Per-parameter norm in fp32 is safe (one tensor rarely overflows); the
        # cross-parameter accumulation is what overflows, so accumulate in fp64.
        param_norm = g.detach().norm(norm_type).to(torch.float64)
        total += param_norm ** norm_type
    return total ** (1.0 / norm_type)


def clip_grad_norm(model, max_norm, norm_type):
    """Clip grads to ``max_norm`` using an fp64 total-norm.

    Mirrors ``clip_grad_norm_`` (scale every grad by ``max_norm/total_norm`` when
    the total exceeds ``max_norm``) but computes the total in fp64 so a large yet
    finite gradient is clipped instead of being mistaken for an explosion. When
    the total norm is genuinely non-finite (real NaN/Inf grad element) the grads
    are left untouched so the caller's guard can detect and skip the step.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = compute_grad_norm(parameters, norm_type)
    if not torch.isfinite(total_norm):
        return total_norm
    max_norm = float(max_norm)
    clip_coef = max_norm / (float(total_norm) + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    return total_norm
