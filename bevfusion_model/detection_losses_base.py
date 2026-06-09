"""
Pure PyTorch/NumPy loss and helper primitives for BEVFusion training.

Ported from:
  - sparsedrive_model/sparsedrive/nn_utils.py (loss classes & weighted_loss helper)
  - bevfusion/mmdet3d/core/utils/gaussian.py (gaussian heatmap drawing)
  - bevfusion/mmdet3d/core/bbox/util.py (bbox normalization)

No mmdet/mmcv imports; self-contained for standalone BEVFusion training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "FocalLoss",
    "GaussianFocalLoss",
    "L1Loss",
    "weighted_loss",
    "gaussian_radius",
    "gaussian_2d",
    "draw_heatmap_gaussian",
    "clip_sigmoid",
    "normalize_bbox",
]


# ============================================================================
# Loss helpers and loss classes (from sparsedrive nn_utils.py)
# ============================================================================

def weighted_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply weight and reduction to a loss tensor.

    Args:
        loss: Loss tensor of any shape.
        weight: Optional weight tensor. If provided and dims don't match loss,
                unsqueeze weight on the last axis until dims match (broadcasting).
        reduction: "mean", "sum", or "none".
        avg_factor: If provided, divide by this instead of counting elements.

    Returns:
        Weighted and reduced loss scalar (or tensor if reduction="none").
    """
    if weight is not None:
        # Broadcasting-aware weight handling: unsqueeze weight on last axis
        # until it matches loss.dim() (handles cases where weight is [N] but
        # loss is [N, H, W] etc.)
        while weight.dim() < loss.dim():
            weight = weight.unsqueeze(-1)
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
    """L1 (absolute difference) loss."""

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        loss = torch.abs(pred - target)
        return weighted_loss(loss, weight, reduction, avg_factor) * self.loss_weight


class FocalLoss(nn.Module):
    """Sigmoid focal loss for object detection.

    Targets are integer class labels [0, num_classes). Converts to one-hot
    internally, applies focal weighting, and computes weighted binary cross-entropy.
    """

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
        num_classes = pred.shape[-1]
        valid = (target >= 0) & (target < num_classes)
        target_onehot = pred.new_zeros(pred.shape)
        target_onehot[valid, target[valid]] = 1
        pred_sigmoid = pred.sigmoid()
        pt = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
        focal_weight = (self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)) * (1 - pt).pow(self.gamma)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_onehot, reduction="none") * focal_weight
        return weighted_loss(loss, weight, reduction, avg_factor) * self.loss_weight


class GaussianFocalLoss(nn.Module):
    """Gaussian focal loss for heatmap-based detection.

    Assumes pred is already sigmoided by caller (values in [0, 1]).
    Target is a heatmap of 0/1 values.
    """

    def __init__(self, alpha=2.0, gamma=4.0, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred.log()) * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -((1 - pred).log()) * pred.pow(self.alpha) * neg_weights * (~pos_weights)
        return weighted_loss(pos_loss + neg_loss, weight, reduction, avg_factor) * self.loss_weight


# ============================================================================
# Gaussian heatmap helpers (from bevfusion mmdet3d/core/utils/gaussian.py)
# ============================================================================

def gaussian_2d(shape, sigma=1):
    """Generate a 2D gaussian map.

    Args:
        shape (list[int]): [height, width] of the map.
        sigma (float): Sigma parameter for gaussian. Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map of shape [height, width].
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Draw a gaussian kernel on a heatmap (in-place).

    Args:
        heatmap (torch.Tensor): Heatmap tensor of shape [..., H, W].
        center (tuple or list): (x, y) center coordinates.
        radius (int): Radius of the gaussian kernel.
        k (int): Scaling factor for the gaussian. Defaults to 1.

    Returns:
        torch.Tensor: The modified heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top : radius + bottom, radius - left : radius + right]
    ).to(heatmap.device, torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Compute the radius of a gaussian kernel to approximate a detection box.

    Uses the 3-case quadratic root formula to find the max radius such that
    the gaussian's footprint overlaps the box with at least min_overlap.

    Args:
        det_size (tuple): (height, width) of the detection box.
        min_overlap (float): Minimum overlap ratio. Defaults to 0.5.

    Returns:
        float or tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


# ============================================================================
# Math helpers
# ============================================================================

def clip_sigmoid(x, eps=1e-4):
    """Clip sigmoid output to avoid log(0) and log(1).

    Uses non-in-place sigmoid so gradients are preserved and the input
    tensor is not mutated (important when input needs gradient).

    Args:
        x: Input tensor (logits).
        eps (float): Clipping epsilon. Defaults to 1e-4.

    Returns:
        torch.Tensor: Clipped sigmoid(x) in range [eps, 1-eps].
    """
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)


def normalize_bbox(bboxes, pc_range=None):
    """Normalize bounding boxes for BEVFusion training.

    Input format: [cx, cy, cz, w, l, h, yaw, (vx, vy)]
    Output format: [cx, cy, log(w), log(l), cz, log(h), sin(yaw), cos(yaw), (vx, vy)]

    Args:
        bboxes (torch.Tensor): Bounding boxes of shape [..., 7 or 9].
            7 = [cx, cy, cz, w, l, h, yaw]
            9 = [cx, cy, cz, w, l, h, yaw, vx, vy]
        pc_range: Unused; kept for signature compatibility.

    Returns:
        torch.Tensor: Normalized bboxes of shape [..., 8 or 10].
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


# ============================================================================
# Self-tests
# ============================================================================

if __name__ == "__main__":
    print("Running self-tests...")

    # Test 1: gaussian heatmap
    print("\n[1] Testing gaussian heatmap drawing...")
    heatmap = torch.zeros(3, 180, 180)
    radius = int(gaussian_radius((torch.tensor(4.0), torch.tensor(4.0)), 0.1))
    print(f"    Computed radius: {radius}")
    heatmap_center = heatmap[0]
    draw_heatmap_gaussian(heatmap_center, (90, 90), radius, k=1)
    peak_val = heatmap_center.max().item()
    print(f"    Peak value after drawing gaussian: {peak_val}")
    assert abs(peak_val - 1.0) < 1e-5, f"Expected peak ~1.0, got {peak_val}"
    print("    PASS")

    # Test 2: bbox normalization
    print("\n[2] Testing normalize_bbox...")
    bboxes = torch.randn(2, 9)
    # Make sizes and heights positive for log
    bboxes[:, 3:6] = torch.abs(bboxes[:, 3:6]) + 0.1  # w, l, h > 0
    output = normalize_bbox(bboxes, pc_range=None)
    print(f"    Input shape: {bboxes.shape}, Output shape: {output.shape}")
    assert output.shape == (2, 10), f"Expected [2,10], got {output.shape}"
    # Verify log relationship: output[..., 2:3] = log(input[..., 3:4])
    assert torch.allclose(output[:, 2:3], torch.log(bboxes[:, 3:4])), "log(w) mismatch"
    assert torch.allclose(output[:, 3:4], torch.log(bboxes[:, 4:5])), "log(l) mismatch"
    assert torch.allclose(output[:, 5:6], torch.log(bboxes[:, 5:6])), "log(h) mismatch"
    print("    PASS")

    # Test 3: Loss functions
    print("\n[3] Testing loss functions...")

    # L1Loss
    l1_loss = L1Loss(reduction="mean", loss_weight=1.0)
    pred = torch.randn(4, 10)
    target = torch.randn(4, 10)
    loss_val = l1_loss(pred, target)
    assert loss_val.numel() == 1, "L1Loss should return scalar"
    assert torch.isfinite(loss_val).all(), "L1Loss output contains non-finite values"
    print(f"    L1Loss: {loss_val.item():.6f} - PASS")

    # FocalLoss
    focal_loss = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0)
    pred = torch.randn(8, 5)  # [batch, num_classes]
    target = torch.randint(0, 5, (8,))  # class labels
    loss_val = focal_loss(pred, target)
    assert loss_val.numel() == 1, "FocalLoss should return scalar"
    assert torch.isfinite(loss_val).all(), "FocalLoss output contains non-finite values"
    print(f"    FocalLoss: {loss_val.item():.6f} - PASS")

    # GaussianFocalLoss
    gauss_loss = GaussianFocalLoss(alpha=2.0, gamma=4.0, reduction="mean", loss_weight=1.0)
    pred = torch.sigmoid(torch.randn(4, 64, 64))  # already sigmoided
    target = torch.randint(0, 2, (4, 64, 64)).float()
    loss_val = gauss_loss(pred, target)
    assert loss_val.numel() == 1, "GaussianFocalLoss should return scalar"
    assert torch.isfinite(loss_val).all(), "GaussianFocalLoss output contains non-finite values"
    print(f"    GaussianFocalLoss: {loss_val.item():.6f} - PASS")

    print("\n" + "="*60)
    print("All self-tests PASSED!")
    print("="*60)
