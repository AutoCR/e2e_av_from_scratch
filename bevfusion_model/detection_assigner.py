"""
Pure-PyTorch DETR-style Hungarian matcher for BEVFusion TransFusionHead training loss.

Ported from BEVFusion mmdet3d Hungarian assigner with focal loss cost and 3D IoU cost.
No mmdet/mmcv dependencies.
"""

import sys
from pathlib import Path

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bevfusion_model.detection_losses_base import normalize_bbox
from bevfusion_model.iou3d import bbox_overlaps_3d, IOU3D_CUDA_AVAILABLE

__all__ = ["FocalLossCost", "BBoxBEVL1Cost", "IoU3DCost", "HungarianAssigner3D", "build_assigner"]


class FocalLossCost:
    """Focal loss cost for classification matching in Hungarian algorithm.

    Computes the focal loss cost between predicted class logits and ground-truth labels.
    """

    def __init__(self, weight=0.15, alpha=0.25, gamma=2, eps=1e-12):
        """
        Args:
            weight: Cost weight scalar.
            alpha: Focal loss alpha (positive class weight).
            gamma: Focal loss gamma (modulation exponent).
            eps: Small epsilon to prevent log(0).
        """
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """Compute focal loss cost.

        Args:
            cls_pred: Raw logits [num_query, num_classes].
            gt_labels: Ground-truth class labels [num_gt]; integer in [0, num_classes).

        Returns:
            cost tensor [num_query, num_gt] where cost[i, j] is the focal loss cost
            of query i matching to gt j.
        """
        cls_pred_sigmoid = cls_pred.sigmoid()  # [num_query, num_classes]

        # Negative cost: cost of predicting background (class-agnostic)
        # neg_cost = -(1 - p + eps).log() * (1 - alpha) * p^gamma
        neg_cost = -(1 - cls_pred_sigmoid + self.eps).log() * (1 - self.alpha) * cls_pred_sigmoid.pow(self.gamma)

        # Positive cost: cost of predicting the ground-truth class
        # pos_cost = -(p + eps).log() * alpha * (1 - p)^gamma
        pos_cost = -(cls_pred_sigmoid + self.eps).log() * self.alpha * (1 - cls_pred_sigmoid).pow(self.gamma)

        # Select costs for ground-truth classes
        # pos_cost[:, gt_labels] shape [num_query, num_gt]
        # neg_cost[:, gt_labels] shape [num_query, num_gt]
        cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]

        return cost * self.weight


class BBoxBEVL1Cost:
    """BEV L1 distance cost between predicted and ground-truth bounding boxes.

    Normalizes xy center coordinates by point cloud range, then computes L1 distance
    over the full 10-d normalized representation.
    """

    def __init__(self, weight=0.25):
        """
        Args:
            weight: Cost weight scalar.
        """
        self.weight = weight

    def __call__(self, bboxes_norm, gt_bboxes_norm, pc_range):
        """Compute BEV L1 cost.

        Args:
            bboxes_norm: Normalized predicted boxes [num_query, 10].
            gt_bboxes_norm: Normalized ground-truth boxes [num_gt, 10].
            pc_range: Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax].

        Returns:
            cost tensor [num_query, num_gt].
        """
        # Normalize xy center coordinates to [0, 1] within pc_range
        pc_start = torch.tensor(pc_range[0:2], dtype=bboxes_norm.dtype, device=bboxes_norm.device)
        pc_size = torch.tensor(pc_range[3:5], dtype=bboxes_norm.dtype, device=bboxes_norm.device) - pc_start

        # Make copies and normalize only xy (indices 0, 1)
        bboxes_norm_xy = bboxes_norm.clone()
        bboxes_norm_xy[:, :2] = (bboxes_norm_xy[:, :2] - pc_start) / pc_size

        gt_bboxes_norm_xy = gt_bboxes_norm.clone()
        gt_bboxes_norm_xy[:, :2] = (gt_bboxes_norm_xy[:, :2] - pc_start) / pc_size

        # L1 distance over full 10-d normalized vector
        cost = torch.cdist(bboxes_norm_xy, gt_bboxes_norm_xy, p=1)

        return cost * self.weight


class IoU3DCost:
    """3D IoU cost for bounding box matching in Hungarian algorithm.

    Takes precomputed 3D IoU and negates it as a cost.
    """

    def __init__(self, weight=0.25):
        """
        Args:
            weight: Cost weight scalar.
        """
        self.weight = weight

    def __call__(self, iou):
        """Compute IoU cost.

        Args:
            iou: Precomputed 3D IoU tensor [num_query, num_gt].

        Returns:
            cost tensor [num_query, num_gt] = -iou * weight.
        """
        return -iou * self.weight


class HungarianAssigner3D:
    """Hungarian assignment matcher for 3D object detection.

    Combines three cost terms (classification, regression, IoU) and solves
    the optimal bipartite matching problem using scipy's linear_sum_assignment.
    """

    def __init__(self, cls_cost, reg_cost, iou_cost, pc_range):
        """
        Args:
            cls_cost: FocalLossCost instance.
            reg_cost: BBoxBEVL1Cost instance.
            iou_cost: IoU3DCost instance.
            pc_range: Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax] as list.
        """
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.iou_cost = iou_cost
        self.pc_range = pc_range

    def assign(self, bbox_pred_world, cls_pred, gt_bboxes_world, gt_labels):
        """Assign queries to ground-truth boxes using Hungarian matching.

        Args:
            bbox_pred_world: Predicted boxes [num_query, 9] in format
                [cx, cy, cz, w, l, h, yaw, vx, vy] in lidar/world coordinates.
            cls_pred: Raw class logits [num_query, num_classes].
            gt_bboxes_world: Ground-truth boxes [num_gt, 9] in same format.
            gt_labels: Ground-truth class labels [num_gt]; long tensor.

        Returns:
            Tuple of:
            - assigned_gt_inds: [num_query] long tensor, 0 = background, >0 = 1-indexed gt index.
            - assigned_labels: [num_query] long tensor, -1 = unassigned, >=0 = gt label.
            - max_overlaps: [num_query] float tensor, 3D IoU of matched pairs (0 if unmatched).
        """
        num_query = bbox_pred_world.shape[0]
        num_gt = gt_bboxes_world.shape[0]

        # Initialize output tensors
        assigned_gt_inds = torch.zeros(num_query, dtype=torch.long, device=bbox_pred_world.device)
        assigned_labels = torch.full((num_query,), -1, dtype=torch.long, device=bbox_pred_world.device)
        max_overlaps = torch.zeros(num_query, dtype=bbox_pred_world.dtype, device=bbox_pred_world.device)

        # Early exit if no ground truth or no queries
        if num_gt == 0 or num_query == 0:
            return assigned_gt_inds, assigned_labels, max_overlaps

        # Compute classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)  # [num_query, num_gt]

        # Compute regression cost (BEV L1 distance)
        bbox_pred_norm = normalize_bbox(bbox_pred_world)  # [num_query, 10]
        gt_bboxes_norm = normalize_bbox(gt_bboxes_world)  # [num_gt, 10]
        reg_cost = self.reg_cost(bbox_pred_norm, gt_bboxes_norm, self.pc_range)  # [num_query, num_gt]

        # Compute IoU cost
        if IOU3D_CUDA_AVAILABLE and bbox_pred_world.is_cuda:
            # Use CUDA-accelerated 3D IoU (only when CUDA available and tensors on CUDA)
            iou = bbox_overlaps_3d(bbox_pred_world[:, :7], gt_bboxes_world[:, :7])  # [num_query, num_gt]
            iou_cost = self.iou_cost(iou)  # [num_query, num_gt]
        else:
            # CPU or no CUDA: IoU disabled (returns zeros)
            iou = None
            iou_cost = 0  # Broadcasts as scalar

        # Combine all costs
        cost = cls_cost + reg_cost + iou_cost

        # Replace NaN/Inf with large finite numbers for numerical stability
        cost = torch.nan_to_num(cost, nan=1e5, posinf=1e5, neginf=-1e5)

        # Solve Hungarian assignment on CPU
        cost_cpu = cost.detach().cpu().numpy()
        row, col = linear_sum_assignment(cost_cpu)

        # Move assignment indices to same device as input
        row = torch.tensor(row, dtype=torch.long, device=bbox_pred_world.device)
        col = torch.tensor(col, dtype=torch.long, device=bbox_pred_world.device)

        # Update output tensors with matches
        assigned_gt_inds[row] = col + 1  # 1-indexed
        assigned_labels[row] = gt_labels[col]

        # Set max overlaps if IoU was computed
        if iou is not None:
            max_overlaps[row] = iou[row, col]

        return assigned_gt_inds, assigned_labels, max_overlaps


def build_assigner(assigner_cfg):
    """Build a HungarianAssigner3D from config dict.

    Args:
        assigner_cfg: Config dict with keys:
            - cls_cost: Dict with keys (gamma, alpha, weight)
            - reg_cost: Dict with key (weight)
            - iou_cost: Dict with key (weight)
            - pc_range: List [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
        HungarianAssigner3D instance.
    """
    cls_cost = FocalLossCost(**assigner_cfg["cls_cost"])
    reg_cost = BBoxBEVL1Cost(**assigner_cfg["reg_cost"])
    iou_cost = IoU3DCost(**assigner_cfg["iou_cost"])

    return HungarianAssigner3D(
        cls_cost=cls_cost,
        reg_cost=reg_cost,
        iou_cost=iou_cost,
        pc_range=assigner_cfg["pc_range"],
    )


if __name__ == "__main__":
    import sys

    # Self-test: build assigner from config and run matching
    from bevfusion_model.configs.bevfusion_hyperparams import get_training_hyperparams

    cfg = get_training_hyperparams()["detection_head"]["train_cfg"]["assigner"]
    asg = build_assigner(cfg)

    torch.manual_seed(0)
    num_q, num_gt, num_cls = 200, 5, 5

    # Create valid bbox predictions (positive dimensions for log)
    bbox_pred = torch.randn(num_q, 9)
    bbox_pred[:, 3:6] = bbox_pred[:, 3:6].abs() + 1.0  # w, l, h > 0

    # Create valid gt boxes
    gt = torch.randn(num_gt, 9)
    gt[:, 3:6] = gt[:, 3:6].abs() + 1.0  # w, l, h > 0

    # Class predictions (raw logits)
    cls_pred = torch.randn(num_q, num_cls)

    # Ground-truth labels
    gt_labels = torch.randint(0, num_cls, (num_gt,))

    # Run assignment
    ag, al, mo = asg.assign(bbox_pred, cls_pred, gt, gt_labels)

    # Verify outputs
    assert ag.shape == (num_q,), f"assigned_gt_inds shape mismatch: {ag.shape}"
    assert al.shape == (num_q,), f"assigned_labels shape mismatch: {al.shape}"
    assert mo.shape == (num_q,), f"max_overlaps shape mismatch: {mo.shape}"

    num_matched = (ag > 0).sum().item()
    assert num_matched == num_gt, f"Expected {num_gt} matches, got {num_matched}"

    # Verify label consistency
    valid_mask = ag > 0
    assert al[valid_mask].min() >= 0, "assigned_labels has negative values for matched queries"

    print(f"assigner OK: matched {num_matched} queries to {num_gt} gts")

    # Test empty gt case
    ag2, al2, mo2 = asg.assign(bbox_pred, cls_pred, torch.zeros(0, 9), torch.zeros(0, dtype=torch.long))
    assert (ag2 == 0).all(), "Empty gt should assign all to background"
    print("empty-gt OK")

    print("\nAll self-tests PASSED!")
