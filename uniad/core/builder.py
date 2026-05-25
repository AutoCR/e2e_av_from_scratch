import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAssigner, CheckpointCompatibleModule


class GenericLoss(nn.Module):
    def __init__(self, use_sigmoid=False, loss_weight=1.0, **kwargs):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.cfg = kwargs

    def forward(self, pred, target=None, *args, **kwargs):
        if target is None:
            return pred.sum() * 0
        if self.use_sigmoid:
            return F.binary_cross_entropy_with_logits(pred, target.float(), reduction="mean") * self.loss_weight
        return F.l1_loss(pred, target, reduction="mean") * self.loss_weight


class GenericAssigner(BaseAssigner):
    def __init__(self, **kwargs):
        self.cfg = kwargs


class GenericSampler:
    def __init__(self, **kwargs):
        self.cfg = kwargs


class GenericCost:
    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight
        self.cfg = kwargs

    def __call__(self, pred, target):
        if pred.numel() == 0 or target.numel() == 0:
            return pred.new_zeros((pred.shape[0], target.shape[0]))
        pred = pred.reshape(pred.shape[0], -1).float()
        target = target.reshape(target.shape[0], -1).float()
        width = min(pred.shape[-1], target.shape[-1])
        return torch.cdist(pred[:, :width], target[:, :width], p=1) * self.weight


class BackbonePlaceholder(CheckpointCompatibleModule):
    pass


class NeckPlaceholder(CheckpointCompatibleModule):
    pass


def _cfg_args(cfg):
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        return cfg
    args = copy.deepcopy(cfg)
    args.pop("type", None)
    return args


def build_head(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = copy.deepcopy(cfg)
    head_type = cfg.pop("type")
    if head_type == "BEVFormerTrackHead":
        from uniad.models.dense_heads.track_head import BEVFormerTrackHead

        return BEVFormerTrackHead(**cfg)
    if head_type == "PansegformerHead":
        from uniad.models.dense_heads.panseg_head import PansegformerHead

        return PansegformerHead(**cfg)
    if head_type == "OccHead":
        from uniad.models.dense_heads.occ_head import OccHead

        return OccHead(**cfg)
    if head_type == "MotionHead":
        from uniad.models.dense_heads.motion_head import MotionHead

        return MotionHead(**cfg)
    if head_type == "PlanningHeadSingleMode":
        from uniad.models.dense_heads.planning_head import PlanningHeadSingleMode

        return PlanningHeadSingleMode(**cfg)
    raise KeyError(f"Unsupported head: {head_type}")


def build_loss(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = copy.deepcopy(cfg)
    loss_type = cfg.pop("type")
    if loss_type in ("FocalLoss", "L1Loss", "GIoULoss", "CrossEntropyLoss"):
        return GenericLoss(**cfg)
    if loss_type == "DiceLoss":
        from uniad.models.losses.dice_loss import DiceLoss

        return DiceLoss(**cfg)
    if loss_type == "DiceLossWithMasks":
        from uniad.models.losses.occflow_loss import DiceLossWithMasks

        return DiceLossWithMasks(**cfg)
    if loss_type == "FieryBinarySegmentationLoss":
        from uniad.models.losses.occflow_loss import FieryBinarySegmentationLoss

        return FieryBinarySegmentationLoss(**cfg)
    if loss_type == "TrajLoss":
        from uniad.models.losses.traj_loss import TrajLoss

        return TrajLoss(**cfg)
    if loss_type == "PlanningLoss":
        from uniad.models.losses.planning_loss import PlanningLoss

        return PlanningLoss(**cfg)
    if loss_type == "CollisionLoss":
        from uniad.models.losses.planning_loss import CollisionLoss

        return CollisionLoss(**cfg)
    if loss_type == "ClipMatcher":
        from uniad.models.losses.track_loss import ClipMatcher

        return ClipMatcher(**cfg)
    if loss_type == "MTPLoss":
        from uniad.models.losses.mtp_loss import MTPLoss

        return MTPLoss(**cfg)
    raise KeyError(f"Unsupported loss: {loss_type}")


def build_bbox_coder(cfg):
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    coder_type = cfg.pop("type")
    if coder_type == "NMSFreeCoder":
        from uniad.core.bbox.coders.nms_free_coder import NMSFreeCoder

        return NMSFreeCoder(**cfg)
    if coder_type == "DETRTrack3DCoder":
        from uniad.core.bbox.coders.detr3d_track_coder import DETRTrack3DCoder

        return DETRTrack3DCoder(**cfg)
    raise KeyError(f"Unsupported bbox coder: {coder_type}")


def build_assigner(cfg):
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    assigner_type = cfg.pop("type", None)
    if assigner_type == "HungarianAssigner3D":
        from uniad.core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D

        return HungarianAssigner3D(**cfg)
    if assigner_type == "HungarianAssigner3DTrack":
        from uniad.core.bbox.assigners.hungarian_assigner_3d_track import HungarianAssigner3DTrack

        return HungarianAssigner3DTrack(**cfg)
    if assigner_type == "HungarianAssigner_filter":
        from uniad.models.dense_heads.seg_head_plugin.seg_assigner import HungarianAssigner_filter

        return HungarianAssigner_filter(**cfg)
    if assigner_type == "HungarianAssigner_multi_info":
        from uniad.models.dense_heads.seg_head_plugin.seg_assigner import HungarianAssigner_multi_info

        return HungarianAssigner_multi_info(**cfg)
    if assigner_type in ("HungarianAssigner", None):
        return GenericAssigner(**cfg)
    raise KeyError(f"Unsupported assigner: {assigner_type}")


def build_match_cost(cfg):
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    cost_type = cfg.pop("type", None)
    if cost_type == "BBox3DL1Cost":
        from uniad.core.bbox.match_costs.match_cost import BBox3DL1Cost

        return BBox3DL1Cost(**cfg)
    if cost_type == "DiceCost":
        from uniad.core.bbox.match_costs.match_cost import DiceCost

        return DiceCost(**cfg)
    if cost_type in ("FocalLossCost", "BBoxL1Cost", "IoUCost", "ClassificationCost", None):
        return GenericCost(**cfg)
    raise KeyError(f"Unsupported match cost: {cost_type}")


def build_sampler(cfg, context=None):
    if cfg is None:
        return GenericSampler()
    cfg = copy.deepcopy(cfg)
    sampler_type = cfg.pop("type", None)
    if sampler_type == "PseudoSampler_segformer":
        from uniad.models.dense_heads.seg_head_plugin.seg_assigner import PseudoSampler_segformer

        return PseudoSampler_segformer(**cfg)
    return GenericSampler(**cfg)


def build_backbone(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = copy.deepcopy(cfg)
    backbone_type = cfg.pop("type")
    if backbone_type == "ResNet":
        from uniad.core.image_modules import ResNet

        return ResNet(**cfg)
    return BackbonePlaceholder({"type": backbone_type, **cfg})


def build_neck(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = copy.deepcopy(cfg)
    neck_type = cfg.pop("type")
    if neck_type == "FPN":
        from uniad.core.image_modules import FPN

        return FPN(**cfg)
    return NeckPlaceholder({"type": neck_type, **cfg})


def build_transformer(cfg):
    from .transformer import build_transformer as _build_transformer

    return _build_transformer(cfg)


def build_transformer_layer_sequence(cfg):
    from .transformer import build_transformer_layer_sequence as _build_transformer_layer_sequence

    return _build_transformer_layer_sequence(cfg)
