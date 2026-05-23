import torch

from uniad.core.ops import multi_scale_deformable_attn_pytorch


def nms_bev(boxes, scores, thresh=0.5):
    return torch.argsort(scores, descending=True)


__all__ = ["multi_scale_deformable_attn_pytorch", "nms_bev"]
