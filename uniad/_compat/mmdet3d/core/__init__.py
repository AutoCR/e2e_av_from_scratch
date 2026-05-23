import torch

from uniad.core.bbox import bbox3d2result


def xywhr2xyxyr(boxes):
    return boxes


__all__ = ["bbox3d2result", "xywhr2xyxyr"]
