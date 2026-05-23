import torch


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result = dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
    if attrs is not None:
        result["attrs_3d"] = attrs
    return result


def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = bbox.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def bbox_xyxy_to_cxcywh(bbox):
    x1, y1, x2, y2 = bbox.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def bbox_overlaps(*args, **kwargs):
    raise NotImplementedError("bbox_overlaps is not ported for standalone inference.")
