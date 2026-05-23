import torch


def bbox_overlaps_nearest_3d(bboxes1, bboxes2, *args, **kwargs):
    return bboxes1.new_zeros((bboxes1.shape[0], bboxes2.shape[0]))


def bbox_overlaps_3d(bboxes1, bboxes2, *args, **kwargs):
    return bboxes1.new_zeros((bboxes1.shape[0], bboxes2.shape[0]))


__all__ = ["bbox_overlaps_3d", "bbox_overlaps_nearest_3d"]
