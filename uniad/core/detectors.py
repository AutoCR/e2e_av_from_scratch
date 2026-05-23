from .base import BaseModule
from .builder import build_backbone, build_head, build_neck


class MVXTwoStageDetector(BaseModule):
    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.pts_bbox_head = build_head(pts_bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained

    @property
    def with_img_neck(self):
        return self.img_neck is not None

    def init_weights(self):
        for module in (self.img_backbone, self.img_neck, self.pts_bbox_head):
            if hasattr(module, "init_weights"):
                module.init_weights()
