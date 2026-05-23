import torch.nn as nn

from .base import BaseModule
from .builder import build_loss, build_transformer
from .positional_encoding import build_positional_encoding


class _LossProxy(nn.Module):
    use_sigmoid = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Loss forward is not needed for standalone inference.")


class AnchorFreeHead(BaseModule):
    def __init__(self, init_cfg=None, **kwargs):
        super().__init__(init_cfg)


class DETRHead(AnchorFreeHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        transformer=None,
        positional_encoding=None,
        loss_cls=None,
        loss_bbox=None,
        loss_iou=None,
        train_cfg=None,
        test_cfg=None,
        sync_cls_avg_factor=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.embed_dims = transformer.get("embed_dims", 256) if isinstance(transformer, dict) else 256
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding) if positional_encoding else None
        self.loss_cls = build_loss(loss_cls) if loss_cls else _LossProxy()
        self.loss_bbox = build_loss(loss_bbox) if loss_bbox else _LossProxy()
        self.loss_iou = build_loss(loss_iou) if loss_iou else _LossProxy()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self._init_layers()

    def _init_layers(self):
        return None
