import copy
import torch
import torch.nn as nn

from .base import BaseModule
from .tensor_utils import inverse_sigmoid


class FFN(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        act_cfg=None,
        dropout_layer=None,
        add_identity=True,
        **kwargs,
    ):
        super().__init__()
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(nn.ReLU(inplace=act_cfg.get("inplace", True) if act_cfg else True))
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(in_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if self.add_identity:
            out = out + (x if identity is None else identity)
        return out


class MultiheadAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout or attn_drop, batch_first=batch_first)
        self.dropout = nn.Dropout(proj_drop or dropout or 0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if key is None:
            key = value if value is not None else query
        if value is None:
            value = key
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return (identity if identity is not None else query) + self.dropout(out)


class MultiScaleDeformableAttention(MultiheadAttention):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4, **kwargs):
        super().__init__(embed_dims=embed_dims, num_heads=num_heads, **kwargs)
        self.num_levels = num_levels
        self.num_points = num_points

    def init_weight(self):
        return None

    def init_weights(self):
        return None


class TransformerLayerSequence(BaseModule):
    def __init__(self, transformerlayers=None, num_layers=1, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        if transformerlayers is None:
            self.layers = nn.ModuleList()
        else:
            self.layers = nn.ModuleList([build_transformer_layer(transformerlayers) for _ in range(num_layers)])
        self.embed_dims = self.layers[0].embed_dims if self.layers else kwargs.get("embed_dims", 256)

    def forward(self, query, *args, **kwargs):
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
        return query


class DeformableDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        valid_ratios=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points
            if reference_points is not None and valid_ratios is not None:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                else:
                    reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

            if reg_branches is not None and reference_points is not None:
                output_for_reg = output.permute(1, 0, 2)
                tmp = reg_branches[layer_idx](output_for_reg)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                else:
                    new_reference_points = tmp[..., :2] + inverse_sigmoid(reference_points)
                reference_points = new_reference_points.sigmoid().detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class BaseTransformerLayer(BaseModule):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=None,
        feedforward_channels=1024,
        ffn_dropout=0.0,
        operation_order=("self_attn", "norm", "ffn", "norm"),
        batch_first=False,
        norm_cfg=None,
        **kwargs,
    ):
        super().__init__()
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        self.attentions = nn.ModuleList([build_attention(cfg) for cfg in (attn_cfgs or [])])
        self.embed_dims = self.attentions[0].embed_dims if self.attentions else kwargs.get("embed_dims", 256)
        if ffn_cfgs is None:
            ffn_cfgs = dict(embed_dims=self.embed_dims, feedforward_channels=feedforward_channels, ffn_drop=ffn_dropout)
        self.ffns = nn.ModuleList([build_feedforward_network(ffn_cfgs) for _ in range(operation_order.count("ffn"))])
        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dims) for _ in range(operation_order.count("norm"))])
        self.operation_order = operation_order
        self.num_attn = num_attn
        self.pre_norm = operation_order[0] == "norm"
        self.batch_first = batch_first

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None, attn_masks=None, **kwargs):
        identity = query
        attn_i = norm_i = ffn_i = 0
        if attn_masks is None:
            attn_masks = [None] * self.num_attn
        for layer in self.operation_order:
            if layer == "self_attn":
                self_attn_kwargs = dict(kwargs)
                self_attn_kwargs.pop("key_padding_mask", None)
                query = self.attentions[attn_i](query, query, query, identity if self.pre_norm else None, query_pos=query_pos, key_pos=query_pos, attn_mask=attn_masks[attn_i], **self_attn_kwargs)
                identity = query
                attn_i += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_i](query, key, value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=key_pos, attn_mask=attn_masks[attn_i], **kwargs)
                identity = query
                attn_i += 1
            elif layer == "norm":
                query = self.norms[norm_i](query)
                norm_i += 1
            elif layer == "ffn":
                query = self.ffns[ffn_i](query, identity if self.pre_norm else None)
                identity = query
                ffn_i += 1
        return query


def build_attention(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = dict(cfg)
    layer_type = cfg.pop("type", "MultiheadAttention")
    if layer_type == "MultiheadAttention":
        return MultiheadAttention(**cfg)
    if layer_type == "MultiScaleDeformableAttention":
        return MultiScaleDeformableAttention(**cfg)
    if layer_type == "TemporalSelfAttention":
        from uniad.models.modules.temporal_self_attention import TemporalSelfAttention

        return TemporalSelfAttention(**cfg)
    if layer_type == "SpatialCrossAttention":
        from uniad.models.modules.spatial_cross_attention import SpatialCrossAttention

        return SpatialCrossAttention(**cfg)
    if layer_type == "MSDeformableAttention3D":
        from uniad.models.modules.spatial_cross_attention import MSDeformableAttention3D

        return MSDeformableAttention3D(**cfg)
    if layer_type == "CustomMSDeformableAttention":
        from uniad.models.modules.decoder import CustomMSDeformableAttention

        return CustomMSDeformableAttention(**cfg)
    if layer_type == "MotionDeformableAttention":
        from uniad.models.dense_heads.motion_head_plugin.motion_deformable_attn import MotionDeformableAttention

        return MotionDeformableAttention(**cfg)
    if layer_type == "CustomModeMultiheadAttention":
        from uniad.models.dense_heads.motion_head_plugin.motion_deformable_attn import CustomModeMultiheadAttention

        return CustomModeMultiheadAttention(**cfg)
    raise KeyError(f"Unsupported attention: {layer_type}")


def build_transformer(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = dict(cfg)
    transformer_type = cfg.pop("type", None)
    if transformer_type == "PerceptionTransformer":
        from uniad.models.modules.transformer import PerceptionTransformer

        return PerceptionTransformer(**cfg)
    if transformer_type == "SegDeformableTransformer":
        from uniad.models.dense_heads.seg_head_plugin.seg_deformable_transformer import SegDeformableTransformer

        return SegDeformableTransformer(**cfg)
    if transformer_type == "SegMaskHead":
        from uniad.models.dense_heads.seg_head_plugin.seg_mask_head import SegMaskHead

        return SegMaskHead(**cfg)
    raise KeyError(f"Unsupported transformer: {transformer_type}")


def build_transformer_layer(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = dict(cfg)
    layer_type = cfg.pop("type", "BaseTransformerLayer")
    if layer_type in ("BaseTransformerLayer", "DetrTransformerDecoderLayer"):
        return BaseTransformerLayer(**cfg)
    if layer_type == "BEVFormerLayer":
        from uniad.models.modules.encoder import BEVFormerLayer

        return BEVFormerLayer(**cfg)
    if layer_type == "MotionTransformerAttentionLayer":
        from uniad.models.dense_heads.motion_head_plugin.motion_deformable_attn import MotionTransformerAttentionLayer

        return MotionTransformerAttentionLayer(**cfg)
    raise KeyError(f"Unsupported transformer layer: {layer_type}")


def build_transformer_layer_sequence(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    cfg = dict(cfg)
    sequence_type = cfg.pop("type", None)
    if sequence_type in ("TransformerLayerSequence", "DetrTransformerEncoder", "DetrTransformerDecoder"):
        return TransformerLayerSequence(**cfg)
    if sequence_type == "DeformableDetrTransformerDecoder":
        return DeformableDetrTransformerDecoder(**cfg)
    if sequence_type == "BEVFormerEncoder":
        from uniad.models.modules.encoder import BEVFormerEncoder

        return BEVFormerEncoder(**cfg)
    if sequence_type == "DetectionTransformerDecoder":
        from uniad.models.modules.decoder import DetectionTransformerDecoder

        return DetectionTransformerDecoder(**cfg)
    if sequence_type == "MotionTransformerDecoder":
        from uniad.models.dense_heads.motion_head_plugin.modules import MotionTransformerDecoder

        return MotionTransformerDecoder(**cfg)
    raise KeyError(f"Unsupported transformer sequence: {sequence_type}")


def build_feedforward_network(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("type", None)
    return FFN(**cfg)


def build_positional_encoding(cfg):
    from uniad.core.positional_encoding import build_positional_encoding as _build

    return _build(cfg)


def build_norm_layer(cfg, num_features, postfix=""):
    layer_type = (cfg or {}).get("type", "LN")
    if layer_type in ("LN", "LayerNorm"):
        return "ln" + str(postfix), nn.LayerNorm(num_features)
    if layer_type in ("BN", "BN2d", "BatchNorm2d"):
        return "bn" + str(postfix), nn.BatchNorm2d(num_features)
    raise KeyError(f"Unsupported norm layer: {layer_type}")


def build_activation_layer(cfg):
    cfg = cfg or {"type": "ReLU"}
    if cfg.get("type") == "ReLU":
        return nn.ReLU(inplace=cfg.get("inplace", False))
    if cfg.get("type") == "GELU":
        return nn.GELU()
    raise KeyError(f"Unsupported activation: {cfg.get('type')}")


def build_dropout(cfg):
    return nn.Dropout((cfg or {}).get("drop_prob", (cfg or {}).get("p", 0.0)))
