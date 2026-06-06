from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn

from .attention import MultiheadFlashAttention
from .blocks import AsymmetricFFN, DeformableFeatureAggregation
from .detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DRefinementModule,
)
from .detection3d_decoder import SparseBox3DDecoder
from .detection3d_losses import SparseBox3DLoss
from .detection3d_target import SparseBox3DTarget
from .dist_utils import reduce_mean
from .instance_bank import InstanceBank
from .map_blocks import (
    SparsePoint3DEncoder,
    SparsePoint3DKeyPointsGenerator,
    SparsePoint3DRefinementModule,
)
from .map_decoder import SparsePoint3DDecoder
from .map_loss import LinesL1Loss, SparseLineLoss
from .map_match_cost import FocalLossCost, LinesL1Cost, MapQueriesCost
from .map_target import HungarianLinesAssigner, SparsePoint3DTarget
from .nn_utils import CrossEntropyLoss, FocalLoss, GaussianFocalLoss, L1Loss

__all__ = ["Sparse4DDetHead", "Sparse4DMap", "convert_sparse4d_state_dict"]


def _convert_flat_layer_state_dict(old_sd, flat_index_map):
    new_sd = OrderedDict()
    for key, value in old_sd.items():
        if not key.startswith("layers."):
            new_sd[key] = value
            continue

        parts = key.split(".", 2)
        if len(parts) < 3:
            continue

        layer_idx = int(parts[1])
        if layer_idx not in flat_index_map:
            continue

        new_sd[f"{flat_index_map[layer_idx]}.{parts[2]}"] = value
    return new_sd


def _decoupled_head_indices(embed_dims, num_heads, device):
    old_head_dim = embed_dims // num_heads
    new_head_dim = old_head_dim * 2
    indices = [
        head_idx * new_head_dim + dim_idx
        for head_idx in range(num_heads)
        for dim_idx in range(old_head_dim)
    ]
    return torch.tensor(indices, device=device)


def _expand_coupled_in_proj_weight(weight, embed_dims, num_heads):
    indices = _decoupled_head_indices(embed_dims, num_heads, weight.device)
    expanded = weight.new_zeros((6 * embed_dims, 2 * embed_dims))
    for proj_idx in range(3):
        old_rows = slice(proj_idx * embed_dims, (proj_idx + 1) * embed_dims)
        new_rows = indices + proj_idx * 2 * embed_dims
        expanded[new_rows, :embed_dims] = weight[old_rows]
        if proj_idx < 2:
            expanded[new_rows, embed_dims:] = weight[old_rows]
    return expanded


def _expand_coupled_in_proj_bias(bias, embed_dims, num_heads):
    indices = _decoupled_head_indices(embed_dims, num_heads, bias.device)
    expanded = bias.new_zeros(6 * embed_dims)
    for proj_idx in range(3):
        old_rows = slice(proj_idx * embed_dims, (proj_idx + 1) * embed_dims)
        new_rows = indices + proj_idx * 2 * embed_dims
        expanded[new_rows] = bias[old_rows]
    return expanded


def _expand_coupled_out_proj_weight(weight, embed_dims, num_heads):
    indices = _decoupled_head_indices(embed_dims, num_heads, weight.device)
    expanded = weight.new_zeros((2 * embed_dims, 2 * embed_dims))
    expanded[:embed_dims, indices] = weight
    return expanded


def _expand_coupled_out_proj_bias(bias, embed_dims):
    expanded = bias.new_zeros(2 * embed_dims)
    expanded[:embed_dims] = bias
    return expanded


def _decoupled_fc_before_weight(embed_dims, template):
    weight = template.new_zeros((2 * embed_dims, embed_dims))
    weight[:embed_dims] = torch.eye(
        embed_dims, dtype=template.dtype, device=template.device
    )
    return weight


def _decoupled_fc_after_weight(embed_dims, template):
    weight = template.new_zeros((embed_dims, 2 * embed_dims))
    weight[:, :embed_dims] = torch.eye(
        embed_dims, dtype=template.dtype, device=template.device
    )
    return weight


def _adapt_coupled_map_state_dict(state_dict, embed_dims, num_heads):
    adapted = OrderedDict()
    template = None
    found_coupled_attention = False

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            adapted[key] = value
            continue

        if value.is_floating_point() and template is None:
            template = value

        if key.endswith(".attn.in_proj_weight") and value.shape == (
            3 * embed_dims,
            embed_dims,
        ):
            value = _expand_coupled_in_proj_weight(value, embed_dims, num_heads)
            found_coupled_attention = True
        elif key.endswith(".attn.in_proj_bias") and value.shape == (
            3 * embed_dims,
        ):
            value = _expand_coupled_in_proj_bias(value, embed_dims, num_heads)
            found_coupled_attention = True
        elif key.endswith(".attn.out_proj.weight") and value.shape == (
            embed_dims,
            embed_dims,
        ):
            value = _expand_coupled_out_proj_weight(value, embed_dims, num_heads)
            found_coupled_attention = True
        elif key.endswith(".attn.out_proj.bias") and value.shape == (embed_dims,):
            value = _expand_coupled_out_proj_bias(value, embed_dims)
            found_coupled_attention = True

        adapted[key] = value

    if found_coupled_attention and template is not None:
        adapted.setdefault(
            "fc_before.weight", _decoupled_fc_before_weight(embed_dims, template)
        )
        adapted.setdefault(
            "fc_after.weight", _decoupled_fc_after_weight(embed_dims, template)
        )
    return adapted


class Sparse4DDetHead(nn.Module):
    def __init__(self, hyperparams: dict):
        super().__init__()
        embed_dims = hyperparams["embed_dims"]
        class_names = hyperparams["class_names"]
        num_classes = len(class_names)
        num_levels = len(hyperparams["strides"])
        kmeans_dir = hyperparams["kmeans_dir"]
        attn_dims = embed_dims * 2
        sf_count = max(hyperparams["num_single_frame_decoder"], 0)
        temp_count = hyperparams["num_decoder"] - sf_count

        self.hyperparams = hyperparams
        self.num_decoder = hyperparams["num_decoder"]
        self.gt_cls_key = hyperparams["det_gt_cls_key"]
        self.gt_reg_key = hyperparams["det_gt_reg_key"]
        self.gt_id_key = hyperparams["det_gt_id_key"]
        self.with_instance_id = hyperparams["det_with_instance_id"]
        self.task_prefix = hyperparams["det_task_prefix"]
        self.cls_threshold_to_reg = hyperparams["det_cls_threshold_to_reg"]
        self.dn_loss_weight = hyperparams["det_dn_loss_weight"]
        self.reg_weights = hyperparams["det_loss_reg_weights"]
        self.num_single_frame_decoder = sf_count
        self.num_temporal_decoder = temp_count

        self.instance_bank = InstanceBank(
            num_anchor=hyperparams["det_num_anchor"],
            embed_dims=embed_dims,
            anchor=f"{kmeans_dir}/{hyperparams['det_anchor_file']}",
            anchor_handler=SparseBox3DKeyPointsGenerator(),
            num_temp_instances=(
                hyperparams["det_num_temp_instances"] if hyperparams["temporal"] else -1
            ),
            confidence_decay=hyperparams["det_confidence_decay"],
            feat_grad=hyperparams["det_feat_grad"],
            anchor_grad=hyperparams.get("det_anchor_grad", True),
        )
        self.anchor_encoder = SparseBox3DEncoder(
            vel_dims=hyperparams["det_encoder_vel_dims"],
            embed_dims=hyperparams["det_encoder_embed_dims_decoupled"],
            mode="cat",
            output_fc=False,
            in_loops=hyperparams["det_encoder_in_loops"],
            out_loops=hyperparams["det_encoder_out_loops_decoupled"],
        )
        self.sampler = SparseBox3DTarget(
            num_dn_groups=hyperparams["det_num_dn_groups"],
            num_temp_dn_groups=hyperparams["det_num_temp_dn_groups"],
            dn_noise_scale=hyperparams["det_dn_noise_scale"],
            max_dn_gt=hyperparams["det_max_dn_gt"],
            add_neg_dn=hyperparams["det_add_neg_dn"],
            cls_weight=hyperparams["det_target_cls_weight"],
            box_weight=hyperparams["det_target_box_weight"],
            reg_weights=hyperparams["det_target_reg_weights"],
            cls_wise_reg_weights={
                class_names.index(name): weights
                for name, weights in hyperparams["det_cls_wise_reg_weights"].items()
            },
        )
        self.decoder = SparseBox3DDecoder()
        self.loss_cls = FocalLoss(
            use_sigmoid=True,
            gamma=hyperparams["det_loss_gamma"],
            alpha=hyperparams["det_loss_alpha"],
            loss_weight=hyperparams["det_loss_cls_weight"],
        )
        self.loss_reg = SparseBox3DLoss(
            loss_box=L1Loss(loss_weight=hyperparams["det_loss_reg_weight"]),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True),
            loss_yawness=GaussianFocalLoss(),
            cls_allow_reverse=[
                class_names.index(name) for name in hyperparams["det_cls_allow_reverse"]
            ],
        )
        self.embed_dims = self.instance_bank.embed_dims
        self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
        self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)

        self.sf_gnns = nn.ModuleList(
            ([nn.Identity()] if sf_count > 0 else [])
            + [
                MultiheadFlashAttention(
                    embed_dims=attn_dims,
                    num_heads=hyperparams["num_groups"],
                    batch_first=hyperparams["attention_batch_first"],
                    dropout=hyperparams["drop_out"],
                )
                for _ in range(max(sf_count - 1, 0))
            ]
        )
        self.sf_norm1s = nn.ModuleList(
            ([nn.Identity()] if sf_count > 0 else [])
            + [nn.LayerNorm(embed_dims) for _ in range(max(sf_count - 1, 0))]
        )
        self.sf_deformables = nn.ModuleList(
            [
                DeformableFeatureAggregation(
                    embed_dims=embed_dims,
                    num_groups=hyperparams["num_groups"],
                    num_levels=num_levels,
                    num_cams=hyperparams["num_cams"],
                    attn_drop=hyperparams["deformable_attn_drop"],
                    use_deformable_func=hyperparams["use_deformable_func"],
                    use_camera_embed=hyperparams["deformable_use_camera_embed"],
                    residual_mode=hyperparams["deformable_residual_mode"],
                    kps_generator=SparseBox3DKeyPointsGenerator(
                        num_learnable_pts=hyperparams["det_keypoint_num_learnable_pts"],
                        fix_scale=hyperparams["det_keypoint_fix_scale"],
                    ),
                )
                for _ in range(sf_count)
            ]
        )
        self.sf_ffns = nn.ModuleList(
            [
                AsymmetricFFN(
                    in_channels=embed_dims * 2,
                    pre_norm=hyperparams["ffn_pre_norm"],
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 4,
                    num_fcs=hyperparams["det_ffn_num_fcs"],
                    ffn_drop=hyperparams["drop_out"],
                    act_cfg=hyperparams["ffn_act_cfg"],
                )
                for _ in range(sf_count)
            ]
        )
        self.sf_norm2s = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(sf_count)])
        self.sf_refines = nn.ModuleList(
            [
                SparseBox3DRefinementModule(
                    embed_dims=embed_dims,
                    num_cls=num_classes,
                    refine_yaw=hyperparams["det_refine_yaw"],
                    with_quality_estimation=hyperparams["with_quality_estimation"],
                )
                for _ in range(sf_count)
            ]
        )

        self.temp_gnns = nn.ModuleList(
            [
                MultiheadFlashAttention(
                    embed_dims=attn_dims,
                    num_heads=hyperparams["num_groups"],
                    batch_first=hyperparams["attention_batch_first"],
                    dropout=hyperparams["drop_out"],
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_local_gnns = nn.ModuleList(
            [
                MultiheadFlashAttention(
                    embed_dims=attn_dims,
                    num_heads=hyperparams["num_groups"],
                    batch_first=hyperparams["attention_batch_first"],
                    dropout=hyperparams["drop_out"],
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_norm1s = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(temp_count)])
        self.temp_deformables = nn.ModuleList(
            [
                DeformableFeatureAggregation(
                    embed_dims=embed_dims,
                    num_groups=hyperparams["num_groups"],
                    num_levels=num_levels,
                    num_cams=hyperparams["num_cams"],
                    attn_drop=hyperparams["deformable_attn_drop"],
                    use_deformable_func=hyperparams["use_deformable_func"],
                    use_camera_embed=hyperparams["deformable_use_camera_embed"],
                    residual_mode=hyperparams["deformable_residual_mode"],
                    kps_generator=SparseBox3DKeyPointsGenerator(
                        num_learnable_pts=hyperparams["det_keypoint_num_learnable_pts"],
                        fix_scale=hyperparams["det_keypoint_fix_scale"],
                    ),
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_ffns = nn.ModuleList(
            [
                AsymmetricFFN(
                    in_channels=embed_dims * 2,
                    pre_norm=hyperparams["ffn_pre_norm"],
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 4,
                    num_fcs=hyperparams["det_ffn_num_fcs"],
                    ffn_drop=hyperparams["drop_out"],
                    act_cfg=hyperparams["ffn_act_cfg"],
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_norm2s = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(temp_count)])
        self.temp_refines = nn.ModuleList(
            [
                SparseBox3DRefinementModule(
                    embed_dims=embed_dims,
                    num_cls=num_classes,
                    refine_yaw=hyperparams["det_refine_yaw"],
                    with_quality_estimation=hyperparams["with_quality_estimation"],
                )
                for _ in range(temp_count)
            ]
        )

    def init_weights(self):
        non_refine_lists = [
            self.sf_gnns,
            self.sf_norm1s,
            self.sf_deformables,
            self.sf_ffns,
            self.sf_norm2s,
            self.temp_gnns,
            self.temp_local_gnns,
            self.temp_norm1s,
            self.temp_deformables,
            self.temp_ffns,
            self.temp_norm2s,
        ]
        for module_list in non_refine_lists:
            for module in module_list:
                for p in module.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for p in self.fc_before.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.fc_after.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training:
            if self.gt_id_key in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x[self.gt_id_key]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )

        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        prediction = []
        classification = []
        quality = []
        cls = None

        for decoder_idx in range(self.num_single_frame_decoder):
            if decoder_idx > 0:
                query = torch.cat([instance_feature, anchor_embed], dim=-1)
                value = self.fc_before(instance_feature)
                instance_feature = self.fc_after(
                    self.sf_gnns[decoder_idx](
                        query,
                        None,
                        value,
                        query_pos=None,
                        key_pos=None,
                        attn_mask=attn_mask,
                    )
                )
                instance_feature = self.sf_norm1s[decoder_idx](instance_feature)

            instance_feature = self.sf_deformables[decoder_idx](
                instance_feature, anchor, anchor_embed, feature_maps, metas
            )
            instance_feature = self.sf_ffns[decoder_idx](instance_feature)
            instance_feature = self.sf_norm2s[decoder_idx](instance_feature)
            anchor, cls, qt = self.sf_refines[decoder_idx](
                instance_feature,
                anchor,
                anchor_embed,
                time_interval=time_interval,
                return_cls=True,
            )
            prediction.append(anchor)
            classification.append(cls)
            quality.append(qt)
            anchor_embed = self.anchor_encoder(anchor)

        if self.num_single_frame_decoder > 0:
            instance_feature, anchor = self.instance_bank.update(
                instance_feature, anchor, cls
            )
            if (
                dn_metas is not None
                and self.sampler.num_temp_dn_groups > 0
                and dn_id_target is not None
            ):
                (
                    instance_feature,
                    anchor,
                    temp_dn_reg_target,
                    temp_dn_cls_target,
                    temp_valid_mask,
                    dn_id_target,
                ) = self.sampler.update_dn(
                    instance_feature,
                    anchor,
                    dn_reg_target,
                    dn_cls_target,
                    valid_mask,
                    dn_id_target,
                    self.instance_bank.num_anchor,
                    self.instance_bank.mask,
                )
            anchor_embed = self.anchor_encoder(anchor)

        for decoder_idx in range(self.num_temporal_decoder):
            query = torch.cat([instance_feature, anchor_embed], dim=-1)
            if temp_instance_feature is not None:
                key = torch.cat(
                    [temp_instance_feature, temp_anchor_embed], dim=-1
                )
                value = self.fc_before(temp_instance_feature)
            else:
                key = None
                value = None
            instance_feature = self.fc_after(
                self.temp_gnns[decoder_idx](
                    query,
                    key,
                    value,
                    query_pos=None,
                    key_pos=None,
                    attn_mask=(
                        attn_mask if temp_instance_feature is None else None
                    ),
                )
            )

            query = torch.cat([instance_feature, anchor_embed], dim=-1)
            value = self.fc_before(instance_feature)
            instance_feature = self.fc_after(
                self.temp_local_gnns[decoder_idx](
                    query,
                    None,
                    value,
                    query_pos=None,
                    key_pos=None,
                    attn_mask=attn_mask,
                )
            )
            instance_feature = self.temp_norm1s[decoder_idx](instance_feature)
            instance_feature = self.temp_deformables[decoder_idx](
                instance_feature, anchor, anchor_embed, feature_maps, metas
            )
            instance_feature = self.temp_ffns[decoder_idx](instance_feature)
            instance_feature = self.temp_norm2s[decoder_idx](instance_feature)
            anchor, cls, qt = self.temp_refines[decoder_idx](
                instance_feature,
                anchor,
                anchor_embed,
                time_interval=time_interval,
                return_cls=True,
            )
            prediction.append(anchor)
            classification.append(cls)
            quality.append(qt)
            anchor_embed = self.anchor_encoder(anchor)
            if temp_anchor_embed is not None:
                temp_anchor_embed = anchor_embed[
                    :, : self.instance_bank.num_temp_instances
                ]

        output = {}
        if dn_metas is not None:
            output.update(
                {
                    "dn_prediction": [
                        x[:, num_free_instance:] for x in prediction
                    ],
                    "dn_classification": [
                        x[:, num_free_instance:] for x in classification
                    ],
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            prediction = [x[:, :num_free_instance] for x in prediction]
            classification = [x[:, :num_free_instance] for x in classification]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask

            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor_embed = anchor_embed[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
            }
        )
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if self.with_instance_id:
            output["instance_id"] = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
        return output

    def loss(self, model_outs, data, feature_maps=None):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_for_match = torch.nan_to_num(
                cls.float(), nan=0.0, posinf=80.0, neginf=-80.0
            ).clamp_(-80.0, 80.0)
            reg_for_match = torch.nan_to_num(
                reg.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls_for_match,
                reg_for_match,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                mask = torch.logical_and(
                    mask,
                    cls_for_match.max(dim=-1).values.sigmoid()
                    > self.cls_threshold_to_reg,
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            reg = torch.nan_to_num(reg, nan=0.0, posinf=0.0, neginf=0.0)
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]
                # Guard against NaN/Inf in quality predictions.
                qt = torch.nan_to_num(
                    qt, nan=0.0, posinf=80.0, neginf=-80.0
                ).clamp_(-80.0, 80.0)

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                prefix=f"{self.task_prefix}_",
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"{self.task_prefix}_loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )

    @staticmethod
    def _flat_layer_key_map(hyperparams):
        sf_count = max(hyperparams["num_single_frame_decoder"], 0)
        temp_count = hyperparams["num_decoder"] - sf_count
        flat_index_map = {}
        flat_i = 0

        for decoder_idx in range(sf_count):
            if decoder_idx > 0:
                flat_index_map[flat_i] = f"sf_gnns.{decoder_idx}"
                flat_i += 1
                flat_index_map[flat_i] = f"sf_norm1s.{decoder_idx}"
                flat_i += 1
            flat_index_map[flat_i] = f"sf_deformables.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_ffns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_norm2s.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_refines.{decoder_idx}"
            flat_i += 1

        for decoder_idx in range(temp_count):
            flat_index_map[flat_i] = f"temp_gnns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_local_gnns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_norm1s.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_deformables.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_ffns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_norm2s.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_refines.{decoder_idx}"
            flat_i += 1
        return flat_index_map

    @classmethod
    def convert_state_dict(cls, old_sd, hyperparams):
        return _convert_flat_layer_state_dict(
            old_sd, cls._flat_layer_key_map(hyperparams)
        )


class Sparse4DMap(nn.Module):
    def __init__(self, hyperparams: dict):
        super().__init__()
        embed_dims = hyperparams["embed_dims"]
        num_map_classes = len(hyperparams["map_class_names"])
        num_levels = len(hyperparams["strides"])
        kmeans_dir = hyperparams["kmeans_dir"]
        attn_dims = embed_dims * 2
        sf_count = max(hyperparams["num_single_frame_decoder_map"], 0)
        temp_count = hyperparams["num_decoder"] - sf_count

        self.hyperparams = hyperparams
        self.num_decoder = hyperparams["num_decoder"]
        self.gt_cls_key = hyperparams["map_gt_cls_key"]
        self.gt_reg_key = hyperparams["map_gt_reg_key"]
        self.gt_id_key = hyperparams["map_gt_id_key"]
        self.with_instance_id = hyperparams["map_with_instance_id"]
        self.task_prefix = hyperparams["map_task_prefix"]
        self.cls_threshold_to_reg = hyperparams["map_cls_threshold_to_reg"]
        self.dn_loss_weight = hyperparams["det_dn_loss_weight"]
        self.reg_weights = hyperparams["map_reg_weights"]
        self.num_single_frame_decoder = sf_count
        self.num_temporal_decoder = temp_count

        self.instance_bank = InstanceBank(
            num_anchor=hyperparams["map_num_anchor"],
            embed_dims=embed_dims,
            anchor=f"{kmeans_dir}/{hyperparams['map_anchor_file']}",
            anchor_handler=SparsePoint3DKeyPointsGenerator(),
            num_temp_instances=(
                hyperparams["num_map_temp_instances"]
                if hyperparams["temporal_map"]
                else -1
            ),
            confidence_decay=hyperparams["map_confidence_decay"],
            feat_grad=hyperparams["map_feat_grad"],
            anchor_grad=hyperparams.get("map_anchor_grad", True),
        )
        self.anchor_encoder = SparsePoint3DEncoder(
            embed_dims=embed_dims,
            num_sample=hyperparams["num_sample"],
        )
        self.sampler = SparsePoint3DTarget(
            assigner=HungarianLinesAssigner(
                cost=MapQueriesCost(
                    cls_cost=FocalLossCost(weight=hyperparams["map_loss_cls_weight"]),
                    reg_cost=LinesL1Cost(
                        weight=hyperparams["map_loss_reg_weight"],
                        beta=hyperparams["map_loss_beta"],
                        permute=True,
                    ),
                )
            ),
            num_cls=num_map_classes,
            num_sample=hyperparams["num_sample"],
            roi_size=hyperparams["roi_size"],
        )
        self.decoder = SparsePoint3DDecoder()
        self.loss_cls = FocalLoss(
            use_sigmoid=True,
            gamma=hyperparams["det_loss_gamma"],
            alpha=hyperparams["det_loss_alpha"],
            loss_weight=hyperparams["map_loss_cls_weight"],
        )
        self.loss_reg = SparseLineLoss(
            loss_line=LinesL1Loss(
                loss_weight=hyperparams["map_loss_reg_weight"],
                beta=hyperparams["map_loss_beta"],
            ),
            num_sample=hyperparams["num_sample"],
            roi_size=hyperparams["roi_size"],
        )
        self.embed_dims = self.instance_bank.embed_dims
        self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
        self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)

        self.sf_gnns = nn.ModuleList(
            [
                MultiheadFlashAttention(
                    embed_dims=attn_dims,
                    num_heads=hyperparams["num_groups"],
                    batch_first=hyperparams["attention_batch_first"],
                    dropout=hyperparams["drop_out"],
                )
                for _ in range(sf_count)
            ]
        )
        self.sf_norm1s = nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(sf_count)]
        )
        self.sf_deformables = nn.ModuleList(
            [
                DeformableFeatureAggregation(
                    embed_dims=embed_dims,
                    num_groups=hyperparams["num_groups"],
                    num_levels=num_levels,
                    num_cams=hyperparams["num_cams"],
                    attn_drop=hyperparams["deformable_attn_drop"],
                    use_deformable_func=hyperparams["use_deformable_func"],
                    use_camera_embed=hyperparams["deformable_use_camera_embed"],
                    residual_mode=hyperparams["deformable_residual_mode"],
                    kps_generator=SparsePoint3DKeyPointsGenerator(
                        embed_dims=embed_dims,
                        num_sample=hyperparams["num_sample"],
                        num_learnable_pts=hyperparams["map_keypoint_num_learnable_pts"],
                        fix_height=hyperparams["map_keypoint_fix_height"],
                        ground_height=hyperparams["map_keypoint_ground_height"],
                    ),
                )
                for _ in range(sf_count)
            ]
        )
        self.sf_ffns = nn.ModuleList(
            [
                AsymmetricFFN(
                    in_channels=embed_dims * 2,
                    pre_norm=hyperparams["ffn_pre_norm"],
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 4,
                    num_fcs=hyperparams["det_ffn_num_fcs"],
                    ffn_drop=hyperparams["drop_out"],
                    act_cfg=hyperparams["ffn_act_cfg"],
                )
                for _ in range(sf_count)
            ]
        )
        self.sf_norm2s = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(sf_count)])
        self.sf_refines = nn.ModuleList(
            [
                SparsePoint3DRefinementModule(
                    embed_dims=embed_dims,
                    num_sample=hyperparams["num_sample"],
                    num_cls=num_map_classes,
                )
                for _ in range(sf_count)
            ]
        )

        self.temp_gnns = nn.ModuleList(
            [
                MultiheadFlashAttention(
                    embed_dims=attn_dims,
                    num_heads=hyperparams["num_groups"],
                    batch_first=hyperparams["attention_batch_first"],
                    dropout=hyperparams["drop_out"],
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_local_gnns = nn.ModuleList(
            [
                MultiheadFlashAttention(
                    embed_dims=attn_dims,
                    num_heads=hyperparams["num_groups"],
                    batch_first=hyperparams["attention_batch_first"],
                    dropout=hyperparams["drop_out"],
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_norm1s = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(temp_count)])
        self.temp_deformables = nn.ModuleList(
            [
                DeformableFeatureAggregation(
                    embed_dims=embed_dims,
                    num_groups=hyperparams["num_groups"],
                    num_levels=num_levels,
                    num_cams=hyperparams["num_cams"],
                    attn_drop=hyperparams["deformable_attn_drop"],
                    use_deformable_func=hyperparams["use_deformable_func"],
                    use_camera_embed=hyperparams["deformable_use_camera_embed"],
                    residual_mode=hyperparams["deformable_residual_mode"],
                    kps_generator=SparsePoint3DKeyPointsGenerator(
                        embed_dims=embed_dims,
                        num_sample=hyperparams["num_sample"],
                        num_learnable_pts=hyperparams["map_keypoint_num_learnable_pts"],
                        fix_height=hyperparams["map_keypoint_fix_height"],
                        ground_height=hyperparams["map_keypoint_ground_height"],
                    ),
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_ffns = nn.ModuleList(
            [
                AsymmetricFFN(
                    in_channels=embed_dims * 2,
                    pre_norm=hyperparams["ffn_pre_norm"],
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 4,
                    num_fcs=hyperparams["det_ffn_num_fcs"],
                    ffn_drop=hyperparams["drop_out"],
                    act_cfg=hyperparams["ffn_act_cfg"],
                )
                for _ in range(temp_count)
            ]
        )
        self.temp_norm2s = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(temp_count)])
        self.temp_refines = nn.ModuleList(
            [
                SparsePoint3DRefinementModule(
                    embed_dims=embed_dims,
                    num_sample=hyperparams["num_sample"],
                    num_cls=num_map_classes,
                )
                for _ in range(temp_count)
            ]
        )

    def init_weights(self):
        non_refine_lists = [
            self.sf_gnns,
            self.sf_norm1s,
            self.sf_deformables,
            self.sf_ffns,
            self.sf_norm2s,
            self.temp_gnns,
            self.temp_local_gnns,
            self.temp_norm1s,
            self.temp_deformables,
            self.temp_ffns,
            self.temp_norm2s,
        ]
        for module_list in non_refine_lists:
            for module in module_list:
                for p in module.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for p in self.fc_before.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.fc_after.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(batch_size, metas)

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        prediction = []
        classification = []
        quality = []
        cls = None

        for decoder_idx in range(self.num_single_frame_decoder):
            query = torch.cat([instance_feature, anchor_embed], dim=-1)
            value = self.fc_before(instance_feature)
            instance_feature = self.fc_after(
                self.sf_gnns[decoder_idx](
                    query,
                    None,
                    value,
                    query_pos=None,
                    key_pos=None,
                )
            )
            instance_feature = self.sf_norm1s[decoder_idx](instance_feature)
            instance_feature = self.sf_deformables[decoder_idx](
                instance_feature, anchor, anchor_embed, feature_maps, metas
            )
            instance_feature = self.sf_ffns[decoder_idx](instance_feature)
            instance_feature = self.sf_norm2s[decoder_idx](instance_feature)
            anchor, cls, qt = self.sf_refines[decoder_idx](
                instance_feature,
                anchor,
                anchor_embed,
                time_interval=time_interval,
                return_cls=True,
            )
            prediction.append(anchor)
            classification.append(cls)
            quality.append(qt)
            anchor_embed = self.anchor_encoder(anchor)

        if self.num_single_frame_decoder > 0:
            instance_feature, anchor = self.instance_bank.update(
                instance_feature, anchor, cls
            )
            anchor_embed = self.anchor_encoder(anchor)

        for decoder_idx in range(self.num_temporal_decoder):
            query = torch.cat([instance_feature, anchor_embed], dim=-1)
            if temp_instance_feature is not None:
                key = torch.cat(
                    [temp_instance_feature, temp_anchor_embed], dim=-1
                )
                value = self.fc_before(temp_instance_feature)
            else:
                key = None
                value = None
            instance_feature = self.fc_after(
                self.temp_gnns[decoder_idx](
                    query,
                    key,
                    value,
                    query_pos=None,
                    key_pos=None,
                )
            )

            query = torch.cat([instance_feature, anchor_embed], dim=-1)
            value = self.fc_before(instance_feature)
            instance_feature = self.fc_after(
                self.temp_local_gnns[decoder_idx](
                    query,
                    None,
                    value,
                    query_pos=None,
                    key_pos=None,
                )
            )
            instance_feature = self.temp_norm1s[decoder_idx](instance_feature)
            instance_feature = self.temp_deformables[decoder_idx](
                instance_feature, anchor, anchor_embed, feature_maps, metas
            )
            instance_feature = self.temp_ffns[decoder_idx](instance_feature)
            instance_feature = self.temp_norm2s[decoder_idx](instance_feature)
            anchor, cls, qt = self.temp_refines[decoder_idx](
                instance_feature,
                anchor,
                anchor_embed,
                time_interval=time_interval,
                return_cls=True,
            )
            prediction.append(anchor)
            classification.append(cls)
            quality.append(qt)
            anchor_embed = self.anchor_encoder(anchor)
            if temp_anchor_embed is not None:
                temp_anchor_embed = anchor_embed[
                    :, : self.instance_bank.num_temp_instances
                ]

        output = {
            "classification": classification,
            "prediction": prediction,
            "quality": quality,
            "instance_feature": instance_feature,
            "anchor_embed": anchor_embed,
        }
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if self.with_instance_id:
            output["instance_id"] = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
        return output

    def loss(self, model_outs, data, feature_maps=None):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            # Guard against NaN/Inf poisoning the Hungarian cost matrix
            cls_for_match = torch.nan_to_num(cls, nan=0.0, posinf=0.0, neginf=0.0)
            reg_for_match = torch.nan_to_num(reg, nan=0.0, posinf=0.0, neginf=0.0)
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls_for_match,
                reg_for_match,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                mask = torch.logical_and(
                    mask,
                    cls.max(dim=-1).values.sigmoid()
                    > self.cls_threshold_to_reg,
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            reg = torch.nan_to_num(reg, nan=0.0, posinf=0.0, neginf=0.0)
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]
                # Guard against NaN/Inf in quality predictions.
                qt = torch.nan_to_num(
                    qt, nan=0.0, posinf=80.0, neginf=-80.0
                ).clamp_(-80.0, 80.0)

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )

    @staticmethod
    def _flat_layer_key_map(hyperparams):
        sf_count = max(hyperparams["num_single_frame_decoder_map"], 0)
        temp_count = hyperparams["num_decoder"] - sf_count
        flat_index_map = {}
        flat_i = 0

        for decoder_idx in range(sf_count):
            flat_index_map[flat_i] = f"sf_gnns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_norm1s.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_deformables.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_ffns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_norm2s.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"sf_refines.{decoder_idx}"
            flat_i += 1

        for decoder_idx in range(temp_count):
            flat_index_map[flat_i] = f"temp_gnns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_local_gnns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_norm1s.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_deformables.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_ffns.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_norm2s.{decoder_idx}"
            flat_i += 1
            flat_index_map[flat_i] = f"temp_refines.{decoder_idx}"
            flat_i += 1
        return flat_index_map

    @classmethod
    def convert_state_dict(cls, old_sd, hyperparams):
        converted = _convert_flat_layer_state_dict(
            old_sd, cls._flat_layer_key_map(hyperparams)
        )
        return _adapt_coupled_map_state_dict(
            converted,
            embed_dims=hyperparams["embed_dims"],
            num_heads=hyperparams["num_groups"],
        )


def convert_sparse4d_state_dict(state_dict, hyperparams):
    """Convert raw SparseDrive det/map head layer keys to split-head keys.

    Handles both non-DDP and DDP prefixes while leaving already-converted keys
    unchanged.
    """
    head_specs = (
        ("head.det_head.", Sparse4DDetHead),
        ("head.map_head.", Sparse4DMap),
        ("module.head.det_head.", Sparse4DDetHead),
        ("module.head.map_head.", Sparse4DMap),
    )
    converted = OrderedDict()
    head_state_dicts = {prefix: OrderedDict() for prefix, _ in head_specs}
    metadata = getattr(state_dict, "_metadata", None)

    for key, value in state_dict.items():
        matched = False
        for prefix, _ in head_specs:
            if key.startswith(prefix):
                head_state_dicts[prefix][key[len(prefix):]] = value
                matched = True
                break
        if not matched:
            converted[key] = value

    for prefix, head_cls in head_specs:
        head_state_dict = head_state_dicts[prefix]
        if not head_state_dict:
            continue
        for key, value in head_cls.convert_state_dict(head_state_dict, hyperparams).items():
            converted[f"{prefix}{key}"] = value

    if metadata is not None:
        converted._metadata = metadata
    return converted
