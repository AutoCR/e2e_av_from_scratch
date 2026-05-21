from typing import List, Optional, Union

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

__all__ = ["Sparse4DHead"]


class Sparse4DHead(nn.Module):
    def __init__(
        self,
        hyperparams: dict,
        task: str = "det",
    ):
        super().__init__()
        (
            instance_bank,
            anchor_encoder,
            graph_model_factory,
            norm_layer_factory,
            ffn_factory,
            deformable_model,
            refine_layer_factory,
            num_decoder,
            num_single_frame_decoder,
            temp_graph_model,
            decouple_attn,
            loss_cls,
            loss_reg,
            decoder,
            sampler,
            gt_cls_key,
            gt_reg_key,
            gt_id_key,
            with_instance_id,
            task_prefix,
            reg_weights,
            cls_threshold_to_reg,
            dn_loss_weight,
            operation_order,
        ) = self._build_components(hyperparams, task)
        self.hyperparams = hyperparams
        self.task = task
        self.num_decoder = num_decoder
        self.decouple_attn = decouple_attn
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.gt_id_key = gt_id_key
        self.with_instance_id = with_instance_id
        self.task_prefix = task_prefix
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        self.instance_bank = instance_bank
        self.anchor_encoder = anchor_encoder
        self.sampler = sampler
        self.decoder = decoder
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.embed_dims = self.instance_bank.embed_dims

        # Shared projection layers kept at head level to match original state_dict keys.
        # When decouple_attn=False these are identity (no parameters).
        if decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        def _make(x):
            return x() if callable(x) else x

        sf_count = max(num_single_frame_decoder, 0)
        temp_count = num_decoder - sf_count
        self.num_single_frame_decoder = sf_count

        # Determine whether SF steps include gnn+norm1 by inspecting the first op
        # of the passed operation_order (absorbed from config kwargs).
        # det_head uses _det_operation_order()[2:] → first op is "deformable" → no gnn.
        # map_head uses _map_operation_order()   → first op is "gnn"        → has gnn.
        if operation_order is not None and sf_count > 0:
            self.include_sf_gnn = operation_order[0] == "gnn"
        else:
            self.include_sf_gnn = sf_count == 0  # irrelevant when no SF steps

        # Single-frame decoder submodules.
        if self.include_sf_gnn:
            self.sf_gnns = nn.ModuleList([graph_model_factory() for _ in range(sf_count)])
            self.sf_norm1s = nn.ModuleList([norm_layer_factory() for _ in range(sf_count)])
        self.sf_deformables = nn.ModuleList([_make(deformable_model) for _ in range(sf_count)])
        self.sf_ffns = nn.ModuleList([ffn_factory() for _ in range(sf_count)])
        self.sf_norm2s = nn.ModuleList([norm_layer_factory() for _ in range(sf_count)])
        self.sf_refines = nn.ModuleList([refine_layer_factory() for _ in range(sf_count)])

        # Temporal decoder submodules.
        self.temp_gnns = nn.ModuleList([_make(temp_graph_model) for _ in range(temp_count)])
        self.temp_local_gnns = nn.ModuleList([graph_model_factory() for _ in range(temp_count)])
        self.temp_norm1s = nn.ModuleList([norm_layer_factory() for _ in range(temp_count)])
        self.temp_deformables = nn.ModuleList([_make(deformable_model) for _ in range(temp_count)])
        self.temp_ffns = nn.ModuleList([ffn_factory() for _ in range(temp_count)])
        self.temp_norm2s = nn.ModuleList([norm_layer_factory() for _ in range(temp_count)])
        self.temp_refines = nn.ModuleList([refine_layer_factory() for _ in range(temp_count)])

    @staticmethod
    def _attn_factory(hyperparams, dims):
        return lambda: MultiheadFlashAttention(
            embed_dims=dims,
            num_heads=hyperparams["num_groups"],
            batch_first=hyperparams["attention_batch_first"],
            dropout=hyperparams["drop_out"],
        )

    @staticmethod
    def _det_operation_order(hyperparams):
        return (
            ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
            * hyperparams["num_single_frame_decoder"]
            + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
            * (hyperparams["num_decoder"] - hyperparams["num_single_frame_decoder"])
        )[2:]

    @staticmethod
    def _map_operation_order(hyperparams):
        return (
            ["gnn", "norm", "deformable", "ffn", "norm", "refine"]
            * hyperparams["num_single_frame_decoder_map"]
            + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
            * (hyperparams["num_decoder"] - hyperparams["num_single_frame_decoder_map"])
        )

    @classmethod
    def _build_components(cls, hyperparams, task):
        if task == "det":
            return cls._build_det_components(hyperparams)
        if task == "map":
            return cls._build_map_components(hyperparams)
        raise ValueError(f"Unsupported Sparse4DHead task: {task}")

    @classmethod
    def _build_det_components(cls, hyperparams):
        embed_dims = hyperparams["embed_dims"]
        decouple_attn = hyperparams["decouple_attn"]
        class_names = hyperparams["class_names"]
        num_classes = len(class_names)
        kmeans_dir = hyperparams["kmeans_dir"]
        num_levels = len(hyperparams["strides"])
        bank_kps = SparseBox3DKeyPointsGenerator()

        def deformable_factory():
            return DeformableFeatureAggregation(
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

        attn_dims = embed_dims * 2 if decouple_attn else embed_dims
        temp_graph_model = (
            cls._attn_factory(hyperparams, attn_dims)
            if hyperparams["temporal"]
            else None
        )
        return (
            InstanceBank(
                num_anchor=hyperparams["det_num_anchor"],
                embed_dims=embed_dims,
                anchor=f"{kmeans_dir}/{hyperparams['det_anchor_file']}",
                anchor_handler=bank_kps,
                num_temp_instances=(
                    hyperparams["det_num_temp_instances"]
                    if hyperparams["temporal"]
                    else -1
                ),
                confidence_decay=hyperparams["det_confidence_decay"],
                feat_grad=hyperparams["det_feat_grad"],
            ),
            SparseBox3DEncoder(
                vel_dims=hyperparams["det_encoder_vel_dims"],
                embed_dims=(
                    hyperparams["det_encoder_embed_dims_decoupled"]
                    if decouple_attn
                    else hyperparams["det_encoder_embed_dims_coupled"]
                ),
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=hyperparams["det_encoder_in_loops"],
                out_loops=(
                    hyperparams["det_encoder_out_loops_decoupled"]
                    if decouple_attn
                    else hyperparams["det_encoder_out_loops_coupled"]
                ),
            ),
            cls._attn_factory(hyperparams, attn_dims),
            lambda: nn.LayerNorm(embed_dims),
            lambda: AsymmetricFFN(
                in_channels=embed_dims * 2,
                pre_norm=hyperparams["ffn_pre_norm"],
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=hyperparams["det_ffn_num_fcs"],
                ffn_drop=hyperparams["drop_out"],
                act_cfg=hyperparams["ffn_act_cfg"],
            ),
            deformable_factory,
            lambda: SparseBox3DRefinementModule(
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=hyperparams["det_refine_yaw"],
                with_quality_estimation=hyperparams["with_quality_estimation"],
            ),
            hyperparams["num_decoder"],
            hyperparams["num_single_frame_decoder"],
            temp_graph_model,
            decouple_attn,
            FocalLoss(
                use_sigmoid=True,
                gamma=hyperparams["det_loss_gamma"],
                alpha=hyperparams["det_loss_alpha"],
                loss_weight=hyperparams["det_loss_cls_weight"],
            ),
            SparseBox3DLoss(
                loss_box=L1Loss(loss_weight=hyperparams["det_loss_reg_weight"]),
                loss_centerness=CrossEntropyLoss(use_sigmoid=True),
                loss_yawness=GaussianFocalLoss(),
                cls_allow_reverse=[
                    class_names.index(name)
                    for name in hyperparams["det_cls_allow_reverse"]
                ],
            ),
            SparseBox3DDecoder(),
            SparseBox3DTarget(
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
                    for name, weights in hyperparams[
                        "det_cls_wise_reg_weights"
                    ].items()
                },
            ),
            hyperparams["det_gt_cls_key"],
            hyperparams["det_gt_reg_key"],
            hyperparams["det_gt_id_key"],
            hyperparams["det_with_instance_id"],
            hyperparams["det_task_prefix"],
            hyperparams["det_loss_reg_weights"],
            hyperparams["det_cls_threshold_to_reg"],
            hyperparams["det_dn_loss_weight"],
            cls._det_operation_order(hyperparams),
        )

    @classmethod
    def _build_map_components(cls, hyperparams):
        embed_dims = hyperparams["embed_dims"]
        decouple_attn = hyperparams["decouple_attn_map"]
        num_map_classes = len(hyperparams["map_class_names"])
        kmeans_dir = hyperparams["kmeans_dir"]
        num_levels = len(hyperparams["strides"])
        bank_kps = SparsePoint3DKeyPointsGenerator()

        def deformable_factory():
            return DeformableFeatureAggregation(
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

        attn_dims = embed_dims * 2 if decouple_attn else embed_dims
        temp_graph_model = (
            cls._attn_factory(hyperparams, attn_dims)
            if hyperparams["temporal_map"]
            else None
        )
        return (
            InstanceBank(
                num_anchor=hyperparams["map_num_anchor"],
                embed_dims=embed_dims,
                anchor=f"{kmeans_dir}/{hyperparams['map_anchor_file']}",
                anchor_handler=bank_kps,
                num_temp_instances=(
                    hyperparams["num_map_temp_instances"]
                    if hyperparams["temporal_map"]
                    else -1
                ),
                confidence_decay=hyperparams["map_confidence_decay"],
                feat_grad=hyperparams["map_feat_grad"],
            ),
            SparsePoint3DEncoder(
                embed_dims=embed_dims,
                num_sample=hyperparams["num_sample"],
            ),
            cls._attn_factory(hyperparams, attn_dims),
            lambda: nn.LayerNorm(embed_dims),
            lambda: AsymmetricFFN(
                in_channels=embed_dims * 2,
                pre_norm=hyperparams["ffn_pre_norm"],
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=hyperparams["det_ffn_num_fcs"],
                ffn_drop=hyperparams["drop_out"],
                act_cfg=hyperparams["ffn_act_cfg"],
            ),
            deformable_factory,
            lambda: SparsePoint3DRefinementModule(
                embed_dims=embed_dims,
                num_sample=hyperparams["num_sample"],
                num_cls=num_map_classes,
            ),
            hyperparams["num_decoder"],
            hyperparams["num_single_frame_decoder_map"],
            temp_graph_model,
            decouple_attn,
            FocalLoss(
                use_sigmoid=True,
                gamma=hyperparams["det_loss_gamma"],
                alpha=hyperparams["det_loss_alpha"],
                loss_weight=hyperparams["map_loss_cls_weight"],
            ),
            SparseLineLoss(
                loss_line=LinesL1Loss(
                    loss_weight=hyperparams["map_loss_reg_weight"],
                    beta=hyperparams["map_loss_beta"],
                ),
                num_sample=hyperparams["num_sample"],
                roi_size=hyperparams["roi_size"],
            ),
            SparsePoint3DDecoder(),
            SparsePoint3DTarget(
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
            ),
            hyperparams["map_gt_cls_key"],
            hyperparams["map_gt_reg_key"],
            hyperparams["map_gt_id_key"],
            hyperparams["map_with_instance_id"],
            hyperparams["map_task_prefix"],
            hyperparams["map_reg_weights"],
            hyperparams["map_cls_threshold_to_reg"],
            hyperparams["det_dn_loss_weight"],
            cls._map_operation_order(hyperparams),
        )

    def init_weights(self):
        non_refine_lists = [
            self.sf_deformables, self.sf_ffns, self.sf_norm2s,
            self.temp_gnns, self.temp_local_gnns, self.temp_norm1s,
            self.temp_deformables, self.temp_ffns, self.temp_norm2s,
        ]
        if self.include_sf_gnn:
            non_refine_lists += [self.sf_gnns, self.sf_norm1s]
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

    def single_frame_decoder_forward(
        self,
        i,
        instance_feature,
        anchor,
        anchor_embed,
        feature_maps,
        metas,
        attn_mask,
        time_interval,
    ):
        if self.include_sf_gnn:
            if self.decouple_attn:
                query = torch.cat([instance_feature, anchor_embed], dim=-1)
                value = self.fc_before(instance_feature)
                instance_feature = self.fc_after(
                    self.sf_gnns[i](
                        query, None, value,
                        query_pos=None, key_pos=None,
                        attn_mask=attn_mask,
                    )
                )
            else:
                instance_feature = self.sf_gnns[i](
                    instance_feature, None, instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            instance_feature = self.sf_norm1s[i](instance_feature)
        instance_feature = self.sf_deformables[i](
            instance_feature, anchor, anchor_embed, feature_maps, metas
        )
        instance_feature = self.sf_ffns[i](instance_feature)
        instance_feature = self.sf_norm2s[i](instance_feature)
        anchor, cls, qt = self.sf_refines[i](
            instance_feature,
            anchor,
            anchor_embed,
            time_interval=time_interval,
            return_cls=True,
        )
        return instance_feature, anchor, cls, qt

    def temp_decoder_forward(
        self,
        i,
        instance_feature,
        anchor,
        anchor_embed,
        feature_maps,
        metas,
        temp_instance_feature,
        temp_anchor_embed,
        attn_mask,
        time_interval,
    ):
        if self.decouple_attn:
            query = torch.cat([instance_feature, anchor_embed], dim=-1)
            if temp_instance_feature is not None:
                key = torch.cat([temp_instance_feature, temp_anchor_embed], dim=-1)
                value = self.fc_before(temp_instance_feature)
            else:
                key = None
                value = None
            instance_feature = self.fc_after(
                self.temp_gnns[i](
                    query, key, value,
                    query_pos=None, key_pos=None,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
            )
        else:
            instance_feature = self.temp_gnns[i](
                instance_feature,
                temp_instance_feature,
                temp_instance_feature,
                query_pos=anchor_embed,
                key_pos=temp_anchor_embed,
                attn_mask=attn_mask if temp_instance_feature is None else None,
            )

        if self.decouple_attn:
            query = torch.cat([instance_feature, anchor_embed], dim=-1)
            value = self.fc_before(instance_feature)
            instance_feature = self.fc_after(
                self.temp_local_gnns[i](
                    query, None, value,
                    query_pos=None, key_pos=None,
                    attn_mask=attn_mask,
                )
            )
        else:
            instance_feature = self.temp_local_gnns[i](
                instance_feature, None, instance_feature,
                query_pos=anchor_embed,
                attn_mask=attn_mask,
            )
        instance_feature = self.temp_norm1s[i](instance_feature)
        instance_feature = self.temp_deformables[i](
            instance_feature, anchor, anchor_embed, feature_maps, metas
        )
        instance_feature = self.temp_ffns[i](instance_feature)
        instance_feature = self.temp_norm2s[i](instance_feature)
        anchor, cls, qt = self.temp_refines[i](
            instance_feature,
            anchor,
            anchor_embed,
            time_interval=time_interval,
            return_cls=True,
        )
        return instance_feature, anchor, cls, qt

    @classmethod
    def convert_state_dict(cls, old_sd, operation_order):
        """Convert an original SparseDrive state_dict (flat ``layers.N.*`` keys)
        to the new flat ModuleList key structure.

        ``fc_before.*`` and ``fc_after.*`` are kept at head level and require no remapping.

        Args:
            old_sd: state_dict from original SparseDrive Sparse4DHead.
            operation_order: The ``operation_order`` list that was used by the
                original model (e.g. from ``_det_operation_order()``).

        Returns:
            new_sd: dict with updated keys compatible with this model.
        """
        # Split operation_order into per-decoder-step chunks (split on "refine").
        steps = []
        current = []
        for op in operation_order:
            current.append(op)
            if op == "refine":
                steps.append(current)
                current = []

        # Map flat layer index → new key prefix.
        flat_index_map = {}
        sf_idx = 0
        temp_idx = 0
        flat_i = 0

        for step_ops in steps:
            is_temporal = "temp_gnn" in step_ops
            prev_op = None
            for op in step_ops:
                if op == "norm":
                    if prev_op in ("gnn", "temp_gnn"):
                        if is_temporal:
                            attr = f"temp_norm1s.{temp_idx}"
                        else:
                            attr = f"sf_norm1s.{sf_idx}"
                    else:
                        if is_temporal:
                            attr = f"temp_norm2s.{temp_idx}"
                        else:
                            attr = f"sf_norm2s.{sf_idx}"
                elif op == "temp_gnn":
                    attr = f"temp_gnns.{temp_idx}"
                elif op == "gnn":
                    if is_temporal:
                        attr = f"temp_local_gnns.{temp_idx}"
                    else:
                        attr = f"sf_gnns.{sf_idx}"
                elif op == "deformable":
                    if is_temporal:
                        attr = f"temp_deformables.{temp_idx}"
                    else:
                        attr = f"sf_deformables.{sf_idx}"
                elif op == "ffn":
                    if is_temporal:
                        attr = f"temp_ffns.{temp_idx}"
                    else:
                        attr = f"sf_ffns.{sf_idx}"
                elif op == "refine":
                    if is_temporal:
                        attr = f"temp_refines.{temp_idx}"
                    else:
                        attr = f"sf_refines.{sf_idx}"
                else:
                    attr = None
                if attr is not None:
                    flat_index_map[flat_i] = attr
                flat_i += 1
                prev_op = op

            if is_temporal:
                temp_idx += 1
            else:
                sf_idx += 1

        new_sd = {}
        for key, value in old_sd.items():
            if key.startswith("layers."):
                parts = key.split(".", 2)
                layer_idx = int(parts[1])
                rest = parts[2] if len(parts) > 2 else ""
                if layer_idx in flat_index_map:
                    prefix = flat_index_map[layer_idx]
                    new_key = f"{prefix}.{rest}" if rest else prefix
                    new_sd[new_key] = value
            else:
                new_sd[key] = value

        return new_sd

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= get instance info ============
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

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
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

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        cls = None
        for i in range(self.num_single_frame_decoder):
            instance_feature, anchor, cls, qt = self.single_frame_decoder_forward(
                i,
                instance_feature,
                anchor,
                anchor_embed,
                feature_maps,
                metas,
                attn_mask=attn_mask,
                time_interval=time_interval,
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

        for i in range(len(self.temp_gnns)):
            instance_feature, anchor, cls, qt = self.temp_decoder_forward(
                i,
                instance_feature,
                anchor,
                anchor_embed,
                feature_maps,
                metas,
                temp_instance_feature=temp_instance_feature,
                temp_anchor_embed=temp_anchor_embed,
                attn_mask=attn_mask,
                time_interval=time_interval,
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

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
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

            # cache dn_metas for temporal denoising
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

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if self.with_instance_id:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            reg_target_full = reg_target.clone()
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
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
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

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

        # ===================== denoising losses ======================
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
