import copy

from uniad.configs.uniad_base_e2e_hparams import UNIAD_BASE_E2E_HPARAMS
from uniad.models.detectors.uniad_e2e import UniAD


def _merge_hparams(hparams=None, **overrides):
    merged = copy.deepcopy(UNIAD_BASE_E2E_HPARAMS)
    if hparams:
        merged.update(copy.deepcopy(hparams))
    merged.update(overrides)
    return merged


def _dummy_anchor_cfg(hparams):
    if not hparams.get("dummy_motion_anchors", False):
        return {}
    return dict(
        dummy_motion_anchors=True,
        dummy_motion_anchor_seed=hparams.get("dummy_motion_anchor_seed", 0),
        dummy_motion_anchor_scale=hparams.get("dummy_motion_anchor_scale", 1.0),
    )


def _loss_cfgs():
    return dict(
        focal=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        l1_track=dict(type="L1Loss", loss_weight=0.25),
        giou_zero=dict(type="GIoULoss", loss_weight=0.0),
        l1_map=dict(type="L1Loss", loss_weight=5.0),
        giou_map=dict(type="GIoULoss", loss_weight=2.0),
        dice_map=dict(type="DiceLoss", loss_weight=2.0),
    )


def _perception_transformer(pc_range, embed_dims, ffn_dims, num_levels):
    return dict(
        type="PerceptionTransformer",
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        embed_dims=embed_dims,
        encoder=dict(
            type="BEVFormerEncoder",
            num_layers=6,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type="BEVFormerLayer",
                attn_cfgs=[
                    dict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                    dict(
                        type="SpatialCrossAttention",
                        pc_range=pc_range,
                        deformable_attention=dict(
                            type="MSDeformableAttention3D",
                            embed_dims=embed_dims,
                            num_points=8,
                            num_levels=num_levels,
                        ),
                        embed_dims=embed_dims,
                    ),
                ],
                feedforward_channels=ffn_dims,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        decoder=dict(
            type="DetectionTransformerDecoder",
            num_layers=6,
            return_intermediate=True,
            transformerlayers=dict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=[
                    dict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                    dict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
                ],
                feedforward_channels=ffn_dims,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
    )


def _seg_transformer(embed_dims, ffn_dims, num_levels):
    return dict(
        type="SegDeformableTransformer",
        encoder=dict(
            type="DetrTransformerEncoder",
            num_layers=6,
            transformerlayers=dict(
                type="BaseTransformerLayer",
                attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=embed_dims, num_levels=num_levels),
                feedforward_channels=ffn_dims,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
        ),
        decoder=dict(
            type="DeformableDetrTransformerDecoder",
            num_layers=6,
            return_intermediate=True,
            transformerlayers=dict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=[
                    dict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                    dict(type="MultiScaleDeformableAttention", embed_dims=embed_dims, num_levels=num_levels),
                ],
                feedforward_channels=ffn_dims,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
    )


def _occ_decoder(embed_dims):
    return dict(
        type="DetrTransformerDecoder",
        return_intermediate=True,
        num_layers=5,
        transformerlayers=dict(
            type="DetrTransformerDecoderLayer",
            attn_cfgs=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                dropout_layer=None,
                batch_first=False,
            ),
            ffn_cfgs=dict(
                embed_dims=embed_dims,
                feedforward_channels=2048,
                num_fcs=2,
                act_cfg=dict(type="ReLU", inplace=True),
                ffn_drop=0.0,
                dropout_layer=None,
                add_identity=True,
            ),
            feedforward_channels=2048,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        ),
        init_cfg=None,
    )


def _motion_transformer(pc_range, embed_dims, ffn_dims, predict_steps):
    return dict(
        type="MotionTransformerDecoder",
        pc_range=pc_range,
        embed_dims=embed_dims,
        num_layers=3,
        transformerlayers=dict(
            type="MotionTransformerAttentionLayer",
            batch_first=True,
            attn_cfgs=[
                dict(
                    type="MotionDeformableAttention",
                    num_steps=predict_steps,
                    embed_dims=embed_dims,
                    num_levels=1,
                    num_heads=8,
                    num_points=4,
                    sample_index=-1,
                )
            ],
            feedforward_channels=ffn_dims,
            ffn_dropout=0.1,
            operation_order=("cross_attn", "norm", "ffn", "norm"),
        ),
    )


def _model_kwargs_from_hparams(hparams):
    pc_range = hparams["point_cloud_range"]
    voxel_size = hparams["voxel_size"]
    bev_h = hparams["bev_h"]
    bev_w = hparams["bev_w"]
    canvas_size = (bev_h, bev_w)
    embed_dims = hparams["embed_dims"]
    ffn_dims = hparams["ffn_dims"]
    num_levels = hparams["num_feature_levels"]
    losses = _loss_cfgs()

    return dict(
        gt_iou_threshold=hparams["train_gt_iou_threshold"],
        queue_length=hparams["queue_length"],
        use_grid_mask=True,
        video_test_mode=True,
        num_query=hparams["num_query"],
        num_classes=hparams["num_classes"],
        vehicle_id_list=hparams["vehicle_id_list"],
        pc_range=pc_range,
        img_backbone=dict(
            type="ResNet",
            depth=101,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=4,
            norm_cfg=dict(type="BN2d", requires_grad=False),
            norm_eval=True,
            style="caffe",
            dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
        ),
        img_neck=dict(
            type="FPN",
            in_channels=[512, 1024, 2048],
            out_channels=embed_dims,
            start_level=0,
            add_extra_convs="on_output",
            num_outs=4,
            relu_before_extra_convs=True,
        ),
        freeze_img_backbone=hparams["freeze_img_backbone"],
        freeze_img_neck=hparams["freeze_img_neck"],
        freeze_bn=hparams["freeze_bn"],
        freeze_bev_encoder=hparams["freeze_bev_encoder"],
        score_thresh=hparams["score_thresh"],
        filter_score_thresh=hparams["filter_score_thresh"],
        qim_args=dict(qim_type="QIMBase", merger_dropout=0, update_query_pos=True, fp_ratio=0.3, random_drop=0.1),
        mem_args=dict(memory_bank_type="MemoryBank", memory_bank_score_thresh=0.0, memory_bank_len=4),
        loss_cfg=dict(
            type="ClipMatcher",
            num_classes=hparams["num_classes"],
            weight_dict=None,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3DTrack",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                pc_range=pc_range,
            ),
            loss_cls=copy.deepcopy(losses["focal"]),
            loss_bbox=copy.deepcopy(losses["l1_track"]),
        ),
        pts_bbox_head=dict(
            type="BEVFormerTrackHead",
            bev_h=bev_h,
            bev_w=bev_w,
            num_query=hparams["num_query"],
            num_classes=hparams["num_classes"],
            in_channels=embed_dims,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            past_steps=hparams["past_steps"],
            fut_steps=hparams["fut_steps"],
            transformer=_perception_transformer(pc_range, embed_dims, ffn_dims, num_levels),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=pc_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=hparams["num_classes"],
            ),
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=embed_dims // 2,
                row_num_embed=bev_h,
                col_num_embed=bev_w,
            ),
            loss_cls=copy.deepcopy(losses["focal"]),
            loss_bbox=copy.deepcopy(losses["l1_track"]),
            loss_iou=copy.deepcopy(losses["giou_zero"]),
        ),
        seg_head=dict(
            type="PansegformerHead",
            bev_h=bev_h,
            bev_w=bev_w,
            canvas_size=canvas_size,
            pc_range=pc_range,
            num_query=300,
            num_classes=4,
            num_things_classes=3,
            num_stuff_classes=1,
            in_channels=2048,
            sync_cls_avg_factor=True,
            as_two_stage=False,
            with_box_refine=True,
            transformer=_seg_transformer(embed_dims, ffn_dims, num_levels),
            positional_encoding=dict(type="SinePositionalEncoding", num_feats=embed_dims // 2, normalize=True, offset=-0.5),
            loss_cls=copy.deepcopy(losses["focal"]),
            loss_bbox=copy.deepcopy(losses["l1_map"]),
            loss_iou=copy.deepcopy(losses["giou_map"]),
            loss_mask=copy.deepcopy(losses["dice_map"]),
            thing_transformer_head=dict(type="SegMaskHead", d_model=embed_dims, nhead=8, num_decoder_layers=4),
            stuff_transformer_head=dict(type="SegMaskHead", d_model=embed_dims, nhead=8, num_decoder_layers=6, self_attn=True),
            train_cfg=dict(
                assigner=dict(
                    type="HungarianAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ),
                assigner_with_mask=dict(
                    type="HungarianAssigner_multi_info",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                    iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
                    mask_cost=dict(type="DiceCost", weight=2.0),
                ),
                sampler=dict(type="PseudoSampler"),
                sampler_with_mask=dict(type="PseudoSampler_segformer"),
            ),
        ),
        occ_head=dict(
            type="OccHead",
            grid_conf=hparams["occflow_grid_conf"],
            ignore_index=255,
            bev_proj_dim=256,
            bev_proj_nlayers=4,
            attn_mask_thresh=0.3,
            transformer_decoder=_occ_decoder(embed_dims),
            query_dim=embed_dims,
            query_mlp_layers=3,
            aux_loss_weight=1.0,
            loss_mask=dict(
                type="FieryBinarySegmentationLoss",
                use_top_k=True,
                top_k_ratio=0.25,
                future_discount=0.95,
                loss_weight=5.0,
                ignore_index=255,
            ),
            loss_dice=dict(
                type="DiceLossWithMasks",
                use_sigmoid=True,
                activate=True,
                reduction="mean",
                naive_dice=True,
                eps=1.0,
                ignore_index=255,
                loss_weight=1.0,
            ),
            pan_eval=True,
            test_seg_thresh=0.1,
            test_with_track_score=True,
        ),
        motion_head=dict(
            type="MotionHead",
            bev_h=bev_h,
            bev_w=bev_w,
            num_query=300,
            num_classes=hparams["num_classes"],
            predict_steps=hparams["predict_steps"],
            predict_modes=hparams["predict_modes"],
            embed_dims=embed_dims,
            loss_traj=dict(
                type="TrajLoss",
                use_variance=True,
                cls_loss_weight=0.5,
                nll_loss_weight=0.5,
                loss_weight_minade=0.0,
                loss_weight_minfde=0.25,
            ),
            num_cls_fcs=3,
            pc_range=pc_range,
            group_id_list=hparams["group_id_list"],
            num_anchor=hparams["motion_num_anchor"],
            use_nonlinear_optimizer=hparams["use_nonlinear_optimizer"],
            anchor_info_path=hparams["motion_anchor_info_path"],
            **_dummy_anchor_cfg(hparams),
            transformerlayers=_motion_transformer(pc_range, embed_dims, ffn_dims, hparams["predict_steps"]),
        ),
        planning_head=dict(
            type="PlanningHeadSingleMode",
            embed_dims=embed_dims,
            planning_steps=hparams["planning_steps"],
            loss_planning=dict(type="PlanningLoss"),
            loss_collision=[
                dict(type="CollisionLoss", delta=0.0, weight=2.5),
                dict(type="CollisionLoss", delta=0.5, weight=1.0),
                dict(type="CollisionLoss", delta=1.0, weight=0.25),
            ],
            use_col_optim=hparams["use_col_optim"],
            planning_eval=True,
            with_adapter=True,
        ),
        task_loss_weight=hparams["task_loss_weight"],
        train_cfg=dict(
            pts=dict(
                grid_size=[512, 512, 1],
                voxel_size=voxel_size,
                point_cloud_range=pc_range,
                out_size_factor=4,
                assigner=dict(
                    type="HungarianAssigner3D",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                    iou_cost=dict(type="IoUCost", weight=0.0),
                    pc_range=pc_range,
                ),
            )
        ),
    )


def build_uniad(hparams=None, **overrides):
    return UniAD(**_model_kwargs_from_hparams(_merge_hparams(hparams, **overrides)))
