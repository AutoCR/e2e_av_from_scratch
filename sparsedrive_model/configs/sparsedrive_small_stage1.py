import torch.nn as nn

from sparsedrive import (
    AsymmetricFFN,
    CrossEntropyLoss,
    DeformableFeatureAggregation,
    DenseDepthNet,
    FPN,
    FocalLoss,
    FocalLossCost,
    GaussianFocalLoss,
    HierarchicalPlanningDecoder,
    HungarianLinesAssigner,
    InstanceBank,
    InstanceQueue,
    L1Loss,
    LinesL1Cost,
    LinesL1Loss,
    MapQueriesCost,
    MotionPlanningHead,
    MotionPlanningRefinementModule,
    MotionTarget,
    MultiheadFlashAttention,
    PlanningTarget,
    Sparse4DHead,
    SparseBox3DDecoder,
    SparseBox3DEncoder,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DLoss,
    SparseBox3DMotionDecoder,
    SparseBox3DRefinementModule,
    SparseBox3DTarget,
    SparseDrive,
    SparseDriveHead,
    SparseLineLoss,
    SparsePoint3DDecoder,
    SparsePoint3DEncoder,
    SparsePoint3DKeyPointsGenerator,
    SparsePoint3DRefinementModule,
    SparsePoint3DTarget,
    TimmResNet50,
)

version = "trainval"
length = {"trainval": 28130, "mini": 323}
total_batch_size = 64
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20
input_shape = (704, 256)

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
map_class_names = ["ped_crossing", "divider", "boundary"]
num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)
num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
num_map_temp_instances = 0
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

task_config = dict(with_det=True, with_map=True, with_motion_plan=False)


def _attn_factory(dims):
    return lambda: MultiheadFlashAttention(
        embed_dims=dims,
        num_heads=num_groups,
        batch_first=True,
        dropout=drop_out,
    )


def _det_operation_order():
    return (
        ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame_decoder
        + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
        * (num_decoder - num_single_frame_decoder)
    )[2:]


def _map_operation_order():
    return (
        ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame_decoder_map
        + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
        * (num_decoder - num_single_frame_decoder_map)
    )


def _make_det_head():
    bank_kps = SparseBox3DKeyPointsGenerator()
    def deformable_factory():
        return DeformableFeatureAggregation(
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=SparseBox3DKeyPointsGenerator(
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        )
    return Sparse4DHead(
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=InstanceBank(
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="data/kmeans/kmeans_det_900.npy",
            anchor_handler=bank_kps,
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=SparseBox3DEncoder(
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=_det_operation_order(),
        temp_graph_model=_attn_factory(embed_dims * 2 if decouple_attn else embed_dims) if temporal else None,
        graph_model_factory=_attn_factory(embed_dims * 2 if decouple_attn else embed_dims),
        norm_layer_factory=lambda: nn.LayerNorm(embed_dims),
        ffn_factory=lambda: AsymmetricFFN(
            in_channels=embed_dims * 2,
            pre_norm="LN",
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg="ReLU",
        ),
        deformable_model=deformable_factory,
        refine_layer_factory=lambda: SparseBox3DRefinementModule(
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=SparseBox3DTarget(
            num_dn_groups=0,
            num_temp_dn_groups=0,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={class_names.index("traffic_cone"): [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]},
        ),
        loss_cls=FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_reg=SparseBox3DLoss(
            loss_box=L1Loss(loss_weight=0.25),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True),
            loss_yawness=GaussianFocalLoss(),
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=SparseBox3DDecoder(),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    )


def _make_map_head():
    bank_kps = SparsePoint3DKeyPointsGenerator()
    def deformable_factory():
        return DeformableFeatureAggregation(
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=SparsePoint3DKeyPointsGenerator(
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_learnable_pts=3,
                fix_height=(0, 0.5, -0.5, 1, -1),
                ground_height=-1.84023,
            ),
        )
    return Sparse4DHead(
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn_map,
        instance_bank=InstanceBank(
            num_anchor=100,
            embed_dims=embed_dims,
            anchor="data/kmeans/kmeans_map_100.npy",
            anchor_handler=bank_kps,
            num_temp_instances=num_map_temp_instances if temporal_map else -1,
            confidence_decay=0.6,
            feat_grad=True,
        ),
        anchor_encoder=SparsePoint3DEncoder(embed_dims=embed_dims, num_sample=num_sample),
        num_single_frame_decoder=num_single_frame_decoder_map,
        operation_order=_map_operation_order(),
        temp_graph_model=_attn_factory(embed_dims * 2 if decouple_attn_map else embed_dims) if temporal_map else None,
        graph_model_factory=_attn_factory(embed_dims * 2 if decouple_attn_map else embed_dims),
        norm_layer_factory=lambda: nn.LayerNorm(embed_dims),
        ffn_factory=lambda: AsymmetricFFN(
            in_channels=embed_dims * 2,
            pre_norm="LN",
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg="ReLU",
        ),
        deformable_model=deformable_factory,
        refine_layer_factory=lambda: SparsePoint3DRefinementModule(
            embed_dims=embed_dims,
            num_sample=num_sample,
            num_cls=num_map_classes,
        ),
        sampler=SparsePoint3DTarget(
            assigner=HungarianLinesAssigner(
                cost=MapQueriesCost(
                    cls_cost=FocalLossCost(weight=1.0),
                    reg_cost=LinesL1Cost(weight=10.0, beta=0.01, permute=True),
                )
            ),
            num_cls=num_map_classes,
            num_sample=num_sample,
            roi_size=roi_size,
        ),
        loss_cls=FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_reg=SparseLineLoss(
            loss_line=LinesL1Loss(loss_weight=10.0, beta=0.01),
            num_sample=num_sample,
            roi_size=roi_size,
        ),
        decoder=SparsePoint3DDecoder(),
        reg_weights=[1.0] * 40,
        gt_cls_key="gt_map_labels",
        gt_reg_key="gt_map_pts",
        gt_id_key="map_instance_id",
        with_instance_id=False,
        task_prefix="map",
    )


def _make_motion_plan_head():
    return MotionPlanningHead(
        fut_ts=fut_ts,
        fut_mode=fut_mode,
        ego_fut_ts=ego_fut_ts,
        ego_fut_mode=ego_fut_mode,
        motion_anchor=f"data/kmeans/kmeans_motion_{fut_mode}.npy",
        plan_anchor=f"data/kmeans/kmeans_plan_{ego_fut_mode}.npy",
        embed_dims=embed_dims,
        decouple_attn=decouple_attn_motion,
        instance_queue=InstanceQueue(
            embed_dims=embed_dims,
            queue_length=queue_length,
            tracking_threshold=0.2,
            feature_map_scale=(input_shape[1] / strides[-1], input_shape[0] / strides[-1]),
        ),
        operation_order=(["temp_gnn", "gnn", "norm", "cross_gnn", "norm", "ffn", "norm"] * 3 + ["refine"]),
        temp_graph_model=_attn_factory(embed_dims * 2 if decouple_attn_motion else embed_dims),
        graph_model_factory=_attn_factory(embed_dims * 2 if decouple_attn_motion else embed_dims),
        cross_graph_model_factory=_attn_factory(embed_dims),
        norm_layer_factory=lambda: nn.LayerNorm(embed_dims),
        ffn_factory=lambda: AsymmetricFFN(
            in_channels=embed_dims,
            pre_norm="LN",
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 2,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg="ReLU",
        ),
        refine_layer_factory=lambda: MotionPlanningRefinementModule(
            embed_dims=embed_dims,
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
        ),
        motion_sampler=MotionTarget(),
        motion_loss_cls=FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
        motion_loss_reg=L1Loss(loss_weight=0.2),
        planning_sampler=PlanningTarget(ego_fut_ts=ego_fut_ts, ego_fut_mode=ego_fut_mode),
        plan_loss_cls=FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.5),
        plan_loss_reg=L1Loss(loss_weight=1.0),
        plan_loss_status=L1Loss(loss_weight=1.0),
        motion_decoder=SparseBox3DMotionDecoder(),
        planning_decoder=HierarchicalPlanningDecoder(
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
            use_rescore=True,
        ),
        num_det=50,
        num_map=10,
    )


def build():
    img_backbone = TimmResNet50(pretrained="model_weights/resnet50-19c8e357.pth", with_cp=True)
    img_neck = FPN(
        num_outs=num_levels,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        in_channels=[256, 512, 1024, 2048],
    )
    head = SparseDriveHead(
        task_config=task_config,
        det_head=_make_det_head(),
        map_head=_make_map_head(),
        motion_plan_head=_make_motion_plan_head() if task_config["with_motion_plan"] else None,
    )
    return SparseDrive(
        img_backbone=img_backbone,
        img_neck=img_neck,
        head=head,
        use_grid_mask=True,
        use_deformable_func=use_deformable_func,
        depth_branch=DenseDepthNet(embed_dims=embed_dims, num_depth_layers=num_depth_layers, loss_weight=0.2),
    )
