"""
BEVFusion camera+LiDAR detection hyperparameters.

Organized by module: dataset, camera encoder, lidar encoder, fuser, decoder, detection head.
All hyperparams are plain Python dicts/lists - no mmdet/mmcv dependencies.

Checkpoint: model_weights/bevfusion/bevfusion-det.pth
Config: swint_v0p075/convfuser.yaml (camera+lidar detection with TransFusionHead)
"""

# ==============================================================================
# DATASET
# ==============================================================================

DATASET = {
    "name": "nuscenes",
    "voxel_size": [0.075, 0.075, 0.2],
    "point_cloud_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    "image_size": [256, 704],
    "object_classes": [
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
    ],
}

# ==============================================================================
# CAMERA ENCODER
# ==============================================================================

CAMERA_BACKBONE = {
    "type": "SwinTransformer",
    "in_channels": 3,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.2,
    "patch_size": 4,
    "out_indices": (0, 1, 2),
}

CAMERA_NECK = {
    "type": "GeneralizedLSSFPN",
    "in_channels": [192, 384, 768],
    "out_channels": 256,
    "start_level": 0,
    "num_outs": 2,
    "upsample_mode": "bilinear",
    "upsample_align_corners": False,
}

CAMERA_VTRANSFORM = {
    "type": "DepthLSSTransform",
    "in_channels": 256,
    "out_channels": 80,
    "image_size": [256, 704],
    "feature_size": [32, 88],
    # swint_v0p075 overrides the base camera+lidar config to match the
    # 1440x1440 LiDAR grid and TransFusion head's 180x180 BEV positions.
    "xbound": [-54.0, 54.0, 0.3],
    "ybound": [-54.0, 54.0, 0.3],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 0.5],
    "downsample": 2,
    "use_depth_classifier": True,
}

CAMERA_ENCODER = {
    "backbone": CAMERA_BACKBONE,
    "neck": CAMERA_NECK,
    "vtransform": CAMERA_VTRANSFORM,
}

# ==============================================================================
# LIDAR ENCODER
# ==============================================================================

LIDAR_VOXELIZE = {
    "type": "HardVoxelization",
    "voxel_size": [0.075, 0.075, 0.2],
    "point_cloud_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    "max_num_points": 10,
    "max_voxels": [120000, 160000],
}

LIDAR_BACKBONE = {
    "type": "SparseEncoder",
    "in_channels": 5,
    "sparse_shape": [1440, 1440, 41],
    "output_channels": 128,
    "order": ["conv", "norm", "act"],
    "encoder_channels": [
        [16, 16, 32],
        [32, 32, 64],
        [64, 64, 128],
        [128, 128],
    ],
    "encoder_paddings": [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, [1, 1, 0]],
        [0, 0],
    ],
    "block_type": "basicblock",
    "layer_nums": [1, 1, 1, 1],
    "layer_strides": [1, 2, 2, 2],
    "norm_cfg": {"type": "BN"},
}

LIDAR_ENCODER = {
    "voxelize": LIDAR_VOXELIZE,
    "backbone": LIDAR_BACKBONE,
}

# ==============================================================================
# FUSER
# ==============================================================================

FUSER = {
    "type": "ConvFuser",
    "in_channels": [80, 256],
    "out_channels": 256,
}

# ==============================================================================
# DECODER
# ==============================================================================

DECODER_BACKBONE = {
    "type": "SECOND",
    "in_channels": 256,
    "out_channels": [128, 256],
    "layer_nums": [5, 5],
    "layer_strides": [1, 2],
    "norm_cfg": {"type": "BN"},
    "conv_cfg": {"type": "Conv2d"},
}

DECODER_NECK = {
    "type": "SECONDFPN",
    "in_channels": [128, 256],
    "out_channels": [256, 256],
    "upsample_strides": [1, 2],
    "norm_cfg": {"type": "BN"},
    "use_conv_for_no_stride": False,
}

# Flat decoder config for BEVFusion model instantiation
DECODER = {
    "in_channels": 256,
    "backbone_out_channels": [128, 256],
    "backbone_layer_nums": [5, 5],
    "backbone_layer_strides": [1, 2],
    "neck_out_channels": [256, 256],
    "neck_upsample_strides": [1, 2],
}

# ==============================================================================
# DETECTION HEAD
# ==============================================================================

# NuScenes detection tasks: 10 classes in 6 task groups
DETECTION_TASKS = [
    {"num_class": 1, "class_names": ["car"]},
    {"num_class": 2, "class_names": ["truck", "construction_vehicle"]},
    {"num_class": 2, "class_names": ["bus", "trailer"]},
    {"num_class": 1, "class_names": ["barrier"]},
    {"num_class": 2, "class_names": ["motorcycle", "bicycle"]},
    {"num_class": 2, "class_names": ["pedestrian", "traffic_cone"]},
]

DETECTION_HEAD_TRAIN_CFG = {
    "point_cloud_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    "grid_size": [1440, 1440, 41],
    "voxel_size": [0.075, 0.075, 0.2],
    "out_size_factor": 8,
    "gaussian_overlap": 0.1,
    "min_radius": 2,
    "dataset": "nuScenes",
}

DETECTION_HEAD_TEST_CFG = {
    "dataset": "nuScenes",
    "grid_size": [1440, 1440, 41],
    "out_size_factor": 8,
    "voxel_size": [0.075, 0.075],
    "pc_range": [-54.0, -54.0],
    "nms_type": None,
    "post_center_limit_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    "score_threshold": 0.0,
}

BBOX_CODER = {
    "type": "TransFusionBBoxCoder",
    "pc_range": [-54.0, -54.0],
    "voxel_size": [0.075, 0.075],
    "out_size_factor": 8,
    "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    "score_threshold": 0.0,
    "code_size": 10,
}

DETECTION_HEAD = {
    "type": "TransFusionHead",
    "num_proposals": 200,
    "auxiliary": True,
    "in_channels": 512,
    "hidden_channel": 128,
    "num_classes": 10,
    "num_decoder_layers": 1,
    "num_heads": 8,
    "nms_kernel_size": 3,
    "ffn_channel": 256,
    "dropout": 0.1,
    "activation": "relu",
    "bn_momentum": 0.1,
    "common_heads": {
        "center": [2, 2],
        "height": [1, 2],
        "dim": [3, 2],
        "rot": [2, 2],
        "vel": [2, 2],
    },
    "num_heatmap_convs": 2,
    "tasks": DETECTION_TASKS,
    "train_cfg": DETECTION_HEAD_TRAIN_CFG,
    "test_cfg": DETECTION_HEAD_TEST_CFG,
    "bbox_coder": BBOX_CODER,
}

# ==============================================================================
# MERGED HYPERPARAMS
# ==============================================================================


def get_fusion_hyperparams() -> dict:
    """
    Returns merged hyperparameters dict for full camera+LiDAR fusion model.

    Returns:
        dict: Complete hyperparams with keys:
              - dataset, camera_encoder, lidar_encoder, fuser, decoder, detection_head
    """
    return {
        "dataset": DATASET,
        "camera_encoder": CAMERA_ENCODER,
        "lidar_encoder": LIDAR_ENCODER,
        "fuser": FUSER,
        "decoder": DECODER,
        "detection_head": DETECTION_HEAD,
    }
