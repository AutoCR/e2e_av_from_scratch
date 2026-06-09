"""
BEVFusion camera+LiDAR detection hyperparameters.

Organized by module: dataset, camera encoder, lidar encoder, fuser, decoder, detection head.
All hyperparams are plain Python dicts/lists - no mmdet/mmcv dependencies.

Checkpoint: model_weights/bevfusion/bevfusion-det.pth
Config: swint_v0p075/convfuser.yaml (camera+lidar detection with TransFusionHead)
"""

from copy import deepcopy

# ==============================================================================
# DATASET
# ==============================================================================

# NAVSIM 5-class object set (training only): order must match dataset label remap
NAVSIM_OBJECT_CLASSES = ["car", "barrier", "bicycle", "pedestrian", "traffic_cone"]

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
# TRAINING (NAVSIM 5-CLASS)
# ==============================================================================

# 5-class detection tasks for NAVSIM (single task group)
DETECTION_TASKS_5CLASS = [
    {"num_class": 5, "class_names": NAVSIM_OBJECT_CLASSES}
]

# Training head config for NAVSIM (5-class variant)
DETECTION_HEAD_TRAIN_CFG_5 = {
    "point_cloud_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    "grid_size": [1440, 1440, 41],
    "voxel_size": [0.075, 0.075, 0.2],
    "out_size_factor": 8,
    "gaussian_overlap": 0.1,
    "min_radius": 2,
    "dataset": "nuScenes",
    "code_weights": [1.0]*8 + [0.2]*2,
    "pos_weight": -1,
    "loss_cls": {"gamma": 2.0, "alpha": 0.25, "loss_weight": 1.0},
    "loss_heatmap": {"loss_weight": 1.0},
    "loss_bbox": {"loss_weight": 0.25},
    "assigner": {
        "cls_cost": {"gamma": 2.0, "alpha": 0.25, "weight": 0.15},
        "reg_cost": {"weight": 0.25},
        "iou_cost": {"weight": 0.25},
        "pc_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    },
}

# 5-class detection head for NAVSIM training (to be dynamically created by get_training_hyperparams)
# This dict is assembled per-call to avoid module-state mutation
def _make_detection_head_train_5class():
    """Create a fresh 5-class detection head for training."""
    return {
        "type": "TransFusionHead",
        "num_proposals": 200,
        "auxiliary": True,
        "in_channels": 512,
        "hidden_channel": 128,
        "num_classes": 5,
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
        "tasks": DETECTION_TASKS_5CLASS,
        "train_cfg": DETECTION_HEAD_TRAIN_CFG_5,
        "test_cfg": DETECTION_HEAD_TEST_CFG,
        "bbox_coder": BBOX_CODER,
    }

# ==============================================================================
# TRAINING RECIPE & RUNTIME CONFIG
# ==============================================================================

TRAINING_RECIPE = {
    "lr": 1e-4,
    "weight_decay": 0.01,
    "backbone_lr_mult": 1.0,
    "grad_clip_max_norm": 35.0,
    "grad_clip_norm_type": 2.0,
    "warmup_iters": 500,
    "warmup_ratio": 1.0 / 3.0,
    "min_lr_ratio": 1e-3,
    "num_epochs": 6,
    "total_batch_size": 4,
    "num_workers": 4,
    "fp16_loss_scale": 512.0,
    "log_interval": 50,
    "ckpt_epoch_interval": 1,
    "eval_epoch_interval": 1,
}

RUNTIME_CONFIG = {
    "splits": {
        "train": {
            "dir": "trainval",
            "log_names_yaml": "navsim/planning/script/config/training/default_train_val_test_log_split.yaml",
            "log_names_key": "train_logs",
            "tokens_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
            "tokens_key": "tokens",
        },
        "val": {
            "dir": "trainval",
            "log_names_yaml": "navsim/planning/script/config/training/default_train_val_test_log_split.yaml",
            "log_names_key": "val_logs",
            "tokens_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
            "tokens_key": "tokens",
        },
        "test": {
            "dir": "trainval",
            "log_names_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
            "log_names_key": "log_names",
            "tokens_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
            "tokens_key": "tokens",
        },
    },
    "openscene_data_root": "/Users/chenran/Code/navsim/dataset",
    "nuplan_maps_root": "/Users/chenran/Code/navsim/dataset/maps",
    "output_dir": "bevfusion_model/outputs/train_navsim",
    "resume_from": None,
    "seed": 0,
    "num_workers": 4,
    "device": "auto",
    "quick_smoke": False,
    "camera_order": (
        "CAM_F0",
        "CAM_L0",
        "CAM_L1",
        "CAM_R0",
        "CAM_R1",
        "CAM_L2",
        "CAM_R2",
        "CAM_B0",
    ),
    "image_hw": [256, 704],
}

# ==============================================================================
# MERGED HYPERPARAMS
# ==============================================================================


def get_fusion_hyperparams() -> dict:
    """
    Returns merged hyperparameters dict for full camera+LiDAR fusion model (inference).
    Uses the 10-class NuScenes detection head unchanged.

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


def get_training_hyperparams() -> dict:
    """
    Returns merged hyperparameters dict for BEVFusion training on NAVSIM.
    Uses 5-class detection head (car, barrier, bicycle, pedestrian, traffic_cone).

    Returns:
        dict: Complete hyperparams with keys:
              - dataset, camera_encoder, lidar_encoder, fuser, decoder, detection_head
              Dataset has NAVSIM_OBJECT_CLASSES; detection_head is 5-class.
    """
    # Deep-copy to prevent callers from mutating module state
    return {
        "dataset": {
            **deepcopy(DATASET),
            "object_classes": deepcopy(NAVSIM_OBJECT_CLASSES),
        },
        "camera_encoder": deepcopy(CAMERA_ENCODER),
        "lidar_encoder": deepcopy(LIDAR_ENCODER),
        "fuser": deepcopy(FUSER),
        "decoder": deepcopy(DECODER),
        "detection_head": deepcopy(_make_detection_head_train_5class()),
    }


def get_training_recipe() -> dict:
    """Return a deep-copy of the training recipe (optimizer, scheduler, batch config)."""
    return deepcopy(TRAINING_RECIPE)


def get_runtime_config() -> dict:
    """Return a deep-copy of the runtime/environment configuration."""
    return deepcopy(RUNTIME_CONFIG)
