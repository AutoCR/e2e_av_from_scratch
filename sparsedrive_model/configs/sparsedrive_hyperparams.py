from copy import deepcopy

# =============================================================================
# SECTION 1: RUNTIME & ENVIRONMENT
# =============================================================================
# Edit this section to configure your local environment before running.

RUNTIME_CONFIG = {
    "stage": "stage1",              # "stage1" (det+map pre-train) or "stage2" (full fine-tune)
    "splits": {
        "train": "mini",            # "mini" (323 scenes) or "trainval" (28130 scenes)
        "val": "mini",
        "test": "mini",
    },
    "openscene_data_root": "/Users/chenran/Code/navsim/dataset",
    "nuplan_maps_root": "/Users/chenran/Code/navsim/dataset/maps",
    "output_dir": "sparsedrive_model/outputs/train_navsim",
    "load_from": None,              # Path to a checkpoint to init weights (stage2: "ckpt/sparsedrive_stage1.pth")
    "resume_from": None,            # Path to a full checkpoint to resume interrupted training
    "seed": 0,
    "num_workers": 4,
    "device": "cpu",                # "auto", "cuda", "mps", or "cpu"
    "quick_smoke": True,           # True → 1-iter smoke test with 2 scenes
}

# =============================================================================
# SECTION 2: TRAINING SCHEDULE
# =============================================================================
# Epochs, batch size, checkpoint/eval cadence and evaluation modes per stage.

TRAINING_SCHEDULE_STAGE1 = {
    "num_epochs": 100,
    "total_batch_size": 64,
    "num_gpus": 8,                  # Reference GPU count (used only by derive_training_hyperparams)
    "ckpt_epoch_interval": 20,      # Save a checkpoint every N epochs
    "eval_epoch_interval": 20,      # Run validation every N epochs
    "eval_mode": {
        "with_det": True,
        "with_tracking": True,
        "with_map": True,
        "with_motion": False,
        "with_planning": False,
        "tracking_threshold": 0.2,
        "motion_threshhold": 0.2,
    },
}

TRAINING_SCHEDULE_STAGE2 = {
    "num_epochs": 10,
    "total_batch_size": 48,
    "num_gpus": 8,
    "ckpt_epoch_interval": 10,
    "eval_epoch_interval": 10,
    "eval_mode": {
        "with_det": True,
        "with_tracking": True,
        "with_map": True,
        "with_motion": True,
        "with_planning": True,
        "tracking_threshold": 0.2,
        "motion_threshhold": 0.2,
    },
}

# =============================================================================
# SECTION 3: OPTIMIZER & SCHEDULER
# =============================================================================
# Learning rate, weight decay, warmup schedule, gradient clipping, and logging.

OPTIMIZER_CONFIG = {
    "lr": 4e-4,
    "weight_decay": 0.001,
    "backbone_lr_mult": 0.5,        # LR multiplier for backbone parameters
    "grad_clip_max_norm": 25.0,
    "grad_clip_norm_type": 2.0,
    "warmup_iters": 500,
    "warmup_ratio": 1.0 / 3.0,
    "min_lr_ratio": 1e-3,           # Final LR = lr * min_lr_ratio
    "fp16_loss_scale": 32.0,
    "log_interval": 51,             # Print/TensorBoard log every N iterations
    # NAVSIM uses 8 cameras; overrides the model-architecture default of 6.
    "num_cams": 8,
}

# =============================================================================
# SECTION 4: MODEL ARCHITECTURE
# =============================================================================
# Dataset meta, input resolution, and core network settings (backbone, FPN,
# deformable attention, temporal fusion).

MODEL_ARCH = {
    # --- Dataset meta (used by derive_training_hyperparams) ---
    "version": "trainval",
    "length": {"trainval": 28130, "mini": 323},
    # --- Input & class config ---
    "input_shape": (704, 256),      # (W, H)
    "class_names": [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ],
    "map_class_names": ["ped_crossing", "divider", "boundary"],
    "roi_size": (30, 60),
    "num_sample": 20,
    "fut_ts": 12,                   # Future time steps for agents
    "fut_mode": 6,                  # Number of future trajectory modes
    "ego_fut_ts": 6,                # Future time steps for ego
    "ego_fut_mode": 6,              # Number of ego future trajectory modes
    "queue_length": 4,              # Temporal queue length
    # --- Core model dims ---
    "embed_dims": 256,
    "num_groups": 8,
    "num_decoder": 6,
    "num_single_frame_decoder": 1,
    "num_single_frame_decoder_map": 1,
    "num_map_temp_instances": 0,    # Stage 2 overrides this to 33
    # --- Grid mask augmentation ---
    "use_grid_mask": True,
    "grid_mask_use_h": True,
    "grid_mask_use_w": True,
    "grid_mask_rotate": 1,
    "grid_mask_offset": False,
    "grid_mask_ratio": 0.5,
    "grid_mask_mode": 1,
    "grid_mask_prob": 0.7,
    # --- Misc model flags ---
    "use_deformable_func": True,
    "strides": [4, 8, 16, 32],
    "num_depth_layers": 3,
    "depth_loss_weight": 0.2,
    "drop_out": 0.1,
    "temporal": True,
    "temporal_map": True,
    "decouple_attn_motion": True,
    "with_quality_estimation": True,
    "task_config": {"with_det": True, "with_map": True, "with_motion_plan": False},
    # --- Paths ---
    "kmeans_dir": "./sparsedrive_model/data/kmeans_nuscenes",
    "backbone_pretrained": "model_weights/resnet50-19c8e357.pth",
    # --- Backbone ---
    "backbone_with_cp": True,       # Use gradient checkpointing in backbone
    # --- FPN ---
    "fpn_out_channels": 256,
    "fpn_add_extra_convs": "on_output",
    "fpn_in_channels": [256, 512, 1024, 2048],
    # --- Camera / attention ---
    "num_cams": 6,                  # Model default; overridden to 8 for NAVSIM via OPTIMIZER_CONFIG
    "deformable_attn_drop": 0.15,
    "deformable_use_camera_embed": True,
    "deformable_residual_mode": "cat",
    "attention_batch_first": True,
    "ffn_pre_norm": "LN",
    "ffn_act_cfg": "ReLU",
}

# =============================================================================
# SECTION 5: DETECTION HEAD
# =============================================================================
# 3-D object detection head parameters.

DETECTION_HEAD = {
    "det_cls_threshold_to_reg": 0.05,
    "det_dn_loss_weight": 5.0,
    # Ground-truth keys
    "det_gt_cls_key": "gt_labels_3d",
    "det_gt_reg_key": "gt_bboxes_3d",
    "det_gt_id_key": "instance_id",
    "det_with_instance_id": True,
    "det_task_prefix": "det",
    # Anchor / instance settings
    "det_num_anchor": 900,
    "det_num_temp_instances": 600,
    "det_confidence_decay": 0.6,
    "det_feat_grad": False,
    "det_anchor_file": "kmeans_det_900.npy",
    # Keypoint geometry
    "det_keypoint_num_learnable_pts": 6,
    "det_keypoint_fix_scale": [
        [0, 0, 0],
        [0.45, 0, 0],
        [-0.45, 0, 0],
        [0, 0.45, 0],
        [0, -0.45, 0],
        [0, 0, 0.45],
        [0, 0, -0.45],
    ],
    # Encoder dims
    "det_encoder_vel_dims": 3,
    "det_encoder_embed_dims_decoupled": [128, 32, 32, 64],
    "det_encoder_embed_dims_coupled": 256,
    "det_encoder_in_loops": 1,
    "det_encoder_out_loops_decoupled": 4,
    "det_encoder_out_loops_coupled": 2,
    "det_ffn_num_fcs": 2,
    "det_refine_yaw": True,
    # Denoising
    "det_num_dn_groups": 0,
    "det_num_temp_dn_groups": 0,
    "det_dn_noise_scale": [2.0] * 3 + [0.5] * 7,
    "det_max_dn_gt": 32,
    "det_add_neg_dn": True,
    # Loss weights
    "det_target_cls_weight": 2.0,
    "det_target_box_weight": 0.25,
    "det_target_reg_weights": [2.0] * 3 + [0.5] * 3 + [0.0] * 4,
    "det_loss_cls_weight": 2.0,
    "det_loss_reg_weight": 0.25,
    "det_loss_gamma": 2.0,
    "det_loss_alpha": 0.25,
    "det_loss_reg_weights": [2.0] * 3 + [1.0] * 7,
    "det_cls_allow_reverse": ["barrier"],
    "det_cls_wise_reg_weights": {
        "traffic_cone": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    },
}

# =============================================================================
# SECTION 6: MAP HEAD
# =============================================================================
# Online map segmentation head parameters.

MAP_HEAD = {
    "map_cls_threshold_to_reg": 0.05,
    "map_num_anchor": 100,
    "map_confidence_decay": 0.6,
    "map_feat_grad": True,
    "map_anchor_file": "kmeans_map_100.npy",
    # Keypoint geometry
    "map_keypoint_num_learnable_pts": 3,
    "map_keypoint_fix_height": (0, 0.5, -0.5, 1, -1),
    "map_keypoint_ground_height": -1.84023,
    # Loss weights
    "map_loss_cls_weight": 1.0,
    "map_loss_reg_weight": 10.0,
    "map_loss_beta": 0.01,
    "map_reg_weights": [1.0] * 40,
    # Ground-truth keys
    "map_gt_cls_key": "gt_map_labels",
    "map_gt_reg_key": "gt_map_pts",
    "map_gt_id_key": "map_instance_id",
    "map_with_instance_id": False,
    "map_task_prefix": "map",
}

# =============================================================================
# SECTION 7: MOTION & PLANNING HEAD
# =============================================================================
# Agent motion prediction and ego planning head parameters.

MOTION_PLANNING_HEAD = {
    "motion_anchor_file": "kmeans_motion_{fut_mode}.npy",
    "plan_anchor_file": "kmeans_plan_{ego_fut_mode}.npy",
    "motion_tracking_threshold": 0.2,
    "motion_num_decoder_layers": 3,
    "motion_ffn_num_fcs": 2,
    # Loss weights
    "motion_loss_cls_weight": 0.2,
    "motion_loss_reg_weight": 0.2,
    "plan_loss_cls_weight": 0.5,
    "plan_loss_reg_weight": 1.0,
    "plan_loss_status_weight": 1.0,
    "planning_decoder_use_rescore": True,
    # Query counts fed to the planning decoder
    "motion_num_det": 50,
    "motion_num_map": 10,
}

# =============================================================================
# STAGE 2 MODEL OVERRIDES
# =============================================================================
# Model-architecture parameters that differ from Stage 1.
# Training-schedule overrides are already captured in TRAINING_SCHEDULE_STAGE2.

MODEL_STAGE2_OVERRIDES = {
    "num_map_temp_instances": 33,
    "task_config": {"with_det": True, "with_map": True, "with_motion_plan": True},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_runtime_config() -> dict:
    """Return a deep-copy of the runtime/environment configuration."""
    return deepcopy(RUNTIME_CONFIG)


def _build_model_hyperparams(stage2: bool = False) -> dict:
    """Merge all model-architecture sections into one flat dict."""
    hyperparams: dict = {}
    for section in (MODEL_ARCH, DETECTION_HEAD, MAP_HEAD, MOTION_PLANNING_HEAD):
        hyperparams.update(deepcopy(section))
    if stage2:
        hyperparams.update(deepcopy(MODEL_STAGE2_OVERRIDES))
    return hyperparams


def get_stage1_hyperparams() -> dict:
    """Return model hyperparameters for Stage 1 (detection + map pre-training)."""
    return _build_model_hyperparams(stage2=False)


def get_stage2_hyperparams() -> dict:
    """Return model hyperparameters for Stage 2 (full pipeline fine-tuning)."""
    return _build_model_hyperparams(stage2=True)


def get_stage_hyperparams(stage: str, num_cams_override: int = 8) -> tuple[dict, dict]:
    """Return ``(model_hyperparams, training_recipe)`` for the given stage.

    ``training_recipe`` is a flat dict consumed by the runner; it contains all
    optimizer, scheduler, and training-schedule parameters.
    """
    normalized = str(stage).lower().replace("_", "").replace("-", "")
    if normalized in {"1", "stage1"}:
        hyperparams = get_stage1_hyperparams()
        schedule = deepcopy(TRAINING_SCHEDULE_STAGE1)
    elif normalized in {"2", "stage2"}:
        hyperparams = get_stage2_hyperparams()
        schedule = deepcopy(TRAINING_SCHEDULE_STAGE2)
    else:
        raise ValueError(f"Unsupported SparseDrive stage {stage!r}; expected 'stage1' or 'stage2'.")

    recipe = deepcopy(OPTIMIZER_CONFIG)
    recipe.update(schedule)
    recipe["num_cams"] = int(num_cams_override)
    hyperparams["num_cams"] = int(num_cams_override)
    return hyperparams, recipe


def derive_training_hyperparams(hyperparams: dict) -> tuple[int, int]:
    """Derive per-GPU batch size and iterations-per-epoch.

    Kept for backward compatibility.  The runner calculates these inline from
    the actual dataset length; prefer that approach in new code.
    """
    num_gpus = hyperparams.get("num_gpus", 1)
    total_batch_size = hyperparams.get("total_batch_size", 64)
    batch_size = total_batch_size // num_gpus
    num_iters_per_epoch = int(
        hyperparams["length"][hyperparams["version"]] // (num_gpus * batch_size)
    )
    return batch_size, num_iters_per_epoch
