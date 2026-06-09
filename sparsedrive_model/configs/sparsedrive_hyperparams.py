from copy import deepcopy

# =============================================================================
# SECTION 1: RUNTIME & ENVIRONMENT
# =============================================================================
# Edit this section to configure your local environment before running.

RUNTIME_CONFIG = {
    "stage": "stage1",              # "stage1" (det+map pre-train) or "stage2" (full fine-tune)
    # Split config format (per split):
    #   str form:  "mini"  — load all scenes from that directory, no filtering.
    #   dict form:
    #     "dir"            — directory name under navsim_logs/  (required)
    #     "log_names_yaml" — YAML path (relative to repo root) for log-name list (optional)
    #     "log_names_key"  — key in log_names_yaml; default "log_names"
    #     "tokens_yaml"    — YAML path for scene-token whitelist (optional, may be same file)
    #     "tokens_key"     — key in tokens_yaml; default "tokens"
    # Omitting or setting a yaml field to None disables that filter entirely.
    "splits": {
        "train": {
            "dir": "trainval",
            # log names: use train_logs from the official log split (978 of 1192 navtrain logs)
            "log_names_yaml": "navsim/planning/script/config/training/default_train_val_test_log_split.yaml",
            "log_names_key": "train_logs",
            # tokens: navtrain scene whitelist (covers both train and val navtrain logs)
            "tokens_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
            "tokens_key": "tokens",
        },
        "val": {
            "dir": "trainval",
            # log names: use val_logs from the official log split (214 of 1192 navtrain logs)
            "log_names_yaml": "navsim/planning/script/config/training/default_train_val_test_log_split.yaml",
            "log_names_key": "val_logs",
            # tokens: same navtrain whitelist; log_names filter ensures only val-log scenes are returned
            "tokens_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
            "tokens_key": "tokens",
        },
        "test": {
            "dir": "test",
            "log_names_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
            "log_names_key": "log_names",
            "tokens_yaml": "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml",
            "tokens_key": "tokens",
        },
    },
    "openscene_data_root": "/prediction_database/navsim",
    "nuplan_maps_root": "/prediction_database/nuplan/dataset/maps",
    "output_dir": "sparsedrive_model/outputs/train_navsim",
    "load_from": None,              # Path to a checkpoint to init weights (stage2: "ckpt/sparsedrive_stage1.pth")
    "resume_from": None,            # Path to a full checkpoint to resume interrupted training
    "seed": 0,
    "num_workers": 4,
    "device": "auto",                # "auto", "cuda", "mps", or "cpu"
    "quick_smoke": False,
}

# =============================================================================
# SECTION 2: TRAINING SCHEDULE
# =============================================================================
# Epochs, batch size, checkpoint/eval cadence and evaluation modes per stage.

TRAINING_SCHEDULE_STAGE1 = {
    "num_epochs": 100,
    # Per-GPU micro-batch. Launch with `torchrun --nproc_per_node=6`: this gives
    # micro_global_batch = 12*6 = 72, and the runner auto-derives grad_accum_steps
    # = round(64/72) = 1 -> effective batch 72, accum=1 (no accumulation, full
    # throughput). 72 is ~12% hotter than the official batch-64 recipe, but that
    # is acceptable now that backbone_lr_mult=0.25 halves the backbone LR: the
    # earlier divergence ran effective batch 96 AND backbone lr 2e-4, and we have
    # removed the larger risk factor. (6 GPUs cannot divide 64 cleanly -- 6 carries
    # a factor of 3 -- so exact 64 is not reachable with all 6 cards; 72 is the
    # closest full-hardware option.)
    "total_batch_size": 12,
    "num_gpus": 6,                  # Reference GPU count (used only by derive_training_hyperparams)
    "ckpt_epoch_interval": 2,      # Save a checkpoint every N epochs
    "eval_epoch_interval": 20,      # Run validation every N epochs
    "eval_mode": {
        "with_det": True,
        "with_tracking": True,
        "with_map": False,
        "with_motion": False,
        "with_planning": False,
        "tracking_threshold": 0.2,
        "motion_threshhold": 0.2,
    },
}

TRAINING_SCHEDULE_STAGE2 = {
    "num_epochs": 10,
    "total_batch_size": 4,
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
    # LR is paired with ``effective_batch_size`` below. The official SparseDrive
    # recipe trains at batch 64 with lr=4e-4; we reproduce that stable regime via
    # gradient accumulation instead of a literal batch of 64 (see runner).
    "lr": 4e-4,
    "weight_decay": 0.001,
    # Backbone LR multiplier. The pretrained ResNet50 backbone (with gradient
    # checkpointing, 3 cameras @ 704x256) diverged REPEATEDLY around epoch 3 at
    # mult=0.5 (backbone lr=2e-4): img_backbone.conv1/layer1 grads blew up to
    # O(5e4). Halving to 0.25 (backbone lr=1e-4) is the standard remedy for a
    # diverging pretrained backbone -- it slows the layers that explode without
    # touching the heads' lr. Raise back toward 0.5 only after a stable long run.
    "backbone_lr_mult": 0.25,       # LR multiplier for backbone parameters
    "grad_clip_max_norm": 1.0,
    "grad_clip_norm_type": 2.0,
    # Gradient-explosion guard. clip_grad_norm_ bounds step *magnitude* but not
    # *direction*: a pathological batch with an exploding pre-clip norm still
    # produces a clipped-but-garbage-direction update that, repeated, walks the
    # weights into a divergent regime (activations overflow -> softmax/matmul
    # NaNs ->all-NaN grads). When the pre-clip grad norm exceeds this threshold
    # the optimizer step is skipped entirely instead of applied.
    #
    # IMPORTANT: this is an ABSOLUTE floor in grad-norm units, independent of
    # grad_clip_max_norm. Set it explicitly. A previous run left it None, which
    # the runner derived as grad_clip_max_norm * 1000 = 1.0 * 1000 = 1e3 (the
    # stale comment claimed 25000, from when grad_clip_max_norm was 25). A 1e3
    # floor is FAR too tight: clip_grad_norm already bounds the step magnitude to
    # max_norm=1.0, so a pre-clip spike of ~1e3 is fully recoverable, yet it was
    # being skipped -- and a string of such skips tripped the 200-skip abort. The
    # backbone genuinely diverges at ~5e4, so 25000 cleanly separates "large but
    # clippable" from "real explosion". Healthy post-warmup norms are O(1)-O(100).
    "grad_skip_norm": 25000.0,
    # Stall guard. The grad-explosion guard above skips a corrupted step, but if
    # the model has walked into a divergent regime EVERY subsequent step explodes
    # and is skipped -> training is frozen yet keeps burning compute (a real run
    # skipped ~99k consecutive steps for ~1.5 days). Abort once this many
    # optimizer steps are skipped in a row; a single successful step resets the
    # counter, so transient one-off spikes never trip it. None disables the guard.
    # With log_interval=5 a value of 200 catches a true stall within ~1000 iters.
    "grad_skip_abort_after": 200,
    # --- Gradient accumulation ---
    # The dataloader yields micro-batches of ``total_batch_size``; the runner
    # accumulates enough of them to reach ``effective_batch_size`` before each
    # optimizer step. This restores the optimization regime the model was tuned
    # for (effective batch 64) without needing GPU memory for a true batch of 64,
    # and fixes the small-batch training instability (lr too hot for batch 4).
    "effective_batch_size": 64,
    "grad_accum_steps": None,       # None -> auto-derive from effective_batch_size; set an int to override
    "warmup_iters": 1000,           # In optimizer-step units. Raised from 500 to
                                    # ease the early backbone ramp after repeated
                                    # epoch-3 divergences (see backbone_lr_mult).
    "warmup_ratio": 1.0 / 3.0,
    "min_lr_ratio": 1e-3,           # Final LR = lr * min_lr_ratio
    "log_interval": 5,             # Print/TensorBoard log every N optimizer steps
}

# =============================================================================
# NAVSIM CAMERA CONFIGURATION
# =============================================================================
# SINGLE source of truth for which NAVSIM cameras, and therefore how many,
# the NAVSIM training pipeline uses. num_cams is derived as len(CAMERA_ORDER).
# To train with 4 cameras, edit this list to a 4-name subset of the 8 valid
# NAVSIM cameras: CAM_F0, CAM_L0, CAM_L1, CAM_R0, CAM_R1, CAM_L2, CAM_R2,
# CAM_B0. The default keeps all 8 cameras, preserving current behavior.
CAMERA_ORDER = (
    "CAM_F0",
    "CAM_L0",
    # "CAM_L1",
    "CAM_R0",
    # "CAM_R1",
    # "CAM_L2",
    # "CAM_R2",
    # "CAM_B0",
)

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
    "task_config": {"with_det": True, "with_map": False, "with_motion_plan": False},
    # --- Paths ---
    "kmeans_dir": "./sparsedrive_model/data/kmeans",
    "backbone_pretrained": "model_weights/resnet50/resnet50-19c8e357.pth",
    # --- Backbone ---
    "backbone_with_cp": True,       # Use gradient checkpointing in backbone
    # --- FPN ---
    "fpn_out_channels": 256,
    "fpn_add_extra_convs": "on_output",
    "fpn_in_channels": [256, 512, 1024, 2048],
    # --- Camera / attention ---
    "num_cams": 6,                  # Base/nuScenes default; the NAVSIM training path overrides this from CAMERA_ORDER via get_stage_hyperparams().
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
    # The detection anchor is a kmeans prior. Leaving it trainable lets it drift
    # over many iterations until projected keypoints cluster against the depth
    # clamp (z>=0.1) and the 1/z^2 backward produces O(1e18) grads that overflow
    # the grad-norm reduction -> the step is skipped every iteration. Freezing it
    # (the standard SparseDrive setting) keeps the anchor at its kmeans prior and
    # removes that drift-driven explosion. Set True only if you intend to learn
    # the anchor and have verified it stays stable.
    "det_anchor_grad": False,
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
    # Freeze the map anchor (kmeans prior) for the same reason as the detection
    # anchor: a trainable anchor drifts until projected keypoints hit the depth
    # clamp and the 1/z^2 backward produces grads large enough to overflow the
    # grad-norm reduction. Relevant in Stage 2 (with_map=True). See
    # [det_anchor_grad] above. Set True only to learn the anchor deliberately.
    "map_anchor_grad": False,
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


def get_camera_order() -> tuple[str, ...]:
    """Return the configured NAVSIM camera order."""
    return tuple(CAMERA_ORDER)


def get_num_cams() -> int:
    """Return the number of configured NAVSIM cameras."""
    return len(CAMERA_ORDER)


def get_stage_hyperparams(stage: str) -> tuple[dict, dict]:
    """Return ``(model_hyperparams, training_recipe)`` for the given stage.

    ``training_recipe`` is a flat dict consumed by the runner; it contains all
    optimizer, scheduler, and training-schedule parameters. num_cams is derived
    from CAMERA_ORDER.
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
    n_cams = get_num_cams()
    recipe["num_cams"] = n_cams
    hyperparams["num_cams"] = n_cams
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
