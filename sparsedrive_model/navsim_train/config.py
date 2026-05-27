"""SparseDrive-on-NAVSIM runner configuration helpers."""

from copy import deepcopy

from sparsedrive_model.configs.sparsedrive_hyperparams import (
    get_stage1_hyperparams,
    get_stage2_hyperparams,
)

RECIPE_STAGE1 = {
    "num_epochs": 100,
    "total_batch_size": 64,
    "ckpt_epoch_interval": 20,
    "eval_epoch_interval": 20,
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

RECIPE_STAGE2 = {
    "num_epochs": 10,
    "total_batch_size": 48,
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

COMMON_RECIPE = {
    "lr": 4e-4,
    "weight_decay": 0.001,
    "backbone_lr_mult": 0.5,
    "grad_clip_max_norm": 25.0,
    "grad_clip_norm_type": 2.0,
    "warmup_iters": 500,
    "warmup_ratio": 1.0 / 3.0,
    "min_lr_ratio": 1e-3,
    "fp16_loss_scale": 32.0,
    "log_interval": 51,
    "num_cams": 8,
}


def get_stage_hyperparams(stage: str, num_cams_override=8) -> tuple[dict, dict]:
    """Return a deepcopy of stage hyperparams plus the pure-PyTorch recipe."""
    normalized = str(stage).lower().replace("_", "").replace("-", "")
    if normalized in {"1", "stage1"}:
        hyperparams = deepcopy(get_stage1_hyperparams())
        recipe = deepcopy(RECIPE_STAGE1)
    elif normalized in {"2", "stage2"}:
        hyperparams = deepcopy(get_stage2_hyperparams())
        recipe = deepcopy(RECIPE_STAGE2)
    else:
        raise ValueError(f"Unsupported SparseDrive stage {stage!r}; expected 'stage1' or 'stage2'.")

    hyperparams["num_cams"] = int(num_cams_override)
    recipe.update(deepcopy(COMMON_RECIPE))
    recipe["num_cams"] = int(num_cams_override)
    return hyperparams, recipe
