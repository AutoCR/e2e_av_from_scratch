"""
BEVFusion configuration module.

Provides hyperparameters for camera+LiDAR 3D object detection.
"""

from .bevfusion_hyperparams import get_fusion_hyperparams


def build_bevfusion() -> "BEVFusion":
    """
    Build a BEVFusion model with default hyperparameters.

    Returns:
        BEVFusion: Configured BEVFusion model instance.

    Note:
        Imports BEVFusion lazily to avoid circular imports.
    """
    from bevfusion_model.bevfusion import BEVFusion

    return BEVFusion(get_fusion_hyperparams())


__all__ = ["get_fusion_hyperparams", "build_bevfusion"]
