"""Standalone UniAD package."""

from .configs.uniad_base_e2e_hparams import UNIAD_BASE_E2E_HPARAMS
from .core.checkpoint import load_checkpoint


def build_uniad(*args, **kwargs):
    from .factory import build_uniad as _build_uniad

    return _build_uniad(*args, **kwargs)

__all__ = ["UNIAD_BASE_E2E_HPARAMS", "build_uniad", "load_checkpoint"]
