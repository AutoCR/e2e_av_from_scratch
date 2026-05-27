from sparsedrive_model.sparsedrive import SparseDrive

from .sparsedrive_hyperparams import get_stage1_hyperparams, get_stage2_hyperparams


def build_stage1():
    return SparseDrive(get_stage1_hyperparams())


def build_stage2():
    return SparseDrive(get_stage2_hyperparams())


__all__ = ["build_stage1", "build_stage2"]
