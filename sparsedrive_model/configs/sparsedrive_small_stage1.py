from sparsedrive import SparseDrive

from .sparsedrive_hyperparams import (
    derive_training_hyperparams,
    get_stage1_hyperparams,
)


hyperparams = get_stage1_hyperparams()
globals().update(hyperparams)
batch_size, num_iters_per_epoch = derive_training_hyperparams(hyperparams)
num_classes = len(class_names)
num_map_classes = len(map_class_names)
num_levels = len(strides)


def build():
    return SparseDrive(get_stage1_hyperparams())
