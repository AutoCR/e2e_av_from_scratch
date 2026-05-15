from . import sparsedrive_small_stage1 as _stage1
from .sparsedrive_small_stage1 import *  # noqa: F401,F403

total_batch_size = 48
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 10
checkpoint_epoch_interval = 10
task_config = dict(with_det=True, with_map=True, with_motion_plan=True)
num_map_temp_instances = 33
load_from = "ckpt/sparsedrive_stage1.pth"


def build():
    _stage1.total_batch_size = total_batch_size
    _stage1.batch_size = batch_size
    _stage1.num_iters_per_epoch = num_iters_per_epoch
    _stage1.num_epochs = num_epochs
    _stage1.checkpoint_epoch_interval = checkpoint_epoch_interval
    _stage1.task_config = task_config
    _stage1.num_map_temp_instances = num_map_temp_instances
    return _stage1.build()
