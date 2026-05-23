import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
        self.fp16_enabled = False

    def init_weights(self):
        return None


ModuleList = nn.ModuleList
Sequential = nn.Sequential


class CheckpointCompatibleModule(BaseModule):
    """Placeholder module that preserves raw checkpoint tensors.

    It lets the standalone package ingest UniAD checkpoints before every
    OpenMMLab module has a full native forward implementation.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}
        self._checkpoint_tensors = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} is checkpoint-compatible but its "
            "standalone forward path has not been ported yet."
        )

    def load_checkpoint_tensors(self, prefix, state_dict):
        marker = prefix + "." if prefix else ""
        self._checkpoint_tensors = {
            key[len(marker):]: value.detach().clone()
            for key, value in state_dict.items()
            if key.startswith(marker)
        }


class BaseBBoxCoder:
    pass


class BaseAssigner:
    pass


class AssignResult:
    def __init__(self, num_gts, gt_inds, max_overlaps=None, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
