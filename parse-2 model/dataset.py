import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from hydra.core.global_hydra import GlobalHydra
print(os.getcwd())
os.environ["NUPLAN_MAPS_ROOT"] = os.path.expandvars("$HOME/Code/navsim_wrkspace/dataset/maps")
os.environ["NAVSIM_EXP_ROOT"] = os.path.expandvars("$HOME/Code/navsim_wrkspace/exp")
os.environ["NAVSIM_DEVKIT_ROOT"] = os.path.expandvars("$HOME/Code/navsim_wrkspace/GTRS")
os.environ["OPENSCENE_DATA_ROOT"] = os.path.expandvars("$HOME/Code/navsim_wrkspace/dataset")
os.environ["NAVSIM_TRAJPDM_ROOT"] = os.path.expandvars("$HOME/Code/navsim_wrkspace/dataset/traj_pdm_v2")

SPLIT = "mini"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)
print(cfg)
scene_filter: SceneFilter = instantiate(cfg)

openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}", # data_path
    openscene_data_root / f"sensor_blobs/{SPLIT}", # original_sensor_path
    scene_filter,
    openscene_data_root / "warmup_two_stage/sensor_blobs", # synthetic_sensor_path
    openscene_data_root / "warmup_two_stage/synthetic_scene_pickles", # synthetic_scenes_path
    sensor_config=SensorConfig.build_all_sensors(),
)

import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, scene_loader: SceneLoader, max_len=None, random_sample=False):
        self.scene_loader = scene_loader
        if random_sample:
            self.tokens = np.random.choice(scene_loader.tokens, size=max_len, replace=False)
        else:
            self.max_len = min(max_len, len(scene_loader)) if max_len is not None else len(scene_loader)
            self.tokens = scene_loader.tokens[:self.max_len]

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.max_len

    def __getitem__(self, idx):
        # Retrieve a sample from the dataset at the given index
        token = self.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        num_his = scene.scene_metadata.num_history_frames
        cur_frame = scene.frames[num_his]
        inputs = {}
        trajectory = scene.get_future_trajectory()
        inputs['cam'] = cur_frame.cameras
        
        return inputs, trajectory