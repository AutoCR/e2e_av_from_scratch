import os
print(os.getcwd())
os.environ["NUPLAN_MAPS_ROOT"] = os.path.expandvars("$HOME/Code/navsim/dataset/maps")
os.environ["OPENSCENE_DATA_ROOT"] = os.path.expandvars("$HOME/Code/navsim/dataset")

from pathlib import Path

import hydra
from hydra.utils import instantiate

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from hydra.core.global_hydra import GlobalHydra
import torch.nn as nn
from typing import Any, Callable
from diffusion_planner import StateNormalizer

import torch
from torch import optim
from diffusion_planner import ObservationNormalizer
from tqdm import tqdm

from diffusion_planner import DiffusionPlanner, cfg, diffusion_loss_func
from diffusion_planner_dataset import DiffusionPlannerDataset
from torch.utils.data.dataloader import DataLoader

SPLIT = "mini"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
filter_cfg = hydra.compose(config_name=FILTER)
print(filter_cfg)
scene_filter: SceneFilter = instantiate(filter_cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}", # data_path
    openscene_data_root / f"sensor_blobs/{SPLIT}", # original_sensor_path
    scene_filter,
    openscene_data_root / "warmup_two_stage/sensor_blobs", # synthetic_sensor_path
    openscene_data_root / "warmup_two_stage/synthetic_scene_pickles", # synthetic_scenes_path
    sensor_config=SensorConfig.build_all_sensors(),
)

dataset = DiffusionPlannerDataset(
    scene_loader=scene_loader,
    cfg = cfg,
    max_len = 8
)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

model = DiffusionPlanner(cfg)
optimizer = optim.AdamW([{'params': model.parameters(), 'lr': 0.001}])
device = 'cpu'
observation_normalizer = ObservationNormalizer({
    k: {kk: torch.tensor(vv, dtype=torch.float32) for kk, vv in v.items()}
    for k, v in cfg['observation_normalizer'].items()
})
state_normalizer = StateNormalizer(**cfg['state_normalizer'])
for epoch in range(0, 1):
    with tqdm(data_loader, desc='Training', unit='batch') as data_epoch:
        for token, features, targets in data_epoch:
            print(targets.keys())
            for k, v in features.items():
                v = v.to(device)
            for k, v in targets.items():
                v = v.to(device)
            
            ego_future = targets['ego_future_gt'].to(device)
            neighbors_future = targets['neighbors_future_gt'].to(device)
            mask = targets['neighbor_future_mask']
            neighbors_future[mask] = 0
            inputs = observation_normalizer(features)
            optimizer.zero_grad()
            loss = {}

            loss, _ = diffusion_loss_func(
                model,
                inputs,
                model.sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                state_normalizer,
                loss,
                cfg['diffusion_model_type']
            )
            
            loss['loss'] = loss['neighbor_prediction_loss'] + 0.1 * loss['ego_planning_loss']

            total_loss = loss['loss'].item()

            loss['loss'].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

