import os
print(os.getcwd())
os.environ["NUPLAN_MAPS_ROOT"] = os.path.expandvars("/prediction_database/nuplan/dataset/maps")
os.environ["OPENSCENE_DATA_ROOT"] = os.path.expandvars("/prediction_database/navsim")
os.environ["NAVSIM_EXP_ROOT"] = os.path.expandvars("/home/pnc/Code/e2e_av_from_scratch/exp")

import csv
import random
from datetime import datetime
from pathlib import Path

import yaml

import hydra
from hydra.utils import instantiate

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import torch.nn as nn
from typing import Any, Callable
from diffusion_planner import StateNormalizer

import torch
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR
from timm.utils.model_ema import ModelEma
from diffusion_planner import ObservationNormalizer
from tqdm import tqdm

from diffusion_planner import DiffusionPlanner, cfg, diffusion_loss_func
from diffusion_planner_dataset import DiffusionPlannerDataset
from torch.utils.data.dataloader import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_annealing_warmup_restarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    assert epoch >= warm_up_epoch
    warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    fixed = MultiplicativeLR(optimizer, lr_lambda=lambda _: 1.0)
    return SequentialLR(optimizer, schedulers=[warmup, fixed], milestones=[warm_up_epoch])


SEED = 3407
NUM_EPOCHS = 1  # sanity-check; raw repo default is 500
WARM_UP_EPOCHS = 5
LEARNING_RATE = 5e-4
ALPHA_PLANNING_LOSS = 1.0
EMA_DECAY = 0.999
SAVE_EVERY_N_EPOCHS = 1  # raw repo default is 20
LOG_EVERY_N_ITERS = 1
BATCH_SIZE = 8

set_seed(SEED)

exp_root = Path(os.getenv("NAVSIM_EXP_ROOT", Path(__file__).resolve().parent / "runs"))
run_dir = exp_root / f"diffusion_planner/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
ckpt_dir = run_dir / "ckpt"
ckpt_dir.mkdir(parents=True, exist_ok=True)
iter_log_path = run_dir / "iter_loss.csv"
print(f"Logging to {run_dir}")


def save_checkpoint(path: Path, epoch: int, model, ema, optimizer, scheduler, train_loss: float) -> None:
    torch.save(
        {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedule': scheduler.state_dict(),
            'loss': train_loss,
        },
        path,
    )

# Data splits.
#
# Train/val come from the same `trainval/` blob, partitioned by log name using
# default_train_val_test_log_split.yaml — same pattern as
# navsim/planning/script/run_training_aug.py (build_datasets). Test uses the
# dedicated `test/` data dir with no log-name override.
TRAINVAL_DATA_SPLIT = "trainval"
TEST_DATA_SPLIT = "test"
LOG_SPLIT_YAML = Path(__file__).resolve().parents[1] / (
    "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
)
FILTER = "all_scenes"

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
filter_cfg = hydra.compose(config_name=FILTER)
print(filter_cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

log_split = yaml.safe_load(LOG_SPLIT_YAML.read_text())
TRAIN_LOGS = log_split["train_logs"]
VAL_LOGS = log_split["val_logs"]
print(f"Log split: {len(TRAIN_LOGS)} train logs, {len(VAL_LOGS)} val logs")


def build_loader(data_split: str, log_names, batch_size: int, shuffle: bool) -> DataLoader:
    scene_filter: SceneFilter = instantiate(filter_cfg)
    if log_names is not None:
        scene_filter.log_names = list(log_names)
    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{data_split}",
        openscene_data_root / f"sensor_blobs/{data_split}",
        scene_filter,
        openscene_data_root / "warmup_two_stage/sensor_blobs",
        openscene_data_root / "warmup_two_stage/synthetic_scene_pickles",
        sensor_config=SensorConfig.build_all_sensors(),
    )
    dataset = DiffusionPlannerDataset(scene_loader=scene_loader, cfg=cfg)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


train_loader = build_loader(TRAINVAL_DATA_SPLIT, TRAIN_LOGS, batch_size=BATCH_SIZE, shuffle=True)
val_loader = build_loader(TRAINVAL_DATA_SPLIT, VAL_LOGS, batch_size=BATCH_SIZE, shuffle=False)
test_loader = build_loader(TEST_DATA_SPLIT, None, batch_size=BATCH_SIZE, shuffle=False)

device = 'cuda'
model = DiffusionPlanner(cfg).to(device)
optimizer = optim.AdamW([{'params': model.parameters(), 'lr': LEARNING_RATE}])

scheduler_epochs = max(NUM_EPOCHS, WARM_UP_EPOCHS)
scheduler = cosine_annealing_warmup_restarts(optimizer, scheduler_epochs, WARM_UP_EPOCHS)

model_ema = ModelEma(model, decay=EMA_DECAY, device=device)

observation_normalizer = ObservationNormalizer({
    k: {kk: torch.tensor(vv, dtype=torch.float32) for kk, vv in v.items()}
    for k, v in cfg['observation_normalizer'].items()
})
state_normalizer = StateNormalizer(**cfg['state_normalizer'])


@torch.no_grad()
def evaluate(eval_model: nn.Module, loader: DataLoader, desc: str = 'Eval') -> dict:
    """Run a forward-only diffusion-loss pass on `loader` and return mean losses.

    Stays in train mode because the decoder only emits `'score'` under
    `self.training` (diffusion_planner.py:563). `@torch.no_grad()` handles
    autograd; train mode keeps the loss path alive. For an inference metric
    (ADE/FDE via DPM sampling) write a separate function that runs under .eval().
    """
    eval_model.train()
    totals = {'loss': 0.0, 'ego_planning_loss': 0.0, 'neighbor_prediction_loss': 0.0}
    n_batches = 0
    with tqdm(loader, desc=desc, unit='batch') as data_epoch:
        for token, features, targets in data_epoch:
            features = {k: v.to(device) for k, v in features.items()}
            targets = {k: v.to(device) for k, v in targets.items()}

            ego_future = targets['ego_future_gt'].to(device)
            neighbors_future = targets['neighbors_future_gt'].to(device)
            mask = targets['neighbor_future_mask']
            neighbors_future[mask] = 0
            inputs = observation_normalizer(features)

            loss = {}
            loss, _ = diffusion_loss_func(
                eval_model,
                inputs,
                eval_model.sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                state_normalizer,
                loss,
                cfg['diffusion_model_type'],
            )
            loss['loss'] = loss['neighbor_prediction_loss'] + ALPHA_PLANNING_LOSS * loss['ego_planning_loss']

            totals['loss'] += loss['loss'].item()
            totals['ego_planning_loss'] += loss['ego_planning_loss'].item()
            totals['neighbor_prediction_loss'] += loss['neighbor_prediction_loss'].item()
            n_batches += 1
            data_epoch.set_postfix(loss=f"{totals['loss'] / max(n_batches, 1):.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# TODO: port StatePerturbation data augmentation (requires dataset to emit
# heading-as-angle rather than cos/sin, or a rewrite of the augmenter).
epoch_log_path = run_dir / "epoch_loss.csv"
global_step = 0
epoch_mean_loss = float('nan')
with iter_log_path.open('w', newline='') as iter_f, epoch_log_path.open('w', newline='') as epoch_f:
    iter_writer = csv.writer(iter_f)
    iter_writer.writerow(['step', 'epoch', 'iter', 'lr', 'loss', 'ego_planning_loss', 'neighbor_prediction_loss'])
    epoch_writer = csv.writer(epoch_f)
    epoch_writer.writerow([
        'epoch', 'lr',
        'train_loss', 'train_ego_planning_loss', 'train_neighbor_prediction_loss',
        'val_loss', 'val_ego_planning_loss', 'val_neighbor_prediction_loss',
    ])

    for epoch in range(0, NUM_EPOCHS):
        model.train()
        epoch_losses = []
        epoch_ego_losses = []
        epoch_neighbor_losses = []
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch') as data_epoch:
            for iter_idx, (token, features, targets) in enumerate(data_epoch):
                features = {k: v.to(device) for k, v in features.items()}
                targets = {k: v.to(device) for k, v in targets.items()}

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

                loss['loss'] = loss['neighbor_prediction_loss'] + ALPHA_PLANNING_LOSS * loss['ego_planning_loss']

                total_loss = loss['loss'].item()
                ego_loss = loss['ego_planning_loss'].item()
                neighbor_loss = loss['neighbor_prediction_loss'].item()
                epoch_losses.append(total_loss)
                epoch_ego_losses.append(ego_loss)
                epoch_neighbor_losses.append(neighbor_loss)

                loss['loss'].backward()

                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                model_ema.update(model)

                if global_step % LOG_EVERY_N_ITERS == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    iter_writer.writerow([global_step, epoch, iter_idx, current_lr, total_loss, ego_loss, neighbor_loss])
                    iter_f.flush()

                data_epoch.set_postfix(loss=f'{total_loss:.4f}', ego=f'{ego_loss:.4f}', nbr=f'{neighbor_loss:.4f}')
                global_step += 1

        epoch_mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        epoch_mean_ego = sum(epoch_ego_losses) / max(len(epoch_ego_losses), 1)
        epoch_mean_neighbor = sum(epoch_neighbor_losses) / max(len(epoch_neighbor_losses), 1)

        # Val pass against EMA weights (what the raw repo actually evaluates).
        val_metrics = evaluate(model_ema.ema, val_loader, desc=f'Val {epoch + 1}/{NUM_EPOCHS}')

        current_lr = optimizer.param_groups[0]['lr']
        epoch_writer.writerow([
            epoch + 1, current_lr,
            epoch_mean_loss, epoch_mean_ego, epoch_mean_neighbor,
            val_metrics['loss'], val_metrics['ego_planning_loss'], val_metrics['neighbor_prediction_loss'],
        ])
        epoch_f.flush()

        print(
            f"Epoch {epoch + 1} | train loss {epoch_mean_loss:.4f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"(ego {val_metrics['ego_planning_loss']:.4f}, nbr {val_metrics['neighbor_prediction_loss']:.4f})"
        )

        scheduler.step()

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch + 1}_trainloss_{epoch_mean_loss:.4f}.pth"
            save_checkpoint(ckpt_path, epoch, model, model_ema.ema, optimizer, scheduler, epoch_mean_loss)
            save_checkpoint(ckpt_dir / "latest.pth", epoch, model, model_ema.ema, optimizer, scheduler, epoch_mean_loss)
            print(f"Saved checkpoint to {ckpt_path}")


# Final test pass against EMA weights.
test_metrics = evaluate(model_ema.ema, test_loader, desc='Test')
print(f"Final test metrics (EMA): {test_metrics}")
test_log_path = run_dir / "test_metrics.csv"
with test_log_path.open('w', newline='') as test_f:
    test_writer = csv.writer(test_f)
    test_writer.writerow(['loss', 'ego_planning_loss', 'neighbor_prediction_loss'])
    test_writer.writerow([
        test_metrics['loss'],
        test_metrics['ego_planning_loss'],
        test_metrics['neighbor_prediction_loss'],
    ])
print(f"Wrote test metrics to {test_log_path}")


def plot_losses(run_dir: Path, test_metrics: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    iter_steps, iter_loss, iter_ego, iter_nbr = [], [], [], []
    with (run_dir / 'iter_loss.csv').open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            iter_steps.append(int(row['step']))
            iter_loss.append(float(row['loss']))
            iter_ego.append(float(row['ego_planning_loss']))
            iter_nbr.append(float(row['neighbor_prediction_loss']))

    epochs, train_loss, val_loss = [], [], []
    train_ego, train_nbr = [], []
    val_ego, val_nbr = [], []
    with (run_dir / 'epoch_loss.csv').open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            train_ego.append(float(row['train_ego_planning_loss']))
            train_nbr.append(float(row['train_neighbor_prediction_loss']))
            val_ego.append(float(row['val_ego_planning_loss']))
            val_nbr.append(float(row['val_neighbor_prediction_loss']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(iter_steps, iter_loss, label='total', alpha=0.7)
    axes[0].plot(iter_steps, iter_ego, label='ego', alpha=0.7)
    axes[0].plot(iter_steps, iter_nbr, label='neighbor', alpha=0.7)
    axes[0].set_xlabel('global step')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Train loss (per iteration)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_loss, 'o-', label='train')
    axes[1].plot(epochs, val_loss, 's-', label='val')
    axes[1].axhline(test_metrics['loss'], color='r', linestyle='--', alpha=0.6, label=f"test={test_metrics['loss']:.4f}")
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('total loss')
    axes[1].set_title('Total loss (train/val/test)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, train_ego, 'o-', label='train ego', color='C0')
    axes[2].plot(epochs, val_ego, 's-', label='val ego', color='C0', alpha=0.5)
    axes[2].plot(epochs, train_nbr, 'o-', label='train nbr', color='C1')
    axes[2].plot(epochs, val_nbr, 's-', label='val nbr', color='C1', alpha=0.5)
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('loss')
    axes[2].set_title('Ego planning vs. neighbor prediction')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_path = run_dir / 'loss_curves.png'
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote loss plot to {out_path}")


plot_losses(run_dir, test_metrics)

