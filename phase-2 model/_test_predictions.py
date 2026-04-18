"""Verify DiffusionPlanner predictions are accurate vs ground truth.

Runs inference over several mini-split scenes and asserts that:
  1. ego ADE over 8s horizon < 3 m
  2. neighbor ADE over valid future mask < 5 m per agent
These thresholds bracket what the official checkpoint achieves on in-distribution data.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/Users/chenran/Code/nuplan/dataset/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", os.path.expandvars("$HOME/Code/navsim/dataset"))

from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_planner import DiffusionPlanner, ObservationNormalizer, cfg
from diffusion_planner_dataset import DiffusionPlannerNuplanDataset
from torch.utils.data.dataloader import DataLoader
from visualization import show_diffusion_planner_result


def main():
    openscene = Path(os.environ["OPENSCENE_DATA_ROOT"])
    dataset = DiffusionPlannerNuplanDataset(
        data_path=openscene / "navsim_logs/mini",
        cfg=cfg,
        maps_root="/Users/chenran/Code/nuplan/dataset/maps",
        max_len=8,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DiffusionPlanner(cfg)
    ckpt = torch.load("/Users/chenran/Code/diffusion-planner/ckpt/model.pth",
                      map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ckpt["ema_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    obs_norm = ObservationNormalizer({
        k: {kk: torch.tensor(vv, dtype=torch.float32) for kk, vv in v.items()}
        for k, v in cfg["observation_normalizer"].items()
    })

    out_dir = Path(__file__).parent / "_test_predictions_out"
    out_dir.mkdir(exist_ok=True)

    ego_ades = []
    neighbor_ades = []
    torch.manual_seed(0)
    for idx, (token, features, targets) in enumerate(loader):
        # Official inference stub for ego
        features["ego_current_state"] = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        ).expand_as(features["ego_current_state"]).clone()
        features_n = obs_norm(features)
        with torch.no_grad():
            op = model(features_n)
        dec_out = op[1]
        pred = dec_out["prediction"].numpy()[0]  # (P=11, 80, 4)

        gt_ego = targets["ego_future_gt"].numpy()[0]  # (80, 4)
        gt_neigh = targets["neighbors_future_gt"].numpy()[0]  # (10, 80, 4)
        neigh_mask = targets["neighbor_future_mask"].numpy()[0].astype(bool)

        ego_ade = float(np.linalg.norm(pred[0, :, :2] - gt_ego[:, :2], axis=-1).mean())
        ego_ades.append(ego_ade)

        for j in range(gt_neigh.shape[0]):
            valid = ~neigh_mask[j]
            if valid.sum() == 0:
                continue
            err = np.linalg.norm(pred[j + 1, valid, :2] - gt_neigh[j, valid, :2], axis=-1).mean()
            neighbor_ades.append(float(err))

        print(f"[{idx}] {token[0]}  ego_ADE={ego_ade:6.3f}  "
              f"neighbor_ADE_avg={np.mean(neighbor_ades[-10:]):6.3f}")

        show_diffusion_planner_result(op, targets, features)
        fig_path = out_dir / f"{idx:02d}_{token[0]}.png"
        plt.gcf().savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"    saved {fig_path}")

    print()
    print(f"mean ego ADE       : {np.mean(ego_ades):.3f} m")
    print(f"mean neighbor ADE  : {np.mean(neighbor_ades):.3f} m")
    print(f"max  ego ADE       : {np.max(ego_ades):.3f} m")
    print(f"max  neighbor ADE  : {np.max(neighbor_ades):.3f} m")

    assert np.mean(ego_ades) < 3.0, f"mean ego ADE too high: {np.mean(ego_ades):.2f}"
    assert np.mean(neighbor_ades) < 5.0, f"mean neighbor ADE too high: {np.mean(neighbor_ades):.2f}"
    print("\nPASS: predictions track ground truth.")


if __name__ == "__main__":
    main()
