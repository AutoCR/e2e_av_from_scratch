"""Cluster nuScenes ego future trajectories into planning anchors."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from nuscenes_loader import (  # noqa: E402
    DEFAULT_NUSCENES_ROOT,
    DEFAULT_NUSCENES_VERSION,
    future_ego_trajectory,
    iter_scene_sample_sequences,
    load_metadata,
)

GROUP_NAMES = ("straight", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(DEFAULT_NUSCENES_ROOT))
    parser.add_argument("--version", default=DEFAULT_NUSCENES_VERSION)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--future_steps", type=int, default=6)
    parser.add_argument("--turn_thresh", type=float, default=2.0)
    parser.add_argument("--out_dir", default="data/kmeans")
    return parser.parse_args()


def _group_for_trajectory(traj: np.ndarray, turn_thresh: float) -> int:
    final_lateral = float(traj[-1, 1])
    if final_lateral > turn_thresh:
        return 1  # left
    if final_lateral < -turn_thresh:
        return 2  # right
    return 0  # straight


def _collect_grouped_trajectories(metadata, future_steps: int, turn_thresh: float) -> list[list[np.ndarray]]:
    grouped: list[list[np.ndarray]] = [[], [], []]
    skipped_incomplete = 0
    skipped_invalid = 0
    sequences = list(iter_scene_sample_sequences(metadata))

    for sequence in tqdm(sequences, desc="[nuscenes plan] scenes"):
        for sample in sequence.samples:
            traj = future_ego_trajectory(metadata, sample, future_steps=future_steps)
            if traj.shape != (future_steps, 2):
                skipped_incomplete += 1
                continue
            if not np.isfinite(traj).all():
                skipped_invalid += 1
                continue
            grouped[_group_for_trajectory(traj, turn_thresh)].append(traj.astype(np.float64, copy=False))

    print(
        "[nuscenes plan] collected "
        + ", ".join(f"{name}={len(trajs)}" for name, trajs in zip(GROUP_NAMES, grouped))
    )
    if skipped_incomplete:
        print(f"[nuscenes plan] skipped {skipped_incomplete} samples without {future_steps} future steps")
    if skipped_invalid:
        print(f"[nuscenes plan] skipped {skipped_invalid} samples with non-finite trajectories")
    return grouped


def _pad_to_k(trajs: list[np.ndarray], k: int, future_steps: int, group_name: str) -> list[np.ndarray]:
    if len(trajs) >= k:
        return trajs

    padding = k - len(trajs)
    if not trajs:
        print(f"  Warning: group '{group_name}' has no samples — padding with zeros")
        return [np.zeros((future_steps, 2), dtype=np.float64) for _ in range(k)]

    print(f"  Warning: group '{group_name}' has only {len(trajs)} samples — padding by repeating samples")
    repeated = [trajs[i % len(trajs)].copy() for i in range(padding)]
    return trajs + repeated


def _fit_group_clusters(grouped: list[list[np.ndarray]], k: int, future_steps: int) -> np.ndarray:
    clusters_list = []
    for group_name, trajs in zip(GROUP_NAMES, grouped):
        padded = _pad_to_k(trajs, k, future_steps, group_name)
        trajs_arr = np.stack(padded, axis=0).reshape(len(padded), -1)
        cluster = KMeans(n_clusters=k, random_state=42, n_init=10).fit(trajs_arr).cluster_centers_
        cluster = cluster.reshape(k, future_steps, 2)
        clusters_list.append(cluster)
        print(f"  {group_name}: {len(trajs)} samples → {k} clusters")
    return np.stack(clusters_list, axis=0)


def _save_visualization(clusters: np.ndarray, k: int) -> None:
    vis_dir = Path("vis/kmeans")
    vis_dir.mkdir(parents=True, exist_ok=True)
    colors = ("tab:blue", "tab:orange", "tab:green")

    plt.figure(figsize=(6, 6))
    for group_idx, group_name in enumerate(GROUP_NAMES):
        for cluster_idx in range(k):
            label = group_name if cluster_idx == 0 else None
            plt.plot(
                clusters[group_idx, cluster_idx, :, 0],
                clusters[group_idx, cluster_idx, :, 1],
                color=colors[group_idx],
                marker="o",
                markersize=2,
                linewidth=0.8,
                label=label,
            )
    plt.xlabel("x_forward (m)")
    plt.ylabel("y_left (m)")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.savefig(vis_dir / f"nuscenes_plan_{k}.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}.")
    if args.future_steps <= 0:
        raise ValueError(f"--future_steps must be positive, got {args.future_steps}.")
    if args.turn_thresh < 0:
        raise ValueError(f"--turn_thresh must be non-negative, got {args.turn_thresh}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[kmeans_nuscenes_plan] version={args.version}, K={args.k}, "
        f"future_steps={args.future_steps}, turn_thresh={args.turn_thresh}"
    )
    metadata = load_metadata(args.data_path, args.version)
    grouped = _collect_grouped_trajectories(metadata, args.future_steps, args.turn_thresh)
    result = _fit_group_clusters(grouped, args.k, args.future_steps)

    out_path = out_dir / f"kmeans_plan_{args.k}.npy"
    np.save(out_path, result)
    _save_visualization(result, args.k)
    print(f"Saved {result.shape} to {out_path}")
    print(f"Saved visualization to vis/kmeans/nuscenes_plan_{args.k}.png")


if __name__ == "__main__":
    main()
