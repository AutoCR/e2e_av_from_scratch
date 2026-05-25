import argparse
import os
import sys

import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from nuscenes_loader import (  # noqa: E402
    DEFAULT_NUSCENES_ROOT,
    DEFAULT_NUSCENES_VERSION,
    SPARSEDRIVE_CLASS_NAMES,
    annotations_for_sample,
    future_agent_trajectory,
    iter_scene_sample_sequences,
    load_metadata,
)


STATIC_CLASS_NAMES = {"barrier", "traffic_cone"}
DYNAMIC_CLASS_NAMES = tuple(name for name in SPARSEDRIVE_CLASS_NAMES if name not in STATIC_CLASS_NAMES)
CLASS_TO_INDEX = {name: index for index, name in enumerate(SPARSEDRIVE_CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster nuScenes dynamic-object future trajectories by SparseDrive class."
    )
    parser.add_argument("--data_path", default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--version", default=DEFAULT_NUSCENES_VERSION)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--dis_thresh", type=float, default=55.0)
    parser.add_argument("--future_steps", type=int, default=12)
    parser.add_argument("--out_dir", default="data/kmeans")
    parser.add_argument(
        "--scarcity",
        choices=("repeat", "zero"),
        default="repeat",
        help=(
            "How to fill classes with fewer than K full trajectories. "
            "'repeat' repeats scarce samples; 'zero' keeps scarce samples then zero-pads."
        ),
    )
    return parser.parse_args()


def _validate_args(args):
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}.")
    if args.future_steps <= 0:
        raise ValueError(f"--future_steps must be positive, got {args.future_steps}.")
    if args.dis_thresh <= 0.0:
        raise ValueError(f"--dis_thresh must be positive, got {args.dis_thresh}.")


def _collect_trajectories(metadata, dis_thresh, future_steps):
    sequences = tuple(iter_scene_sample_sequences(metadata))
    total_samples = sum(max(0, len(sequence.samples) - future_steps) for sequence in sequences)
    trajectories = {index: [] for index in range(len(SPARSEDRIVE_CLASS_NAMES))}
    stats = {
        "dynamic_annotations": 0,
        "distance_filtered": 0,
        "missing_future": 0,
        "kept": 0,
    }

    progress = tqdm(total=total_samples, desc="[nuscenes motion] samples")
    try:
        for sequence in sequences:
            usable_samples = sequence.samples[:-future_steps]
            for sample in usable_samples:
                for annotation in annotations_for_sample(metadata, sample, mapped_only=True):
                    class_name = annotation.class_name
                    if class_name not in DYNAMIC_CLASS_NAMES:
                        continue

                    stats["dynamic_annotations"] += 1
                    if float(np.linalg.norm(annotation.center_local[:2])) >= dis_thresh:
                        stats["distance_filtered"] += 1
                        continue

                    trajectory = future_agent_trajectory(
                        metadata,
                        sample,
                        annotation.instance_token,
                        future_steps=future_steps,
                        require_full=True,
                    )
                    if trajectory is None or trajectory.shape != (future_steps, 2):
                        stats["missing_future"] += 1
                        continue

                    class_index = CLASS_TO_INDEX[class_name]
                    trajectories[class_index].append(trajectory.astype(np.float32, copy=False))
                    stats["kept"] += 1
                progress.update(1)
    finally:
        progress.close()

    return trajectories, stats, len(sequences), total_samples


def _fill_scarce_class(trajs, class_name, k, future_steps, scarcity, reason):
    count = len(trajs)
    print(
        f"  Warning: {class_name} has {reason} (need {k}); "
        f"filling with {scarcity!r} scarcity behavior."
    )
    if count == 0:
        return np.zeros((k, future_steps, 2), dtype=np.float32)

    samples = np.stack(trajs, axis=0).astype(np.float32, copy=False)
    if scarcity == "zero":
        filled = np.zeros((k, future_steps, 2), dtype=np.float32)
        filled[:count] = samples
        return filled

    repeat_indices = np.arange(k) % count
    return samples[repeat_indices]


def _cluster_or_fill(trajs, class_name, k, future_steps, scarcity):
    if len(trajs) < k:
        return _fill_scarce_class(
            trajs,
            class_name,
            k,
            future_steps,
            scarcity,
            reason=f"only {len(trajs)} full trajectories",
        )

    trajs_arr = np.stack(trajs, axis=0).reshape(len(trajs), -1)
    unique_trajs_arr = np.unique(trajs_arr, axis=0)
    if len(unique_trajs_arr) < k:
        unique_trajs = unique_trajs_arr.reshape(len(unique_trajs_arr), future_steps, 2)
        return _fill_scarce_class(
            list(unique_trajs),
            class_name,
            k,
            future_steps,
            scarcity,
            reason=f"only {len(unique_trajs_arr)} distinct trajectories among {len(trajs)} full trajectories",
        )

    cluster = KMeans(n_clusters=k, random_state=42, n_init=10).fit(trajs_arr).cluster_centers_
    cluster = cluster.reshape(k, future_steps, 2).astype(np.float32)
    print(f"  {class_name}: {len(trajs)} samples → {k} clusters")
    return cluster


def _save_visualization(cluster, class_name, k):
    os.makedirs("vis/kmeans", exist_ok=True)
    plt.figure(figsize=(4, 4))
    for index in range(k):
        plt.plot(cluster[index, :, 0], cluster[index, :, 1], linewidth=1.0, marker="o", markersize=2)
    plt.title(f"nuScenes {class_name} motion intentions (K={k})")
    plt.xlabel("x forward (m)")
    plt.ylabel("y left (m)")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.savefig(f"vis/kmeans/nuscenes_motion_intention_{class_name}_{k}.png", bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    _validate_args(args)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("vis/kmeans", exist_ok=True)

    print(
        f"[kmeans_nuscenes_motion] version={args.version}, K={args.k}, "
        f"future_steps={args.future_steps}, dis_thresh={args.dis_thresh}"
    )
    print(f"[kmeans_nuscenes_motion] dynamic classes: {', '.join(DYNAMIC_CLASS_NAMES)}")

    metadata = load_metadata(args.data_path, args.version)
    trajectories, stats, scene_count, sample_count = _collect_trajectories(
        metadata, args.dis_thresh, args.future_steps
    )
    print(f"[nuscenes motion] scanned {sample_count} usable samples from {scene_count} scenes")
    print(
        "[nuscenes motion] kept={kept}, dynamic_annotations={dynamic_annotations}, "
        "distance_filtered={distance_filtered}, missing_full_future={missing_future}".format(**stats)
    )

    clusters = []
    for class_index, class_name in enumerate(SPARSEDRIVE_CLASS_NAMES):
        class_cluster = _cluster_or_fill(
            trajectories[class_index],
            class_name,
            args.k,
            args.future_steps,
            args.scarcity,
        )
        clusters.append(class_cluster)
        _save_visualization(class_cluster, class_name, args.k)

    result = np.stack(clusters, axis=0).astype(np.float32)
    out_path = os.path.join(args.out_dir, f"kmeans_motion_{args.k}.npy")
    np.save(out_path, result)
    print(f"Saved {result.shape} to {out_path}")


if __name__ == "__main__":
    main()
