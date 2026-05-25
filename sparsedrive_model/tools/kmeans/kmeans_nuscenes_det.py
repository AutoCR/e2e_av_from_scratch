from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from nuscenes_loader import (  # noqa: E402
    DEFAULT_NUSCENES_ROOT,
    DEFAULT_NUSCENES_VERSION,
    NuScenesKMeansMetadata,
    annotations_for_sample,
    iter_scene_sample_sequences,
    load_metadata,
)

_WORKER_METADATA: NuScenesKMeansMetadata | None = None


def _init_worker(metadata: NuScenesKMeansMetadata) -> None:
    global _WORKER_METADATA
    _WORKER_METADATA = metadata


def _num_workers(workers: int | None) -> int:
    if workers is None:
        return 1
    if workers <= 0:
        return cpu_count()
    return workers


def _sample_token(sample: Mapping[str, object]) -> str:
    token = sample.get("token")
    if token is None:
        raise KeyError("sample record is missing required field 'token'.")
    return str(token)


def _process_sequence(task: tuple[tuple[str, ...], float]) -> tuple[np.ndarray, Counter[str]]:
    if _WORKER_METADATA is None:
        raise RuntimeError("nuScenes metadata was not initialized in worker.")

    sample_tokens, dis_thresh = task
    centers: list[np.ndarray] = []
    class_counts: Counter[str] = Counter()
    for sample_token in sample_tokens:
        sample = _WORKER_METADATA.get_sample(sample_token)
        for annotation in annotations_for_sample(_WORKER_METADATA, sample, mapped_only=True):
            center = np.asarray(annotation.center_local, dtype=np.float64)
            if np.linalg.norm(center[:2]) >= dis_thresh:
                continue
            centers.append(center)
            if annotation.class_name is not None:
                class_counts[annotation.class_name] += 1

    if not centers:
        return np.empty((0, 3), dtype=np.float64), class_counts
    return np.stack(centers, axis=0), class_counts


def collect_centers(
    metadata: NuScenesKMeansMetadata,
    dis_thresh: float,
    workers: int = 1,
) -> tuple[np.ndarray, Counter[str]]:
    sequences = tuple(iter_scene_sample_sequences(metadata))
    tasks = [
        (tuple(_sample_token(sample) for sample in sequence.samples), dis_thresh)
        for sequence in sequences
    ]

    centers: list[np.ndarray] = []
    class_counts: Counter[str] = Counter()
    if workers == 1:
        _init_worker(metadata)
        iterator = map(_process_sequence, tasks)
        for batch, counts in tqdm(iterator, total=len(tasks), desc="[nuscenes_det] scenes"):
            if len(batch):
                centers.append(batch)
            class_counts.update(counts)
    else:
        with Pool(workers, initializer=_init_worker, initargs=(metadata,)) as pool:
            for batch, counts in tqdm(
                pool.imap(_process_sequence, tasks),
                total=len(tasks),
                desc="[nuscenes_det] scenes",
            ):
                if len(batch):
                    centers.append(batch)
                class_counts.update(counts)

    if not centers:
        return np.empty((0, 3), dtype=np.float64), class_counts
    return np.concatenate(centers, axis=0), class_counts


def _validate_args(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}.")
    if args.dis_thresh <= 0.0:
        raise ValueError(f"--dis_thresh must be positive, got {args.dis_thresh}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster nuScenes GT box centers into SparseDrive detection anchors."
    )
    parser.add_argument("--data_path", default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--version", default=DEFAULT_NUSCENES_VERSION)
    parser.add_argument("--k", type=int, default=900)
    parser.add_argument("--dis_thresh", type=float, default=55.0)
    parser.add_argument("--out_dir", default="data/kmeans")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use 0 or a negative value for cpu_count().",
    )
    return parser.parse_args()


def _mini_batch_kmeans(centers: np.ndarray, k: int) -> np.ndarray:
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=min(10000, max(k, len(centers))),
        n_init=3,
        verbose=1,
    ).fit(centers).cluster_centers_


def _save_visualization(cluster: np.ndarray, k: int) -> str:
    os.makedirs("vis/kmeans", exist_ok=True)
    vis_path = f"vis/kmeans/nuscenes_det_anchor_{k}.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(cluster[:, 0], cluster[:, 1], s=2)
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(f"nuScenes detection anchors (K={k})")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(vis_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return vis_path


def main() -> None:
    args = parse_args()
    _validate_args(args)
    workers = _num_workers(args.workers)
    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"[kmeans_nuscenes_det] version={args.version}, K={args.k}, "
        f"dis_thresh={args.dis_thresh}, workers={workers}"
    )
    metadata = load_metadata(args.data_path, args.version)
    centers, class_counts = collect_centers(metadata, args.dis_thresh, workers=workers)
    print(f"[nuscenes_det] {len(centers)} mapped GT box centers collected.")
    if class_counts:
        counts_text = ", ".join(f"{name}={count}" for name, count in sorted(class_counts.items()))
        print(f"[nuscenes_det] class counts: {counts_text}")

    if len(centers) < args.k:
        raise ValueError(
            f"Only {len(centers)} mapped nuScenes GT box centers within "
            f"{args.dis_thresh:g}m were collected, but --k={args.k}. "
            "For v1.0-mini, lower --k (for example --k 4 or another value below "
            "the collected count), increase --dis_thresh, or use a larger nuScenes version."
        )

    print(f"[nuscenes_det] Fitting MiniBatchKMeans(K={args.k})...")
    cluster = _mini_batch_kmeans(centers, args.k)
    extras = np.tile(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float64), (args.k, 1))
    result = np.concatenate([cluster, extras], axis=1)

    out_path = os.path.join(args.out_dir, f"kmeans_det_{args.k}.npy")
    np.save(out_path, result)
    vis_path = _save_visualization(cluster, args.k)
    print(f"Saved {result.shape} to {out_path}")
    print(f"Saved visualization to {vis_path}")


if __name__ == "__main__":
    main()
