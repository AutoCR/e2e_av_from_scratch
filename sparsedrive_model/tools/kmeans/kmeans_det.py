import argparse
import os
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from navsim_loader import load_sequences


def _process_seq(args_tuple):
    seq, dis_thresh = args_tuple
    centers = []
    for frame in seq:
        boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64)
        if len(boxes) == 0:
            continue
        xyz = boxes[:, :3]
        dist = np.linalg.norm(xyz[:, :2], axis=1)
        keep = xyz[dist < dis_thresh]
        if len(keep):
            centers.append(keep)
    return centers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chenran/Code/navsim/dataset")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--k", type=int, default=900)
    parser.add_argument("--dis_thresh", type=float, default=55.0)
    parser.add_argument("--out_dir", default="data/kmeans")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count())")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("vis/kmeans", exist_ok=True)

    num_workers = args.workers or cpu_count()
    print(f"[kmeans_det] split={args.split}, K={args.k}, workers={num_workers}")

    sequences = load_sequences(args.data_path, args.split)
    tasks = [(seq, args.dis_thresh) for seq in sequences]

    centers = []
    with Pool(num_workers) as pool:
        for batch in tqdm(pool.imap(_process_seq, tasks),
                          total=len(tasks), desc="[det] sequences"):
            centers.extend(batch)

    centers = np.concatenate(centers, axis=0)
    print(f"[det] {len(centers)} box centers collected. Fitting MiniBatchKMeans(K={args.k})...")

    cluster = MiniBatchKMeans(
        n_clusters=args.k, random_state=42, batch_size=10000, verbose=1
    ).fit(centers).cluster_centers_

    plt.figure()
    plt.scatter(cluster[:, 0], cluster[:, 1], s=2)
    plt.savefig(f"vis/kmeans/det_anchor_{args.k}.png", bbox_inches="tight")
    plt.close()

    extras = np.tile(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float64), (args.k, 1))
    result = np.concatenate([cluster, extras], axis=1)
    out_path = os.path.join(args.out_dir, f"kmeans_det_{args.k}.npy")
    np.save(out_path, result)
    print(f"Saved {result.shape} to {out_path}")


if __name__ == "__main__":
    main()
