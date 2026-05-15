import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from navsim_loader import load_sequences, get_ego_pose, global_to_local_xy

# Maps driving_command index [STRAIGHT, LEFT, RIGHT, TURN_U] → plan group [STRAIGHT=0, LEFT=1, RIGHT=2]
CMD_TO_GROUP = [0, 1, 2, 0]


def _process_seq(args_tuple):
    seq, future_steps = args_tuple

    # Pre-cache all ego poses for this sequence
    poses = [get_ego_pose(frame) for frame in seq]

    trajs = [[], [], []]  # [straight, left, right]

    for t in range(len(seq) - future_steps):
        current_pose = poses[t]
        cmd = np.asarray(seq[t]["driving_command"], dtype=np.int32)
        group = CMD_TO_GROUP[int(np.argmax(cmd))]

        # Vectorized: batch-extract all future global positions, then transform
        future_global = np.stack([poses[t + k][:2] for k in range(1, future_steps + 1)])  # (future_steps, 2)
        future_local = global_to_local_xy(current_pose, future_global)  # (future_steps, 2)
        trajs[group].append(future_local)

    return trajs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chenran/Code/navsim/dataset")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--future_steps", type=int, default=6)
    parser.add_argument("--out_dir", default="data/kmeans")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count())")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("vis/kmeans", exist_ok=True)

    num_workers = args.workers or cpu_count()
    print(f"[kmeans_plan] split={args.split}, K={args.k}, future_steps={args.future_steps}, workers={num_workers}")

    sequences = load_sequences(args.data_path, args.split)
    tasks = [(seq, args.future_steps) for seq in sequences]

    navi_trajs = [[], [], []]
    with Pool(num_workers) as pool:
        for seq_trajs in tqdm(pool.imap(_process_seq, tasks),
                               total=len(tasks), desc="[plan] sequences"):
            for g in range(3):
                navi_trajs[g].extend(seq_trajs[g])

    plt.figure()
    clusters_list = []
    for group_idx, trajs in enumerate(navi_trajs):
        group_name = ["straight", "left", "right"][group_idx]
        if len(trajs) < args.k:
            print(f"  Warning: group '{group_name}' has only {len(trajs)} samples — padding with zeros")
            padding = args.k - len(trajs)
            trajs += [np.zeros((args.future_steps, 2), dtype=np.float64)] * padding

        trajs_arr = np.stack(trajs, axis=0).reshape(len(trajs), -1)
        cluster = KMeans(n_clusters=args.k, random_state=42).fit(trajs_arr).cluster_centers_
        cluster = cluster.reshape(args.k, args.future_steps, 2)
        clusters_list.append(cluster)

        for j in range(args.k):
            plt.scatter(cluster[j, :, 0], cluster[j, :, 1], s=4)

    plt.savefig(f"vis/kmeans/plan_{args.k}.png", bbox_inches="tight")
    plt.close()

    result = np.stack(clusters_list, axis=0)
    out_path = os.path.join(args.out_dir, f"kmeans_plan_{args.k}.npy")
    np.save(out_path, result)
    print(f"Saved {result.shape} to {out_path}")


if __name__ == "__main__":
    main()
