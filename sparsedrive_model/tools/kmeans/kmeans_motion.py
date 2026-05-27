import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from navsim_loader import load_sequences, get_ego_pose, local_to_global_xy, rotation_matrix

CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
DYNAMIC_CLASSES = {"car", "truck", "construction_vehicle", "bus", "trailer",
                   "motorcycle", "bicycle", "pedestrian"}
VEHICLE_INDICES = [1, 2, 3, 4, 6]  # truck, construction_vehicle, bus, trailer, motorcycle
HEADING_IDX = 6

# NavSim/NuPlan uses consolidated class names; map them to SparseDrive taxonomy
# before checking DYNAMIC_CLASSES / looking up CLASSES index.
_NAVSIM_TO_SPARSEDRIVE = {
    "vehicle": "car",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "generic_object": None,
    "czone_sign": None,
}


def _process_seq(args_tuple):
    seq, dis_thresh, future_steps = args_tuple
    num_classes = len(CLASSES)

    # Pre-cache all ego poses for this sequence (avoids repeated pyquaternion calls)
    poses = [get_ego_pose(frame) for frame in seq]

    intention = {i: [] for i in range(num_classes)}

    for t in range(len(seq) - future_steps):
        current_frame = seq[t]
        current_pose = poses[t]

        boxes = np.asarray(current_frame["anns"]["gt_boxes"], dtype=np.float64)
        names = np.asarray(current_frame["anns"]["gt_names"])
        tokens = list(current_frame["anns"]["track_tokens"])

        if len(boxes) == 0:
            continue

        # Build token → list of global positions across future frames
        token_to_future = {tok: [] for tok in tokens}
        for dt in range(1, future_steps + 1):
            future_frame = seq[t + dt]
            fp = poses[t + dt]
            fut_boxes = np.asarray(future_frame["anns"]["gt_boxes"], dtype=np.float64)
            fut_tokens = list(future_frame["anns"]["track_tokens"])
            for fbox, ftok in zip(fut_boxes, fut_tokens):
                if ftok not in token_to_future:
                    continue
                token_to_future[ftok].append(local_to_global_xy(fp, fbox[:2]))

        for box, name, tok in zip(boxes, names, tokens):
            cls_name = _NAVSIM_TO_SPARSEDRIVE.get(str(name), str(name))
            if cls_name is None or cls_name not in DYNAMIC_CLASSES:
                continue
            cls_idx = CLASSES.index(cls_name)

            future_global = token_to_future[tok]
            if len(future_global) < future_steps:
                continue

            dist = np.linalg.norm(box[:2])
            if dist >= dis_thresh:
                continue

            agent_global = local_to_global_xy(current_pose, box[:2])
            agent_heading_global = current_pose[2] + box[HEADING_IDX]
            R = rotation_matrix(-agent_heading_global)

            # Vectorized: stack all future positions, subtract origin, rotate in one matmul
            future_arr = np.stack(future_global[:future_steps], axis=0)  # (future_steps, 2)
            traj_agent = (future_arr - agent_global) @ R.T               # (future_steps, 2)
            intention[cls_idx].append(traj_agent)

    return intention


def _merge_intentions(results, num_classes):
    merged = {i: [] for i in range(num_classes)}
    for result in results:
        for i in range(num_classes):
            merged[i].extend(result[i])
    return merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chenran/Code/navsim/dataset")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--dis_thresh", type=float, default=55.0)
    parser.add_argument("--future_steps", type=int, default=12)
    parser.add_argument("--out_dir", default="data/kmeans")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count())")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("vis/kmeans", exist_ok=True)

    num_workers = args.workers or cpu_count()
    print(f"[kmeans_motion] split={args.split}, K={args.k}, future_steps={args.future_steps}, workers={num_workers}")

    sequences = load_sequences(args.data_path, args.split)
    tasks = [(seq, args.dis_thresh, args.future_steps) for seq in sequences]

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(_process_seq, tasks),
                           total=len(tasks), desc="[motion] sequences"):
            results.append(result)

    intention = _merge_intentions(results, len(CLASSES))

    result = np.zeros((len(CLASSES), args.k, args.future_steps, 2), dtype=np.float64)
    for i, cls_name in enumerate(CLASSES):
        trajs = intention[i]
        if len(trajs) < args.k:
            if i not in VEHICLE_INDICES:
                print(f"  Skipping {cls_name}: only {len(trajs)} samples (need {args.k})")
            continue
        trajs_arr = np.stack(trajs, axis=0).reshape(len(trajs), -1)
        cluster = KMeans(n_clusters=args.k, random_state=42).fit(trajs_arr).cluster_centers_
        cluster = cluster.reshape(args.k, args.future_steps, 2)
        result[i] = cluster

        plt.figure()
        for j in range(args.k):
            plt.scatter(cluster[j, :, 0], cluster[j, :, 1], s=4)
        plt.savefig(f"vis/kmeans/motion_intention_{cls_name}_{args.k}.png", bbox_inches="tight")
        plt.close()
        print(f"  {cls_name}: {len(trajs)} samples → {args.k} clusters")

    car_clusters = result[0]
    if car_clusters.any():
        for vehicle_idx in VEHICLE_INDICES:
            if not result[vehicle_idx].any():
                result[vehicle_idx] = car_clusters
                print(f"  {CLASSES[vehicle_idx]}: copied from car (no NavSim samples)")

    out_path = os.path.join(args.out_dir, f"kmeans_motion_{args.k}.npy")
    np.save(out_path, result)
    print(f"Saved {result.shape} to {out_path}")


if __name__ == "__main__":
    main()
