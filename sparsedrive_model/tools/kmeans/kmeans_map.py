"""Compute map-anchor kmeans in **ego-relative / SparseDrive LIDAR_TOP** coordinates.

Previous version collected global UTM lane centers and clustered them in UTM space,
producing anchor values of magnitude ~1e5–1e7.  Those values overflow fp16 inside the
SparsePoint3DEncoder → NaN loss.

This version:
1. Samples a subset of ego poses from the split.
2. For each pose, queries map baselines within the SparseDrive ROI.
3. Transforms every polyline from global to ego-relative LIDAR_TOP frame:
       x_lidar = -y_ego,  y_lidar = x_ego
   (matches the convention used by vectorize_map_for_frame).
4. Clusters the ego-relative polyline centres → values are bounded by roi_size.
"""

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from navsim_loader import load_sequences, NuplanMapStore, get_ego_pose, rotation_matrix

# SparseDrive ROI (metres, ego-relative, matching roi_size = (30, 60))
ROI_X = 30.0   # half-width: left/right
ROI_Y = 60.0   # half-depth: forward/back


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chenran/Code/navsim/dataset")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--maps_root", default=None,
                        help="Path to NuPlan maps root. Falls back to NUPLAN_MAPS_ROOT env var.")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--num_sample", type=int, default=20)
    parser.add_argument("--out_dir", default="data/kmeans")
    parser.add_argument("--max_frames", type=int, default=5000,
                        help="Max frames to sample ego poses from (for speed).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_linestring(geom, num_points):
    from shapely.geometry import LineString
    if geom is None or geom.is_empty:
        return None
    line = geom if geom.geom_type == "LineString" else LineString(geom.exterior.coords)
    if line.length == 0:
        return None
    dists = np.linspace(0.0, line.length, num_points)
    return np.array([line.interpolate(d).coords[0] for d in dists], dtype=np.float64)


def global_to_lidar_top(pts_global_xy: np.ndarray, ego_t: np.ndarray, ego_R: np.ndarray) -> np.ndarray:
    """Transform global XY → SparseDrive LIDAR_TOP (x_right = -y_ego, y_fwd = x_ego)."""
    # Step 1: global → ego frame  (ego_R is R_ego2global)
    ego_xy = (pts_global_xy - ego_t[:2]) @ ego_R[:2, :2]   # row-vec × R^T equivalent
    # Step 2: ego → LIDAR_TOP convention used by SparseDrive / vectorize_map_for_frame
    return np.stack([-ego_xy[:, 1], ego_xy[:, 0]], axis=1)


def main():
    args = parse_args()
    maps_root = args.maps_root or os.environ.get("NUPLAN_MAPS_ROOT")
    if maps_root is None:
        raise RuntimeError("Provide --maps_root or set NUPLAN_MAPS_ROOT env var.")

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("vis/kmeans", exist_ok=True)

    print(f"[kmeans_map] split={args.split}, K={args.k}, ego-relative LIDAR_TOP frame")

    sequences = load_sequences(args.data_path, args.split)
    all_frames = [f for seq in sequences for f in seq]
    if args.max_frames < len(all_frames):
        all_frames = random.sample(all_frames, args.max_frames)
    print(f"[map] Sampled {len(all_frames)} ego poses from {len(sequences)} sequences")

    map_store = NuplanMapStore(maps_root)
    ego_rel_mean_pts = []

    for frame in tqdm(all_frames, desc="[map] collecting ego-relative line centres"):
        pose = get_ego_pose(frame)   # [x_global, y_global, yaw]
        ego_t_3d = np.array([pose[0], pose[1], 0.0], dtype=np.float64)
        yaw = pose[2]
        R_ego2global = rotation_matrix(yaw)   # 2×2

        map_loc = frame.get("map_location")
        if map_loc is None:
            continue
        map_data = map_store._load_map(map_loc)

        for col in ("lane_baselines", "connector_baselines"):
            if col not in map_data:
                continue
            for _, row in map_data[col].iterrows():
                pts = sample_linestring(row.geometry, args.num_sample)
                if pts is None:
                    continue
                pts_xy = pts[:, :2]
                lidar_xy = global_to_lidar_top(pts_xy, ego_t_3d, R_ego2global)
                cx, cy = lidar_xy.mean(axis=0)
                # Keep only segments whose centre falls inside the ROI
                if abs(cx) <= ROI_X and abs(cy) <= ROI_Y:
                    ego_rel_mean_pts.append(np.array([cx, cy], dtype=np.float32))

    if len(ego_rel_mean_pts) < args.k:
        raise RuntimeError(
            f"Only {len(ego_rel_mean_pts)} valid ego-relative line centres found "
            f"(need at least k={args.k}).  Try a larger --max_frames or --split."
        )

    mean_pts = np.stack(ego_rel_mean_pts, axis=0)
    print(f"[map] {len(mean_pts)} ego-relative lane centres. Fitting KMeans(K={args.k})...")
    print(f"[map] centre range: x [{mean_pts[:,0].min():.1f}, {mean_pts[:,0].max():.1f}]  "
          f"y [{mean_pts[:,1].min():.1f}, {mean_pts[:,1].max():.1f}]")

    centers = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto").fit(mean_pts).cluster_centers_

    # Build anchor vectors: straight-ahead lines of length 8 m centred at each cluster
    delta_y = np.linspace(-4.0, 4.0, args.num_sample)
    delta_x = np.zeros(args.num_sample)
    delta = np.stack([delta_x, delta_y], axis=-1)
    vecs = centers[:, np.newaxis] + delta[np.newaxis]  # (K, num_sample, 2)

    print(f"[map] anchor range: {vecs.min():.2f} – {vecs.max():.2f}  (should be ≈ ROI bounds)")

    plt.figure()
    for i in range(args.k):
        plt.plot(vecs[i, :, 0], vecs[i, :, 1], linewidth=0.5, marker="o", markersize=1)
    plt.title("Map anchors (ego-relative LIDAR_TOP)")
    plt.xlabel("x_right (m)")
    plt.ylabel("y_fwd (m)")
    plt.savefig(f"vis/kmeans/map_anchor_{args.k}.png", bbox_inches="tight")
    plt.close()

    out_path = os.path.join(args.out_dir, f"kmeans_map_{args.k}.npy")
    np.save(out_path, vecs)
    print(f"Saved {vecs.shape} to {out_path}")


if __name__ == "__main__":
    main()
