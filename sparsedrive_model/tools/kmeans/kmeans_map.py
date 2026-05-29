"""Compute map-anchor kmeans in **ego-relative / SparseDrive LIDAR_TOP** coordinates.

Previous version collected global UTM lane centers and clustered them in UTM space,
producing anchor values of magnitude ~1e5–1e7.  Those large values cause numerical
overflow inside the SparsePoint3DEncoder → NaN loss.

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

# Bounding-box radius used for spatial pre-filtering (covers any ROI orientation)
_SPATIAL_RADIUS = (ROI_X ** 2 + ROI_Y ** 2) ** 0.5   # ≈ 67 m


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
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel worker processes (default: 1 = sequential).")
    return parser.parse_args()


def sample_linestring_fast(geom, num_points: int):
    """Sample *num_points* evenly spaced along *geom* using numpy interpolation.

    Avoids Shapely's per-point ``interpolate`` calls; extracts raw coords once
    and uses ``np.interp`` — roughly 10–20× faster per line.
    """
    if geom is None or geom.is_empty:
        return None
    # Extract raw coordinates as a numpy array (works for LineString; fall back for Polygon)
    if geom.geom_type == "LineString":
        coords_xy = np.array(geom.coords, dtype=np.float64)[:, :2]
    else:
        coords_xy = np.array(geom.exterior.coords, dtype=np.float64)[:, :2]

    if len(coords_xy) < 2:
        return None

    diffs = np.diff(coords_xy, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.empty(len(seg_len) + 1, dtype=np.float64)
    cumlen[0] = 0.0
    np.cumsum(seg_len, out=cumlen[1:])
    total_len = cumlen[-1]
    if total_len == 0.0:
        return None

    dists = np.linspace(0.0, total_len, num_points)
    xs = np.interp(dists, cumlen, coords_xy[:, 0])
    ys = np.interp(dists, cumlen, coords_xy[:, 1])
    return np.stack([xs, ys], axis=1)


def global_to_lidar_top(pts_global_xy: np.ndarray, ego_t: np.ndarray, ego_R: np.ndarray) -> np.ndarray:
    """Transform global XY → SparseDrive LIDAR_TOP (x_right = -y_ego, y_fwd = x_ego)."""
    # Step 1: global → ego frame  (ego_R is R_ego2global)
    ego_xy = (pts_global_xy - ego_t[:2]) @ ego_R[:2, :2]   # row-vec × R^T equivalent
    # Step 2: ego → LIDAR_TOP convention used by SparseDrive / vectorize_map_for_frame
    return np.stack([-ego_xy[:, 1], ego_xy[:, 0]], axis=1)


def _collect_centers_for_frame(frame: dict, map_store: NuplanMapStore, num_sample: int) -> list:
    """Return a list of (cx, cy) float32 ego-relative lane centres for one frame."""
    pose = get_ego_pose(frame)   # [x_global, y_global, yaw]
    ego_t_3d = np.array([pose[0], pose[1], 0.0], dtype=np.float64)
    yaw = pose[2]
    R_ego2global = rotation_matrix(yaw)   # 2×2

    map_loc = frame.get("map_location")
    if map_loc is None:
        return []
    map_data = map_store._load_map(map_loc)

    # Spatial pre-filter: bounding box around ego in global UTM coordinates
    ex, ey = pose[0], pose[1]
    r = _SPATIAL_RADIUS
    minx, maxx = ex - r, ex + r
    miny, maxy = ey - r, ey + r

    # Collect sampled mean points from all candidate lanes
    sampled_pts = []   # list of (num_sample, 2) arrays
    for col in ("lane_baselines", "connector_baselines"):
        if col not in map_data:
            continue
        # Spatial index slice — reduces ~10k rows to ~50–100
        candidates = map_data[col].cx[minx:maxx, miny:maxy]
        for row in candidates.itertuples(index=False):
            pts = sample_linestring_fast(row.geometry, num_sample)
            if pts is not None:
                sampled_pts.append(pts)

    if not sampled_pts:
        return []

    # Batch-transform all lane centers in one vectorised call
    all_means = np.stack([p.mean(axis=0) for p in sampled_pts], axis=0)  # (N, 2)
    lidar_means = global_to_lidar_top(all_means, ego_t_3d, R_ego2global)  # (N, 2)

    # ROI filter
    mask = (np.abs(lidar_means[:, 0]) <= ROI_X) & (np.abs(lidar_means[:, 1]) <= ROI_Y)
    valid = lidar_means[mask].astype(np.float32)
    return [valid[i] for i in range(len(valid))]


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

    if args.n_jobs == 1:
        for frame in tqdm(all_frames, desc="[map] collecting ego-relative line centres"):
            ego_rel_mean_pts.extend(_collect_centers_for_frame(frame, map_store, args.num_sample))
    else:
        import concurrent.futures, functools
        # Each worker needs its own map store (file handles are not fork-safe)
        def _worker(frame):
            ms = NuplanMapStore(maps_root)
            return _collect_centers_for_frame(frame, ms, args.num_sample)

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            for centers in tqdm(
                pool.map(_worker, all_frames, chunksize=max(1, len(all_frames) // (args.n_jobs * 4))),
                total=len(all_frames),
                desc="[map] collecting ego-relative line centres",
            ):
                ego_rel_mean_pts.extend(centers)

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
