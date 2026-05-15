import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from navsim_loader import load_sequences, NuplanMapStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chenran/Code/navsim/dataset")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--maps_root", default=None,
                        help="Path to NuPlan maps root. Falls back to NUPLAN_MAPS_ROOT env var.")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--num_sample", type=int, default=20)
    parser.add_argument("--out_dir", default="data/kmeans")
    return parser.parse_args()


def sample_linestring(geom, num_points):
    from shapely.geometry import LineString
    if geom.is_empty:
        return None
    line = geom if geom.geom_type == "LineString" else LineString(geom.exterior.coords)
    if line.length == 0:
        return None
    dists = np.linspace(0.0, line.length, num_points)
    return np.array([line.interpolate(d).coords[0] for d in dists], dtype=np.float64)


def main():
    args = parse_args()
    maps_root = args.maps_root or os.environ.get("NUPLAN_MAPS_ROOT")
    if maps_root is None:
        raise RuntimeError("Provide --maps_root or set NUPLAN_MAPS_ROOT env var.")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("vis/kmeans", exist_ok=True)

    print(f"[kmeans_map] split={args.split}, K={args.k}")

    # Collect unique map locations — no need to iterate all frames for map data
    sequences = load_sequences(args.data_path, args.split)
    map_locations = sorted({frame["map_location"] for seq in sequences for frame in seq})
    print(f"[map] Found {len(map_locations)} unique map locations: {map_locations}")

    map_store = NuplanMapStore(maps_root)
    mean_pts = []

    for map_loc in tqdm(map_locations, desc="[map] map regions"):
        map_data = map_store._load_map(map_loc)

        for _, baseline in tqdm(map_data["lane_baselines"].iterrows(),
                                total=len(map_data["lane_baselines"]),
                                desc="  lanes", leave=False):
            pts = sample_linestring(baseline.geometry, args.num_sample)
            if pts is not None:
                mean_pts.append(pts.mean(axis=0))

        for _, baseline in tqdm(map_data["connector_baselines"].iterrows(),
                                total=len(map_data["connector_baselines"]),
                                desc="  connectors", leave=False):
            pts = sample_linestring(baseline.geometry, args.num_sample)
            if pts is not None:
                mean_pts.append(pts.mean(axis=0))

    mean_pts = np.stack(mean_pts, axis=0)
    print(f"[map] {len(mean_pts)} lane centers collected. Fitting KMeans(K={args.k})...")

    centers = KMeans(n_clusters=args.k, random_state=42).fit(mean_pts).cluster_centers_

    delta_y = np.linspace(-4, 4, args.num_sample)
    delta_x = np.zeros(args.num_sample)
    delta = np.stack([delta_x, delta_y], axis=-1)
    vecs = centers[:, np.newaxis] + delta[np.newaxis]  # (K, num_sample, 2)

    plt.figure()
    for i in range(args.k):
        plt.plot(vecs[i, :, 0], vecs[i, :, 1], linewidth=0.5, marker="o", markersize=1)
    plt.savefig(f"vis/kmeans/map_anchor_{args.k}.png", bbox_inches="tight")
    plt.close()

    out_path = os.path.join(args.out_dir, f"kmeans_map_{args.k}.npy")
    np.save(out_path, vecs)
    print(f"Saved {vecs.shape} to {out_path}")


if __name__ == "__main__":
    main()
