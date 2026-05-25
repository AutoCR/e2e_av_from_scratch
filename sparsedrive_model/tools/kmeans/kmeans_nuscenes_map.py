from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover - only used when sklearn is unavailable.
    KMeans = None

sys.path.insert(0, os.path.dirname(__file__))
from nuscenes_loader import (  # noqa: E402
    DEFAULT_NUSCENES_ROOT,
    DEFAULT_NUSCENES_VERSION,
    NuScenesKMeansMetadata,
    NuScenesMapPolyline,
    global_xy_to_sample_local_xy,
    interpolate_polyline,
    iter_scene_sample_sequences,
    load_metadata,
    local_xy_to_raw_lidar_xy,
    map_centerlines_for_sample,
    map_lines_for_sample,
    sample_lidar_xyz_to_global_xyz,
)


CENTERLINE_LAYERS = ("lane", "lane_connector")
FALLBACK_LINE_LAYERS = ("lane_divider", "road_divider", "traffic_light")
ROI_EPS = 1e-6


@dataclass(frozen=True)
class MapPrimitiveCatalog:
    centerlines: tuple[NuScenesMapPolyline, ...]
    centerline_bboxes: np.ndarray
    fallback_lines: tuple[NuScenesMapPolyline, ...]
    fallback_line_bboxes: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster nuScenes map vectors as ego-local SparseDrive anchors. "
            "Output vectors are raw LIDAR_TOP [x_right, y_forward] meters to match "
            "SparseDrive's map ROI contract roi_size=(width, length) and temporal transforms."
        )
    )
    parser.add_argument("--data_path", default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--version", default=DEFAULT_NUSCENES_VERSION)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--num_sample", type=int, default=20)
    parser.add_argument(
        "--roi_width",
        type=float,
        default=30.0,
        help="Local ROI width in meters along y_left. Default matches roi_size[0].",
    )
    parser.add_argument(
        "--roi_length",
        type=float,
        default=60.0,
        help="Local ROI length in meters along x_forward. Default matches roi_size[1].",
    )
    parser.add_argument(
        "--dedupe_precision",
        type=float,
        default=0.05,
        help="Quantization step in meters for de-duplicating sampled local vectors. Use 0 to disable.",
    )
    parser.add_argument(
        "--max_vectors_per_sample",
        type=int,
        default=0,
        help="Optional cap on kept vectors per sample after de-duplication. Use 0 for no cap.",
    )
    parser.add_argument("--out_dir", default="data/kmeans")
    return parser.parse_args()


def _valid_polyline(points_xy: np.ndarray) -> bool:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        return False
    if not np.isfinite(points).all():
        return False
    return bool(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > 1e-6)


def _valid_sampled_vector(points_xy: np.ndarray, num_sample: int) -> bool:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape != (num_sample, 2) or not np.isfinite(points).all():
        return False
    if num_sample < 2:
        return True
    return bool(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > 1e-6)


def _valid_primitives_with_bboxes(
    primitives: tuple[NuScenesMapPolyline, ...],
) -> tuple[tuple[NuScenesMapPolyline, ...], np.ndarray]:
    valid: list[NuScenesMapPolyline] = []
    bboxes: list[tuple[float, float, float, float]] = []
    for primitive in primitives:
        points = np.asarray(primitive.points_xy, dtype=np.float64)
        if not _valid_polyline(points):
            continue
        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        valid.append(primitive)
        bboxes.append((float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])))
    if not bboxes:
        return tuple(valid), np.empty((0, 4), dtype=np.float64)
    return tuple(valid), np.asarray(bboxes, dtype=np.float64)


def _catalog_for_sample(metadata: NuScenesKMeansMetadata, sample) -> MapPrimitiveCatalog:
    centerlines, centerline_bboxes = _valid_primitives_with_bboxes(
        map_centerlines_for_sample(metadata, sample, layer_names=CENTERLINE_LAYERS, local=False)
    )
    fallback_lines, fallback_line_bboxes = _valid_primitives_with_bboxes(
        map_lines_for_sample(metadata, sample, layer_names=FALLBACK_LINE_LAYERS, local=False)
    )
    return MapPrimitiveCatalog(
        centerlines=centerlines,
        centerline_bboxes=centerline_bboxes,
        fallback_lines=fallback_lines,
        fallback_line_bboxes=fallback_line_bboxes,
    )


def _roi_bounds(roi_width: float, roi_length: float) -> tuple[float, float, float, float]:
    half_length = roi_length / 2.0
    half_width = roi_width / 2.0
    return -half_length, half_length, -half_width, half_width


def _local_bbox_intersects_roi(points_xy: np.ndarray, roi_width: float, roi_length: float) -> bool:
    if points_xy.size == 0:
        return False
    min_forward, max_forward, min_left, max_left = _roi_bounds(roi_width, roi_length)
    return (
        float(np.max(points_xy[:, 0])) >= min_forward - ROI_EPS
        and float(np.min(points_xy[:, 0])) <= max_forward + ROI_EPS
        and float(np.max(points_xy[:, 1])) >= min_left - ROI_EPS
        and float(np.min(points_xy[:, 1])) <= max_left + ROI_EPS
    )


def _clip_segment_to_roi(
    start: np.ndarray,
    end: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray | None:
    min_forward, max_forward, min_left, max_left = bounds
    delta = end - start
    p_values = (-delta[0], delta[0], -delta[1], delta[1])
    q_values = (start[0] - min_forward, max_forward - start[0], start[1] - min_left, max_left - start[1])
    t0 = 0.0
    t1 = 1.0

    for p_value, q_value in zip(p_values, q_values):
        if abs(float(p_value)) <= ROI_EPS:
            if q_value < -ROI_EPS:
                return None
            continue
        ratio = float(q_value / p_value)
        if p_value < 0.0:
            if ratio > t1 + ROI_EPS:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0 - ROI_EPS:
                return None
            t1 = min(t1, ratio)

    if t0 > t1 + ROI_EPS:
        return None

    clipped = np.stack([start + t0 * delta, start + t1 * delta], axis=0).astype(np.float64, copy=False)
    clipped[:, 0] = np.clip(clipped[:, 0], min_forward, max_forward)
    clipped[:, 1] = np.clip(clipped[:, 1], min_left, max_left)
    if float(np.linalg.norm(clipped[1] - clipped[0])) <= 1e-6:
        return None
    return clipped


def _clip_polyline_to_roi(points_xy: np.ndarray, roi_width: float, roi_length: float) -> tuple[np.ndarray, ...]:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 2 or not _local_bbox_intersects_roi(points, roi_width, roi_length):
        return ()

    bounds = _roi_bounds(roi_width, roi_length)
    fragments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        clipped = _clip_segment_to_roi(start, end, bounds)
        if clipped is None:
            if len(current) >= 2:
                fragments.append(np.asarray(current, dtype=np.float64))
            current = []
            continue

        clipped_start, clipped_end = clipped
        if not current:
            current = [clipped_start, clipped_end]
        elif float(np.linalg.norm(current[-1] - clipped_start)) <= 1e-5:
            if float(np.linalg.norm(current[-1] - clipped_end)) > 1e-6:
                current.append(clipped_end)
        else:
            if len(current) >= 2:
                fragments.append(np.asarray(current, dtype=np.float64))
            current = [clipped_start, clipped_end]

    if len(current) >= 2:
        fragments.append(np.asarray(current, dtype=np.float64))

    return tuple(fragment for fragment in fragments if _valid_polyline(fragment))


def _sample_roi_global_bbox(
    metadata: NuScenesKMeansMetadata,
    sample,
    roi_width: float,
    roi_length: float,
) -> tuple[float, float, float, float]:
    min_forward, max_forward, min_left, max_left = _roi_bounds(roi_width, roi_length)
    local_corners = np.asarray(
        [
            [min_forward, min_left],
            [min_forward, max_left],
            [max_forward, min_left],
            [max_forward, max_left],
        ],
        dtype=np.float64,
    )
    raw_lidar_xy = local_xy_to_raw_lidar_xy(local_corners)
    raw_lidar_xyz = np.column_stack([raw_lidar_xy, np.zeros(raw_lidar_xy.shape[0], dtype=np.float64)])
    global_xy = sample_lidar_xyz_to_global_xyz(metadata, sample, raw_lidar_xyz)[:, :2]
    min_xy = np.min(global_xy, axis=0)
    max_xy = np.max(global_xy, axis=0)
    return float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])


def _bbox_intersect_indices(bboxes: np.ndarray, query_bbox: tuple[float, float, float, float]) -> np.ndarray:
    if bboxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    min_x, min_y, max_x, max_y = query_bbox
    mask = (
        (bboxes[:, 2] >= min_x - ROI_EPS)
        & (bboxes[:, 0] <= max_x + ROI_EPS)
        & (bboxes[:, 3] >= min_y - ROI_EPS)
        & (bboxes[:, 1] <= max_y + ROI_EPS)
    )
    return np.flatnonzero(mask)


def _sample_vectors_from_candidates(
    metadata: NuScenesKMeansMetadata,
    sample,
    primitives: tuple[NuScenesMapPolyline, ...],
    candidate_indices: np.ndarray,
    *,
    roi_width: float,
    roi_length: float,
    num_sample: int,
) -> list[np.ndarray]:
    sampled_vectors: list[np.ndarray] = []
    for primitive_index in candidate_indices:
        primitive = primitives[int(primitive_index)]
        local_points = global_xy_to_sample_local_xy(metadata, sample, primitive.points_xy)
        for fragment in _clip_polyline_to_roi(local_points, roi_width, roi_length):
            sampled = interpolate_polyline(fragment, num_sample).astype(np.float64, copy=False)
            if _valid_sampled_vector(sampled, num_sample):
                sampled_vectors.append(_local_xy_to_sparsedrive_map_xy(sampled))
    return sampled_vectors


def _local_xy_to_sparsedrive_map_xy(local_xy: np.ndarray) -> np.ndarray:
    """Convert display-local [x_forward, y_left] to SparseDrive raw map [x_right, y_forward]."""

    points = np.asarray(local_xy, dtype=np.float64)
    converted = np.empty_like(points, dtype=np.float64)
    converted[..., 0] = -points[..., 1]
    converted[..., 1] = points[..., 0]
    return converted


def _sparsedrive_map_xy_to_local_xy(map_xy: np.ndarray) -> np.ndarray:
    """Convert SparseDrive raw map [x_right, y_forward] to display-local [x_forward, y_left]."""

    points = np.asarray(map_xy, dtype=np.float64)
    converted = np.empty_like(points, dtype=np.float64)
    converted[..., 0] = points[..., 1]
    converted[..., 1] = -points[..., 0]
    return converted


def _dedupe_key(vector: np.ndarray, precision: float) -> bytes | None:
    if precision <= 0.0:
        return None
    quantized = np.rint(np.asarray(vector, dtype=np.float64) / precision).astype(np.int32)
    return quantized.tobytes()


def _append_deduped_vectors(
    source_vectors: list[np.ndarray],
    destination: list[np.ndarray],
    seen_keys: set[bytes],
    *,
    dedupe_precision: float,
    max_vectors_per_sample: int,
) -> int:
    kept = 0
    for vector in source_vectors:
        key = _dedupe_key(vector, dedupe_precision)
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)

        destination.append(vector.astype(np.float64, copy=False))
        kept += 1
        if max_vectors_per_sample > 0 and kept >= max_vectors_per_sample:
            break
    return kept


def _collect_sampled_vectors(
    data_path: str,
    version: str,
    num_sample: int,
    *,
    roi_width: float,
    roi_length: float,
    dedupe_precision: float,
    max_vectors_per_sample: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    metadata = load_metadata(data_path, version)
    sequences = tuple(iter_scene_sample_sequences(metadata))
    if not sequences:
        raise RuntimeError(f"No scenes were found in nuScenes metadata {data_path}/{version}.")

    catalogs: dict[str, MapPrimitiveCatalog] = {}
    vectors: list[np.ndarray] = []
    counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    seen_keys: set[bytes] = set()

    for sequence in sequences:
        if not sequence.samples:
            continue

        map_name = sequence.map_name
        if map_name is None:
            stats["samples_without_map"] += len(sequence.samples)
            continue

        if map_name not in catalogs:
            catalogs[map_name] = _catalog_for_sample(metadata, sequence.samples[0])

        catalog = catalogs[map_name]
        for sample in sequence.samples:
            stats["samples"] += 1
            query_bbox = _sample_roi_global_bbox(metadata, sample, roi_width, roi_length)

            centerline_indices = _bbox_intersect_indices(catalog.centerline_bboxes, query_bbox)
            source_name = "centerline"
            source_vectors = _sample_vectors_from_candidates(
                metadata,
                sample,
                catalog.centerlines,
                centerline_indices,
                roi_width=roi_width,
                roi_length=roi_length,
                num_sample=num_sample,
            )

            if not source_vectors:
                fallback_indices = _bbox_intersect_indices(catalog.fallback_line_bboxes, query_bbox)
                source_name = "fallback_line"
                source_vectors = _sample_vectors_from_candidates(
                    metadata,
                    sample,
                    catalog.fallback_lines,
                    fallback_indices,
                    roi_width=roi_width,
                    roi_length=roi_length,
                    num_sample=num_sample,
                )

            if not source_vectors:
                stats["samples_without_vectors"] += 1
                continue

            kept = _append_deduped_vectors(
                source_vectors,
                vectors,
                seen_keys,
                dedupe_precision=dedupe_precision,
                max_vectors_per_sample=max_vectors_per_sample,
            )
            stats[f"{source_name}_sampled"] += len(source_vectors)
            stats[f"{source_name}_kept"] += kept
            if kept:
                counts[map_name] += kept
                stats["samples_with_vectors"] += 1
            else:
                stats["samples_only_duplicates"] += 1

    if not vectors:
        roi_text = f"x_forward=[{-roi_length / 2:g},{roi_length / 2:g}], y_left=[{-roi_width / 2:g},{roi_width / 2:g}]"
        raise RuntimeError(f"No valid local nuScenes map vectors were collected inside ROI {roi_text}.")

    map_names = tuple(sorted(catalogs))
    catalog_text = ", ".join(
        f"{name}=centerlines:{len(catalogs[name].centerlines)},fallback_lines:{len(catalogs[name].fallback_lines)}"
        for name in map_names
    )
    count_text = ", ".join(f"{name}={counts[name]}" for name in map_names)
    print("[map] primitive catalog per map: " + catalog_text)
    print("[map] kept local vectors per map: " + count_text)
    print(
        "[map] samples: "
        f"total={stats['samples']}, with_vectors={stats['samples_with_vectors']}, "
        f"without_vectors={stats['samples_without_vectors']}, duplicates_only={stats['samples_only_duplicates']}"
    )
    print(
        "[map] vectors: "
        f"centerline_sampled={stats['centerline_sampled']}, centerline_kept={stats['centerline_kept']}, "
        f"fallback_line_sampled={stats['fallback_line_sampled']}, fallback_line_kept={stats['fallback_line_kept']}"
    )

    return np.stack(vectors, axis=0), map_names


def _fit_kmeans(vectors: np.ndarray, k: int) -> np.ndarray:
    flat = vectors.reshape(vectors.shape[0], -1)
    if KMeans is not None:
        return KMeans(n_clusters=k, random_state=42, n_init=10).fit(flat).cluster_centers_

    rng = np.random.default_rng(42)
    centers = flat[rng.choice(flat.shape[0], size=k, replace=False)].copy()
    labels = np.full(flat.shape[0], -1, dtype=np.int64)
    for _ in range(100):
        distances = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_index in range(k):
            members = flat[labels == cluster_index]
            if len(members):
                centers[cluster_index] = members.mean(axis=0)
            else:
                centers[cluster_index] = flat[rng.integers(0, flat.shape[0])]
    return centers


def _save_visualization(vecs: np.ndarray, k: int) -> str:
    os.makedirs("vis/kmeans", exist_ok=True)
    path = f"vis/kmeans/nuscenes_map_anchor_{k}.png"
    plt.figure(figsize=(8, 8))
    for i in range(k):
        local_xy = _sparsedrive_map_xy_to_local_xy(vecs[i])
        plt.plot(local_xy[:, 0], local_xy[:, 1], linewidth=0.5, marker="o", markersize=1)
    plt.xlabel("x_forward (m)")
    plt.ylabel("y_left (m)")
    plt.title(f"nuScenes local map anchors (K={k})")
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path


def main():
    args = parse_args()
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}.")
    if args.num_sample <= 0:
        raise ValueError(f"--num_sample must be positive, got {args.num_sample}.")
    if args.roi_width <= 0.0:
        raise ValueError(f"--roi_width must be positive, got {args.roi_width}.")
    if args.roi_length <= 0.0:
        raise ValueError(f"--roi_length must be positive, got {args.roi_length}.")
    if args.dedupe_precision < 0.0:
        raise ValueError(f"--dedupe_precision must be non-negative, got {args.dedupe_precision}.")
    if args.max_vectors_per_sample < 0:
        raise ValueError(f"--max_vectors_per_sample must be non-negative, got {args.max_vectors_per_sample}.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"[kmeans_nuscenes_map] version={args.version}, K={args.k}, num_sample={args.num_sample}, "
        f"ROI x_forward=[{-args.roi_length / 2:g},{args.roi_length / 2:g}], "
        f"y_left=[{-args.roi_width / 2:g},{args.roi_width / 2:g}]"
    )

    vectors, map_names = _collect_sampled_vectors(
        args.data_path,
        args.version,
        args.num_sample,
        roi_width=args.roi_width,
        roi_length=args.roi_length,
        dedupe_precision=args.dedupe_precision,
        max_vectors_per_sample=args.max_vectors_per_sample,
    )
    print(f"[map] Found {len(map_names)} unique map locations: {list(map_names)}")
    print(f"[map] {len(vectors)} local map vectors collected in SparseDrive raw [x_right, y_forward] order.")
    if len(vectors) < args.k:
        raise RuntimeError(
            f"Collected only {len(vectors)} local nuScenes map vectors inside the ROI, but --k={args.k}. "
            "Use a lower --k (especially for v1.0-mini), a larger ROI, disable/decrease de-duplication, "
            "or use a larger nuScenes split."
        )

    print(f"[map] Fitting KMeans(K={args.k}) on flattened ({args.num_sample}, 2) vectors...")
    centers = _fit_kmeans(vectors, args.k)
    vecs = centers.reshape(args.k, args.num_sample, 2).astype(np.float32)

    vis_path = _save_visualization(vecs, args.k)
    out_path = os.path.join(args.out_dir, f"kmeans_map_{args.k}.npy")
    np.save(out_path, vecs)
    print(f"Saved {vecs.shape} to {out_path}")
    print(f"Saved visualization to {vis_path}")


if __name__ == "__main__":
    main()
