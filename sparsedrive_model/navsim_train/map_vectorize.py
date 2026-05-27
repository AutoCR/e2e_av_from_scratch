"""NuPlan map vectorization for SparseDrive-on-NAVSIM training.

This module is deliberately independent of mmcv/mmdet/mmdet3d and does not
import NuPlan map APIs at module import time.  It converts NuPlan map objects
near the ego pose into SparseDrive map targets in raw ``LIDAR_TOP`` XY where
``+x`` is ego-right and ``+y`` is ego-forward.

Shape contract for :func:`vectorize_map_for_frame`:
- ``permute=False``: ``gt_map_pts`` is ``float32`` with shape ``(M, N, 2)``.
- ``permute=True``: mirrors raw SparseDrive ``VectorizeMap`` exactly:
  ``gt_map_pts`` is ``float32`` with shape ``(M, 2 * (N - 1), N, 2)``.  Closed
  rings produce all rotations in both directions; open lines produce forward
  and reverse order followed by ``1e5`` padding permutations.  ``N`` is
  ``sample_num`` (20 in the SparseDrive configs).  If ``map_api is None``, the
  requested NAVSIM sentinel is returned with shape ``(0, N, 2)``.  ``gt_map_labels``
  is always ``int64`` with shape ``(M,)`` and class ids 0=crosswalk, 1=divider,
  2=boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Shapely is part of the NuPlan stack in the intended environment.
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
    from shapely.geometry.base import BaseGeometry
except Exception:  # pragma: no cover - exercised only in stripped environments.
    LineString = MultiLineString = MultiPolygon = Polygon = box = BaseGeometry = None  # type: ignore


_LABEL_PED_CROSSING = 0
_LABEL_DIVIDER = 1
_LABEL_BOUNDARY = 2
_PADDING_VALUE = 1e5
_EPS = 1e-6


@dataclass(frozen=True)
class _Point2D:
    x: float
    y: float


def _empty_result(sample_num: int, permute: bool) -> dict[str, np.ndarray]:
    if permute:
        pts_shape = (0, 2 * (sample_num - 1), sample_num, 2)
    else:
        pts_shape = (0, sample_num, 2)
    return {
        "gt_map_pts": np.zeros(pts_shape, dtype=np.float32),
        "gt_map_labels": np.zeros((0,), dtype=np.int64),
    }


def _none_map_result(sample_num: int) -> dict[str, np.ndarray]:
    return {
        "gt_map_pts": np.zeros((0, sample_num, 2), dtype=np.float32),
        "gt_map_labels": np.zeros((0,), dtype=np.int64),
    }


def _as_quaternion_rotation_matrix(ego_rotation_quat: Any) -> np.ndarray:
    if hasattr(ego_rotation_quat, "rotation_matrix"):
        return np.asarray(ego_rotation_quat.rotation_matrix, dtype=np.float64)

    quat = np.asarray(ego_rotation_quat, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError("ego_rotation_quat must be a pyquaternion.Quaternion or a (4,) wxyz sequence")
    w, x, y, z = quat
    norm = np.linalg.norm(quat)
    if norm <= _EPS:
        raise ValueError("ego_rotation_quat has near-zero norm")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _global_xy_to_lidar_xy(points_xy: np.ndarray, ego_translation: np.ndarray, ego_rotation_matrix: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.reshape(0, 2)
    points3 = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
    points3[:, :2] = points_xy[:, :2]
    ego_xyz = (points3 - ego_translation.reshape(1, 3)) @ ego_rotation_matrix
    return np.stack([-ego_xyz[:, 1], ego_xyz[:, 0]], axis=1)


def _transform_linestring_to_lidar(line: Any, ego_translation: np.ndarray, ego_rotation_matrix: np.ndarray) -> Any:
    coords = np.asarray(line.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return None
    lidar_xy = _global_xy_to_lidar_xy(coords[:, :2], ego_translation, ego_rotation_matrix)
    return LineString(lidar_xy)


def _transform_polygon_to_lidar(poly: Any, ego_translation: np.ndarray, ego_rotation_matrix: np.ndarray) -> Any:
    exterior = np.asarray(poly.exterior.coords, dtype=np.float64)
    if exterior.shape[0] < 4:
        return None
    lidar_exterior = _global_xy_to_lidar_xy(exterior[:, :2], ego_translation, ego_rotation_matrix)
    holes = []
    for interior in poly.interiors:
        coords = np.asarray(interior.coords, dtype=np.float64)
        if coords.shape[0] >= 4:
            holes.append(_global_xy_to_lidar_xy(coords[:, :2], ego_translation, ego_rotation_matrix))
    try:
        transformed = Polygon(lidar_exterior, holes)
        if not transformed.is_valid:
            transformed = transformed.buffer(0)
        return transformed if not transformed.is_empty else None
    except Exception:
        return None


def _iter_polygons(geom: Any) -> Iterable[Any]:
    if geom is None or getattr(geom, "is_empty", True):
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        yield geom
    elif geom_type == "MultiPolygon":
        yield from geom.geoms
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _iter_lines(geom: Any) -> Iterable[Any]:
    if geom is None or getattr(geom, "is_empty", True):
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "LineString":
        if geom.length > _EPS and len(geom.coords) >= 2:
            yield geom
    elif geom_type == "LinearRing":
        line = LineString(geom.coords)
        if line.length > _EPS:
            yield line
    elif geom_type == "Polygon":
        yield LineString(geom.exterior.coords)
    elif geom_type == "MultiLineString":
        for part in geom.geoms:
            yield from _iter_lines(part)
    elif geom_type == "MultiPolygon":
        for part in geom.geoms:
            yield from _iter_lines(part)
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_lines(part)


def _resample_line(line: Any, sample_num: int) -> np.ndarray | None:
    if line is None or line.is_empty or line.length <= _EPS or sample_num <= 0:
        return None
    distances = np.linspace(0.0, float(line.length), sample_num)
    points = np.array([line.interpolate(float(distance)).coords[0][:2] for distance in distances], dtype=np.float32)
    if points.ndim != 2 or points.shape != (sample_num, 2):
        return None
    return points


def _permute_line_raw(line: np.ndarray, padding: float = _PADDING_VALUE) -> np.ndarray:
    is_closed = bool(np.allclose(line[0], line[-1], atol=1e-3))
    num_points = int(line.shape[0])
    permute_num = num_points - 1
    if permute_num <= 0:
        return line.reshape(1, num_points, 2)

    if is_closed:
        pts = line[:-1]
        variants = [np.roll(pts, shift_i, axis=0) for shift_i in range(permute_num)]
        flipped = np.flip(pts, axis=0)
        variants.extend(np.roll(flipped, shift_i, axis=0) for shift_i in range(permute_num))
        arr = np.stack(variants, axis=0).astype(np.float32)
        out = np.zeros((permute_num * 2, num_points, 2), dtype=np.float32)
        out[:, :-1, :] = arr
        out[:, -1, :] = arr[:, 0, :]
        return out

    arr = np.stack([line, np.flip(line, axis=0)], axis=0).astype(np.float32)
    pad_count = permute_num * 2 - 2
    if pad_count > 0:
        pad = np.full((pad_count, num_points, 2), padding, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def _object_polygon(map_object: Any) -> Any:
    polygon = getattr(map_object, "polygon", None)
    return polygon() if callable(polygon) else polygon


def _object_linestring(map_object: Any) -> Any:
    linestring = getattr(map_object, "linestring", None)
    if linestring is not None:
        return linestring() if callable(linestring) else linestring
    discrete_path = getattr(map_object, "discrete_path", None)
    if discrete_path is not None:
        points = [(float(p.x), float(p.y)) for p in discrete_path]
        if len(points) >= 2:
            return LineString(points)
    return None


def _lane_boundary_lines(lane: Any) -> Iterable[Any]:
    for attr in ("left_boundary", "right_boundary"):
        try:
            boundary = getattr(lane, attr, None)
            if callable(boundary):
                boundary = boundary()
        except Exception:
            boundary = None
        line = _object_linestring(boundary)
        if line is not None:
            yield line


def _available_layers(map_api: Any, requested_layers: Sequence[Any]) -> list[Any]:
    try:
        supported = set(map_api.get_available_map_objects())
        return [layer for layer in requested_layers if layer in supported]
    except Exception:
        return list(requested_layers)


def _proximal_objects(map_api: Any, point: _Point2D, radius: float, layers: Sequence[Any]) -> dict[Any, list[Any]]:
    if not layers:
        return {}
    try:
        return map_api.get_proximal_map_objects(point, radius, list(layers))
    except Exception:
        return {layer: [] for layer in layers}


def _append_sampled(
    sampled_lines: list[np.ndarray],
    sampled_labels: list[int],
    line: Any,
    label: int,
    sample_num: int,
) -> None:
    sampled = _resample_line(line, sample_num)
    if sampled is not None:
        sampled_lines.append(sampled)
        sampled_labels.append(label)


def vectorize_map_for_frame(
    map_api: Any,
    ego_translation: Sequence[float] | np.ndarray,
    ego_rotation_quat: Any,
    roi_size: tuple[float, float] = (30, 60),
    sample_num: int = 20,
    permute: bool = True,
) -> dict[str, np.ndarray]:
    """Vectorize NuPlan map objects around one NAVSIM frame for SparseDrive.

    Uses ``map_api.get_proximal_map_objects`` on ``CROSSWALK``, ``LANE``,
    ``LANE_CONNECTOR`` and ``ROADBLOCK`` layers.  Global NuPlan geometries are
    transformed to SparseDrive raw ``LIDAR_TOP`` XY via ``(x_right, y_forward) =
    (-y_ego, x_ego)``, clipped to ``[-W/2, W/2] x [-H/2, H/2]`` and resampled to
    ``sample_num`` points before optional raw SparseDrive permutation padding.
    """
    if map_api is None:
        return _none_map_result(sample_num)
    if LineString is None or Polygon is None or box is None:
        return _empty_result(sample_num, permute)

    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    ego_t = np.zeros(3, dtype=np.float64)
    raw_translation = np.asarray(ego_translation, dtype=np.float64).reshape(-1)
    ego_t[: min(3, raw_translation.size)] = raw_translation[: min(3, raw_translation.size)]
    ego_r = _as_quaternion_rotation_matrix(ego_rotation_quat)

    roi_w, roi_h = float(roi_size[0]), float(roi_size[1])
    roi_box = box(-roi_w / 2.0, -roi_h / 2.0, roi_w / 2.0, roi_h / 2.0)
    query_radius = max(roi_w, roi_h) / 2.0 + 50.0
    query_point = _Point2D(float(ego_t[0]), float(ego_t[1]))

    requested = [
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.ROADBLOCK,
    ]
    layers = _available_layers(map_api, requested)
    objects = _proximal_objects(map_api, query_point, query_radius, layers)

    sampled_lines: list[np.ndarray] = []
    sampled_labels: list[int] = []

    for crosswalk in objects.get(SemanticMapLayer.CROSSWALK, []):
        poly = _object_polygon(crosswalk)
        if poly is None:
            continue
        lidar_poly = _transform_polygon_to_lidar(poly, ego_t, ego_r)
        if lidar_poly is None:
            continue
        clipped = lidar_poly.intersection(roi_box)
        for part in _iter_polygons(clipped):
            if part.area > _EPS and part.exterior.length > _EPS:
                _append_sampled(sampled_lines, sampled_labels, LineString(part.exterior.coords), _LABEL_PED_CROSSING, sample_num)

    seen_dividers: set[bytes] = set()
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        for lane in objects.get(layer, []):
            for boundary in _lane_boundary_lines(lane):
                try:
                    key = boundary.wkb
                except Exception:
                    key = repr(boundary).encode("utf-8")
                if key in seen_dividers:
                    continue
                seen_dividers.add(key)
                lidar_line = _transform_linestring_to_lidar(boundary, ego_t, ego_r)
                if lidar_line is None:
                    continue
                for part in _iter_lines(lidar_line.intersection(roi_box)):
                    _append_sampled(sampled_lines, sampled_labels, part, _LABEL_DIVIDER, sample_num)

    for roadblock in objects.get(SemanticMapLayer.ROADBLOCK, []):
        poly = _object_polygon(roadblock)
        if poly is None:
            continue
        lidar_poly = _transform_polygon_to_lidar(poly, ego_t, ego_r)
        if lidar_poly is None:
            continue
        for part in _iter_lines(LineString(lidar_poly.exterior.coords).intersection(roi_box)):
            _append_sampled(sampled_lines, sampled_labels, part, _LABEL_BOUNDARY, sample_num)

    if not sampled_lines:
        return _empty_result(sample_num, permute)

    labels = np.asarray(sampled_labels, dtype=np.int64)
    if permute:
        pts = np.stack([_permute_line_raw(line) for line in sampled_lines], axis=0).astype(np.float32)
    else:
        pts = np.stack(sampled_lines, axis=0).astype(np.float32)
    return {"gt_map_pts": pts, "gt_map_labels": labels}


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, "/Users/chenran/Code/e2e_av_from_scratch")
    os.environ.setdefault("OPENSCENE_DATA_ROOT", "/Users/chenran/Code/navsim/dataset")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", "/Users/chenran/Code/navsim/dataset/maps")

    from sparsedrive_model.navsim_adapter import build_navsim_scene_loader, build_navsim_sparsedrive_sample

    loader = build_navsim_scene_loader(split="mini", max_scenes=2)
    token = next(iter(loader.tokens))
    sample = build_navsim_sparsedrive_sample(loader, token)
    frame = sample.current_frame
    out = vectorize_map_for_frame(
        sample.map_api,
        frame["ego2global_translation"],
        frame["ego2global_rotation"],
    )
    print({k: (v.shape, v.dtype) for k, v in out.items()})
    print("label counts:", {int(label): int((out["gt_map_labels"] == label).sum()) for label in [0, 1, 2]})
