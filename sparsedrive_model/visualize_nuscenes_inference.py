"""nuScenes SparseDrive BEV visualization.

Raw nuScenes ``LIDAR_TOP`` / SparseDrive XY values in this adapter use +y as
ego-forward and +x as ego-right.  This visualizer converts those raw values to a
display ego frame ``[x_forward, y_left] = [raw_y, -raw_x]`` before plotting.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MatplotlibPath
import numpy as np

try:
    from .prediction_decode import DecodedSparseDrivePrediction, decode_sparsedrive_outputs
except ImportError:  # Supports importing this module from sparsedrive_model/test_nuscenes_mini.py.
    from prediction_decode import DecodedSparseDrivePrediction, decode_sparsedrive_outputs


PathLike = Union[str, os.PathLike[str]]

DEFAULT_LATERAL_EXTENT_METERS = 50.0
DEFAULT_REAR_EXTENT_METERS = 30.0
DEFAULT_FORWARD_EXTENT_METERS = 80.0

EGO_BOX_CONFIG: dict[str, Any] = {
    "facecolor": "#222222",
    "edgecolor": "#000000",
    "alpha": 0.85,
    "linewidth": 1.0,
    "zorder": 8,
}
PREDICTED_OBSTACLE_CONFIG: dict[str, Any] = {
    "facecolor": "#4e79a7",
    "edgecolor": "#1f4e79",
    "alpha": 0.35,
    "linewidth": 1.0,
    "zorder": 4,
}
GT_OBSTACLE_CONFIG: dict[str, Any] = {
    "facecolor": "#59a14f",
    "edgecolor": "#2f6b2f",
    "alpha": 0.25,
    "linewidth": 1.0,
    "zorder": 4,
}
PREDICTED_MAP_CONFIG: dict[str, Any] = {
    "color": "#9467bd",
    "alpha": 0.85,
    "linewidth": 1.2,
    "linestyle": "-",
    "zorder": 2,
}
PREDICTED_EGO_TRAJECTORY_CONFIG: dict[str, Any] = {
    "color": "#d62728",
    "alpha": 0.95,
    "linewidth": 2.0,
    "linestyle": "-",
    "marker": ".",
    "markersize": 4.0,
    "zorder": 7,
}
PREDICTED_OBSTACLE_TRAJECTORY_CONFIG: dict[str, Any] = {
    "color": "#ff7f0e",
    "alpha": 0.9,
    "linewidth": 1.0,
    "linestyle": "-",
    "marker": ".",
    "markersize": 2.5,
    "zorder": 5,
}
GT_EGO_TRAJECTORY_CONFIG: dict[str, Any] = {
    "color": "#2ca02c",
    "alpha": 0.95,
    "linewidth": 2.0,
    "linestyle": "-",
    "marker": ".",
    "markersize": 4.0,
    "zorder": 7,
}
GT_OBSTACLE_TRAJECTORY_CONFIG: dict[str, Any] = {
    "color": "#17becf",
    "alpha": 0.9,
    "linewidth": 1.0,
    "linestyle": "-",
    "marker": ".",
    "markersize": 2.5,
    "zorder": 5,
}
MAP_COLORS = ("#9467bd", "#1f77b4", "#8c564b", "#e377c2", "#7f7f7f")
PREDICTED_MAP_CLASS_NAMES = ("ped_crossing", "divider", "boundary")
GT_STATIC_MAP_POLYGON_CONFIGS: dict[str, dict[str, Any]] = {
    "drivable_area": {
        "facecolor": "#9e9e9e",
        "edgecolor": "none",
        "alpha": 0.10,
        "linewidth": 0.0,
        "zorder": 0,
    },
    "road_segment": {
        "facecolor": "#9ecae1",
        "edgecolor": "none",
        "alpha": 0.12,
        "linewidth": 0.0,
        "zorder": 1,
    },
    "lane": {
        "facecolor": "#74c476",
        "edgecolor": "none",
        "alpha": 0.14,
        "linewidth": 0.0,
        "zorder": 1,
    },
    "walkway": {
        "facecolor": "#fdd0a2",
        "edgecolor": "none",
        "alpha": 0.16,
        "linewidth": 0.0,
        "zorder": 1,
    },
    "ped_crossing": {
        "facecolor": "#fdae6b",
        "edgecolor": "#e6550d",
        "alpha": 0.24,
        "linewidth": 0.35,
        "zorder": 2,
    },
}
GT_STATIC_MAP_LINE_CONFIGS: dict[str, dict[str, Any]] = {
    "lane_divider": {
        "color": "#ffffff",
        "alpha": 0.75,
        "linewidth": 0.75,
        "linestyle": "-",
        "zorder": 3,
    },
    "road_divider": {
        "color": "#4d4d4d",
        "alpha": 0.75,
        "linewidth": 0.95,
        "linestyle": "-",
        "zorder": 3,
    },
}

_MAP_POLYGON_LAYER_FIELDS = {
    "drivable_area": "polygon_tokens",
    "road_segment": "polygon_token",
    "lane": "polygon_token",
    "walkway": "polygon_token",
    "ped_crossing": "polygon_token",
}
_MAP_LINE_LAYER_FIELDS = {
    "lane_divider": "line_token",
    "road_divider": "line_token",
}
_MAP_FILTER_MARGIN_METERS = 5.0
_NUSCENES_MAP_CACHE: dict[Path, "_NuScenesStaticMap"] = {}


_EMPTY_XY = np.empty((0, 2), dtype=np.float64)


@dataclass(frozen=True)
class _MapPolygon:
    token: str
    exterior_xy: np.ndarray
    hole_xys: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _NuScenesStaticMap:
    path: Path
    polygon_layers: dict[str, tuple[_MapPolygon, ...]]
    line_layers: dict[str, tuple[np.ndarray, ...]]


def visualize_nuscenes_inference(
    sample: Any,
    decoded_prediction: DecodedSparseDrivePrediction,
    output_dir: PathLike,
    *,
    summary: Optional[Mapping[str, Any]] = None,
    filename: Optional[str] = None,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """Render and save a two-panel SparseDrive nuScenes inference BEV figure."""

    output_path = _figure_output_path(sample, output_dir, filename)
    fig, axes = plt.subplots(2, 1, figsize=(8, 13))

    _plot_prediction_panel(axes[0], sample, decoded_prediction, summary=summary)
    _plot_ground_truth_panel(axes[1], sample)

    for ax in axes:
        _configure_bev_axis(ax)

    fig.suptitle(f"SparseDrive nuScenes inference: {_sample_token(sample)}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return output_path


def visualize_nuscenes_sparsedrive_predictions(
    *,
    samples: Sequence[Any],
    outputs: Optional[Sequence[Mapping[str, Any]]] = None,
    decoded_predictions: Optional[Sequence[DecodedSparseDrivePrediction]] = None,
    summaries: Optional[Sequence[Mapping[str, Any]]] = None,
    output_dir: PathLike,
) -> list[Path]:
    """Batch-compatible helper used by sparsedrive_model/test_nuscenes_mini.py."""

    predictions = _resolve_decoded_predictions(samples, outputs, decoded_predictions)
    normalized_summaries = _normalize_summaries(summaries, len(samples))
    return [
        visualize_nuscenes_inference(
            sample,
            prediction,
            output_dir,
            summary=summary,
        )
        for sample, prediction, summary in zip(samples, predictions, normalized_summaries)
    ]


def _plot_prediction_panel(
    ax: plt.Axes,
    sample: Any,
    decoded_prediction: DecodedSparseDrivePrediction,
    *,
    summary: Optional[Mapping[str, Any]],
) -> None:
    ax.set_title("Prediction")
    _add_ego_reference(ax, draw_box=True)
    _plot_predicted_map(ax, decoded_prediction)
    _plot_prediction_obstacles(ax, decoded_prediction)
    _plot_xy_path(
        ax,
        _raw_lidar_xy_to_display_ego_xy(
            decoded_prediction.predicted_ego_trajectory,
            "predicted_ego_trajectory",
        ),
        prepend_origin=True,
        **PREDICTED_EGO_TRAJECTORY_CONFIG,
    )
    _add_metadata_text(ax, sample, decoded_prediction, summary)
    _add_prediction_legend(ax)


def _plot_ground_truth_panel(ax: plt.Axes, sample: Any) -> None:
    ax.set_title("Ground truth")
    _plot_nuscenes_static_map(ax, sample)
    _add_ego_reference(ax, draw_box=True)
    _plot_current_annotations(ax, getattr(sample, "current_annotations", None))
    _plot_future_obstacle_trajectories(ax, getattr(sample, "future_obstacle_trajectories", None))
    _plot_xy_path(
        ax,
        _raw_lidar_xy_to_display_ego_xy(
            getattr(sample, "future_lidar_origins", None),
            "sample.future_lidar_origins",
        ),
        prepend_origin=True,
        **GT_EGO_TRAJECTORY_CONFIG,
    )


def _plot_predicted_map(ax: plt.Axes, decoded_prediction: DecodedSparseDrivePrediction) -> None:
    labels = _optional_1d_array(getattr(decoded_prediction, "map_labels", None), "map_labels")
    for polyline_idx, polyline in enumerate(_iter_polylines(decoded_prediction.predicted_map_polylines)):
        points = _raw_lidar_xy_to_display_ego_xy(polyline, f"predicted_map_polylines[{polyline_idx}]")
        if points.shape[0] < 2:
            continue
        config = dict(PREDICTED_MAP_CONFIG)
        if labels is not None and polyline_idx < labels.shape[0] and np.isfinite(labels[polyline_idx]):
            config["color"] = MAP_COLORS[int(labels[polyline_idx]) % len(MAP_COLORS)]
        _plot_xy_path(ax, points, **config)


def _plot_prediction_obstacles(
    ax: plt.Axes,
    decoded_prediction: DecodedSparseDrivePrediction,
) -> None:
    boxes = _to_numpy(decoded_prediction.predicted_obstacle_boxes, "predicted_obstacle_boxes")
    if boxes.size == 0:
        return
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"predicted_obstacle_boxes must be shaped [N, >=7], got {boxes.shape}.")
    boxes = boxes.astype(np.float64, copy=False)

    trajectories = _ensure_trajectory_batch(
        decoded_prediction.predicted_obstacle_trajectories,
        "predicted_obstacle_trajectories",
        expected_count=boxes.shape[0],
    )

    for index, box_values in enumerate(boxes):
        required_values = box_values[:7]
        if not np.isfinite(required_values).all():
            continue
        x_size = max(float(box_values[3]), 0.05)
        y_size = max(float(box_values[4]), 0.05)
        center_xy = _raw_lidar_xy_to_display_ego_xy(
            np.asarray(box_values[:2], dtype=np.float64).reshape(1, 2),
            f"predicted_obstacle_boxes[{index}].center",
        )[0]
        yaw = float(_raw_lidar_yaw_to_display_ego_yaw(box_values[6]))
        _draw_box(ax, center_xy, x_size, y_size, yaw, **PREDICTED_OBSTACLE_CONFIG)

        if trajectories.shape[0] == 0:
            continue
        path = _raw_lidar_xy_to_display_ego_xy(
            trajectories[index],
            f"predicted_obstacle_trajectories[{index}]",
        )
        if path.shape[0] == 0:
            continue
        path = np.concatenate([center_xy.reshape(1, 2), path], axis=0)
        _plot_xy_path(ax, path, **PREDICTED_OBSTACLE_TRAJECTORY_CONFIG)


def _plot_current_annotations(ax: plt.Axes, annotations: Any) -> None:
    for index, annotation in enumerate(_iter_current_annotations(annotations)):
        components = _annotation_box_components(annotation, index)
        if components is None:
            continue
        center_xy, x_size, y_size, yaw = components
        _draw_raw_lidar_box(ax, center_xy, x_size, y_size, yaw, **GT_OBSTACLE_CONFIG)


def _plot_future_obstacle_trajectories(ax: plt.Axes, trajectories: Any) -> None:
    if trajectories is None:
        return
    if isinstance(trajectories, Mapping):
        trajectories = trajectories.values()
    for index, trajectory in enumerate(trajectories):
        points_value = _get_optional_field(trajectory, "points_xy", "points", "trajectory", "xy")
        if points_value is None:
            continue
        points = _raw_lidar_xy_to_display_ego_xy(points_value, f"future_obstacle_trajectories[{index}]")
        if points.shape[0] < 2:
            continue
        _plot_xy_path(ax, points, **GT_OBSTACLE_TRAJECTORY_CONFIG)


def _plot_nuscenes_static_map(ax: plt.Axes, sample: Any) -> None:
    map_name = _sample_map_name(sample)
    if map_name is None:
        _add_gt_map_note(ax, "nuScenes static map unavailable: sample has no map_name")
        return

    dataset_root = _resolve_nuscenes_dataset_root(sample)
    if dataset_root is None:
        _add_gt_map_note(ax, f"nuScenes static map unavailable: no dataset root for {map_name}")
        return

    map_path = _nuscenes_map_path(dataset_root, map_name)
    if map_path is None:
        _add_gt_map_note(ax, f"nuScenes static map unavailable: unsupported map name {map_name!r}")
        return
    if not map_path.is_file():
        _add_gt_map_note(ax, f"nuScenes static map unavailable: {map_name}.json not found")
        return

    t_global_inv, transform_error = _sample_global_to_lidar_transform(sample)
    if t_global_inv is None:
        _add_gt_map_note(ax, f"nuScenes static map unavailable: {transform_error}")
        return

    static_map = _load_cached_nuscenes_static_map(map_path)
    _draw_nuscenes_static_map(ax, static_map, t_global_inv)


def _draw_nuscenes_static_map(
    ax: plt.Axes,
    static_map: _NuScenesStaticMap,
    t_global_inv: np.ndarray,
) -> None:
    for layer_name, config in GT_STATIC_MAP_POLYGON_CONFIGS.items():
        for polygon in static_map.polygon_layers.get(layer_name, ()):
            exterior = _global_xy_to_display_ego_xy(
                polygon.exterior_xy,
                t_global_inv,
                f"{static_map.path.name}:{layer_name}:{polygon.token}.exterior",
            )
            if exterior.shape[0] < 3 or not _display_bbox_intersects_bev(exterior):
                continue
            holes = tuple(
                _global_xy_to_display_ego_xy(
                    hole,
                    t_global_inv,
                    f"{static_map.path.name}:{layer_name}:{polygon.token}.hole[{hole_index}]",
                )
                for hole_index, hole in enumerate(polygon.hole_xys)
            )
            _add_static_map_polygon(ax, exterior, holes, config)

    for layer_name, config in GT_STATIC_MAP_LINE_CONFIGS.items():
        for line_index, line_xy in enumerate(static_map.line_layers.get(layer_name, ())):
            points = _global_xy_to_display_ego_xy(
                line_xy,
                t_global_inv,
                f"{static_map.path.name}:{layer_name}[{line_index}]",
            )
            if points.shape[0] < 2 or not _display_bbox_intersects_bev(points):
                continue
            _plot_xy_path(ax, points, **config)


def _load_cached_nuscenes_static_map(map_path: Path) -> _NuScenesStaticMap:
    cache_key = map_path.expanduser().resolve()
    cached = _NUSCENES_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loaded = _read_nuscenes_static_map(cache_key)
    _NUSCENES_MAP_CACHE[cache_key] = loaded
    return loaded


def _read_nuscenes_static_map(map_path: Path) -> _NuScenesStaticMap:
    with map_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise TypeError(f"nuScenes map file {map_path} must contain a JSON object, got {type(payload).__name__}.")

    node_xy_by_token = _load_map_nodes(payload, map_path)
    polygon_by_token = _load_map_polygons(payload, node_xy_by_token, map_path)
    line_by_token = _load_map_lines(payload, node_xy_by_token, map_path)

    polygon_layers = {
        layer_name: _resolve_layer_primitives(
            polygon_by_token,
            _iter_layer_primitive_tokens(payload, layer_name, token_field, map_path),
            primitive_name="polygon",
            layer_name=layer_name,
            map_path=map_path,
        )
        for layer_name, token_field in _MAP_POLYGON_LAYER_FIELDS.items()
    }
    line_layers = {
        layer_name: _resolve_layer_primitives(
            line_by_token,
            _iter_layer_primitive_tokens(payload, layer_name, token_field, map_path),
            primitive_name="line",
            layer_name=layer_name,
            map_path=map_path,
        )
        for layer_name, token_field in _MAP_LINE_LAYER_FIELDS.items()
    }
    return _NuScenesStaticMap(path=map_path, polygon_layers=polygon_layers, line_layers=line_layers)


def _load_map_nodes(payload: Mapping[str, Any], map_path: Path) -> dict[str, np.ndarray]:
    nodes: dict[str, np.ndarray] = {}
    for index, record in enumerate(_map_record_sequence(payload, "node", map_path)):
        token = _map_token(record, "node", index, map_path)
        if token in nodes:
            raise ValueError(f"nuScenes map file {map_path} has duplicate node token {token!r}.")
        x = _map_float_field(record, "x", "node", index, map_path)
        y = _map_float_field(record, "y", "node", index, map_path)
        nodes[token] = np.array([x, y], dtype=np.float64)
    if not nodes:
        raise ValueError(f"nuScenes map file {map_path} does not contain any node records.")
    return nodes


def _load_map_polygons(
    payload: Mapping[str, Any],
    node_xy_by_token: Mapping[str, np.ndarray],
    map_path: Path,
) -> dict[str, _MapPolygon]:
    polygons: dict[str, _MapPolygon] = {}
    for index, record in enumerate(_map_record_sequence(payload, "polygon", map_path)):
        if _is_empty_polygon_placeholder(record):
            continue
        token = _map_token(record, "polygon", index, map_path)
        if token in polygons:
            raise ValueError(f"nuScenes map file {map_path} has duplicate polygon token {token!r}.")
        exterior_tokens = _map_token_sequence(record, "exterior_node_tokens", "polygon", index, map_path)
        exterior_xy = _node_xy_for_tokens(
            exterior_tokens,
            node_xy_by_token,
            f"nuScenes map file {map_path} polygon[{index}] exterior_node_tokens",
            min_points=3,
        )
        hole_xys = tuple(
            _node_xy_for_tokens(
                _map_hole_node_tokens(hole, hole_index, index, map_path),
                node_xy_by_token,
                f"nuScenes map file {map_path} polygon[{index}] holes[{hole_index}]",
                min_points=3,
            )
            for hole_index, hole in enumerate(_map_holes(record, index, map_path))
        )
        polygons[token] = _MapPolygon(token=token, exterior_xy=exterior_xy, hole_xys=hole_xys)
    return polygons


def _load_map_lines(
    payload: Mapping[str, Any],
    node_xy_by_token: Mapping[str, np.ndarray],
    map_path: Path,
) -> dict[str, np.ndarray]:
    lines: dict[str, np.ndarray] = {}
    for index, record in enumerate(_map_record_sequence(payload, "line", map_path)):
        if _is_empty_line_placeholder(record):
            continue
        token = _map_token(record, "line", index, map_path)
        if token in lines:
            raise ValueError(f"nuScenes map file {map_path} has duplicate line token {token!r}.")
        node_tokens = _map_token_sequence(record, "node_tokens", "line", index, map_path)
        lines[token] = _node_xy_for_tokens(
            node_tokens,
            node_xy_by_token,
            f"nuScenes map file {map_path} line[{index}] node_tokens",
            min_points=2,
        )
    return lines


def _is_empty_polygon_placeholder(record: Mapping[str, Any]) -> bool:
    return (
        str(record.get("token", "")) == ""
        and not record.get("exterior_node_tokens")
        and not record.get("holes")
    )


def _is_empty_line_placeholder(record: Mapping[str, Any]) -> bool:
    return str(record.get("token", "")) == "" and not record.get("node_tokens")


def _iter_layer_primitive_tokens(
    payload: Mapping[str, Any],
    layer_name: str,
    token_field: str,
    map_path: Path,
) -> list[str]:
    records = _map_record_sequence(payload, layer_name, map_path, required=False)
    tokens: list[str] = []
    for index, record in enumerate(records):
        if token_field.endswith("_tokens"):
            tokens.extend(_map_token_sequence(record, token_field, layer_name, index, map_path))
            continue

        token = str(_map_required_field(record, token_field, layer_name, index, map_path))
        if token == "":
            raise ValueError(
                f"nuScenes map file {map_path} {layer_name}[{index}] has empty {token_field!r}."
            )
        tokens.append(token)
    return tokens


def _resolve_layer_primitives(
    primitive_by_token: Mapping[str, Any],
    tokens: Sequence[str],
    *,
    primitive_name: str,
    layer_name: str,
    map_path: Path,
) -> tuple[Any, ...]:
    primitives: list[Any] = []
    for token in tokens:
        try:
            primitives.append(primitive_by_token[token])
        except KeyError as exc:
            raise KeyError(
                f"nuScenes map file {map_path} layer {layer_name!r} references unknown "
                f"{primitive_name} token {token!r}."
            ) from exc
    return tuple(primitives)


def _map_record_sequence(
    payload: Mapping[str, Any],
    table_name: str,
    map_path: Path,
    *,
    required: bool = True,
) -> list[Mapping[str, Any]]:
    records = payload.get(table_name)
    if records is None:
        if required:
            raise KeyError(f"nuScenes map file {map_path} is missing required table {table_name!r}.")
        return []
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(
            f"nuScenes map file {map_path} table {table_name!r} must be a list, "
            f"got {type(records).__name__}."
        )

    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"nuScenes map file {map_path} {table_name}[{index}] must be an object, "
                f"got {type(record).__name__}."
            )
        normalized.append(record)
    return normalized


def _map_token(record: Mapping[str, Any], table_name: str, index: int, map_path: Path) -> str:
    token = str(_map_required_field(record, "token", table_name, index, map_path))
    if token == "":
        raise ValueError(f"nuScenes map file {map_path} {table_name}[{index}] has an empty token.")
    return token


def _map_required_field(
    record: Mapping[str, Any],
    field_name: str,
    table_name: str,
    index: int,
    map_path: Path,
) -> Any:
    if field_name not in record:
        token = record.get("token", "<unknown>")
        raise KeyError(
            f"nuScenes map file {map_path} {table_name}[{index}] ({token!r}) "
            f"is missing required field {field_name!r}."
        )
    return record[field_name]


def _map_float_field(
    record: Mapping[str, Any],
    field_name: str,
    table_name: str,
    index: int,
    map_path: Path,
) -> float:
    value = _map_required_field(record, field_name, table_name, index, map_path)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"nuScenes map file {map_path} {table_name}[{index}] field {field_name!r} "
            f"must be numeric, got {value!r}."
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"nuScenes map file {map_path} {table_name}[{index}] field {field_name!r} "
            f"must be finite, got {value!r}."
        )
    return result


def _map_token_sequence(
    record: Mapping[str, Any],
    field_name: str,
    table_name: str,
    index: int,
    map_path: Path,
) -> list[str]:
    value = _map_required_field(record, field_name, table_name, index, map_path)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"nuScenes map file {map_path} {table_name}[{index}] field {field_name!r} "
            f"must be a list of tokens, got {type(value).__name__}."
        )
    tokens = [str(token) for token in value]
    if any(token == "" for token in tokens):
        raise ValueError(
            f"nuScenes map file {map_path} {table_name}[{index}] field {field_name!r} "
            "contains an empty token."
        )
    return tokens


def _map_holes(record: Mapping[str, Any], polygon_index: int, map_path: Path) -> list[Any]:
    holes = record.get("holes", [])
    if holes is None:
        return []
    if isinstance(holes, (str, bytes)) or not isinstance(holes, Sequence):
        raise TypeError(
            f"nuScenes map file {map_path} polygon[{polygon_index}] field 'holes' "
            f"must be a list, got {type(holes).__name__}."
        )
    return list(holes)


def _map_hole_node_tokens(hole: Any, hole_index: int, polygon_index: int, map_path: Path) -> list[str]:
    if isinstance(hole, Mapping):
        return _map_token_sequence(hole, "node_tokens", "polygon.hole", hole_index, map_path)
    if isinstance(hole, (str, bytes)) or not isinstance(hole, Sequence):
        raise TypeError(
            f"nuScenes map file {map_path} polygon[{polygon_index}] holes[{hole_index}] "
            f"must be an object with node_tokens or a list of tokens, got {type(hole).__name__}."
        )
    tokens = [str(token) for token in hole]
    if any(token == "" for token in tokens):
        raise ValueError(
            f"nuScenes map file {map_path} polygon[{polygon_index}] holes[{hole_index}] "
            "contains an empty token."
        )
    return tokens


def _node_xy_for_tokens(
    tokens: Sequence[str],
    node_xy_by_token: Mapping[str, np.ndarray],
    context: str,
    *,
    min_points: int,
) -> np.ndarray:
    if len(tokens) < min_points:
        raise ValueError(f"{context} must contain at least {min_points} node tokens, got {len(tokens)}.")
    points: list[np.ndarray] = []
    for token in tokens:
        try:
            points.append(node_xy_by_token[str(token)])
        except KeyError as exc:
            raise KeyError(f"{context} references unknown node token {token!r}.") from exc
    return np.stack(points, axis=0).astype(np.float64, copy=False)


def _add_static_map_polygon(
    ax: plt.Axes,
    exterior_xy: np.ndarray,
    hole_xys: Sequence[np.ndarray],
    config: Mapping[str, Any],
) -> None:
    kwargs = dict(config)
    if not hole_xys:
        ax.add_patch(Polygon(_xy_to_plot(exterior_xy), closed=True, **kwargs))
        return

    vertices: list[np.ndarray] = []
    codes: list[int] = []
    _append_polygon_ring(vertices, codes, _oriented_ring(exterior_xy, ccw=True))
    for hole_xy in hole_xys:
        if hole_xy.shape[0] >= 3:
            _append_polygon_ring(vertices, codes, _oriented_ring(hole_xy, ccw=False))
    if not vertices:
        return
    path = MatplotlibPath(np.vstack(vertices), np.asarray(codes, dtype=np.uint8))
    ax.add_patch(PathPatch(path, **kwargs))


def _append_polygon_ring(vertices: list[np.ndarray], codes: list[int], ring_xy: np.ndarray) -> None:
    if ring_xy.shape[0] < 3:
        return
    plot_ring = _xy_to_plot(ring_xy)
    vertices.append(plot_ring[0])
    codes.append(MatplotlibPath.MOVETO)
    for point in plot_ring[1:]:
        vertices.append(point)
        codes.append(MatplotlibPath.LINETO)
    vertices.append(plot_ring[0])
    codes.append(MatplotlibPath.CLOSEPOLY)


def _oriented_ring(ring_xy: np.ndarray, *, ccw: bool) -> np.ndarray:
    area = _signed_polygon_area(ring_xy)
    if area == 0.0:
        return ring_xy
    should_reverse = (area < 0.0) if ccw else (area > 0.0)
    return ring_xy[::-1] if should_reverse else ring_xy


def _signed_polygon_area(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _global_xy_to_display_ego_xy(value: Any, t_global_inv: np.ndarray, name: str) -> np.ndarray:
    global_xy = _ensure_xy_array(value, name)
    if global_xy.shape[0] == 0:
        return _EMPTY_XY.copy()
    homogeneous = np.column_stack(
        [
            global_xy[:, 0],
            global_xy[:, 1],
            np.zeros(global_xy.shape[0], dtype=np.float64),
            np.ones(global_xy.shape[0], dtype=np.float64),
        ]
    )
    raw_lidar_xy = (t_global_inv @ homogeneous.T).T[:, :2]
    return _raw_lidar_xy_to_display_ego_xy(raw_lidar_xy, f"{name} in current lidar")


def _display_bbox_intersects_bev(points_xy: np.ndarray) -> bool:
    if points_xy.shape[0] == 0:
        return False
    min_forward = float(np.min(points_xy[:, 0]))
    max_forward = float(np.max(points_xy[:, 0]))
    min_left = float(np.min(points_xy[:, 1]))
    max_left = float(np.max(points_xy[:, 1]))
    margin = _MAP_FILTER_MARGIN_METERS
    return (
        max_forward >= -DEFAULT_REAR_EXTENT_METERS - margin
        and min_forward <= DEFAULT_FORWARD_EXTENT_METERS + margin
        and max_left >= -DEFAULT_LATERAL_EXTENT_METERS - margin
        and min_left <= DEFAULT_LATERAL_EXTENT_METERS + margin
    )


def _sample_map_name(sample: Any) -> Optional[str]:
    map_name = _get_optional_field(sample, "map_name")
    if map_name is None:
        return None
    map_name_text = str(map_name).strip()
    return map_name_text or None


def _resolve_nuscenes_dataset_root(sample: Any) -> Optional[Path]:
    explicit_root = _get_optional_field(sample, "dataset_root", "nuscenes_dataset_root", "data_root")
    if explicit_root is not None and str(explicit_root).strip():
        return _expand_path(explicit_root)

    for record in _iter_sample_data_records(sample):
        filename = _get_optional_field(record, "filename")
        if filename is None:
            continue
        inferred_root = _infer_dataset_root_from_sample_data_filename(filename)
        if inferred_root is not None:
            return inferred_root

    default_root = _default_nuscenes_dataset_root()
    if default_root is not None:
        return _expand_path(default_root)
    return None


def _nuscenes_map_path(dataset_root: Path, map_name: str) -> Optional[Path]:
    normalized = map_name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        return None
    return dataset_root / "maps" / "expansion" / f"{normalized}.json"


def _iter_sample_data_records(sample: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for field_name in ("lidar_sample_data", "camera_sample_data"):
        value = _get_optional_field(sample, field_name)
        if value is None:
            continue
        if isinstance(value, Mapping):
            if "filename" in value:
                records.append(value)
            else:
                records.extend(record for record in value.values() if isinstance(record, Mapping))
            continue
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            continue
        records.extend(record for record in value if isinstance(record, Mapping))
    return records


def _infer_dataset_root_from_sample_data_filename(filename: Any) -> Optional[Path]:
    filename_text = str(filename)
    if not filename_text:
        return None
    path = _expand_path(filename_text)
    if not path.is_absolute():
        return None
    parts = path.parts
    for marker in ("samples", "sweeps"):
        if marker in parts:
            marker_index = parts.index(marker)
            if marker_index > 0:
                return Path(*parts[:marker_index])
    return None


def _default_nuscenes_dataset_root() -> Optional[Any]:
    try:
        from .nuscenes_adapter import DEFAULT_NUSCENES_ROOT
    except Exception:
        try:
            from nuscenes_adapter import DEFAULT_NUSCENES_ROOT  # type: ignore[no-redef]
        except Exception:
            return None
    return DEFAULT_NUSCENES_ROOT


def _sample_global_to_lidar_transform(sample: Any) -> tuple[Optional[np.ndarray], str]:
    img_metas = _get_optional_field(sample, "img_metas")
    if not isinstance(img_metas, Mapping):
        return None, "sample.img_metas is missing"
    if "T_global_inv" not in img_metas:
        return None, "sample.img_metas['T_global_inv'] is missing"
    try:
        transform = np.asarray(img_metas["T_global_inv"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return None, f"sample.img_metas['T_global_inv'] is invalid: {exc}"
    if transform.shape != (4, 4):
        return None, f"sample.img_metas['T_global_inv'] has shape {transform.shape}, expected (4, 4)"
    if not np.isfinite(transform).all():
        return None, "sample.img_metas['T_global_inv'] contains non-finite values"
    return transform, ""


def _expand_path(path: Any) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(path))))


def _add_gt_map_note(ax: plt.Axes, message: str) -> None:
    ax.text(
        0.01,
        0.98,
        message,
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
        zorder=20,
    )


def _add_ego_reference(ax: plt.Axes, *, draw_box: bool) -> None:
    if draw_box:
        _draw_box(ax, np.array([0.0, 0.0], dtype=np.float64), 4.8, 2.0, 0.0, **EGO_BOX_CONFIG)
    ax.scatter([0.0], [0.0], c="#000000", s=18, marker="x", zorder=10)
    forward_arrow = _xy_to_plot(np.array([[0.0, 0.0], [8.0, 0.0]], dtype=np.float64))
    left_arrow = _xy_to_plot(np.array([[0.0, 0.0], [0.0, 4.0]], dtype=np.float64))
    forward_label = _xy_to_plot(np.array([[8.4, 0.2]], dtype=np.float64))[0]
    left_label = _xy_to_plot(np.array([[0.3, 4.4]], dtype=np.float64))[0]
    ax.arrow(
        forward_arrow[0, 0],
        forward_arrow[0, 1],
        forward_arrow[1, 0] - forward_arrow[0, 0],
        forward_arrow[1, 1] - forward_arrow[0, 1],
        width=0.08,
        head_width=0.8,
        head_length=1.2,
        length_includes_head=True,
        color="#111111",
        alpha=0.85,
        zorder=9,
    )
    ax.arrow(
        left_arrow[0, 0],
        left_arrow[0, 1],
        left_arrow[1, 0] - left_arrow[0, 0],
        left_arrow[1, 1] - left_arrow[0, 1],
        width=0.04,
        head_width=0.5,
        head_length=0.8,
        length_includes_head=True,
        color="#666666",
        alpha=0.65,
        zorder=9,
    )
    ax.text(forward_label[0], forward_label[1], "x fwd", fontsize=7, color="#111111")
    ax.text(left_label[0], left_label[1], "y left", fontsize=7, color="#666666")


def _draw_raw_lidar_box(
    ax: plt.Axes,
    raw_center_xy: np.ndarray,
    x_size: float,
    y_size: float,
    raw_yaw: float,
    **draw_kwargs: Any,
) -> None:
    center_xy = _raw_lidar_xy_to_display_ego_xy(raw_center_xy.reshape(1, 2), "raw box center")[0]
    yaw = float(_raw_lidar_yaw_to_display_ego_yaw(raw_yaw))
    _draw_box(ax, center_xy, x_size, y_size, yaw, **draw_kwargs)


def _draw_box(
    ax: plt.Axes,
    center_xy: np.ndarray,
    x_size: float,
    y_size: float,
    yaw: float,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    """Draw a box whose center/yaw are already in display ego coordinates."""

    if not np.isfinite(center_xy).all() or not np.isfinite([x_size, y_size, yaw]).all():
        return
    if x_size <= 0.0 or y_size <= 0.0:
        return

    corners = _box_corners_xy(center_xy, x_size, y_size, yaw)
    polygon = Polygon(
        _xy_to_plot(corners),
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(polygon)

    front_xy = center_xy + np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64) * (x_size / 2.0)
    _plot_xy_path(
        ax,
        np.stack([center_xy, front_xy], axis=0),
        color=edgecolor,
        alpha=min(1.0, alpha + 0.35),
        linewidth=max(0.75, linewidth),
        linestyle="-",
        zorder=zorder + 1,
    )


def _box_corners_xy(center_xy: np.ndarray, x_size: float, y_size: float, yaw: float) -> np.ndarray:
    half_x = x_size / 2.0
    half_y = y_size / 2.0
    local = np.array(
        [
            [half_x, half_y],
            [half_x, -half_y],
            [-half_x, -half_y],
            [-half_x, half_y],
        ],
        dtype=np.float64,
    )
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return center_xy.reshape(1, 2) + local @ rotation.T


def _plot_xy_path(
    ax: plt.Axes,
    xy: Any,
    *,
    prepend_origin: bool = False,
    **plot_kwargs: Any,
) -> None:
    """Plot display ego XY points with Matplotlib axes [y_left, x_forward]."""

    points = _ensure_xy_array(xy, "xy path")
    if points.shape[0] == 0:
        return
    if prepend_origin:
        points = np.concatenate([np.zeros((1, 2), dtype=np.float64), points], axis=0)
    if points.shape[0] == 1:
        ax.scatter(points[:, 1], points[:, 0], **_scatter_kwargs(plot_kwargs))
        return
    ax.plot(points[:, 1], points[:, 0], **plot_kwargs)


def _ensure_xy_array(value: Any, name: str) -> np.ndarray:
    if value is None:
        return _EMPTY_XY.copy()
    array = _to_numpy(value, name)
    if array.size == 0:
        return _EMPTY_XY.copy()
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must be shaped [N, >=2], got {array.shape}.")
    points = array[:, :2].astype(np.float64, copy=False)
    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.any():
        return _EMPTY_XY.copy()
    if not finite_mask.all():
        points = points[finite_mask]
    return points


def _raw_lidar_xy_to_display_ego_xy(value: Any, name: str) -> np.ndarray:
    """Convert raw LIDAR_TOP/SparseDrive XY to display [x_forward, y_left]."""

    points = _ensure_xy_array(value, name)
    converted = np.empty_like(points, dtype=np.float64)
    converted[:, 0] = points[:, 1]
    converted[:, 1] = -points[:, 0]
    return converted


def _raw_lidar_yaw_to_display_ego_yaw(yaw: Any) -> Any:
    """Convert raw LIDAR_TOP/SparseDrive yaw to display ego-frame yaw."""

    return _normalize_angle(np.asarray(yaw, dtype=np.float64) + np.pi / 2.0)


def _ensure_trajectory_batch(value: Any, name: str, *, expected_count: int) -> np.ndarray:
    if value is None:
        return np.empty((0, 0, 2), dtype=np.float64)
    array = _to_numpy(value, name)
    if array.size == 0:
        return np.empty((0, 0, 2), dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] < 2:
        raise ValueError(f"{name} must be shaped [N, T, >=2], got {array.shape}.")
    if array.shape[0] != expected_count:
        raise ValueError(
            f"{name} first dimension ({array.shape[0]}) must match "
            f"predicted_obstacle_boxes count ({expected_count})."
        )
    return array[..., :2].astype(np.float64, copy=False)


def _optional_1d_array(value: Any, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = _to_numpy(value, name)
    if array.size == 0:
        return np.empty((0,), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    return array.astype(np.float64, copy=False)


def _iter_polylines(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError("predicted_map_polylines must be a sequence of arrays, not a string.")
    if hasattr(value, "detach") and callable(value.detach):
        value = _to_numpy(value, "predicted_map_polylines")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.ndim != 3 or value.shape[-1] < 2:
            raise ValueError(f"predicted_map_polylines must be shaped [M, P, >=2], got {value.shape}.")
        return [value[index] for index in range(value.shape[0])]
    if isinstance(value, Sequence):
        return list(value)
    raise TypeError(f"predicted_map_polylines must be a sequence, got {type(value).__name__}.")


def _iter_current_annotations(annotations: Any) -> list[Any]:
    if annotations is None:
        return []
    if isinstance(annotations, Mapping) and "gt_boxes" in annotations:
        boxes = _to_numpy(annotations["gt_boxes"], "current_annotations['gt_boxes']")
        if boxes.size == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] < 7:
            raise ValueError(
                "current_annotations['gt_boxes'] must be shaped [N, >=7], "
                f"got {boxes.shape}."
            )
        return [{"gt_box": boxes[index]} for index in range(boxes.shape[0])]
    if isinstance(annotations, Mapping):
        return list(annotations.values())
    if isinstance(annotations, (str, bytes)):
        raise TypeError("current_annotations must be a sequence of records, not a string.")
    return list(annotations)


def _annotation_box_components(annotation: Any, index: int) -> Optional[tuple[np.ndarray, float, float, float]]:
    gt_box = _get_optional_field(annotation, "gt_box", "box", "bbox")
    if gt_box is not None:
        box = _to_numpy(gt_box, f"current_annotations[{index}].gt_box").astype(np.float64, copy=False)
        if box.shape[0] < 7:
            raise ValueError(f"current_annotations[{index}].gt_box must have at least 7 values, got {box.shape}.")
        if not np.isfinite(box[:7]).all():
            return None
        return box[:2], max(float(box[3]), 0.05), max(float(box[4]), 0.05), float(box[6])

    center_value = _get_optional_field(annotation, "center", "translation", "center_xy")
    size_value = _get_optional_field(annotation, "size", "wlh", "box_size")
    if center_value is None or size_value is None:
        return None

    center = _to_numpy(center_value, f"current_annotations[{index}].center").astype(np.float64, copy=False).reshape(-1)
    size = _to_numpy(size_value, f"current_annotations[{index}].size").astype(np.float64, copy=False).reshape(-1)
    if center.shape[0] < 2 or size.shape[0] < 2:
        raise ValueError(
            f"current_annotations[{index}] center/size must contain at least two values, "
            f"got {center.shape} and {size.shape}."
        )
    yaw_value = _get_optional_field(annotation, "yaw", "heading")
    yaw = 0.0 if yaw_value is None else float(np.asarray(yaw_value, dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(center[:2]).all() or not np.isfinite(size[:2]).all() or not np.isfinite(yaw):
        return None

    # nuScenes annotation sizes are [width, length, height]; x_size is the box length
    # along its converted heading and y_size is its lateral width.
    y_size = max(float(size[0]), 0.05)
    x_size = max(float(size[1]), 0.05)
    return center[:2], x_size, y_size, yaw


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    if value is None:
        raise ValueError(f"{name} is None.")
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to a numpy array.") from exc


def _get_optional_field(source: Any, *names: str) -> Any:
    for name in names:
        if isinstance(source, Mapping):
            if name in source:
                return source[name]
        elif hasattr(source, name):
            return getattr(source, name)
    return None


def _xy_to_plot(points_xy: np.ndarray) -> np.ndarray:
    """Map display [x_forward, y_left] points to Matplotlib [x_axis, y_axis]."""

    return np.column_stack([points_xy[:, 1], points_xy[:, 0]])


def _scatter_kwargs(plot_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "color" in plot_kwargs:
        kwargs["c"] = plot_kwargs["color"]
    if "alpha" in plot_kwargs:
        kwargs["alpha"] = plot_kwargs["alpha"]
    if "marker" in plot_kwargs:
        kwargs["marker"] = plot_kwargs["marker"]
    if "markersize" in plot_kwargs:
        kwargs["s"] = float(plot_kwargs["markersize"]) ** 2
    if "zorder" in plot_kwargs:
        kwargs["zorder"] = plot_kwargs["zorder"]
    return kwargs


def _normalize_angle(angle: Any) -> Any:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _configure_bev_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(DEFAULT_LATERAL_EXTENT_METERS, -DEFAULT_LATERAL_EXTENT_METERS)
    ax.set_ylim(-DEFAULT_REAR_EXTENT_METERS, DEFAULT_FORWARD_EXTENT_METERS)
    ax.set_xlabel("y left (m)")
    ax.set_ylabel("x forward (m)")
    ax.grid(True, linewidth=0.35, alpha=0.35)


def _add_metadata_text(
    ax: plt.Axes,
    sample: Any,
    decoded_prediction: DecodedSparseDrivePrediction,
    summary: Optional[Mapping[str, Any]],
) -> None:
    token = _sample_token(sample)
    lines = [f"token={token}"]
    scene_name = getattr(sample, "scene_name", None)
    map_name = getattr(sample, "map_name", None)
    if scene_name is not None:
        lines.append(f"scene={scene_name}")
    if map_name is not None:
        lines.append(f"map={map_name}")
    planning = f"planning={decoded_prediction.planning_source}"
    if decoded_prediction.planning_command_index is not None:
        planning += f" cmd={decoded_prediction.planning_command_index}"
    if decoded_prediction.planning_mode_index is not None:
        planning += f" mode={decoded_prediction.planning_mode_index}"
    lines.append(planning)
    lines.append(f"obstacles={_safe_len(decoded_prediction.predicted_obstacle_boxes)}")
    lines.append(f"map_polylines={len(_iter_polylines(decoded_prediction.predicted_map_polylines))}")
    if summary is not None:
        config_name = summary.get("config_name")
        if config_name is not None:
            lines.append(f"config={config_name}")
    ax.text(
        0.01,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.0},
        zorder=20,
    )


def _add_prediction_legend(ax: plt.Axes) -> None:
    handles: list[Any] = [
        Patch(
            facecolor=EGO_BOX_CONFIG["facecolor"],
            edgecolor=EGO_BOX_CONFIG["edgecolor"],
            alpha=EGO_BOX_CONFIG["alpha"],
            label="ego",
        ),
        Line2D(
            [0],
            [0],
            color=PREDICTED_EGO_TRAJECTORY_CONFIG["color"],
            linewidth=PREDICTED_EGO_TRAJECTORY_CONFIG["linewidth"],
            marker=PREDICTED_EGO_TRAJECTORY_CONFIG["marker"],
            markersize=PREDICTED_EGO_TRAJECTORY_CONFIG["markersize"],
            label="pred ego traj",
        ),
        Patch(
            facecolor=PREDICTED_OBSTACLE_CONFIG["facecolor"],
            edgecolor=PREDICTED_OBSTACLE_CONFIG["edgecolor"],
            alpha=PREDICTED_OBSTACLE_CONFIG["alpha"],
            label="pred object box",
        ),
        Line2D(
            [0],
            [0],
            color=PREDICTED_OBSTACLE_TRAJECTORY_CONFIG["color"],
            linewidth=PREDICTED_OBSTACLE_TRAJECTORY_CONFIG["linewidth"],
            marker=PREDICTED_OBSTACLE_TRAJECTORY_CONFIG["marker"],
            markersize=PREDICTED_OBSTACLE_TRAJECTORY_CONFIG["markersize"],
            label="pred object traj",
        ),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=MAP_COLORS[class_index % len(MAP_COLORS)],
            linewidth=PREDICTED_MAP_CONFIG["linewidth"],
            linestyle=PREDICTED_MAP_CONFIG["linestyle"],
            label=f"pred {class_name}",
        )
        for class_index, class_name in enumerate(PREDICTED_MAP_CLASS_NAMES)
    )
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.82,
        facecolor="white",
        edgecolor="none",
    )


def _safe_len(value: Any) -> int:
    try:
        return int(len(value))
    except TypeError:
        return 0


def _resolve_decoded_predictions(
    samples: Sequence[Any],
    outputs: Optional[Sequence[Mapping[str, Any]]],
    decoded_predictions: Optional[Sequence[DecodedSparseDrivePrediction]],
) -> list[DecodedSparseDrivePrediction]:
    if decoded_predictions is not None:
        predictions = list(decoded_predictions)
    else:
        if outputs is None:
            raise ValueError("Either decoded_predictions or outputs must be provided.")
        predictions = [decode_sparsedrive_outputs(outputs, sample_index=index) for index in range(len(samples))]

    if len(predictions) != len(samples):
        raise ValueError(f"Expected {len(samples)} decoded predictions, got {len(predictions)}.")
    return predictions


def _normalize_summaries(
    summaries: Optional[Sequence[Mapping[str, Any]]],
    expected_count: int,
) -> list[Optional[Mapping[str, Any]]]:
    if summaries is None:
        return [None] * expected_count
    normalized = list(summaries)
    if len(normalized) != expected_count:
        raise ValueError(f"Expected {expected_count} summaries, got {len(normalized)}.")
    return normalized


def _figure_output_path(sample: Any, output_dir: PathLike, filename: Optional[str]) -> Path:
    directory = Path(os.path.expandvars(os.path.expanduser(os.fspath(output_dir))))
    if filename is None:
        filename = f"{_sanitize_filename(_sample_token(sample))}.png"
    return directory / filename


def _sample_token(sample: Any) -> str:
    if isinstance(sample, Mapping):
        token = sample.get("token") or sample.get("sample_token")
    else:
        token = getattr(sample, "token", None) or getattr(sample, "sample_token", None)
    return str(token or "sample")


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "sample"


__all__ = [
    "visualize_nuscenes_inference",
    "visualize_nuscenes_sparsedrive_predictions",
]
