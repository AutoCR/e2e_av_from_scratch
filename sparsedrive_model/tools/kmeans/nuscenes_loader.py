"""JSON-backed nuScenes helpers for SparseDrive kmeans scripts.

The loader intentionally avoids ``nuscenes-devkit`` and model-inference code.  It
reads nuScenes metadata/map JSON files directly and exposes the small set of
geometry helpers needed by detection, map, motion, and planning anchor scripts.

Coordinate conventions used here:

* raw nuScenes ``LIDAR_TOP`` XY in this project has +y ego-forward and +x
  ego-right;
* SparseDrive/display local XY uses x-forward/y-left;
* therefore ``local_xy = [raw_y, -raw_x]``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

PathLike = Union[str, os.PathLike[str]]

DEFAULT_NUSCENES_ROOT: PathLike = "/Users/chenran/Code/nuscenes/nuscenes"
DEFAULT_NUSCENES_VERSION = "v1.0-mini"
DEFAULT_VERSION = DEFAULT_NUSCENES_VERSION
DEFAULT_LIDAR_CHANNEL = "LIDAR_TOP"

SPARSEDRIVE_CLASS_NAMES: tuple[str, ...] = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

NUSCENES_CATEGORY_TO_SPARSEDRIVE_CLASS: dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}
NUSCENES_CATEGORY_PREFIX_TO_SPARSEDRIVE_CLASS: tuple[tuple[str, str], ...] = (
    ("vehicle.bus.", "bus"),
    ("human.pedestrian.", "pedestrian"),
)

REQUIRED_METADATA_TABLES: tuple[str, ...] = (
    "sample",
    "sample_data",
    "calibrated_sensor",
    "ego_pose",
    "scene",
    "log",
    "sensor",
    "sample_annotation",
    "instance",
    "category",
)

_MAP_POLYGON_LAYER_FIELDS: Mapping[str, str] = {
    "drivable_area": "polygon_tokens",
    "road_segment": "polygon_token",
    "road_block": "polygon_token",
    "lane": "polygon_token",
    "lane_connector": "polygon_token",
    "walkway": "polygon_token",
    "ped_crossing": "polygon_token",
    "stop_line": "polygon_token",
    "carpark_area": "polygon_token",
}
_MAP_LINE_LAYER_FIELDS: Mapping[str, str] = {
    "lane_divider": "line_token",
    "road_divider": "line_token",
    "traffic_light": "line_token",
}
_MAP_CACHE: dict[Path, "NuScenesMapExpansion"] = {}


@dataclass(frozen=True)
class SceneSampleSequence:
    """A scene and its ordered key samples."""

    scene: Mapping[str, Any]
    samples: tuple[Mapping[str, Any], ...]
    map_name: Optional[str]

    @property
    def scene_token(self) -> str:
        return str(self.scene["token"])

    @property
    def scene_name(self) -> Optional[str]:
        name = self.scene.get("name")
        return str(name) if name is not None else None


@dataclass(frozen=True)
class NuScenesKMeansMetadata:
    """Indexed nuScenes JSON metadata."""

    dataset_root: Path
    version: str
    metadata_dir: Path
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    sample_by_token: Mapping[str, Mapping[str, Any]]
    sample_data_by_token: Mapping[str, Mapping[str, Any]]
    calibrated_sensor_by_token: Mapping[str, Mapping[str, Any]]
    ego_pose_by_token: Mapping[str, Mapping[str, Any]]
    scene_by_token: Mapping[str, Mapping[str, Any]]
    log_by_token: Mapping[str, Mapping[str, Any]]
    sensor_by_token: Mapping[str, Mapping[str, Any]]
    sample_annotation_by_token: Mapping[str, Mapping[str, Any]]
    instance_by_token: Mapping[str, Mapping[str, Any]]
    category_by_token: Mapping[str, Mapping[str, Any]]
    sample_data_by_sample_channel: Mapping[tuple[str, str], Mapping[str, Any]]
    sample_data_records_by_sample_channel: Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]]
    sample_data_channel_by_token: Mapping[str, str]
    sample_annotations_by_sample_token: Mapping[str, tuple[Mapping[str, Any], ...]]
    sample_annotation_by_sample_instance: Mapping[tuple[str, str], Mapping[str, Any]]

    @property
    def samples(self) -> tuple[Mapping[str, Any], ...]:
        return self.tables["sample"]

    @property
    def scenes(self) -> tuple[Mapping[str, Any], ...]:
        return self.tables["scene"]

    @property
    def sample_annotations(self) -> tuple[Mapping[str, Any], ...]:
        return self.tables["sample_annotation"]

    def get_sample(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.sample_by_token, token, "sample")

    def get_scene(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.scene_by_token, token, "scene")

    def get_sample_data(self, sample_token: str, channel: str = DEFAULT_LIDAR_CHANNEL) -> Mapping[str, Any]:
        key = (str(sample_token), str(channel).upper())
        try:
            return self.sample_data_by_sample_channel[key]
        except KeyError as exc:
            available = sorted(found_channel for token, found_channel in self.sample_data_by_sample_channel if token == key[0])
            available_text = ", ".join(available) if available else "none"
            raise KeyError(
                f"sample {sample_token!r} is missing sample_data channel {key[1]!r}; "
                f"available channels: {available_text}."
            ) from exc

    def get_sample_annotation(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.sample_annotation_by_token, token, "sample_annotation")

    def get_sample_annotations(self, sample_token: str) -> tuple[Mapping[str, Any], ...]:
        _lookup_record(self.sample_by_token, sample_token, "sample")
        return self.sample_annotations_by_sample_token.get(str(sample_token), ())

    def get_sample_instance_annotation(self, sample_token: str, instance_token: str) -> Optional[Mapping[str, Any]]:
        _lookup_record(self.sample_by_token, sample_token, "sample")
        _lookup_record(self.instance_by_token, instance_token, "instance")
        return self.sample_annotation_by_sample_instance.get((str(sample_token), str(instance_token)))

    def get_calibrated_sensor(self, sample_data_record: Mapping[str, Any]) -> Mapping[str, Any]:
        token = str(_require_field(sample_data_record, "calibrated_sensor_token", "sample_data"))
        return _lookup_record(self.calibrated_sensor_by_token, token, "calibrated_sensor")

    def get_ego_pose(self, sample_data_record: Mapping[str, Any]) -> Mapping[str, Any]:
        token = str(_require_field(sample_data_record, "ego_pose_token", "sample_data"))
        return _lookup_record(self.ego_pose_by_token, token, "ego_pose")

    def get_instance(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.instance_by_token, token, "instance")

    def get_category(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.category_by_token, token, "category")

    def category_name_for_instance(self, instance_token: str) -> str:
        instance = self.get_instance(str(instance_token))
        category_token = str(_require_field(instance, "category_token", "instance"))
        category = self.get_category(category_token)
        return str(_require_field(category, "name", "category"))

    def category_name_for_annotation(self, annotation_record: Mapping[str, Any]) -> str:
        instance_token = str(_require_field(annotation_record, "instance_token", "sample_annotation"))
        return self.category_name_for_instance(instance_token)

    def class_name_for_annotation(self, annotation_record: Mapping[str, Any]) -> Optional[str]:
        return sparsedrive_class_name(self.category_name_for_annotation(annotation_record))

    def sample_scene(self, sample: Union[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        sample_record = _coerce_sample_record(self, sample)
        scene_token = str(_require_field(sample_record, "scene_token", "sample"))
        return self.get_scene(scene_token)

    def sample_map_name(self, sample: Union[str, Mapping[str, Any]]) -> Optional[str]:
        scene = self.sample_scene(sample)
        log_token = scene.get("log_token")
        if log_token is None:
            return None
        log = _lookup_record(self.log_by_token, str(log_token), "log")
        location = log.get("location")
        return str(location) if location is not None and str(location) else None


@dataclass(frozen=True)
class NuScenesAnnotation:
    """Current sample annotation transformed into the current LIDAR_TOP frame."""

    token: str
    sample_token: str
    instance_token: str
    category_token: str
    category_name: str
    class_name: Optional[str]
    center_global: np.ndarray
    center_raw_lidar: np.ndarray
    center_local: np.ndarray
    size: np.ndarray
    yaw_raw_lidar: float
    yaw_local: float
    num_lidar_pts: int
    num_radar_pts: int
    record: Mapping[str, Any]


@dataclass(frozen=True)
class NuScenesMapPolygon:
    token: str
    exterior_xy: np.ndarray
    hole_xys: tuple[np.ndarray, ...]
    record: Mapping[str, Any]


@dataclass(frozen=True)
class NuScenesMapLine:
    token: str
    points_xy: np.ndarray
    record: Mapping[str, Any]


@dataclass(frozen=True)
class NuScenesMapPolygonPrimitive:
    layer_name: str
    token: str
    exterior_xy: np.ndarray
    hole_xys: tuple[np.ndarray, ...]
    record: Mapping[str, Any]


@dataclass(frozen=True)
class NuScenesMapPolyline:
    layer_name: str
    token: str
    points_xy: np.ndarray
    record: Mapping[str, Any]
    source: str


@dataclass(frozen=True)
class NuScenesMapExpansion:
    """Parsed nuScenes map expansion JSON."""

    dataset_root: Path
    map_name: str
    path: Path
    nodes: Mapping[str, np.ndarray]
    polygons_by_token: Mapping[str, NuScenesMapPolygon]
    lines_by_token: Mapping[str, NuScenesMapLine]
    polygon_layers: Mapping[str, tuple[NuScenesMapPolygon, ...]]
    line_layers: Mapping[str, tuple[NuScenesMapLine, ...]]
    lane_centerlines: Mapping[str, NuScenesMapPolyline]
    lane_connector_centerlines: Mapping[str, NuScenesMapPolyline]
    payload: Mapping[str, Any]

    def polygon_primitives(
        self,
        layer_names: Optional[Sequence[str]] = None,
    ) -> tuple[NuScenesMapPolygonPrimitive, ...]:
        layers = _normalize_layer_names(layer_names, self.polygon_layers)
        primitives: list[NuScenesMapPolygonPrimitive] = []
        for layer_name in layers:
            for polygon in self.polygon_layers.get(layer_name, ()):
                primitives.append(
                    NuScenesMapPolygonPrimitive(
                        layer_name=layer_name,
                        token=polygon.token,
                        exterior_xy=polygon.exterior_xy,
                        hole_xys=polygon.hole_xys,
                        record=polygon.record,
                    )
                )
        return tuple(primitives)

    def line_primitives(self, layer_names: Optional[Sequence[str]] = None) -> tuple[NuScenesMapPolyline, ...]:
        layers = _normalize_layer_names(layer_names, self.line_layers)
        primitives: list[NuScenesMapPolyline] = []
        for layer_name in layers:
            for line in self.line_layers.get(layer_name, ()):
                primitives.append(
                    NuScenesMapPolyline(
                        layer_name=layer_name,
                        token=line.token,
                        points_xy=line.points_xy,
                        record=line.record,
                        source="line",
                    )
                )
        return tuple(primitives)

    def lane_centerline_primitives(
        self,
        layer_names: Sequence[str] = ("lane", "lane_connector"),
    ) -> tuple[NuScenesMapPolyline, ...]:
        primitives: list[NuScenesMapPolyline] = []
        if "lane" in layer_names:
            primitives.extend(self.lane_centerlines.values())
        if "lane_connector" in layer_names:
            primitives.extend(self.lane_connector_centerlines.values())
        return tuple(primitives)


__all__ = [
    "DEFAULT_LIDAR_CHANNEL",
    "DEFAULT_NUSCENES_ROOT",
    "DEFAULT_NUSCENES_VERSION",
    "DEFAULT_VERSION",
    "NUSCENES_CATEGORY_PREFIX_TO_SPARSEDRIVE_CLASS",
    "NUSCENES_CATEGORY_TO_SPARSEDRIVE_CLASS",
    "NuScenesAnnotation",
    "NuScenesKMeansMetadata",
    "NuScenesMapExpansion",
    "NuScenesMapLine",
    "NuScenesMapPolygon",
    "NuScenesMapPolygonPrimitive",
    "NuScenesMapPolyline",
    "SPARSEDRIVE_CLASS_NAMES",
    "SceneSampleSequence",
    "annotations_for_sample",
    "future_agent_trajectory",
    "future_ego_trajectory",
    "future_sample_tokens",
    "global_xy_to_sample_local_xy",
    "global_xyz_to_sample_lidar_xyz",
    "interpolate_polyline",
    "iter_scene_sample_sequences",
    "load_map_expansion",
    "load_map_for_sample",
    "load_metadata",
    "local_xy_to_raw_lidar_xy",
    "local_yaw_to_raw_lidar_yaw",
    "map_centerlines_for_sample",
    "map_lines_for_sample",
    "map_names_for_metadata",
    "normalize_angle",
    "raw_lidar_xy_to_local_xy",
    "raw_lidar_xyz_to_local_xyz",
    "raw_lidar_yaw_to_local_yaw",
    "sample_global_to_lidar_transform",
    "sample_lidar_to_global_transform",
    "sample_lidar_xyz_to_global_xyz",
    "sparsedrive_class_name",
]


def sparsedrive_class_name(category_name: str) -> Optional[str]:
    """Return the SparseDrive class for a nuScenes category, or ``None``."""

    name = str(category_name)
    mapped = NUSCENES_CATEGORY_TO_SPARSEDRIVE_CLASS.get(name)
    if mapped is not None:
        return mapped
    for prefix, class_name in NUSCENES_CATEGORY_PREFIX_TO_SPARSEDRIVE_CLASS:
        if name.startswith(prefix):
            return class_name
    return None


def load_metadata(
    dataset_root: PathLike = DEFAULT_NUSCENES_ROOT,
    version: str = DEFAULT_NUSCENES_VERSION,
) -> NuScenesKMeansMetadata:
    """Load and index nuScenes metadata JSON tables."""

    root = _expand_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"nuScenes dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"nuScenes dataset root is not a directory: {root}")

    metadata_dir = root / version
    if not metadata_dir.exists():
        raise FileNotFoundError(f"nuScenes metadata directory does not exist: {metadata_dir}")
    if not metadata_dir.is_dir():
        raise NotADirectoryError(f"nuScenes metadata path is not a directory: {metadata_dir}")

    tables = {name: _read_metadata_table(metadata_dir, name) for name in REQUIRED_METADATA_TABLES}
    sample_by_token = _index_by_token(tables["sample"], "sample")
    sample_data_by_token = _index_by_token(tables["sample_data"], "sample_data")
    calibrated_sensor_by_token = _index_by_token(tables["calibrated_sensor"], "calibrated_sensor")
    ego_pose_by_token = _index_by_token(tables["ego_pose"], "ego_pose")
    scene_by_token = _index_by_token(tables["scene"], "scene")
    log_by_token = _index_by_token(tables["log"], "log")
    sensor_by_token = _index_by_token(tables["sensor"], "sensor")
    sample_annotation_by_token = _index_by_token(tables["sample_annotation"], "sample_annotation")
    instance_by_token = _index_by_token(tables["instance"], "instance")
    category_by_token = _index_by_token(tables["category"], "category")

    sample_data_channel_by_token, grouped_sample_data = _group_sample_data_by_sample_channel(
        sample_data=tables["sample_data"],
        sample_by_token=sample_by_token,
        calibrated_sensor_by_token=calibrated_sensor_by_token,
        sensor_by_token=sensor_by_token,
    )
    selected_sample_data = {
        key: _select_sample_data_record(records, sample_by_token[key[0]], key[1])
        for key, records in grouped_sample_data.items()
    }
    sample_annotations_by_sample_token, sample_annotation_by_sample_instance = _group_sample_annotations_by_sample_and_instance(
        sample_annotations=tables["sample_annotation"],
        sample_by_token=sample_by_token,
        instance_by_token=instance_by_token,
        category_by_token=category_by_token,
    )

    return NuScenesKMeansMetadata(
        dataset_root=root,
        version=version,
        metadata_dir=metadata_dir,
        tables=tables,
        sample_by_token=sample_by_token,
        sample_data_by_token=sample_data_by_token,
        calibrated_sensor_by_token=calibrated_sensor_by_token,
        ego_pose_by_token=ego_pose_by_token,
        scene_by_token=scene_by_token,
        log_by_token=log_by_token,
        sensor_by_token=sensor_by_token,
        sample_annotation_by_token=sample_annotation_by_token,
        instance_by_token=instance_by_token,
        category_by_token=category_by_token,
        sample_data_by_sample_channel=selected_sample_data,
        sample_data_records_by_sample_channel=dict(grouped_sample_data),
        sample_data_channel_by_token=sample_data_channel_by_token,
        sample_annotations_by_sample_token=sample_annotations_by_sample_token,
        sample_annotation_by_sample_instance=sample_annotation_by_sample_instance,
    )


def iter_scene_sample_sequences(metadata: NuScenesKMeansMetadata) -> Iterator[SceneSampleSequence]:
    """Yield ordered key-sample sequences by scene."""

    for scene in metadata.scenes:
        scene_token = str(_require_field(scene, "token", "scene"))
        cursor_token = str(_require_field(scene, "first_sample_token", "scene"))
        last_token = str(scene.get("last_sample_token", ""))
        seen: set[str] = set()
        samples: list[Mapping[str, Any]] = []

        while cursor_token:
            if cursor_token in seen:
                raise ValueError(f"scene {scene_token!r} sample chain contains a cycle at {cursor_token!r}.")
            seen.add(cursor_token)
            sample = metadata.get_sample(cursor_token)
            sample_scene_token = str(_require_field(sample, "scene_token", "sample"))
            if sample_scene_token != scene_token:
                raise ValueError(
                    f"scene {scene_token!r} sample chain reached sample {cursor_token!r} "
                    f"from scene {sample_scene_token!r}."
                )
            samples.append(sample)
            if cursor_token == last_token:
                break
            next_token = str(sample.get("next", ""))
            if not next_token:
                break
            cursor_token = next_token

        yield SceneSampleSequence(scene=scene, samples=tuple(samples), map_name=metadata.sample_map_name(samples[0]) if samples else None)


def sample_lidar_to_global_transform(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    channel: str = DEFAULT_LIDAR_CHANNEL,
) -> np.ndarray:
    """Return the selected sample-data sensor-to-global transform."""

    sample_record = _coerce_sample_record(metadata, sample)
    sample_token = str(_require_field(sample_record, "token", "sample"))
    sample_data = metadata.get_sample_data(sample_token, channel)
    return sample_data_to_global_transform(metadata, sample_data)


def sample_global_to_lidar_transform(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    channel: str = DEFAULT_LIDAR_CHANNEL,
) -> np.ndarray:
    """Return the selected sample-data global-to-sensor transform."""

    return np.linalg.inv(sample_lidar_to_global_transform(metadata, sample, channel=channel))


def sample_data_to_global_transform(
    metadata: NuScenesKMeansMetadata,
    sample_data_record: Mapping[str, Any],
) -> np.ndarray:
    pose = metadata.get_ego_pose(sample_data_record)
    calibration = metadata.get_calibrated_sensor(sample_data_record)
    ego_to_global = _transform_from_translation_quaternion(
        translation=_require_field(pose, "translation", "ego_pose"),
        quaternion=_require_field(pose, "rotation", "ego_pose"),
        context=f"ego_pose {pose.get('token')!r}",
    )
    sensor_to_ego = _transform_from_translation_quaternion(
        translation=_require_field(calibration, "translation", "calibrated_sensor"),
        quaternion=_require_field(calibration, "rotation", "calibrated_sensor"),
        context=f"calibrated_sensor {calibration.get('token')!r}",
    )
    return ego_to_global @ sensor_to_ego


def raw_lidar_xy_to_local_xy(value: Any) -> np.ndarray:
    """Convert raw LIDAR_TOP XY to SparseDrive local [x_forward, y_left]."""

    points = _xy_like(value, "raw_lidar_xy")
    converted = np.empty_like(points, dtype=np.float64)
    converted[..., 0] = points[..., 1]
    converted[..., 1] = -points[..., 0]
    return converted


def local_xy_to_raw_lidar_xy(value: Any) -> np.ndarray:
    """Convert SparseDrive local [x_forward, y_left] to raw LIDAR_TOP XY."""

    points = _xy_like(value, "local_xy")
    converted = np.empty_like(points, dtype=np.float64)
    converted[..., 0] = -points[..., 1]
    converted[..., 1] = points[..., 0]
    return converted


def raw_lidar_xyz_to_local_xyz(value: Any) -> np.ndarray:
    """Convert raw LIDAR_TOP XYZ to local [x_forward, y_left, z]."""

    points = _xyz_like(value, "raw_lidar_xyz")
    converted = np.empty_like(points, dtype=np.float64)
    converted[..., 0] = points[..., 1]
    converted[..., 1] = -points[..., 0]
    converted[..., 2] = points[..., 2]
    return converted


def raw_lidar_yaw_to_local_yaw(yaw: Any) -> Any:
    """Convert raw LIDAR_TOP yaw to local x-forward/y-left yaw."""

    return normalize_angle(np.asarray(yaw, dtype=np.float64) + np.pi / 2.0)


def local_yaw_to_raw_lidar_yaw(yaw: Any) -> Any:
    """Convert local x-forward/y-left yaw to raw LIDAR_TOP yaw."""

    return normalize_angle(np.asarray(yaw, dtype=np.float64) - np.pi / 2.0)


def normalize_angle(angle: Any) -> Any:
    result = np.arctan2(np.sin(angle), np.cos(angle))
    if np.ndim(result) == 0:
        return float(result)
    return result


def global_xyz_to_sample_lidar_xyz(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    global_xyz: Any,
    channel: str = DEFAULT_LIDAR_CHANNEL,
) -> np.ndarray:
    transform = sample_global_to_lidar_transform(metadata, sample, channel=channel)
    return transform_xyz(transform, global_xyz)


def sample_lidar_xyz_to_global_xyz(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    lidar_xyz: Any,
    channel: str = DEFAULT_LIDAR_CHANNEL,
) -> np.ndarray:
    transform = sample_lidar_to_global_transform(metadata, sample, channel=channel)
    return transform_xyz(transform, lidar_xyz)


def global_xy_to_sample_local_xy(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    global_xy: Any,
    channel: str = DEFAULT_LIDAR_CHANNEL,
) -> np.ndarray:
    xy = _xy_like(global_xy, "global_xy")
    if xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    original_shape = xy.shape
    flat = xy.reshape(-1, 2)
    global_xyz = np.column_stack([flat, np.zeros(flat.shape[0], dtype=np.float64)])
    raw_xyz = global_xyz_to_sample_lidar_xyz(metadata, sample, global_xyz, channel=channel)
    local_xy = raw_lidar_xy_to_local_xy(raw_xyz[:, :2])
    return local_xy.reshape(original_shape)


def transform_xyz(transform: Any, xyz: Any) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {matrix.shape}.")
    points = _xyz_like(xyz, "xyz")
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    original_shape = points.shape
    flat = points.reshape(-1, 3)
    homogeneous = np.column_stack([flat, np.ones(flat.shape[0], dtype=np.float64)])
    transformed = (matrix @ homogeneous.T).T[:, :3]
    return transformed.reshape(original_shape)


def annotations_for_sample(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    *,
    mapped_only: bool = True,
) -> tuple[NuScenesAnnotation, ...]:
    """Return current annotations transformed into raw lidar and local frames."""

    sample_record = _coerce_sample_record(metadata, sample)
    sample_token = str(_require_field(sample_record, "token", "sample"))
    global_to_lidar = sample_global_to_lidar_transform(metadata, sample_record)
    annotations: list[NuScenesAnnotation] = []
    for record in metadata.get_sample_annotations(sample_token):
        annotation = _annotation_to_current_lidar(metadata, record, global_to_lidar)
        if mapped_only and annotation.class_name is None:
            continue
        annotations.append(annotation)
    return tuple(annotations)


def future_sample_tokens(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    future_steps: int,
) -> tuple[str, ...]:
    """Return up to ``future_steps`` sample tokens by following ``sample.next``."""

    if future_steps < 0:
        raise ValueError(f"future_steps must be non-negative, got {future_steps}.")
    cursor = _coerce_sample_record(metadata, sample)
    tokens: list[str] = []
    for _ in range(future_steps):
        next_token = str(cursor.get("next", ""))
        if not next_token:
            break
        try:
            cursor = metadata.get_sample(next_token)
        except KeyError:
            break
        tokens.append(next_token)
    return tuple(tokens)


def future_ego_trajectory(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    future_steps: int = 6,
    *,
    include_current: bool = False,
) -> np.ndarray:
    """Return future ego LIDAR origins in current local x-forward/y-left frame."""

    if future_steps < 0:
        raise ValueError(f"future_steps must be non-negative, got {future_steps}.")
    sample_record = _coerce_sample_record(metadata, sample)
    current_global_to_lidar = sample_global_to_lidar_transform(metadata, sample_record)
    points: list[np.ndarray] = []
    if include_current:
        points.append(np.zeros(2, dtype=np.float64))
    cursor = sample_record
    for _ in range(future_steps):
        next_token = str(cursor.get("next", ""))
        if not next_token:
            break
        try:
            future_sample = metadata.get_sample(next_token)
            future_lidar_to_global = sample_lidar_to_global_transform(metadata, future_sample)
        except (KeyError, ValueError):
            break
        future_origin_global = future_lidar_to_global[:3, 3]
        future_origin_raw = transform_xyz(current_global_to_lidar, future_origin_global.reshape(1, 3))[0]
        points.append(raw_lidar_xy_to_local_xy(future_origin_raw[:2]))
        cursor = future_sample
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.stack(points, axis=0).astype(np.float32)


def future_agent_trajectory(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    instance_token: str,
    future_steps: int = 12,
    *,
    include_current: bool = False,
    require_full: bool = False,
) -> Optional[np.ndarray]:
    """Return an agent future path in the current agent-local frame.

    ``None`` is returned when the instance is not annotated in the current sample
    or when ``require_full=True`` and a future match is missing.
    """

    if future_steps < 0:
        raise ValueError(f"future_steps must be non-negative, got {future_steps}.")
    sample_record = _coerce_sample_record(metadata, sample)
    sample_token = str(_require_field(sample_record, "token", "sample"))
    current_record = metadata.get_sample_instance_annotation(sample_token, str(instance_token))
    if current_record is None:
        return None

    current_global_to_lidar = sample_global_to_lidar_transform(metadata, sample_record)
    current_annotation = _annotation_to_current_lidar(metadata, current_record, current_global_to_lidar)
    current_center_local = current_annotation.center_local[:2]
    current_yaw_local = current_annotation.yaw_local
    points: list[np.ndarray] = []
    if include_current:
        points.append(np.zeros(2, dtype=np.float64))

    cursor = sample_record
    for _ in range(future_steps):
        next_token = str(cursor.get("next", ""))
        if not next_token:
            break
        try:
            future_sample = metadata.get_sample(next_token)
        except KeyError:
            break
        future_record = metadata.get_sample_instance_annotation(next_token, str(instance_token))
        if future_record is None:
            break
        future_center_raw = _annotation_center_in_current_lidar(future_record, current_global_to_lidar)
        future_center_local = raw_lidar_xy_to_local_xy(future_center_raw[:2])
        delta_local = future_center_local - current_center_local
        points.append(_rotate_xy(delta_local.reshape(1, 2), -current_yaw_local)[0])
        cursor = future_sample

    required_count = future_steps + (1 if include_current else 0)
    if require_full and len(points) < required_count:
        return None
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.stack(points, axis=0).astype(np.float32)


def load_map_expansion(
    dataset_root: PathLike,
    map_name: str,
    *,
    use_cache: bool = True,
) -> NuScenesMapExpansion:
    """Load a nuScenes ``maps/expansion/{map_name}.json`` file."""

    root = _expand_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"nuScenes dataset root does not exist: {root}")
    normalized_name = _validate_map_name(map_name)
    map_path = root / "maps" / "expansion" / f"{normalized_name}.json"
    if not map_path.is_file():
        raise FileNotFoundError(f"nuScenes map expansion JSON not found: {map_path}")
    cache_key = map_path.expanduser().resolve()
    if use_cache and cache_key in _MAP_CACHE:
        return _MAP_CACHE[cache_key]
    loaded = _read_map_expansion(root, normalized_name, cache_key)
    if use_cache:
        _MAP_CACHE[cache_key] = loaded
    return loaded


def load_map_for_sample(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    *,
    use_cache: bool = True,
) -> Optional[NuScenesMapExpansion]:
    map_name = metadata.sample_map_name(sample)
    if map_name is None:
        return None
    return load_map_expansion(metadata.dataset_root, map_name, use_cache=use_cache)


def map_names_for_metadata(metadata: NuScenesKMeansMetadata) -> tuple[str, ...]:
    names: set[str] = set()
    for scene in metadata.scenes:
        log_token = scene.get("log_token")
        if log_token is None:
            continue
        log = _lookup_record(metadata.log_by_token, str(log_token), "log")
        location = log.get("location")
        if location is not None and str(location):
            names.add(str(location))
    return tuple(sorted(names))


def map_centerlines_for_sample(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    *,
    radius: Optional[float] = None,
    layer_names: Sequence[str] = ("lane", "lane_connector"),
    local: bool = True,
) -> tuple[NuScenesMapPolyline, ...]:
    expansion = load_map_for_sample(metadata, sample)
    if expansion is None:
        return ()
    primitives = expansion.lane_centerline_primitives(layer_names=layer_names)
    return _map_polylines_for_sample(metadata, sample, primitives, radius=radius, local=local)


def map_lines_for_sample(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    *,
    radius: Optional[float] = None,
    layer_names: Optional[Sequence[str]] = None,
    local: bool = True,
) -> tuple[NuScenesMapPolyline, ...]:
    expansion = load_map_for_sample(metadata, sample)
    if expansion is None:
        return ()
    primitives = expansion.line_primitives(layer_names=layer_names)
    return _map_polylines_for_sample(metadata, sample, primitives, radius=radius, local=local)


def interpolate_polyline(points_xy: Any, num_points: int) -> np.ndarray:
    """Sample ``num_points`` points uniformly along a polyline."""

    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}.")
    points = _xy_like(points_xy, "points_xy")
    if points.size == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    points = points.reshape(-1, 2)
    if points.shape[0] == 1:
        return np.repeat(points.astype(np.float32), num_points, axis=0)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-9 or not np.isfinite(total):
        return np.repeat(points[:1].astype(np.float32), num_points, axis=0)
    targets = np.linspace(0.0, total, num_points)
    sampled = np.column_stack(
        [
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        ]
    )
    return sampled.astype(np.float32)


def _map_polylines_for_sample(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
    primitives: Sequence[NuScenesMapPolyline],
    *,
    radius: Optional[float],
    local: bool,
) -> tuple[NuScenesMapPolyline, ...]:
    if radius is not None and radius < 0:
        raise ValueError(f"radius must be non-negative when provided, got {radius}.")
    result: list[NuScenesMapPolyline] = []
    for primitive in primitives:
        points = primitive.points_xy
        if local:
            points = global_xy_to_sample_local_xy(metadata, sample, points)
        if radius is not None:
            if points.size == 0:
                continue
            distances = np.linalg.norm(points.reshape(-1, 2), axis=1)
            if float(np.min(distances)) > radius:
                continue
        result.append(
            NuScenesMapPolyline(
                layer_name=primitive.layer_name,
                token=primitive.token,
                points_xy=points.astype(np.float64, copy=False),
                record=primitive.record,
                source=primitive.source,
            )
        )
    return tuple(result)


def _annotation_to_current_lidar(
    metadata: NuScenesKMeansMetadata,
    annotation_record: Mapping[str, Any],
    current_global_to_lidar: np.ndarray,
) -> NuScenesAnnotation:
    token = str(_require_field(annotation_record, "token", "sample_annotation"))
    sample_token = str(_require_field(annotation_record, "sample_token", "sample_annotation"))
    instance_token = str(_require_field(annotation_record, "instance_token", "sample_annotation"))
    instance = metadata.get_instance(instance_token)
    category_token = str(_require_field(instance, "category_token", "instance"))
    category_name = metadata.category_name_for_instance(instance_token)
    class_name = sparsedrive_class_name(category_name)
    context = f"sample_annotation {token!r}"

    center_global = _require_float_vector(
        _require_field(annotation_record, "translation", "sample_annotation"),
        3,
        f"{context} translation",
    )
    center_raw_lidar = transform_xyz(current_global_to_lidar, center_global.reshape(1, 3))[0]
    if not np.isfinite(center_raw_lidar).all():
        raise ValueError(f"{context} center in current lidar contains non-finite values.")
    center_local = raw_lidar_xyz_to_local_xyz(center_raw_lidar)
    size = _require_float_vector(
        _require_field(annotation_record, "size", "sample_annotation"),
        3,
        f"{context} size",
    )
    if (size <= 0.0).any():
        raise ValueError(f"{context} size values must be positive [w, l, h], got {size.tolist()}.")
    global_box_rotation = _quaternion_to_rotation_matrix(
        _require_field(annotation_record, "rotation", "sample_annotation"),
        f"{context} rotation",
    )
    lidar_box_rotation = current_global_to_lidar[:3, :3] @ global_box_rotation
    yaw_raw = _yaw_from_rotation_matrix(lidar_box_rotation, f"{context} rotation in current lidar")
    yaw_local = float(raw_lidar_yaw_to_local_yaw(yaw_raw))

    return NuScenesAnnotation(
        token=token,
        sample_token=sample_token,
        instance_token=instance_token,
        category_token=category_token,
        category_name=category_name,
        class_name=class_name,
        center_global=center_global.astype(np.float32),
        center_raw_lidar=center_raw_lidar.astype(np.float32),
        center_local=center_local.astype(np.float32),
        size=size.astype(np.float32),
        yaw_raw_lidar=yaw_raw,
        yaw_local=yaw_local,
        num_lidar_pts=_optional_int(annotation_record.get("num_lidar_pts", 0), f"{context} num_lidar_pts"),
        num_radar_pts=_optional_int(annotation_record.get("num_radar_pts", 0), f"{context} num_radar_pts"),
        record=annotation_record,
    )


def _annotation_center_in_current_lidar(
    annotation_record: Mapping[str, Any],
    current_global_to_lidar: np.ndarray,
) -> np.ndarray:
    token = str(annotation_record.get("token", "<unknown>"))
    center_global = _require_float_vector(
        _require_field(annotation_record, "translation", "sample_annotation"),
        3,
        f"sample_annotation {token!r} translation",
    )
    center_lidar = transform_xyz(current_global_to_lidar, center_global.reshape(1, 3))[0]
    if not np.isfinite(center_lidar).all():
        raise ValueError(f"sample_annotation {token!r} center in current lidar contains non-finite values.")
    return center_lidar.astype(np.float32)


def _read_metadata_table(metadata_dir: Path, table_name: str) -> tuple[Mapping[str, Any], ...]:
    path = metadata_dir / f"{table_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"nuScenes metadata table not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"nuScenes metadata table {path} is not valid JSON: {exc}") from exc
    if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
        raise TypeError(f"nuScenes metadata table {path} must contain a JSON list, got {type(payload).__name__}.")
    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise TypeError(f"nuScenes metadata table {path} record {index} must be an object, got {type(record).__name__}.")
        records.append(record)
    return tuple(records)


def _index_by_token(records: Sequence[Mapping[str, Any]], table_name: str) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for record_index, record in enumerate(records):
        token = str(_require_field(record, "token", table_name))
        if not token:
            raise ValueError(f"{table_name}[{record_index}] has an empty token.")
        if token in index:
            raise ValueError(f"{table_name} contains duplicate token {token!r}.")
        index[token] = record
    return index


def _lookup_record(index: Mapping[str, Mapping[str, Any]], token: str, table_name: str) -> Mapping[str, Any]:
    try:
        return index[str(token)]
    except KeyError as exc:
        raise KeyError(f"unknown {table_name} token {token!r}.") from exc


def _require_field(record: Mapping[str, Any], field: str, table_name: str) -> Any:
    if field not in record:
        token = record.get("token", "<unknown>")
        raise KeyError(f"{table_name} record {token!r} is missing required field {field!r}.")
    return record[field]


def _group_sample_data_by_sample_channel(
    *,
    sample_data: Sequence[Mapping[str, Any]],
    sample_by_token: Mapping[str, Mapping[str, Any]],
    calibrated_sensor_by_token: Mapping[str, Mapping[str, Any]],
    sensor_by_token: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[Mapping[str, Any], ...]]]:
    channel_by_token: dict[str, str] = {}
    grouped_lists: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in sample_data:
        token = str(_require_field(record, "token", "sample_data"))
        sample_token = str(_require_field(record, "sample_token", "sample_data"))
        _lookup_record(sample_by_token, sample_token, "sample")
        calibrated_sensor_token = str(_require_field(record, "calibrated_sensor_token", "sample_data"))
        calibrated_sensor = _lookup_record(calibrated_sensor_by_token, calibrated_sensor_token, "calibrated_sensor")
        sensor_token = str(_require_field(calibrated_sensor, "sensor_token", "calibrated_sensor"))
        sensor = _lookup_record(sensor_by_token, sensor_token, "sensor")
        channel = str(_require_field(sensor, "channel", "sensor")).upper()
        channel_by_token[token] = channel
        grouped_lists[(sample_token, channel)].append(record)
    grouped = {key: tuple(records) for key, records in grouped_lists.items()}
    return channel_by_token, grouped


def _select_sample_data_record(
    records: Sequence[Mapping[str, Any]],
    sample_record: Mapping[str, Any],
    channel: str,
) -> Mapping[str, Any]:
    if not records:
        sample_token = sample_record.get("token", "<unknown>")
        raise KeyError(f"sample {sample_token!r} has no sample_data records for channel {channel!r}.")
    sample_timestamp = _optional_float(sample_record.get("timestamp", 0.0), "sample timestamp")

    def score(record: Mapping[str, Any]) -> tuple[int, float, float]:
        timestamp = _optional_float(record.get("timestamp", sample_timestamp), "sample_data timestamp")
        key_frame_penalty = 0 if bool(record.get("is_key_frame", False)) else 1
        return key_frame_penalty, abs(timestamp - sample_timestamp), timestamp

    return sorted(records, key=score)[0]


def _group_sample_annotations_by_sample_and_instance(
    *,
    sample_annotations: Sequence[Mapping[str, Any]],
    sample_by_token: Mapping[str, Mapping[str, Any]],
    instance_by_token: Mapping[str, Mapping[str, Any]],
    category_by_token: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, tuple[Mapping[str, Any], ...]], dict[tuple[str, str], Mapping[str, Any]]]:
    by_sample_lists: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_sample_instance: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in sample_annotations:
        sample_token = str(_require_field(record, "sample_token", "sample_annotation"))
        instance_token = str(_require_field(record, "instance_token", "sample_annotation"))
        _lookup_record(sample_by_token, sample_token, "sample")
        instance = _lookup_record(instance_by_token, instance_token, "instance")
        category_token = str(_require_field(instance, "category_token", "instance"))
        _lookup_record(category_by_token, category_token, "category")
        by_sample_lists[sample_token].append(record)
        key = (sample_token, instance_token)
        if key in by_sample_instance:
            raise ValueError(
                f"sample {sample_token!r} has multiple annotations for instance {instance_token!r}."
            )
        by_sample_instance[key] = record
    by_sample = {token: tuple(records) for token, records in by_sample_lists.items()}
    return by_sample, by_sample_instance


def _read_map_expansion(dataset_root: Path, map_name: str, map_path: Path) -> NuScenesMapExpansion:
    try:
        with map_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"nuScenes map file {map_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"nuScenes map file {map_path} must contain a JSON object, got {type(payload).__name__}.")

    nodes = _load_map_nodes(payload, map_path)
    polygons = _load_map_polygons(payload, nodes, map_path)
    lines = _load_map_lines(payload, nodes, map_path)
    polygon_layers = {
        layer_name: _resolve_layer_primitives(
            polygons,
            _iter_layer_primitive_tokens(payload, layer_name, token_field, map_path),
            primitive_name="polygon",
            layer_name=layer_name,
            map_path=map_path,
        )
        for layer_name, token_field in _MAP_POLYGON_LAYER_FIELDS.items()
    }
    line_layers = {
        layer_name: _resolve_layer_primitives(
            lines,
            _iter_layer_primitive_tokens(payload, layer_name, token_field, map_path),
            primitive_name="line",
            layer_name=layer_name,
            map_path=map_path,
        )
        for layer_name, token_field in _MAP_LINE_LAYER_FIELDS.items()
    }
    lane_centerlines = _load_lane_centerlines("lane", payload, polygons, map_path)
    lane_connector_centerlines = _load_lane_centerlines("lane_connector", payload, polygons, map_path)
    return NuScenesMapExpansion(
        dataset_root=dataset_root,
        map_name=map_name,
        path=map_path,
        nodes=nodes,
        polygons_by_token=polygons,
        lines_by_token=lines,
        polygon_layers=polygon_layers,
        line_layers=line_layers,
        lane_centerlines=lane_centerlines,
        lane_connector_centerlines=lane_connector_centerlines,
        payload=payload,
    )


def _load_map_nodes(payload: Mapping[str, Any], map_path: Path) -> dict[str, np.ndarray]:
    nodes: dict[str, np.ndarray] = {}
    for index, record in enumerate(_map_record_sequence(payload, "node", map_path)):
        token = _map_token(record, "node", index, map_path)
        if token in nodes:
            raise ValueError(f"nuScenes map file {map_path} has duplicate node token {token!r}.")
        nodes[token] = np.array(
            [
                _map_float_field(record, "x", "node", index, map_path),
                _map_float_field(record, "y", "node", index, map_path),
            ],
            dtype=np.float64,
        )
    if not nodes:
        raise ValueError(f"nuScenes map file {map_path} does not contain any node records.")
    return nodes


def _load_map_polygons(
    payload: Mapping[str, Any],
    node_xy_by_token: Mapping[str, np.ndarray],
    map_path: Path,
) -> dict[str, NuScenesMapPolygon]:
    polygons: dict[str, NuScenesMapPolygon] = {}
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
        polygons[token] = NuScenesMapPolygon(token=token, exterior_xy=exterior_xy, hole_xys=hole_xys, record=record)
    return polygons


def _load_map_lines(
    payload: Mapping[str, Any],
    node_xy_by_token: Mapping[str, np.ndarray],
    map_path: Path,
) -> dict[str, NuScenesMapLine]:
    lines: dict[str, NuScenesMapLine] = {}
    for index, record in enumerate(_map_record_sequence(payload, "line", map_path)):
        if _is_empty_line_placeholder(record):
            continue
        token = _map_token(record, "line", index, map_path)
        if token in lines:
            raise ValueError(f"nuScenes map file {map_path} has duplicate line token {token!r}.")
        node_tokens = _map_token_sequence(record, "node_tokens", "line", index, map_path)
        points_xy = _node_xy_for_tokens(
            node_tokens,
            node_xy_by_token,
            f"nuScenes map file {map_path} line[{index}] node_tokens",
            min_points=2,
        )
        lines[token] = NuScenesMapLine(token=token, points_xy=points_xy, record=record)
    return lines


def _load_lane_centerlines(
    layer_name: str,
    payload: Mapping[str, Any],
    polygons_by_token: Mapping[str, NuScenesMapPolygon],
    map_path: Path,
) -> dict[str, NuScenesMapPolyline]:
    arcline_paths = payload.get("arcline_path_3", {})
    if arcline_paths is None:
        arcline_paths = {}
    if not isinstance(arcline_paths, Mapping):
        raise TypeError(f"nuScenes map file {map_path} field 'arcline_path_3' must be an object when present.")
    centerlines: dict[str, NuScenesMapPolyline] = {}
    for index, record in enumerate(_map_record_sequence(payload, layer_name, map_path, required=False)):
        token = _map_token(record, layer_name, index, map_path)
        points_xy = _centerline_from_arcline_records(arcline_paths.get(token), map_path, layer_name, index)
        if points_xy is None:
            polygon_token = str(record.get("polygon_token", ""))
            polygon = polygons_by_token.get(polygon_token)
            if polygon is None:
                continue
            points_xy = _polygon_representative_vector(polygon.exterior_xy)
        if points_xy.shape[0] < 2:
            continue
        centerlines[token] = NuScenesMapPolyline(
            layer_name=layer_name,
            token=token,
            points_xy=points_xy.astype(np.float64, copy=False),
            record=record,
            source="arcline_path_3" if arcline_paths.get(token) is not None else "polygon_representative",
        )
    return centerlines


def _centerline_from_arcline_records(
    records: Any,
    map_path: Path,
    layer_name: str,
    layer_index: int,
) -> Optional[np.ndarray]:
    if records is None:
        return None
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(
            f"nuScenes map file {map_path} arcline_path_3 for {layer_name}[{layer_index}] "
            f"must be a list, got {type(records).__name__}."
        )
    points: list[np.ndarray] = []
    for segment_index, segment in enumerate(records):
        if not isinstance(segment, Mapping):
            raise TypeError(
                f"nuScenes map file {map_path} arcline_path_3 {layer_name}[{layer_index}][{segment_index}] "
                f"must be an object, got {type(segment).__name__}."
            )
        start_pose = _map_pose_xy(segment, "start_pose", map_path, layer_name, layer_index, segment_index)
        end_pose = _map_pose_xy(segment, "end_pose", map_path, layer_name, layer_index, segment_index)
        if not points or np.linalg.norm(points[-1] - start_pose) > 1e-6:
            points.append(start_pose)
        points.append(end_pose)
    if len(points) < 2:
        return None
    return _dedupe_consecutive_points(np.stack(points, axis=0).astype(np.float64))


def _polygon_representative_vector(exterior_xy: np.ndarray) -> np.ndarray:
    points = _xy_like(exterior_xy, "polygon exterior").reshape(-1, 2)
    if points.shape[0] < 2:
        return points.astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return points[[0, -1]].astype(np.float64)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centered @ direction
    start = points.mean(axis=0) + direction * float(np.min(projections))
    end = points.mean(axis=0) + direction * float(np.max(projections))
    return np.stack([start, end], axis=0).astype(np.float64)


def _map_pose_xy(
    segment: Mapping[str, Any],
    field_name: str,
    map_path: Path,
    layer_name: str,
    layer_index: int,
    segment_index: int,
) -> np.ndarray:
    value = segment.get(field_name)
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"nuScenes map file {map_path} arcline_path_3 {layer_name}[{layer_index}][{segment_index}] "
            f"field {field_name!r} must be a numeric pose."
        ) from exc
    if pose.shape[0] < 2 or not np.isfinite(pose[:2]).all():
        raise ValueError(
            f"nuScenes map file {map_path} arcline_path_3 {layer_name}[{layer_index}][{segment_index}] "
            f"field {field_name!r} must contain finite x/y values."
        )
    return pose[:2].astype(np.float64)


def _dedupe_consecutive_points(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] <= 1:
        return points_xy
    keep = [0]
    for index in range(1, points_xy.shape[0]):
        if np.linalg.norm(points_xy[index] - points_xy[keep[-1]]) > 1e-6:
            keep.append(index)
    return points_xy[keep]


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
            f"nuScenes map file {map_path} table {table_name!r} must be a list, got {type(records).__name__}."
        )
    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"nuScenes map file {map_path} {table_name}[{index}] must be an object, got {type(record).__name__}."
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
            f"nuScenes map file {map_path} {table_name}[{index}] ({token!r}) is missing required field {field_name!r}."
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
            f"nuScenes map file {map_path} {table_name}[{index}] field {field_name!r} must be numeric, got {value!r}."
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"nuScenes map file {map_path} {table_name}[{index}] field {field_name!r} must be finite, got {value!r}."
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
            f"nuScenes map file {map_path} {table_name}[{index}] field {field_name!r} contains an empty token."
        )
    return tokens


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
        if token:
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
                f"nuScenes map file {map_path} layer {layer_name!r} references unknown {primitive_name} token {token!r}."
            ) from exc
    return tuple(primitives)


def _map_holes(record: Mapping[str, Any], polygon_index: int, map_path: Path) -> list[Any]:
    holes = record.get("holes", [])
    if holes is None:
        return []
    if isinstance(holes, (str, bytes)) or not isinstance(holes, Sequence):
        raise TypeError(
            f"nuScenes map file {map_path} polygon[{polygon_index}] field 'holes' must be a list, got {type(holes).__name__}."
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
        raise ValueError(f"nuScenes map file {map_path} polygon[{polygon_index}] holes[{hole_index}] contains an empty token.")
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


def _is_empty_polygon_placeholder(record: Mapping[str, Any]) -> bool:
    return str(record.get("token", "")) == "" and not record.get("exterior_node_tokens") and not record.get("holes")


def _is_empty_line_placeholder(record: Mapping[str, Any]) -> bool:
    return str(record.get("token", "")) == "" and not record.get("node_tokens")


def _normalize_layer_names(layer_names: Optional[Sequence[str]], available: Mapping[str, Any]) -> tuple[str, ...]:
    if layer_names is None:
        return tuple(available.keys())
    return tuple(str(name) for name in layer_names)


def _validate_map_name(map_name: str) -> str:
    normalized = str(map_name).strip()
    if not normalized:
        raise ValueError("map_name must be non-empty.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ValueError(f"map_name contains unsupported characters: {map_name!r}.")
    return normalized


def _expand_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _coerce_sample_record(
    metadata: NuScenesKMeansMetadata,
    sample: Union[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    return metadata.get_sample(str(sample))


def _xy_like(value: Any, name: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a numeric array with last dimension 2.") from exc
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"{name} must have last dimension 2, got shape {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values.")
    return points.astype(np.float64, copy=False)


def _xyz_like(value: Any, name: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a numeric array with last dimension 3.") from exc
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have last dimension 3, got shape {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values.")
    return points.astype(np.float64, copy=False)


def _require_float_vector(value: Any, size: int, context: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be convertible to a float vector of length {size}.") from exc
    if vector.shape != (size,):
        raise ValueError(f"{context} must have shape ({size},), got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{context} contains non-finite values: {value!r}.")
    return vector


def _optional_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be integer-compatible, got {value!r}.") from exc


def _optional_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be numeric, got {value!r}.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite, got {value!r}.")
    return result


def _transform_from_translation_quaternion(
    translation: Sequence[float],
    quaternion: Sequence[float],
    context: str,
) -> np.ndarray:
    translation_np = _require_float_vector(translation, 3, f"{context} translation")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_to_rotation_matrix(quaternion, context)
    transform[:3, 3] = translation_np
    return transform


def _quaternion_to_rotation_matrix(quaternion: Sequence[float], context: str) -> np.ndarray:
    quat = _require_float_vector(quaternion, 4, f"{context} quaternion")
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{context} quaternion has invalid norm {norm}: {quaternion!r}.")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw_from_rotation_matrix(rotation_matrix: np.ndarray, context: str) -> float:
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"{context} must have shape (3, 3), got {rotation.shape}.")
    if not np.isfinite(rotation).all():
        raise ValueError(f"{context} contains non-finite values.")
    return float(normalize_angle(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))))


def _rotate_xy(points_xy: np.ndarray, angle: float) -> np.ndarray:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float64)
    return (rotation @ points_xy.reshape(-1, 2).T).T.reshape(points_xy.shape)
