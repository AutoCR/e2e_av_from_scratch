from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image

PathLike = Union[str, os.PathLike[str]]

DEFAULT_NUSCENES_ROOT: PathLike = "/Users/chenran/Code/nuscenes/nuscenes"
DEFAULT_VERSION = "v1.0-mini"
DEFAULT_IMAGE_HW = (256, 704)
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_NUM_FUTURE_STEPS = 6
DEFAULT_LATERAL_COMMAND_THRESHOLD_METERS = 2.0
DEFAULT_GT_EGO_FUT_CMD = (1.0, 0.0, 0.0)
DEFAULT_LIDAR_CHANNEL = "LIDAR_TOP"

DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

VALID_NUSCENES_CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

REQUIRED_METADATA_TABLES = (
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


@dataclass(frozen=True)
class NuScenesMetadata:
    dataset_root: Path
    version: str
    metadata_dir: Path
    samples: Sequence[Mapping[str, Any]]
    sample_data: Sequence[Mapping[str, Any]]
    calibrated_sensors: Sequence[Mapping[str, Any]]
    ego_poses: Sequence[Mapping[str, Any]]
    scenes: Sequence[Mapping[str, Any]]
    logs: Sequence[Mapping[str, Any]]
    sensors: Sequence[Mapping[str, Any]]
    sample_annotations: Sequence[Mapping[str, Any]]
    instances: Sequence[Mapping[str, Any]]
    categories: Sequence[Mapping[str, Any]]
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
    sample_data_records_by_sample_channel: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]
    sample_data_channel_by_token: Mapping[str, str]
    sample_annotations_by_sample_token: Mapping[str, Sequence[Mapping[str, Any]]]
    sample_annotation_by_sample_instance: Mapping[tuple[str, str], Mapping[str, Any]]

    def get_sample(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.sample_by_token, token, "sample")

    def get_scene(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.scene_by_token, token, "scene")

    def get_sample_data(self, sample_token: str, channel: str) -> Mapping[str, Any]:
        normalized_channel = channel.upper()
        key = (sample_token, normalized_channel)
        try:
            return self.sample_data_by_sample_channel[key]
        except KeyError as exc:
            available = sorted(
                found_channel
                for found_sample_token, found_channel in self.sample_data_by_sample_channel
                if found_sample_token == sample_token
            )
            available_text = ", ".join(available) if available else "none"
            raise KeyError(
                f"Sample {sample_token!r} is missing required sample_data channel "
                f"{normalized_channel!r}; available channels: {available_text}."
            ) from exc

    def get_sample_annotation(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.sample_annotation_by_token, token, "sample_annotation")

    def get_sample_annotations(self, sample_token: str) -> tuple[Mapping[str, Any], ...]:
        _lookup_record(self.sample_by_token, sample_token, "sample")
        return tuple(self.sample_annotations_by_sample_token.get(sample_token, ()))

    def get_sample_instance_annotation(
        self,
        sample_token: str,
        instance_token: str,
    ) -> Optional[Mapping[str, Any]]:
        _lookup_record(self.sample_by_token, sample_token, "sample")
        _lookup_record(self.instance_by_token, instance_token, "instance")
        return self.sample_annotation_by_sample_instance.get((sample_token, instance_token))

    def get_calibrated_sensor(self, sample_data_record: Mapping[str, Any]) -> Mapping[str, Any]:
        token = _require_field(sample_data_record, "calibrated_sensor_token", "sample_data")
        return _lookup_record(self.calibrated_sensor_by_token, str(token), "calibrated_sensor")

    def get_ego_pose(self, sample_data_record: Mapping[str, Any]) -> Mapping[str, Any]:
        token = _require_field(sample_data_record, "ego_pose_token", "sample_data")
        return _lookup_record(self.ego_pose_by_token, str(token), "ego_pose")

    def get_instance(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.instance_by_token, token, "instance")

    def get_category(self, token: str) -> Mapping[str, Any]:
        return _lookup_record(self.category_by_token, token, "category")

    def category_name_for_instance(self, instance_token: str) -> str:
        instance = self.get_instance(instance_token)
        category_token = str(_require_field(instance, "category_token", "instance"))
        category = self.get_category(category_token)
        return str(_require_field(category, "name", "category"))

    def category_name_for_annotation(self, annotation_record: Mapping[str, Any]) -> str:
        instance_token = str(_require_field(annotation_record, "instance_token", "sample_annotation"))
        return self.category_name_for_instance(instance_token)


@dataclass(frozen=True)
class NuScenesLidarAnnotation:
    """Current nuScenes box metadata transformed into raw current LIDAR_TOP.

    In this adapter's nuScenes samples, raw LIDAR_TOP XY uses +x to ego-right
    and +y to ego-forward; visualization converts it to x-forward/y-left.
    """

    token: str
    sample_token: str
    instance_token: str
    category_token: str
    category_name: str
    center: np.ndarray
    size: np.ndarray
    yaw: float
    num_lidar_pts: int
    num_radar_pts: int
    record: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.category_name


@dataclass(frozen=True)
class NuScenesFutureObstacleTrajectory:
    """Matched obstacle center path in raw current LIDAR_TOP for BEV plotting."""

    instance_token: str
    category_token: str
    category_name: str
    current_annotation_token: str
    sample_tokens: tuple[str, ...]
    annotation_tokens: tuple[str, ...]
    points_xy: np.ndarray

    @property
    def name(self) -> str:
        return self.category_name


@dataclass
class SparseDriveNuScenesSample:
    token: str
    img: torch.Tensor
    projection_mat: torch.Tensor
    image_wh: torch.Tensor
    timestamp: torch.Tensor
    img_metas: dict[str, np.ndarray]
    gt_ego_fut_cmd: torch.Tensor
    sample: Mapping[str, Any]
    lidar_sample_data: Mapping[str, Any]
    camera_sample_data: Mapping[str, Mapping[str, Any]]
    future_sample_tokens: tuple[str, ...]
    future_lidar_origins: torch.Tensor
    scene_name: Optional[str]
    map_name: Optional[str]
    current_annotations: Sequence[NuScenesLidarAnnotation]
    future_obstacle_trajectories: Sequence[NuScenesFutureObstacleTrajectory]
    dataset_root: Optional[Path] = None

    def to_model_inputs(self, batched: bool = True) -> dict[str, Any]:
        if batched:
            return collate_nuscenes_sparsedrive_samples([self])
        return {
            "img": self.img,
            "projection_mat": self.projection_mat,
            "image_wh": self.image_wh,
            "timestamp": self.timestamp,
            "img_metas": self.img_metas,
            "gt_ego_fut_cmd": self.gt_ego_fut_cmd,
        }

    @property
    def gt_payload(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "sample": self.sample,
            "lidar_sample_data": self.lidar_sample_data,
            "camera_sample_data": self.camera_sample_data,
            "dataset_root": self.dataset_root,
            "future_sample_tokens": self.future_sample_tokens,
            "future_lidar_origins": self.future_lidar_origins,
            "scene_name": self.scene_name,
            "map_name": self.map_name,
            "current_annotations": self.current_annotations,
            "future_obstacle_trajectories": self.future_obstacle_trajectories,
        }


def image_hw_from_sparsedrive_input_shape(input_shape: Sequence[int]) -> tuple[int, int]:
    try:
        width, height = tuple(int(dim) for dim in input_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SparseDrive input_shape must contain exactly two dimensions in "
            f"(width, height) order, got {input_shape!r}."
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"SparseDrive input_shape dimensions must be positive, got {input_shape!r}.")
    return height, width


def normalize_camera_order(camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER) -> tuple[str, ...]:
    normalized = tuple(str(camera).upper() for camera in camera_order)
    if len(normalized) != 6:
        raise ValueError(f"SparseDrive requires exactly six cameras, got {len(normalized)}: {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Camera order must not contain duplicates: {normalized}")

    invalid = [camera for camera in normalized if camera not in VALID_NUSCENES_CAMERA_CHANNELS]
    if invalid:
        valid = ", ".join(VALID_NUSCENES_CAMERA_CHANNELS)
        raise ValueError(f"Unknown nuScenes camera channel(s) {invalid}; valid channels: {valid}.")
    return normalized


def load_nuscenes_metadata(
    dataset_root: PathLike = DEFAULT_NUSCENES_ROOT,
    version: str = DEFAULT_VERSION,
) -> NuScenesMetadata:
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

    tables = {table_name: _read_metadata_table(metadata_dir, table_name) for table_name in REQUIRED_METADATA_TABLES}

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
        key: _select_sample_data_record(records, sample_by_token[key[0]]) for key, records in grouped_sample_data.items()
    }
    sample_annotations_by_sample_token, sample_annotation_by_sample_instance = (
        _group_sample_annotations_by_sample_and_instance(
            sample_annotations=tables["sample_annotation"],
            sample_by_token=sample_by_token,
            instance_by_token=instance_by_token,
            category_by_token=category_by_token,
        )
    )

    return NuScenesMetadata(
        dataset_root=root,
        version=version,
        metadata_dir=metadata_dir,
        samples=tables["sample"],
        sample_data=tables["sample_data"],
        calibrated_sensors=tables["calibrated_sensor"],
        ego_poses=tables["ego_pose"],
        scenes=tables["scene"],
        logs=tables["log"],
        sensors=tables["sensor"],
        sample_annotations=tables["sample_annotation"],
        instances=tables["instance"],
        categories=tables["category"],
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


def build_nuscenes_sparsedrive_sample(
    metadata: NuScenesMetadata,
    token: str,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    image_mean: Sequence[float] = DEFAULT_IMAGE_MEAN,
    image_std: Sequence[float] = DEFAULT_IMAGE_STD,
    num_future_steps: int = DEFAULT_NUM_FUTURE_STEPS,
    lateral_command_threshold_meters: float = DEFAULT_LATERAL_COMMAND_THRESHOLD_METERS,
) -> SparseDriveNuScenesSample:
    if num_future_steps < 0:
        raise ValueError(f"num_future_steps must be non-negative, got {num_future_steps}.")
    if lateral_command_threshold_meters < 0:
        raise ValueError(
            "lateral_command_threshold_meters must be non-negative, "
            f"got {lateral_command_threshold_meters}."
        )

    sample_record = metadata.get_sample(token)
    lidar_sample_data = metadata.get_sample_data(token, DEFAULT_LIDAR_CHANNEL)
    _require_key_frame_sample_data(lidar_sample_data, DEFAULT_LIDAR_CHANNEL, token)
    lidar_to_global = _sample_data_to_global(metadata, lidar_sample_data)

    img, projection_mat, image_wh, camera_sample_data = _load_camera_tensors(
        metadata=metadata,
        sample_record=sample_record,
        lidar_to_global=lidar_to_global,
        camera_order=camera_order,
        image_hw=image_hw,
        image_mean=image_mean,
        image_std=image_std,
    )

    future_lidar_origins, future_sample_tokens = _future_lidar_origins_in_current_lidar(
        metadata=metadata,
        sample_record=sample_record,
        current_lidar_to_global=lidar_to_global,
        num_future_steps=num_future_steps,
    )
    gt_ego_fut_cmd = _build_gt_ego_fut_cmd(
        future_lidar_origins=future_lidar_origins,
        lateral_command_threshold_meters=lateral_command_threshold_meters,
    )
    current_global_to_lidar = np.linalg.inv(lidar_to_global)
    current_annotations = _current_annotations_in_current_lidar(
        metadata=metadata,
        sample_token=token,
        current_global_to_lidar=current_global_to_lidar,
    )
    future_obstacle_trajectories = _future_obstacle_trajectories_in_current_lidar(
        metadata=metadata,
        current_annotations=current_annotations,
        future_sample_tokens=future_sample_tokens,
        current_global_to_lidar=current_global_to_lidar,
    )
    t_global = lidar_to_global.astype(np.float32)
    t_global_inv = current_global_to_lidar.astype(np.float32)
    scene_name, map_name = _scene_context(metadata, sample_record)

    return SparseDriveNuScenesSample(
        token=token,
        img=img,
        projection_mat=projection_mat,
        image_wh=image_wh,
        timestamp=torch.tensor(_timestamp_seconds(lidar_sample_data["timestamp"]), dtype=torch.float32),
        img_metas={"T_global": t_global, "T_global_inv": t_global_inv},
        gt_ego_fut_cmd=gt_ego_fut_cmd,
        sample=sample_record,
        lidar_sample_data=lidar_sample_data,
        camera_sample_data=camera_sample_data,
        dataset_root=metadata.dataset_root,
        future_sample_tokens=future_sample_tokens,
        future_lidar_origins=future_lidar_origins,
        scene_name=scene_name,
        map_name=map_name,
        current_annotations=current_annotations,
        future_obstacle_trajectories=future_obstacle_trajectories,
    )


def load_nuscenes_sparsedrive_sample(
    token: str,
    dataset_root: PathLike = DEFAULT_NUSCENES_ROOT,
    version: str = DEFAULT_VERSION,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    metadata: Optional[NuScenesMetadata] = None,
) -> SparseDriveNuScenesSample:
    metadata = metadata or load_nuscenes_metadata(dataset_root=dataset_root, version=version)
    return build_nuscenes_sparsedrive_sample(
        metadata=metadata,
        token=token,
        camera_order=camera_order,
        image_hw=image_hw,
    )


def load_nuscenes_sparsedrive_samples(
    dataset_root: PathLike = DEFAULT_NUSCENES_ROOT,
    version: str = DEFAULT_VERSION,
    tokens: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = 1,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    metadata: Optional[NuScenesMetadata] = None,
) -> list[SparseDriveNuScenesSample]:
    metadata = metadata or load_nuscenes_metadata(dataset_root=dataset_root, version=version)
    selected_tokens = list(tokens) if tokens is not None else [str(sample["token"]) for sample in metadata.samples]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive or None, got {max_samples!r}.")
        selected_tokens = selected_tokens[:max_samples]
    return [
        build_nuscenes_sparsedrive_sample(
            metadata=metadata,
            token=token,
            camera_order=camera_order,
            image_hw=image_hw,
        )
        for token in selected_tokens
    ]


def collate_nuscenes_sparsedrive_samples(samples: Sequence[SparseDriveNuScenesSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one SparseDriveNuScenesSample is required to build a batch.")
    return {
        "img": torch.stack([sample.img for sample in samples], dim=0),
        "projection_mat": torch.stack([sample.projection_mat for sample in samples], dim=0),
        "image_wh": torch.stack([sample.image_wh for sample in samples], dim=0),
        "timestamp": torch.stack([sample.timestamp.reshape(()) for sample in samples], dim=0),
        "img_metas": [sample.img_metas for sample in samples],
        "gt_ego_fut_cmd": torch.stack([sample.gt_ego_fut_cmd for sample in samples], dim=0),
    }


def sample_to_device(data: Any, device: Union[str, torch.device], non_blocking: bool = False) -> Any:
    if torch.is_tensor(data):
        return data.to(device=device, non_blocking=non_blocking)
    if isinstance(data, SparseDriveNuScenesSample):
        data.img = sample_to_device(data.img, device, non_blocking)
        data.projection_mat = sample_to_device(data.projection_mat, device, non_blocking)
        data.image_wh = sample_to_device(data.image_wh, device, non_blocking)
        data.timestamp = sample_to_device(data.timestamp, device, non_blocking)
        data.gt_ego_fut_cmd = sample_to_device(data.gt_ego_fut_cmd, device, non_blocking)
        data.future_lidar_origins = sample_to_device(data.future_lidar_origins, device, non_blocking)
        return data
    if isinstance(data, Mapping):
        return {key: sample_to_device(value, device, non_blocking) for key, value in data.items()}
    if isinstance(data, list):
        return [sample_to_device(value, device, non_blocking) for value in data]
    if isinstance(data, tuple):
        return tuple(sample_to_device(value, device, non_blocking) for value in data)
    return data


def _expand_path(path: PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(path))))


def _read_metadata_table(metadata_dir: Path, table_name: str) -> list[Mapping[str, Any]]:
    path = metadata_dir / f"{table_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required nuScenes metadata file is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"nuScenes metadata file {path} must contain a JSON list, got {type(data).__name__}.")
    return data


def _index_by_token(records: Sequence[Mapping[str, Any]], table_name: str) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for record in records:
        token = _require_field(record, "token", table_name)
        token = str(token)
        if token in index:
            raise ValueError(f"Duplicate token {token!r} found in nuScenes {table_name}.json.")
        index[token] = record
    return index


def _lookup_record(index: Mapping[str, Mapping[str, Any]], token: str, table_name: str) -> Mapping[str, Any]:
    try:
        return index[token]
    except KeyError as exc:
        raise KeyError(f"nuScenes {table_name} record not found for token {token!r}.") from exc


def _require_field(record: Mapping[str, Any], field: str, table_name: str) -> Any:
    if field not in record:
        token = record.get("token", "<unknown>")
        raise KeyError(f"nuScenes {table_name} record {token!r} is missing required field {field!r}.")
    return record[field]


def _group_sample_data_by_sample_channel(
    sample_data: Sequence[Mapping[str, Any]],
    sample_by_token: Mapping[str, Mapping[str, Any]],
    calibrated_sensor_by_token: Mapping[str, Mapping[str, Any]],
    sensor_by_token: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[tuple[str, str], list[Mapping[str, Any]]]]:
    channel_by_token: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)

    for record in sample_data:
        sample_token = str(_require_field(record, "sample_token", "sample_data"))
        if sample_token not in sample_by_token:
            token = record.get("token", "<unknown>")
            raise KeyError(f"sample_data record {token!r} references unknown sample token {sample_token!r}.")

        calibrated_sensor_token = str(_require_field(record, "calibrated_sensor_token", "sample_data"))
        calibrated_sensor = _lookup_record(
            calibrated_sensor_by_token,
            calibrated_sensor_token,
            "calibrated_sensor",
        )
        sensor_token = str(_require_field(calibrated_sensor, "sensor_token", "calibrated_sensor"))
        sensor = _lookup_record(sensor_by_token, sensor_token, "sensor")
        channel = str(_require_field(sensor, "channel", "sensor")).upper()

        token = str(_require_field(record, "token", "sample_data"))
        channel_by_token[token] = channel
        grouped[(sample_token, channel)].append(record)

    return channel_by_token, grouped


def _group_sample_annotations_by_sample_and_instance(
    sample_annotations: Sequence[Mapping[str, Any]],
    sample_by_token: Mapping[str, Mapping[str, Any]],
    instance_by_token: Mapping[str, Mapping[str, Any]],
    category_by_token: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[tuple[str, str], Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_sample_instance: dict[tuple[str, str], Mapping[str, Any]] = {}

    for record in sample_annotations:
        token = str(_require_field(record, "token", "sample_annotation"))
        sample_token = str(_require_field(record, "sample_token", "sample_annotation"))
        if sample_token not in sample_by_token:
            raise KeyError(
                f"sample_annotation record {token!r} references unknown sample token {sample_token!r}."
            )

        instance_token = str(_require_field(record, "instance_token", "sample_annotation"))
        instance = _lookup_record(instance_by_token, instance_token, "instance")
        category_token = str(_require_field(instance, "category_token", "instance"))
        _lookup_record(category_by_token, category_token, "category")

        key = (sample_token, instance_token)
        if key in by_sample_instance:
            first_token = by_sample_instance[key].get("token", "<unknown>")
            raise ValueError(
                f"Duplicate sample_annotation records for sample {sample_token!r} "
                f"and instance {instance_token!r}: {first_token!r}, {token!r}."
            )

        grouped[sample_token].append(record)
        by_sample_instance[key] = record

    return dict(grouped), by_sample_instance


def _select_sample_data_record(
    records: Sequence[Mapping[str, Any]],
    sample_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    sample_timestamp = float(_require_field(sample_record, "timestamp", "sample"))
    return min(
        records,
        key=lambda record: (
            0 if bool(record.get("is_key_frame", False)) else 1,
            abs(float(record.get("timestamp", sample_timestamp)) - sample_timestamp),
            float(record.get("timestamp", sample_timestamp)),
            str(record.get("token", "")),
        ),
    )


def _timestamp_seconds(timestamp: Any) -> float:
    value = float(timestamp)
    abs_value = abs(value)
    if abs_value > 1.0e17:
        return value / 1.0e9
    if abs_value > 1.0e12:
        return value / 1.0e6
    return value


def _resize_crop_image(
    image: Image.Image,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
) -> tuple[Image.Image, np.ndarray]:
    target_h, target_w = image_hw
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"image_hw dimensions must be positive, got {image_hw!r}.")

    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = max(target_w, int(math.ceil(src_w * scale)))
    resized_h = max(target_h, int(math.ceil(src_h * scale)))
    crop_left = (resized_w - target_w) // 2
    crop_top = (resized_h - target_h) // 2

    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = image.resize((resized_w, resized_h), resampling)
    cropped = resized.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))

    image_transform = np.array(
        [[scale, 0.0, -float(crop_left)], [0.0, scale, -float(crop_top)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cropped, image_transform @ intrinsics


def _image_to_tensor(
    image: Image.Image,
    mean: Sequence[float] = DEFAULT_IMAGE_MEAN,
    std: Sequence[float] = DEFAULT_IMAGE_STD,
) -> torch.Tensor:
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    image_np = (image_np - mean_np) / std_np
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().to(torch.float32)


def _load_camera_tensors(
    metadata: NuScenesMetadata,
    sample_record: Mapping[str, Any],
    lidar_to_global: np.ndarray,
    camera_order: Sequence[str],
    image_hw: tuple[int, int],
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Mapping[str, Any]]]:
    camera_tensors: list[torch.Tensor] = []
    projection_mats: list[torch.Tensor] = []
    image_wh: list[torch.Tensor] = []
    camera_records: dict[str, Mapping[str, Any]] = {}
    sample_token = str(_require_field(sample_record, "token", "sample"))

    for camera_name in normalize_camera_order(camera_order):
        camera_sample_data = metadata.get_sample_data(sample_token, camera_name)
        _require_key_frame_sample_data(camera_sample_data, camera_name, sample_token)
        camera_calibration = metadata.get_calibrated_sensor(camera_sample_data)
        intrinsics = np.asarray(
            _require_field(camera_calibration, "camera_intrinsic", "calibrated_sensor"),
            dtype=np.float64,
        )
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Camera {camera_name} calibration {camera_calibration.get('token')!r} has "
                f"invalid camera_intrinsic shape {intrinsics.shape}; expected (3, 3)."
            )

        filename = str(_require_field(camera_sample_data, "filename", "sample_data"))
        image_path = metadata.dataset_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Camera image not found for {camera_name} in sample {sample_token}: {image_path}")

        with Image.open(image_path) as raw_image:
            resized_image, adjusted_intrinsics = _resize_crop_image(raw_image.convert("RGB"), intrinsics, image_hw)

        camera_to_global = _sample_data_to_global(metadata, camera_sample_data)
        projection_mat = _projection_from_lidar_to_camera(
            adjusted_intrinsics=adjusted_intrinsics,
            lidar_to_global=lidar_to_global,
            camera_to_global=camera_to_global,
        )

        camera_tensors.append(_image_to_tensor(resized_image, image_mean, image_std))
        projection_mats.append(torch.from_numpy(projection_mat))
        image_wh.append(torch.tensor([image_hw[1], image_hw[0]], dtype=torch.float32))
        camera_records[camera_name] = camera_sample_data

    return (
        torch.stack(camera_tensors, dim=0),
        torch.stack(projection_mats, dim=0),
        torch.stack(image_wh, dim=0),
        camera_records,
    )


def _require_key_frame_sample_data(sample_data_record: Mapping[str, Any], channel: str, sample_token: str) -> None:
    if not bool(sample_data_record.get("is_key_frame", False)):
        raise KeyError(
            f"Sample {sample_token!r} has no key-frame sample_data record for required channel "
            f"{channel!r}; selected non-key-frame record {sample_data_record.get('token')!r}."
        )


def _projection_from_lidar_to_camera(
    adjusted_intrinsics: np.ndarray,
    lidar_to_global: np.ndarray,
    camera_to_global: np.ndarray,
) -> np.ndarray:
    lidar_to_camera = np.linalg.inv(camera_to_global) @ lidar_to_global
    return (adjusted_intrinsics @ lidar_to_camera[:3, :]).astype(np.float32)


def _sample_data_to_global(metadata: NuScenesMetadata, sample_data_record: Mapping[str, Any]) -> np.ndarray:
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


def _current_annotations_in_current_lidar(
    metadata: NuScenesMetadata,
    sample_token: str,
    current_global_to_lidar: np.ndarray,
) -> tuple[NuScenesLidarAnnotation, ...]:
    return tuple(
        _annotation_to_current_lidar(
            metadata=metadata,
            annotation_record=annotation_record,
            current_global_to_lidar=current_global_to_lidar,
        )
        for annotation_record in metadata.get_sample_annotations(sample_token)
    )


def _annotation_to_current_lidar(
    metadata: NuScenesMetadata,
    annotation_record: Mapping[str, Any],
    current_global_to_lidar: np.ndarray,
) -> NuScenesLidarAnnotation:
    token = str(_require_field(annotation_record, "token", "sample_annotation"))
    sample_token = str(_require_field(annotation_record, "sample_token", "sample_annotation"))
    instance_token = str(_require_field(annotation_record, "instance_token", "sample_annotation"))
    instance = metadata.get_instance(instance_token)
    category_token = str(_require_field(instance, "category_token", "instance"))
    category_name = metadata.category_name_for_instance(instance_token)
    context = f"sample_annotation {token!r}"

    center = _annotation_center_in_current_lidar(
        annotation_record=annotation_record,
        current_global_to_lidar=current_global_to_lidar,
    )
    size = _require_float_vector(
        _require_field(annotation_record, "size", "sample_annotation"),
        3,
        f"{context} size",
    ).astype(np.float32)
    if (size <= 0.0).any():
        raise ValueError(f"{context} size values must be positive [w, l, h], got {size.tolist()}.")

    global_box_rotation = _quaternion_to_rotation_matrix(
        _require_field(annotation_record, "rotation", "sample_annotation"),
        f"{context} rotation",
    )
    lidar_box_rotation = current_global_to_lidar[:3, :3] @ global_box_rotation
    yaw = _yaw_from_rotation_matrix(lidar_box_rotation, f"{context} rotation in current lidar")

    return NuScenesLidarAnnotation(
        token=token,
        sample_token=sample_token,
        instance_token=instance_token,
        category_token=category_token,
        category_name=category_name,
        center=center.astype(np.float32),
        size=size,
        yaw=yaw,
        num_lidar_pts=_optional_int(annotation_record.get("num_lidar_pts", 0), f"{context} num_lidar_pts"),
        num_radar_pts=_optional_int(annotation_record.get("num_radar_pts", 0), f"{context} num_radar_pts"),
        record=annotation_record,
    )


def _future_obstacle_trajectories_in_current_lidar(
    metadata: NuScenesMetadata,
    current_annotations: Sequence[NuScenesLidarAnnotation],
    future_sample_tokens: Sequence[str],
    current_global_to_lidar: np.ndarray,
) -> tuple[NuScenesFutureObstacleTrajectory, ...]:
    if not future_sample_tokens:
        return tuple()

    trajectories: list[NuScenesFutureObstacleTrajectory] = []
    for current_annotation in current_annotations:
        points_xy: list[np.ndarray] = [current_annotation.center[:2].astype(np.float32)]
        sample_tokens: list[str] = [current_annotation.sample_token]
        annotation_tokens: list[str] = [current_annotation.token]

        for future_sample_token in future_sample_tokens:
            future_annotation = metadata.get_sample_instance_annotation(
                str(future_sample_token),
                current_annotation.instance_token,
            )
            if future_annotation is None:
                break

            center = _annotation_center_in_current_lidar(
                annotation_record=future_annotation,
                current_global_to_lidar=current_global_to_lidar,
            )
            points_xy.append(center[:2].astype(np.float32))
            sample_tokens.append(str(_require_field(future_annotation, "sample_token", "sample_annotation")))
            annotation_tokens.append(str(_require_field(future_annotation, "token", "sample_annotation")))
        else:
            if len(points_xy) > 1:
                trajectories.append(
                    NuScenesFutureObstacleTrajectory(
                        instance_token=current_annotation.instance_token,
                        category_token=current_annotation.category_token,
                        category_name=current_annotation.category_name,
                        current_annotation_token=current_annotation.token,
                        sample_tokens=tuple(sample_tokens),
                        annotation_tokens=tuple(annotation_tokens),
                        points_xy=np.stack(points_xy, axis=0).astype(np.float32),
                    )
                )

    return tuple(trajectories)


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
    center_homogeneous = np.concatenate([center_global, np.array([1.0], dtype=np.float64)])
    center_lidar = current_global_to_lidar @ center_homogeneous
    center = center_lidar[:3]
    if not np.isfinite(center).all():
        raise ValueError(f"sample_annotation {token!r} center in current lidar contains non-finite values.")
    return center.astype(np.float32)


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
        raise TypeError(f"{context} must be an integer-compatible value, got {value!r}.") from exc


def _transform_from_translation_quaternion(
    translation: Sequence[float],
    quaternion: Sequence[float],
    context: str,
) -> np.ndarray:
    translation_np = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation_np.shape != (3,):
        raise ValueError(f"{context} translation must have shape (3,), got {translation_np.shape}.")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_to_rotation_matrix(quaternion, context)
    transform[:3, 3] = translation_np
    return transform


def _quaternion_to_rotation_matrix(quaternion: Sequence[float], context: str) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"{context} quaternion must have shape (4,) in [w, x, y, z] order, got {quat.shape}.")
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
    return _normalize_angle(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))


def _normalize_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _future_lidar_origins_in_current_lidar(
    metadata: NuScenesMetadata,
    sample_record: Mapping[str, Any],
    current_lidar_to_global: np.ndarray,
    num_future_steps: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    current_global_to_lidar = np.linalg.inv(current_lidar_to_global)
    origins: list[np.ndarray] = []
    future_tokens: list[str] = []
    cursor = sample_record

    for _ in range(num_future_steps):
        next_token = str(cursor.get("next", ""))
        if not next_token:
            break
        future_sample = metadata.get_sample(next_token)
        future_lidar_sample_data = metadata.get_sample_data(next_token, DEFAULT_LIDAR_CHANNEL)
        _require_key_frame_sample_data(future_lidar_sample_data, DEFAULT_LIDAR_CHANNEL, next_token)
        future_lidar_to_global = _sample_data_to_global(metadata, future_lidar_sample_data)
        future_origin_global = np.concatenate([future_lidar_to_global[:3, 3], np.array([1.0], dtype=np.float64)])
        future_origin_current_lidar = current_global_to_lidar @ future_origin_global
        origins.append(future_origin_current_lidar[:3].astype(np.float32))
        future_tokens.append(next_token)
        cursor = future_sample

    if not origins:
        return torch.zeros((0, 3), dtype=torch.float32), tuple()
    return torch.from_numpy(np.stack(origins, axis=0).astype(np.float32)), tuple(future_tokens)


def _build_gt_ego_fut_cmd(
    future_lidar_origins: torch.Tensor,
    lateral_command_threshold_meters: float,
) -> torch.Tensor:
    """Build SparseDrive [straight, left, right] command from raw LIDAR_TOP offsets.

    Raw nuScenes LIDAR_TOP XY here has +x to ego-right and +y to ego-forward, so
    lateral turn direction must be derived from raw x, not raw y.
    """

    if future_lidar_origins.numel() == 0:
        return torch.tensor(DEFAULT_GT_EGO_FUT_CMD, dtype=torch.float32)

    final_right_offset = float(future_lidar_origins[-1, 0].item())
    sparse_idx = (
        1
        if final_right_offset < -lateral_command_threshold_meters
        else 2
        if final_right_offset > lateral_command_threshold_meters
        else 0
    )
    command = np.zeros(3, dtype=np.float32)
    command[sparse_idx] = 1.0
    return torch.from_numpy(command)


def _scene_context(metadata: NuScenesMetadata, sample_record: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    scene_token = str(_require_field(sample_record, "scene_token", "sample"))
    scene = metadata.get_scene(scene_token)
    scene_name = scene.get("name")
    log_token = scene.get("log_token")
    if log_token is None:
        return str(scene_name) if scene_name is not None else None, None
    log = _lookup_record(metadata.log_by_token, str(log_token), "log")
    map_name = log.get("location")
    return (
        str(scene_name) if scene_name is not None else None,
        str(map_name) if map_name is not None else None,
    )


__all__ = [
    "DEFAULT_CAMERA_ORDER",
    "DEFAULT_GT_EGO_FUT_CMD",
    "DEFAULT_IMAGE_HW",
    "DEFAULT_IMAGE_MEAN",
    "DEFAULT_IMAGE_STD",
    "DEFAULT_LATERAL_COMMAND_THRESHOLD_METERS",
    "DEFAULT_LIDAR_CHANNEL",
    "DEFAULT_NUSCENES_ROOT",
    "DEFAULT_NUM_FUTURE_STEPS",
    "DEFAULT_VERSION",
    "NuScenesFutureObstacleTrajectory",
    "NuScenesLidarAnnotation",
    "NuScenesMetadata",
    "SparseDriveNuScenesSample",
    "build_nuscenes_sparsedrive_sample",
    "collate_nuscenes_sparsedrive_samples",
    "image_hw_from_sparsedrive_input_shape",
    "load_nuscenes_metadata",
    "load_nuscenes_sparsedrive_sample",
    "load_nuscenes_sparsedrive_samples",
    "normalize_camera_order",
    "sample_to_device",
]
