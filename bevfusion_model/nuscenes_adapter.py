from __future__ import annotations

import json
import os
from collections import defaultdict
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
DEFAULT_TEST_RESIZE = 0.48
DEFAULT_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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
    "sensor",
)


@dataclass(frozen=True)
class BEVFusionNuScenesSample:
    token: str
    img: torch.Tensor  # (N, 3, H, W) float32
    points: torch.Tensor  # (P, 5) float32 [x,y,z,intensity,rel_timestamp] (10-sweep aggregated)
    camera2ego: torch.Tensor  # (N, 4, 4)
    lidar2ego: torch.Tensor  # (4, 4)
    lidar2camera: torch.Tensor  # (N, 4, 4)
    lidar2image: torch.Tensor  # (N, 4, 4)
    camera_intrinsics: torch.Tensor  # (N, 4, 4)
    camera2lidar: torch.Tensor  # (N, 4, 4)
    img_aug_matrix: torch.Tensor  # (N, 4, 4)
    scene_name: str
    timestamp: float
    sample: dict


def _expand_path(path: PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(path))))


def _read_metadata_table(metadata_dir: Path, table_name: str) -> list[dict[str, Any]]:
    path = metadata_dir / f"{table_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required nuScenes metadata file is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"nuScenes metadata file {path} must contain a JSON list, got {type(data).__name__}.")
    return data


def _index_by_token(records: list[dict[str, Any]], table_name: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        token = str(record.get("token", ""))
        if not token:
            raise ValueError(f"nuScenes {table_name} record missing token field")
        if token in index:
            raise ValueError(f"Duplicate token {token!r} found in nuScenes {table_name}.json.")
        index[token] = record
    return index


def _lookup_record(index: dict[str, dict[str, Any]], token: str, table_name: str) -> dict[str, Any]:
    try:
        return index[token]
    except KeyError as exc:
        raise KeyError(f"nuScenes {table_name} record not found for token {token!r}.") from exc


def load_nuscenes_metadata(
    dataset_root: PathLike = DEFAULT_NUSCENES_ROOT,
    version: str = DEFAULT_VERSION,
) -> dict[str, Any]:
    """Load and index nuScenes JSON metadata tables."""
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

    tables = {
        table_name: _read_metadata_table(metadata_dir, table_name)
        for table_name in REQUIRED_METADATA_TABLES
    }

    sample_by_token = _index_by_token(tables["sample"], "sample")
    sample_data_by_token = _index_by_token(tables["sample_data"], "sample_data")
    calibrated_sensor_by_token = _index_by_token(tables["calibrated_sensor"], "calibrated_sensor")
    ego_pose_by_token = _index_by_token(tables["ego_pose"], "ego_pose")
    scene_by_token = _index_by_token(tables["scene"], "scene")
    sensor_by_token = _index_by_token(tables["sensor"], "sensor")

    sample_data_by_sample_channel = _group_sample_data_by_sample_channel(
        sample_data=tables["sample_data"],
        sensor_by_token=sensor_by_token,
        calibrated_sensor_by_token=calibrated_sensor_by_token,
    )

    return {
        "dataset_root": root,
        "version": version,
        "sample_by_token": sample_by_token,
        "sample_data_by_token": sample_data_by_token,
        "calibrated_sensor_by_token": calibrated_sensor_by_token,
        "ego_pose_by_token": ego_pose_by_token,
        "scene_by_token": scene_by_token,
        "sensor_by_token": sensor_by_token,
        "sample_data_by_sample_channel": sample_data_by_sample_channel,
    }


def _group_sample_data_by_sample_channel(
    sample_data: list[dict[str, Any]],
    sensor_by_token: dict[str, dict[str, Any]],
    calibrated_sensor_by_token: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group sample_data records by (sample_token, channel), selecting key frames."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in sample_data:
        sample_token = str(record.get("sample_token", ""))
        calibrated_sensor_token = str(record.get("calibrated_sensor_token", ""))

        if not sample_token or not calibrated_sensor_token:
            continue

        calibrated_sensor = calibrated_sensor_by_token.get(calibrated_sensor_token)
        if not calibrated_sensor:
            continue

        sensor_token = str(calibrated_sensor.get("sensor_token", ""))
        sensor = sensor_by_token.get(sensor_token)
        if not sensor:
            continue

        channel = str(sensor.get("channel", "")).upper()
        grouped[(sample_token, channel)].append(record)

    result = {}
    for key, records in grouped.items():
        result[key] = _select_sample_data_record(records)
    return result


def _select_sample_data_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select key frame record, or the closest non-key-frame."""
    if not records:
        raise ValueError("No sample_data records to select from")
    return min(
        records,
        key=lambda r: (
            0 if bool(r.get("is_key_frame", False)) else 1,
            str(r.get("token", "")),
        ),
    )


def quaternion_to_rotation_matrix(q: list | np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = np.array(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")

    norm = np.linalg.norm(q)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid quaternion norm: {norm}")

    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(translation: list | np.ndarray, rotation_quat: list | np.ndarray) -> np.ndarray:
    """Build 4x4 transform from translation + quaternion [w, x, y, z]."""
    translation = np.array(translation, dtype=np.float64)
    rotation = quaternion_to_rotation_matrix(rotation_quat)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def load_lidar_points(filepath: PathLike) -> np.ndarray:
    """Load .pcd.bin file as (P, 5) array [x, y, z, intensity, ring]."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {filepath}")

    pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    if pts.shape[1] != 5:
        raise ValueError(f"Expected shape (N, 5), got {pts.shape}")
    return pts


def load_lidar_points_multisweep(
    dataset_root: Path,
    lidar_sample_data: dict[str, Any],
    keyframe_lidar_to_global: np.ndarray,
    keyframe_timestamp_s: float,
    metadata: dict[str, Any],
    sweeps_num: int = 9,
) -> np.ndarray:
    """Load the keyframe LiDAR sweep plus prior sweeps, aggregated into the
    keyframe LiDAR frame.

    Mirrors mmdet3d ``LoadPointsFromMultiSweeps`` (the official BEVFusion test
    pipeline uses ``sweeps_num=9`` => 10 sweeps total). Channel 4 of the
    nuScenes ``.pcd.bin`` is the LiDAR *ring* index, but the model's 5-channel
    ``conv_input`` was trained with channel 4 holding a *relative timestamp*
    (0 for the keyframe). We therefore overwrite channel 4 accordingly.

    Args:
        dataset_root: nuScenes root path.
        lidar_sample_data: keyframe LIDAR_TOP ``sample_data`` record.
        keyframe_lidar_to_global: (4, 4) keyframe lidar-to-global transform.
        keyframe_timestamp_s: keyframe timestamp in seconds.
        metadata: indexed nuScenes metadata.
        sweeps_num: number of *prior* sweeps to aggregate (default 9).

    Returns:
        (P, 5) float32 points in the keyframe LiDAR frame, channel 4 = relative
        timestamp (keyframe_ts - sweep_ts).
    """
    sample_data_by_token = metadata["sample_data_by_token"]
    calibrated_sensor_by_token = metadata["calibrated_sensor_by_token"]
    ego_pose_by_token = metadata["ego_pose_by_token"]

    def _remove_close(pts: np.ndarray, radius: float = 1.0) -> np.ndarray:
        not_close = ~((np.abs(pts[:, 0]) < radius) & (np.abs(pts[:, 1]) < radius))
        return pts[not_close]

    global_to_keyframe_lidar = np.linalg.inv(keyframe_lidar_to_global)

    # Keyframe sweep: channel 4 zeroed (relative timestamp = 0).
    keyframe_path = dataset_root / str(lidar_sample_data.get("filename", ""))
    keyframe_points = load_lidar_points(keyframe_path).copy()
    keyframe_points[:, 4] = 0.0
    sweeps = [keyframe_points]

    # Walk the ``prev`` chain to collect prior (non-key-frame) sweeps.
    current = lidar_sample_data
    for _ in range(sweeps_num):
        prev_token = str(current.get("prev", ""))
        if not prev_token:
            break
        sweep_sd = sample_data_by_token.get(prev_token)
        if not sweep_sd:
            break
        current = sweep_sd

        sweep_filename = str(sweep_sd.get("filename", ""))
        sweep_path = dataset_root / sweep_filename
        if not sweep_path.is_file():
            continue

        sweep_points = _remove_close(load_lidar_points(sweep_path).copy())

        # sweep sensor -> global -> keyframe lidar
        sweep_cs = calibrated_sensor_by_token.get(
            str(sweep_sd.get("calibrated_sensor_token", ""))
        )
        sweep_ep = ego_pose_by_token.get(str(sweep_sd.get("ego_pose_token", "")))
        if not sweep_cs or not sweep_ep:
            continue

        sweep_sensor_to_ego = make_transform(
            sweep_cs.get("translation", [0, 0, 0]),
            sweep_cs.get("rotation", [1, 0, 0, 0]),
        )
        sweep_ego_to_global = make_transform(
            sweep_ep.get("translation", [0, 0, 0]),
            sweep_ep.get("rotation", [1, 0, 0, 0]),
        )
        sensor_to_keyframe_lidar = (
            global_to_keyframe_lidar @ sweep_ego_to_global @ sweep_sensor_to_ego
        )

        xyz = sweep_points[:, :3].astype(np.float64)
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        xyz_keyframe = (sensor_to_keyframe_lidar @ xyz_h.T).T[:, :3]
        sweep_points[:, :3] = xyz_keyframe.astype(np.float32)

        sweep_ts = float(sweep_sd.get("timestamp", 0)) / 1e6
        sweep_points[:, 4] = keyframe_timestamp_s - sweep_ts
        sweeps.append(sweep_points)

    # pad_empty_sweeps: scene-start keyframes have no prior sweeps. The reference
    # pipeline duplicates the keyframe to keep point density consistent.
    if len(sweeps) == 1:
        padded = _remove_close(keyframe_points)
        for _ in range(sweeps_num):
            sweeps.append(padded.copy())

    return np.concatenate(sweeps, axis=0).astype(np.float32)


def load_and_resize_image(
    filepath: PathLike,
    target_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
) -> np.ndarray:
    """Load image and resize to target_hw = (H, W), return as (H, W, 3) uint8."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    with Image.open(filepath) as img:
        img_rgb = img.convert("RGB")
        target_h, target_w = target_hw
        img_resized = img_rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(img_resized, dtype=np.uint8)


def load_and_augment_image(
    filepath: PathLike,
    target_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    resize: float = DEFAULT_TEST_RESIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply BEVFusion test-time ImageAug3D resize+crop and return img_aug_matrix."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    with Image.open(filepath) as img:
        img_rgb = img.convert("RGB")
        orig_w, orig_h = img_rgb.size
        target_h, target_w = target_hw

        resize_dims = (int(orig_w * resize), int(orig_h * resize))
        new_w, new_h = resize_dims
        crop_h = int(new_h) - target_h
        crop_w = int(max(0, new_w - target_w) / 2)
        crop = (crop_w, crop_h, crop_w + target_w, crop_h + target_h)

        img_aug = img_rgb.resize(resize_dims, Image.Resampling.BILINEAR).crop(crop)

    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = resize
    transform[1, 1] = resize
    transform[0, 3] = -crop_w
    transform[1, 3] = -crop_h
    return np.array(img_aug, dtype=np.uint8), transform


def normalize_images(
    images: np.ndarray,
    mean: np.ndarray = DEFAULT_IMAGE_MEAN,
    std: np.ndarray = DEFAULT_IMAGE_STD,
) -> torch.Tensor:
    """Normalize images to (N, 3, H, W) tensor with ImageNet normalization."""
    if images.dtype != np.float32:
        images = images.astype(np.float32) / 255.0
    else:
        if images.max() > 1.0:
            images = images / 255.0

    images = (images - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]

    result = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(torch.float32)
    return result


def load_bevfusion_sample(
    token: str,
    metadata: dict[str, Any],
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    camera_order: tuple[str, ...] = DEFAULT_CAMERA_ORDER,
) -> BEVFusionNuScenesSample:
    """Load a single BEVFusion-ready sample."""
    sample = metadata["sample_by_token"].get(token)
    if not sample:
        raise KeyError(f"Sample token not found: {token}")

    sample_data_by_sample_channel = metadata["sample_data_by_sample_channel"]
    calibrated_sensor_by_token = metadata["calibrated_sensor_by_token"]
    ego_pose_by_token = metadata["ego_pose_by_token"]
    scene_by_token = metadata["scene_by_token"]
    dataset_root = metadata["dataset_root"]

    lidar_sample_data = sample_data_by_sample_channel.get((token, DEFAULT_LIDAR_CHANNEL))
    if not lidar_sample_data:
        raise KeyError(f"No LiDAR data for sample {token}")

    lidar_ego_pose_token = str(lidar_sample_data.get("ego_pose_token", ""))
    lidar_ego_pose = ego_pose_by_token.get(lidar_ego_pose_token)
    if not lidar_ego_pose:
        raise KeyError(f"Missing ego_pose for LiDAR: {lidar_ego_pose_token}")

    lidar_calibrated_sensor_token = str(lidar_sample_data.get("calibrated_sensor_token", ""))
    lidar_calibrated_sensor = calibrated_sensor_by_token.get(lidar_calibrated_sensor_token)
    if not lidar_calibrated_sensor:
        raise KeyError(f"Missing calibrated_sensor for LiDAR: {lidar_calibrated_sensor_token}")

    ego_to_global = make_transform(
        lidar_ego_pose.get("translation", [0, 0, 0]),
        lidar_ego_pose.get("rotation", [1, 0, 0, 0]),
    )
    lidar_to_ego = make_transform(
        lidar_calibrated_sensor.get("translation", [0, 0, 0]),
        lidar_calibrated_sensor.get("rotation", [1, 0, 0, 0]),
    )
    lidar2ego_np = lidar_to_ego
    lidar_to_global = ego_to_global @ lidar_to_ego

    images = []
    camera2ego_list = []
    camera_intrinsics_list = []
    lidar2camera_list = []
    lidar2image_list = []
    camera2lidar_list = []
    img_aug_matrix_list = []

    for camera_name in camera_order:
        camera_sample_data = sample_data_by_sample_channel.get((token, camera_name))
        if not camera_sample_data:
            raise KeyError(f"No camera data for {camera_name} in sample {token}")

        camera_ego_pose_token = str(camera_sample_data.get("ego_pose_token", ""))
        camera_ego_pose = ego_pose_by_token.get(camera_ego_pose_token)
        if not camera_ego_pose:
            raise KeyError(f"Missing ego_pose for {camera_name}: {camera_ego_pose_token}")

        camera_calibrated_sensor_token = str(camera_sample_data.get("calibrated_sensor_token", ""))
        camera_calibrated_sensor = calibrated_sensor_by_token.get(camera_calibrated_sensor_token)
        if not camera_calibrated_sensor:
            raise KeyError(f"Missing calibrated_sensor for {camera_name}: {camera_calibrated_sensor_token}")

        camera_ego_to_global = make_transform(
            camera_ego_pose.get("translation", [0, 0, 0]),
            camera_ego_pose.get("rotation", [1, 0, 0, 0]),
        )
        camera_to_ego = make_transform(
            camera_calibrated_sensor.get("translation", [0, 0, 0]),
            camera_calibrated_sensor.get("rotation", [1, 0, 0, 0]),
        )
        camera2ego_list.append(camera_to_ego)

        intrinsic_3x3 = np.array(
            camera_calibrated_sensor.get("camera_intrinsic", np.eye(3)),
            dtype=np.float64,
        )
        if intrinsic_3x3.shape != (3, 3):
            raise ValueError(f"Invalid intrinsic shape for {camera_name}: {intrinsic_3x3.shape}")

        intrinsic_4x4 = np.eye(4, dtype=np.float64)
        intrinsic_4x4[:3, :3] = intrinsic_3x3
        camera_intrinsics_list.append(intrinsic_4x4)

        camera_to_global = camera_ego_to_global @ camera_to_ego
        camera2lidar = np.linalg.inv(lidar_to_global) @ camera_to_global
        lidar2camera = np.linalg.inv(camera2lidar)
        lidar2camera_list.append(lidar2camera)

        camera2lidar_list.append(camera2lidar)

        lidar2image = intrinsic_4x4 @ lidar2camera
        lidar2image_list.append(lidar2image)

        filename = str(camera_sample_data.get("filename", ""))
        image_path = dataset_root / filename
        img_array, img_aug_matrix = load_and_augment_image(image_path, image_hw)
        images.append(img_array)
        img_aug_matrix_list.append(img_aug_matrix)

    stacked_images = np.stack(images, axis=0)
    img_tensor = normalize_images(stacked_images)

    lidar_keyframe_ts = float(lidar_sample_data.get("timestamp", sample.get("timestamp", 0))) / 1e6
    points_array = load_lidar_points_multisweep(
        dataset_root=dataset_root,
        lidar_sample_data=lidar_sample_data,
        keyframe_lidar_to_global=lidar_to_global,
        keyframe_timestamp_s=lidar_keyframe_ts,
        metadata=metadata,
        sweeps_num=9,
    )

    scene_token = str(sample.get("scene_token", ""))
    scene = scene_by_token.get(scene_token, {})
    scene_name = str(scene.get("name", "unknown"))

    timestamp = float(sample.get("timestamp", 0)) / 1e6

    return BEVFusionNuScenesSample(
        token=token,
        img=img_tensor,
        points=torch.from_numpy(points_array).to(torch.float32),
        camera2ego=torch.from_numpy(np.stack(camera2ego_list, axis=0)).to(torch.float32),
        lidar2ego=torch.from_numpy(lidar2ego_np).to(torch.float32),
        lidar2camera=torch.from_numpy(np.stack(lidar2camera_list, axis=0)).to(torch.float32),
        lidar2image=torch.from_numpy(np.stack(lidar2image_list, axis=0)).to(torch.float32),
        camera_intrinsics=torch.from_numpy(np.stack(camera_intrinsics_list, axis=0)).to(torch.float32),
        camera2lidar=torch.from_numpy(np.stack(camera2lidar_list, axis=0)).to(torch.float32),
        img_aug_matrix=torch.from_numpy(np.stack(img_aug_matrix_list, axis=0)).to(torch.float32),
        scene_name=scene_name,
        timestamp=timestamp,
        sample=sample,
    )


def load_bevfusion_samples(
    dataset_root: PathLike = DEFAULT_NUSCENES_ROOT,
    version: str = DEFAULT_VERSION,
    max_samples: Optional[int] = None,
    camera_order: tuple[str, ...] = DEFAULT_CAMERA_ORDER,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    tokens: Optional[list[str]] = None,
) -> list[BEVFusionNuScenesSample]:
    """Load multiple samples."""
    metadata = load_nuscenes_metadata(dataset_root=dataset_root, version=version)

    if tokens is not None:
        selected_tokens = list(tokens)
    else:
        selected_tokens = list(metadata["sample_by_token"].keys())

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive or None, got {max_samples}")
        selected_tokens = selected_tokens[:max_samples]

    samples = []
    for token in selected_tokens:
        sample = load_bevfusion_sample(
            token,
            metadata=metadata,
            image_hw=image_hw,
            camera_order=camera_order,
        )
        samples.append(sample)

    return samples


def collate_bevfusion_samples(samples: list[BEVFusionNuScenesSample]) -> dict[str, Any]:
    """Batch samples into model-ready tensors."""
    if not samples:
        raise ValueError("At least one sample required")

    batch_size = len(samples)
    num_cameras = samples[0].img.shape[0]

    return {
        "img": torch.stack([s.img for s in samples], dim=0),  # (B, N, 3, H, W)
        "points": [s.points for s in samples],  # list of (P_i, 5)
        "camera2ego": torch.stack([s.camera2ego for s in samples], dim=0),  # (B, N, 4, 4)
        "lidar2ego": torch.stack([s.lidar2ego for s in samples], dim=0),  # (B, 4, 4)
        "lidar2camera": torch.stack([s.lidar2camera for s in samples], dim=0),  # (B, N, 4, 4)
        "lidar2image": torch.stack([s.lidar2image for s in samples], dim=0),  # (B, N, 4, 4)
        "camera_intrinsics": torch.stack([s.camera_intrinsics for s in samples], dim=0),  # (B, N, 4, 4)
        "camera2lidar": torch.stack([s.camera2lidar for s in samples], dim=0),  # (B, N, 4, 4)
        "img_aug_matrix": torch.stack([s.img_aug_matrix for s in samples], dim=0),  # (B, N, 4, 4)
        "lidar_aug_matrix": torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1),
        "metas": [
            {"token": s.token, "scene_name": s.scene_name}
            for s in samples
        ],
    }


def sample_to_device(batch: dict[str, Any], device: Union[str, torch.device]) -> dict[str, Any]:
    """Move tensors to device, keeping points as list."""
    result = {}
    for key, value in batch.items():
        if key == "points":
            result[key] = [p.to(device) for p in value]
        elif isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


__all__ = [
    "BEVFusionNuScenesSample",
    "DEFAULT_CAMERA_ORDER",
    "DEFAULT_IMAGE_HW",
    "DEFAULT_IMAGE_MEAN",
    "DEFAULT_IMAGE_STD",
    "DEFAULT_LIDAR_CHANNEL",
    "DEFAULT_NUSCENES_ROOT",
    "DEFAULT_VERSION",
    "VALID_NUSCENES_CAMERA_CHANNELS",
    "collate_bevfusion_samples",
    "load_and_resize_image",
    "load_and_augment_image",
    "load_bevfusion_sample",
    "load_bevfusion_samples",
    "load_lidar_points",
    "load_nuscenes_metadata",
    "make_transform",
    "normalize_images",
    "quaternion_to_rotation_matrix",
    "sample_to_device",
]


if __name__ == "__main__":
    metadata = load_nuscenes_metadata(DEFAULT_NUSCENES_ROOT, DEFAULT_VERSION)
    samples = metadata["sample_by_token"]
    first_token = list(samples.keys())[0]
    sample = load_bevfusion_sample(first_token, metadata)
    print("img shape:", sample.img.shape)
    print("points shape:", sample.points.shape)
    print("camera2ego shape:", sample.camera2ego.shape)
    print("lidar2image shape:", sample.lidar2image.shape)
    print("Success!")
