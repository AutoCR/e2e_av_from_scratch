from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import navsim.common.dataclasses as navsim_dataclasses
import numpy as np
import torch
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from PIL import Image
from pyquaternion import Quaternion

PathLike = Union[str, os.PathLike[str]]

DEFAULT_OPENSCENE_DATA_ROOT: Optional[PathLike] = None
DEFAULT_NUPLAN_MAPS_ROOT: Optional[PathLike] = None
DEFAULT_SPLIT = "mini"
DEFAULT_NUM_HISTORY_FRAMES = 4
DEFAULT_NUM_FUTURE_FRAMES = 10
# SparseDrive configs store input_shape as (width, height); this adapter uses
# image_hw as (height, width) for PIL/tensor preprocessing.
DEFAULT_IMAGE_HW = (256, 704)
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_MAP_VERSION = "nuplan-maps-v1.0"

VALID_NAVSIM_CAMERA_NAMES = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)

# SparseDrive's original nuScenes recipe uses six cameras; keep that default for
# backward compatibility. The NAVSIM training port can opt into all eight cameras
# via DEFAULT_CAMERA_ORDER_8.
DEFAULT_CAMERA_ORDER = (
    "CAM_F0",  # front
    "CAM_R0",  # front/right side
    "CAM_L0",  # front/left side
    "CAM_B0",  # rear
    "CAM_L2",  # rear/left side
    "CAM_R2",  # rear/right side
)
DEFAULT_CAMERA_ORDER_8 = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_R0",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
)

# SparseDrive's planning branch indexes commands as [straight, left, right].
DEFAULT_GT_EGO_FUT_CMD = (1.0, 0.0, 0.0)
NAVSIM_COMMAND_TO_SPARSEDRIVE = {0: 0, 1: 1, 2: 2, 3: 0}


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


@dataclass(frozen=True)
class NavSimPathConfig:
    openscene_data_root: Path
    nuplan_maps_root: Path


@dataclass
class SparseDriveNavSimSample:
    token: str
    img: torch.Tensor
    projection_mat: torch.Tensor
    image_wh: torch.Tensor
    timestamp: torch.Tensor
    img_metas: dict[str, np.ndarray]
    gt_ego_fut_cmd: torch.Tensor
    scene_frames: Sequence[Mapping[str, Any]]
    current_frame: Mapping[str, Any]
    future_frames: Sequence[Mapping[str, Any]]
    future_ego_trajectory: torch.Tensor
    current_annotations: Optional[Mapping[str, Any]]
    map_api: Any
    map_name: Optional[str]
    ego_pose: np.ndarray

    def to_model_inputs(self, batched: bool = True) -> dict[str, Any]:
        if batched:
            return collate_navsim_sparsedrive_samples([self])
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
            "scene_frames": self.scene_frames,
            "current_frame": self.current_frame,
            "future_ego_trajectory": self.future_ego_trajectory,
            "current_annotations": self.current_annotations,
            "future_frames": self.future_frames,
            "map_api": self.map_api,
            "map_name": self.map_name,
            "ego_pose": self.ego_pose,
        }


def _expand_path(path: PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(path))))


def _resolve_required_path(env_name: str, explicit: Optional[PathLike], default: Optional[PathLike]) -> Path:
    value = explicit if explicit is not None else os.environ.get(env_name)
    if value is None:
        value = default
    if value is None or str(value).strip() == "":
        raise RuntimeError(
            f"{env_name} is required. Set the {env_name} environment variable, "
            "pass it to resolve_navsim_paths/build_navsim_scene_loader, or edit "
            f"DEFAULT_{env_name}."
        )
    path = _expand_path(value)
    if not path.exists():
        raise FileNotFoundError(f"{env_name} path does not exist: {path}")
    return path


def resolve_navsim_paths(
    openscene_data_root: Optional[PathLike] = None,
    nuplan_maps_root: Optional[PathLike] = None,
) -> NavSimPathConfig:
    openscene_root = _resolve_required_path(
        "OPENSCENE_DATA_ROOT", openscene_data_root, DEFAULT_OPENSCENE_DATA_ROOT
    )
    maps_root = _resolve_required_path("NUPLAN_MAPS_ROOT", nuplan_maps_root, DEFAULT_NUPLAN_MAPS_ROOT)
    return NavSimPathConfig(openscene_data_root=openscene_root, nuplan_maps_root=maps_root)


def _set_navsim_maps_root(maps_root: Path) -> None:
    os.environ["NUPLAN_MAPS_ROOT"] = str(maps_root)
    navsim_dataclasses.NUPLAN_MAPS_ROOT = str(maps_root)


def _normalize_camera_name(camera_name: str) -> str:
    normalized = camera_name.upper()
    if not normalized.startswith("CAM_"):
        normalized = f"CAM_{normalized}"
    if normalized not in VALID_NAVSIM_CAMERA_NAMES:
        valid = ", ".join(VALID_NAVSIM_CAMERA_NAMES)
        raise ValueError(f"Unknown NAVSIM camera '{camera_name}'. Valid names: {valid}")
    return normalized


def normalize_camera_order(camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER) -> tuple[str, ...]:
    normalized = tuple(_normalize_camera_name(camera) for camera in camera_order)
    if not (1 <= len(normalized) <= len(VALID_NAVSIM_CAMERA_NAMES)):
        raise ValueError(
            f"SparseDrive requires between 1 and {len(VALID_NAVSIM_CAMERA_NAMES)} cameras, "
            f"got {len(normalized)}: {normalized}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Camera order must not contain duplicates: {normalized}")
    return normalized


def build_navsim_sensor_config(camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER) -> SensorConfig:
    selected = set(normalize_camera_order(camera_order))
    kwargs = {camera.lower(): camera in selected for camera in VALID_NAVSIM_CAMERA_NAMES}
    kwargs["lidar_pc"] = False
    return SensorConfig(**kwargs)


def build_navsim_scene_loader(
    split: str = DEFAULT_SPLIT,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    openscene_data_root: Optional[PathLike] = None,
    nuplan_maps_root: Optional[PathLike] = None,
    num_history_frames: int = DEFAULT_NUM_HISTORY_FRAMES,
    num_future_frames: int = DEFAULT_NUM_FUTURE_FRAMES,
    frame_interval: Optional[int] = None,
    has_route: bool = True,
    max_scenes: Optional[int] = None,
    log_names: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
) -> SceneLoader:
    paths = resolve_navsim_paths(openscene_data_root, nuplan_maps_root)
    _set_navsim_maps_root(paths.nuplan_maps_root)

    data_path = paths.openscene_data_root / "navsim_logs" / split
    sensor_path = paths.openscene_data_root / "sensor_blobs" / split
    if not data_path.exists():
        raise FileNotFoundError(f"NAVSIM log path does not exist: {data_path}")
    if not sensor_path.exists():
        raise FileNotFoundError(f"NAVSIM sensor path does not exist: {sensor_path}")

    scene_filter = SceneFilter(
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        frame_interval=frame_interval,
        has_route=has_route,
        max_scenes=max_scenes,
        log_names=list(log_names) if log_names is not None else None,
        tokens=list(tokens) if tokens is not None else None,
        include_synthetic_scenes=False,
    )
    return SceneLoader(
        data_path=data_path,
        original_sensor_path=sensor_path,
        scene_filter=scene_filter,
        sensor_config=build_navsim_sensor_config(camera_order),
    )


def _frame_pose(frame: Mapping[str, Any]) -> np.ndarray:
    translation = np.asarray(frame["ego2global_translation"][:2], dtype=np.float64)
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def _relative_poses(origin_pose: np.ndarray, global_poses: np.ndarray) -> np.ndarray:
    theta = -origin_pose[2]
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    relative = global_poses - origin_pose[None]
    relative[:, :2] = relative[:, :2] @ rotation.T
    relative[:, 2] = np.arctan2(np.sin(relative[:, 2]), np.cos(relative[:, 2]))
    return relative


def _future_ego_trajectory(
    frame_list: Sequence[Mapping[str, Any]], current_index: int, num_future_frames: int
) -> torch.Tensor:
    future_frames = list(frame_list[current_index + 1 : current_index + 1 + num_future_frames])
    if not future_frames:
        return torch.zeros((0, 3), dtype=torch.float32)
    current_pose = _frame_pose(frame_list[current_index])
    future_global_poses = np.stack([_frame_pose(frame) for frame in future_frames], axis=0)
    return torch.from_numpy(_relative_poses(current_pose, future_global_poses).astype(np.float32))


def _ego_to_global_matrix(frame: Mapping[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(*frame["ego2global_rotation"]).rotation_matrix
    transform[:3, 3] = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    return transform.astype(np.float32)


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


def _camera_projection(camera_info: Mapping[str, Any], intrinsics: np.ndarray) -> np.ndarray:
    sensor2lidar_rotation = np.asarray(camera_info["sensor2lidar_rotation"], dtype=np.float64)
    sensor2lidar_translation = np.asarray(camera_info["sensor2lidar_translation"], dtype=np.float64).reshape(3)
    lidar2sensor_rotation = sensor2lidar_rotation.T
    lidar2sensor_translation = -lidar2sensor_rotation @ sensor2lidar_translation
    lidar2sensor = np.concatenate(
        [lidar2sensor_rotation, lidar2sensor_translation[:, None]],
        axis=1,
    )
    return (intrinsics @ lidar2sensor).astype(np.float32)


def _load_camera_tensors(
    current_frame: Mapping[str, Any],
    sensor_root: Path,
    camera_order: Sequence[str],
    image_hw: tuple[int, int],
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    camera_tensors: list[torch.Tensor] = []
    projection_mats: list[torch.Tensor] = []
    image_wh: list[torch.Tensor] = []
    cameras = current_frame["cams"]

    for camera_name in normalize_camera_order(camera_order):
        if camera_name not in cameras:
            raise KeyError(f"Frame {current_frame.get('token')} is missing camera {camera_name}")
        camera_info = cameras[camera_name]
        image_path = sensor_root / camera_info["data_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Camera image not found for {camera_name}: {image_path}")

        intrinsics = np.asarray(camera_info["cam_intrinsic"], dtype=np.float64)
        if intrinsics.shape != (3, 3):
            intrinsics = intrinsics[:3, :3]

        with Image.open(image_path) as raw_image:
            resized_image, adjusted_intrinsics = _resize_crop_image(raw_image.convert("RGB"), intrinsics, image_hw)
        camera_tensors.append(_image_to_tensor(resized_image, image_mean, image_std))
        projection_mats.append(torch.from_numpy(_camera_projection(camera_info, adjusted_intrinsics)))
        image_wh.append(torch.tensor([image_hw[1], image_hw[0]], dtype=torch.float32))

    return torch.stack(camera_tensors, dim=0), torch.stack(projection_mats, dim=0), torch.stack(image_wh, dim=0)


def _build_gt_ego_fut_cmd(
    current_frame: Mapping[str, Any], future_ego_trajectory: torch.Tensor
) -> torch.Tensor:
    driving_command = current_frame.get("driving_command")
    if driving_command is not None:
        command = np.asarray(driving_command)
        if command.size:
            navsim_idx = int(np.argmax(command))
            sparse_idx = NAVSIM_COMMAND_TO_SPARSEDRIVE.get(navsim_idx, 0)
            result = np.zeros(3, dtype=np.float32)
            result[sparse_idx] = 1.0
            return torch.from_numpy(result)

    if future_ego_trajectory.numel() > 0:
        final_xy = future_ego_trajectory[-1, :2].cpu().numpy()
        sparse_idx = 1 if final_xy[1] > 2.0 else 2 if final_xy[1] < -2.0 else 0
        result = np.zeros(3, dtype=np.float32)
        result[sparse_idx] = 1.0
        return torch.from_numpy(result)

    return torch.tensor(DEFAULT_GT_EGO_FUT_CMD, dtype=torch.float32)


def _load_map_api(map_name: Optional[str], maps_root: Optional[Path]) -> Any:
    if map_name is None or maps_root is None:
        return None
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    return get_maps_api(str(maps_root), DEFAULT_MAP_VERSION, map_name)


def build_navsim_sparsedrive_sample(
    scene_loader: SceneLoader,
    token: str,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    image_mean: Sequence[float] = DEFAULT_IMAGE_MEAN,
    image_std: Sequence[float] = DEFAULT_IMAGE_STD,
    maps_root: Optional[PathLike] = None,
    include_map_api: bool = True,
) -> SparseDriveNavSimSample:
    if token not in scene_loader.scene_frames_dicts:
        raise KeyError(f"Token {token} is not an original raw NAVSIM scene token in scene_loader.scene_frames_dicts")

    frame_list = scene_loader.scene_frames_dicts[token]
    current_index = scene_loader._scene_filter.num_history_frames - 1
    current_frame = frame_list[current_index]
    num_future_frames = scene_loader._scene_filter.num_future_frames
    future_frames = list(frame_list[current_index + 1 : current_index + 1 + num_future_frames])

    sensor_root = Path(scene_loader._original_sensor_path)
    img, projection_mat, image_wh = _load_camera_tensors(
        current_frame=current_frame,
        sensor_root=sensor_root,
        camera_order=camera_order,
        image_hw=image_hw,
        image_mean=image_mean,
        image_std=image_std,
    )

    future_trajectory = _future_ego_trajectory(frame_list, current_index, num_future_frames)
    t_global = _ego_to_global_matrix(current_frame)
    t_global_inv = np.linalg.inv(t_global).astype(np.float32)
    map_name = current_frame.get("map_location")
    maps_root_path = None
    if include_map_api:
        maps_root_path = _resolve_required_path("NUPLAN_MAPS_ROOT", maps_root, DEFAULT_NUPLAN_MAPS_ROOT)
        _set_navsim_maps_root(maps_root_path)

    return SparseDriveNavSimSample(
        token=token,
        img=img,
        projection_mat=projection_mat,
        image_wh=image_wh,
        timestamp=torch.tensor(_timestamp_seconds(current_frame["timestamp"]), dtype=torch.float32),
        img_metas={"T_global": t_global, "T_global_inv": t_global_inv},
        gt_ego_fut_cmd=_build_gt_ego_fut_cmd(current_frame, future_trajectory),
        scene_frames=frame_list,
        current_frame=current_frame,
        future_frames=future_frames,
        future_ego_trajectory=future_trajectory,
        current_annotations=current_frame.get("anns"),
        map_api=_load_map_api(map_name, maps_root_path),
        map_name=map_name,
        ego_pose=_frame_pose(current_frame).astype(np.float32),
    )


def load_navsim_sparsedrive_samples(
    scene_loader: SceneLoader,
    tokens: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = 1,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    maps_root: Optional[PathLike] = None,
    include_map_api: bool = True,
) -> list[SparseDriveNavSimSample]:
    selected_tokens = list(tokens) if tokens is not None else list(scene_loader.tokens)
    if max_samples is not None:
        selected_tokens = selected_tokens[:max_samples]
    return [
        build_navsim_sparsedrive_sample(
            scene_loader=scene_loader,
            token=token,
            camera_order=camera_order,
            image_hw=image_hw,
            maps_root=maps_root,
            include_map_api=include_map_api,
        )
        for token in selected_tokens
    ]


def collate_navsim_sparsedrive_samples(samples: Sequence[SparseDriveNavSimSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one SparseDriveNavSimSample is required to build a batch")
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
    if isinstance(data, SparseDriveNavSimSample):
        data.img = sample_to_device(data.img, device, non_blocking)
        data.projection_mat = sample_to_device(data.projection_mat, device, non_blocking)
        data.image_wh = sample_to_device(data.image_wh, device, non_blocking)
        data.timestamp = sample_to_device(data.timestamp, device, non_blocking)
        data.gt_ego_fut_cmd = sample_to_device(data.gt_ego_fut_cmd, device, non_blocking)
        data.future_ego_trajectory = sample_to_device(data.future_ego_trajectory, device, non_blocking)
        return data
    if isinstance(data, Mapping):
        return {key: sample_to_device(value, device, non_blocking) for key, value in data.items()}
    if isinstance(data, list):
        return [sample_to_device(value, device, non_blocking) for value in data]
    if isinstance(data, tuple):
        return tuple(sample_to_device(value, device, non_blocking) for value in data)
    return data


__all__ = [
    "DEFAULT_CAMERA_ORDER",
    "DEFAULT_CAMERA_ORDER_8",
    "DEFAULT_GT_EGO_FUT_CMD",
    "DEFAULT_IMAGE_HW",
    "NavSimPathConfig",
    "SparseDriveNavSimSample",
    "build_navsim_scene_loader",
    "build_navsim_sensor_config",
    "build_navsim_sparsedrive_sample",
    "collate_navsim_sparsedrive_samples",
    "image_hw_from_sparsedrive_input_shape",
    "load_navsim_sparsedrive_samples",
    "normalize_camera_order",
    "resolve_navsim_paths",
    "sample_to_device",
]
