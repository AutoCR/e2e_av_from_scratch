from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.enums import BoundingBoxIndex
from PIL import Image, ImageDraw
from pyquaternion import Quaternion


NAVSIM_CAMERA_NAMES = (
    "CAM_F0",
    "CAM_R0",
    "CAM_L0",
    "CAM_B0",
    "CAM_L2",
    "CAM_R2",
)

IMG_NORM_CFG = {
    "mean": np.array([103.530, 116.280, 123.675], dtype=np.float32),
    "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "to_rgb": False,
}


@dataclass(frozen=True)
class NavsimUniADConfig:
    dataset_root: Path = Path("/Users/chenran/Code/navsim/dataset")
    split: str = "mini"
    map_version: str = "nuplan-maps-v1.0"
    num_history_frames: int = 4
    num_future_frames: int = 10
    frame_interval: int | None = None
    max_scenes: int | None = 1
    sample_index: int = 0
    image_hw: tuple[int, int] = (128, 224)
    pad_divisor: int = 32
    camera_order: tuple[str, ...] = NAVSIM_CAMERA_NAMES
    bev_hw: tuple[int, int] = (20, 20)
    bev_extent_m: float = 102.4
    planning_steps: int = 6
    occ_num_frames: int = 7

    @property
    def log_path(self) -> Path:
        return self.dataset_root / "navsim_logs" / self.split

    @property
    def sensor_path(self) -> Path:
        return self.dataset_root / "sensor_blobs" / self.split

    @property
    def maps_root(self) -> Path:
        return self.dataset_root / "maps"


@dataclass
class NavsimUniADFrameInput:
    token: str
    frame: Mapping[str, Any]
    img: torch.Tensor
    img_metas: list[list[dict[str, Any]]]
    l2g_t: torch.Tensor
    l2g_r_mat: torch.Tensor
    timestamp: list[torch.Tensor]
    gt_lane_labels: list[torch.Tensor]
    gt_lane_masks: list[torch.Tensor]
    gt_segmentation: list[torch.Tensor]
    gt_instance: list[torch.Tensor]
    gt_occ_img_is_valid: list[torch.Tensor]
    sdc_planning: torch.Tensor
    sdc_planning_mask: torch.Tensor
    command: torch.Tensor

    def to_model_kwargs(self, device: torch.device | str) -> dict[str, Any]:
        return {
            "img": [self.img.to(device)],
            "img_metas": copy.deepcopy(self.img_metas),
            "l2g_t": self.l2g_t.to(device),
            "l2g_r_mat": self.l2g_r_mat.to(device),
            "timestamp": [self.timestamp[0].to(device)],
            "gt_lane_labels": [self.gt_lane_labels[0].to(device)],
            "gt_lane_masks": [self.gt_lane_masks[0].to(device)],
            "gt_segmentation": [self.gt_segmentation[0].to(device)],
            "gt_instance": [self.gt_instance[0].to(device)],
            "gt_occ_img_is_valid": [self.gt_occ_img_is_valid[0].to(device)],
            "sdc_planning": self.sdc_planning.to(device),
            "sdc_planning_mask": self.sdc_planning_mask.to(device),
            "command": self.command.to(device),
            "return_loss": False,
        }


@dataclass
class NavsimUniADSample:
    token: str
    scene_frames: Sequence[Mapping[str, Any]]
    current_index: int
    frame_inputs: list[NavsimUniADFrameInput]
    current_frame: Mapping[str, Any]
    future_frames: Sequence[Mapping[str, Any]]
    ego_future_trajectory: torch.Tensor
    obstacle_future_trajectories: list["ObstacleTrajectory"]
    map_api: Any
    map_name: str

    @property
    def final_input(self) -> NavsimUniADFrameInput:
        return self.frame_inputs[-1]


@dataclass(frozen=True)
class ObstacleTrajectory:
    token: str
    name: str
    points: np.ndarray


class LiDARInstance3DBoxes:
    """Small subset of mmdet3d LiDARInstance3DBoxes used by UniAD inference."""

    def __init__(self, tensor: torch.Tensor | np.ndarray, box_dim: int = 9, origin: tuple[float, float, float] = (0.5, 0.5, 0.5)):
        del origin
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
        if tensor.numel() == 0:
            tensor = tensor.reshape(0, box_dim)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        if tensor.shape[-1] < box_dim:
            padding = tensor.new_zeros((*tensor.shape[:-1], box_dim - tensor.shape[-1]))
            tensor = torch.cat([tensor, padding], dim=-1)
        self.tensor = tensor[..., :box_dim]
        self.box_dim = box_dim

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, item: Any) -> "LiDARInstance3DBoxes":
        return LiDARInstance3DBoxes(self.tensor[item], self.box_dim)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def yaw(self) -> torch.Tensor:
        return self.tensor[:, 6]

    @property
    def gravity_center(self) -> torch.Tensor:
        return self.tensor[:, :3]

    @property
    def bev(self) -> torch.Tensor:
        return self.tensor[:, [0, 1, 3, 4, 6]]

    def to(self, *args: Any, **kwargs: Any) -> "LiDARInstance3DBoxes":
        return LiDARInstance3DBoxes(self.tensor.to(*args, **kwargs), self.box_dim)

    def cpu(self) -> "LiDARInstance3DBoxes":
        return self.to("cpu")

    def clone(self) -> "LiDARInstance3DBoxes":
        return LiDARInstance3DBoxes(self.tensor.clone(), self.box_dim)


def build_scene_loader(config: NavsimUniADConfig) -> SceneLoader:
    if len(config.camera_order) != 6:
        raise ValueError(f"UniAD expects exactly six cameras, got {config.camera_order}")
    if not config.log_path.exists():
        raise FileNotFoundError(f"NAVSIM log path does not exist: {config.log_path}")
    if not config.sensor_path.exists():
        raise FileNotFoundError(f"NAVSIM sensor path does not exist: {config.sensor_path}")

    scene_filter = SceneFilter(
        num_history_frames=config.num_history_frames,
        num_future_frames=config.num_future_frames,
        frame_interval=config.frame_interval,
        has_route=True,
        max_scenes=config.max_scenes,
        include_synthetic_scenes=False,
    )
    return SceneLoader(
        data_path=config.log_path,
        original_sensor_path=config.sensor_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )


def load_navsim_uniad_sample(config: NavsimUniADConfig) -> NavsimUniADSample:
    loader = build_scene_loader(config)
    tokens = loader.tokens
    if not tokens:
        raise RuntimeError(f"No NAVSIM scenes loaded from {config.log_path}")
    if config.sample_index >= len(tokens):
        raise IndexError(f"sample_index={config.sample_index} but only {len(tokens)} scene(s) loaded")

    token = tokens[config.sample_index]
    frame_list = loader.scene_frames_dicts[token]
    current_index = loader._scene_filter.num_history_frames - 1
    current_frame = frame_list[current_index]
    future_frames = frame_list[current_index + 1 : current_index + 1 + config.num_future_frames]
    ego_future = build_ego_future_trajectory(frame_list, current_index, config.planning_steps)
    command = infer_uniad_command(ego_future, current_frame)
    map_api = load_map_api(config, current_frame["map_location"])
    frame_inputs = [
        build_frame_input(config, frame_list, frame_idx, current_index, ego_future, command, map_api)
        for frame_idx in range(current_index + 1)
    ]
    return NavsimUniADSample(
        token=token,
        scene_frames=frame_list,
        current_index=current_index,
        frame_inputs=frame_inputs,
        current_frame=current_frame,
        future_frames=future_frames,
        ego_future_trajectory=ego_future,
        obstacle_future_trajectories=build_obstacle_future_trajectories(current_frame, future_frames),
        map_api=map_api,
        map_name=current_frame["map_location"],
    )


def build_frame_input(
    config: NavsimUniADConfig,
    frame_list: Sequence[Mapping[str, Any]],
    frame_idx: int,
    final_current_index: int,
    final_ego_future: torch.Tensor,
    final_command: int,
    map_api: Any,
) -> NavsimUniADFrameInput:
    frame = frame_list[frame_idx]
    image_tensor, meta = build_images_and_meta(config, frame)
    l2g_r, l2g_t = lidar_to_global_row_transform(frame)
    gt_lane_labels, gt_lane_masks = build_map_lane_gt(config, frame, map_api)
    gt_segmentation, gt_instance, gt_occ_img_is_valid = build_occupancy_gt(config, frame_list, frame_idx)

    if frame_idx == final_current_index:
        sdc_planning = final_ego_future
        command = final_command
    else:
        sdc_planning = build_ego_future_trajectory(frame_list, frame_idx, config.planning_steps)
        command = infer_uniad_command(sdc_planning, frame)

    sdc_planning_tensor = torch.zeros((1, config.planning_steps, 3), dtype=torch.float32)
    sdc_planning_mask = torch.zeros((1, config.planning_steps, 2), dtype=torch.float32)
    valid_steps = min(config.planning_steps, int(sdc_planning.shape[0]))
    if valid_steps:
        sdc_planning_tensor[0, :valid_steps] = sdc_planning[:valid_steps]
        sdc_planning_mask[0, :valid_steps] = 1.0

    return NavsimUniADFrameInput(
        token=frame["token"],
        frame=frame,
        img=image_tensor.unsqueeze(0),
        img_metas=[[meta]],
        l2g_t=torch.from_numpy(l2g_t).reshape(1, 3).to(torch.float32),
        l2g_r_mat=torch.from_numpy(l2g_r).reshape(1, 3, 3).to(torch.float32),
        timestamp=[torch.tensor(timestamp_seconds(frame["timestamp"]), dtype=torch.float32)],
        gt_lane_labels=[gt_lane_labels],
        gt_lane_masks=[gt_lane_masks],
        gt_segmentation=[gt_segmentation],
        gt_instance=[gt_instance],
        gt_occ_img_is_valid=[gt_occ_img_is_valid],
        sdc_planning=sdc_planning_tensor,
        sdc_planning_mask=sdc_planning_mask,
        command=torch.tensor([command], dtype=torch.long),
    )


def build_images_and_meta(config: NavsimUniADConfig, frame: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    images = []
    filenames = []
    ori_shapes = []
    img_shapes = []
    pad_shapes = []
    lidar2img = []
    target_h, target_w = config.image_hw
    pad_h = math.ceil(target_h / config.pad_divisor) * config.pad_divisor
    pad_w = math.ceil(target_w / config.pad_divisor) * config.pad_divisor

    for camera_name in config.camera_order:
        camera_info = frame["cams"][camera_name]
        image_path = config.sensor_path / camera_info["data_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing NAVSIM image for {camera_name}: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            src_w, src_h = image.size
            resized = image.resize((target_w, target_h), getattr(Image.Resampling, "BILINEAR", Image.BILINEAR))
        scale_x = target_w / src_w
        scale_y = target_h / src_h
        ori_shapes.append((src_h, src_w, 3))
        img_shapes.append((target_h, target_w, 3))
        pad_shapes.append((pad_h, pad_w, 3))
        intrinsics = np.asarray(camera_info["cam_intrinsic"], dtype=np.float64).copy()
        intrinsics[0] *= scale_x
        intrinsics[1] *= scale_y

        image_bgr = np.asarray(resized, dtype=np.float32)[..., ::-1]
        image_bgr = (image_bgr - IMG_NORM_CFG["mean"]) / IMG_NORM_CFG["std"]
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        padded[:target_h, :target_w] = image_bgr
        images.append(torch.from_numpy(padded).permute(2, 0, 1).contiguous())
        filenames.append(str(image_path))
        lidar2img.append(build_lidar2img(camera_info, intrinsics))

    l2g_r, l2g_t = lidar_to_global_row_transform(frame)
    meta = {
        "sample_idx": frame["token"],
        "scene_token": frame.get("scene_token", frame.get("scene_name", frame["token"])),
        "filename": filenames,
        "cam_names": list(config.camera_order),
        "ori_shape": ori_shapes,
        "img_shape": img_shapes,
        "pad_shape": pad_shapes,
        "img_norm_cfg": {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in IMG_NORM_CFG.items()},
        "lidar2img": np.stack(lidar2img, axis=0).astype(np.float32),
        "can_bus": build_can_bus(frame),
        "l2g_r_mat": l2g_r.astype(np.float32),
        "l2g_t": l2g_t.astype(np.float32),
        "pts_filename": frame.get("lidar_path") or frame["token"],
        "box_type_3d": LiDARInstance3DBoxes,
    }
    return torch.stack(images, dim=0).to(torch.float32), meta


def build_lidar2img(camera_info: Mapping[str, Any], intrinsics: np.ndarray) -> np.ndarray:
    sensor2lidar_rotation = np.asarray(camera_info["sensor2lidar_rotation"], dtype=np.float64)
    sensor2lidar_translation = np.asarray(camera_info["sensor2lidar_translation"], dtype=np.float64)
    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = intrinsics
    return viewpad @ lidar2cam_rt.T


def build_can_bus(frame: Mapping[str, Any]) -> np.ndarray:
    can_bus = np.zeros(18, dtype=np.float32)
    raw_can_bus = np.asarray(frame.get("can_bus", can_bus), dtype=np.float32).reshape(-1)
    can_bus[: min(18, raw_can_bus.size)] = raw_can_bus[:18]
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float32)
    rotation_quat = Quaternion(*np.asarray(frame["ego2global_rotation"], dtype=np.float64))
    rotation = np.asarray(rotation_quat.elements, dtype=np.float32)
    yaw_degrees = float(rotation_quat.yaw_pitch_roll[0] / np.pi * 180.0)
    if yaw_degrees < 0.0:
        yaw_degrees += 360.0
    can_bus[:3] = translation
    can_bus[3:7] = rotation
    can_bus[-2] = yaw_degrees / 180.0 * math.pi
    can_bus[-1] = yaw_degrees
    return can_bus


def lidar_to_global_row_transform(frame: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    transform = frame_to_global_transform(frame)
    return transform[:3, :3].T.astype(np.float32), transform[:3, 3].astype(np.float32)


def timestamp_seconds(timestamp: Any) -> float:
    value = float(timestamp)
    if abs(value) > 1.0e17:
        return value / 1.0e9
    if abs(value) > 1.0e12:
        return value / 1.0e6
    return value


def _yaw_from_rotation_matrix(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def frame_to_global_transform(frame: Mapping[str, Any]) -> np.ndarray:
    if "lidar2global" in frame:
        transform = np.asarray(frame["lidar2global"], dtype=np.float64)
        if transform.shape == (4, 4):
            return transform.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(*frame["ego2global_rotation"]).rotation_matrix
    transform[:3, 3] = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    return transform


def current_to_target_local_transform(current_frame: Mapping[str, Any], target_frame: Mapping[str, Any]) -> np.ndarray:
    current_pose = frame_pose(current_frame)
    target_pose = frame_pose(target_frame)
    target_xy = global_to_local_xy(current_frame, target_pose[:2])
    target_yaw = float(np.arctan2(np.sin(target_pose[2] - current_pose[2]), np.cos(target_pose[2] - current_pose[2])))
    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = np.array(
        [[math.cos(target_yaw), -math.sin(target_yaw)], [math.sin(target_yaw), math.cos(target_yaw)]],
        dtype=np.float64,
    )
    transform[:2, 3] = target_xy
    return transform


def frame_pose(frame: Mapping[str, Any]) -> np.ndarray:
    transform = frame_to_global_transform(frame)
    translation = transform[:3, 3]
    yaw = _yaw_from_rotation_matrix(transform[:3, :3])
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def local_to_global_xy(frame: Mapping[str, Any], xy: np.ndarray) -> np.ndarray:
    xy_array = np.asarray(xy, dtype=np.float64)
    pose = frame_pose(frame)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x = pose[0] + c * xy_array[..., 0] - s * xy_array[..., 1]
    y = pose[1] + s * xy_array[..., 0] + c * xy_array[..., 1]
    return np.stack([x, y], axis=-1)


def global_to_local_xy(frame: Mapping[str, Any], xy: np.ndarray) -> np.ndarray:
    xy_array = np.asarray(xy, dtype=np.float64)
    pose = frame_pose(frame)
    dx = xy_array[..., 0] - pose[0]
    dy = xy_array[..., 1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x = c * dx + s * dy
    y = -s * dx + c * dy
    return np.stack([x, y], axis=-1)


def build_ego_future_trajectory(
    frame_list: Sequence[Mapping[str, Any]],
    current_index: int,
    planning_steps: int,
) -> torch.Tensor:
    current_frame = frame_list[current_index]
    future_frames = frame_list[current_index + 1 : current_index + 1 + planning_steps]
    if not future_frames:
        return torch.zeros((0, 3), dtype=torch.float32)
    relative_poses = []
    for future_frame in future_frames:
        relative_transform = current_to_target_local_transform(current_frame, future_frame)
        relative_poses.append(
            [
                relative_transform[0, 3],
                relative_transform[1, 3],
                _yaw_from_rotation_matrix(relative_transform[:3, :3]),
            ]
        )
    return torch.from_numpy(np.asarray(relative_poses, dtype=np.float32))


def infer_uniad_command(ego_future_trajectory: torch.Tensor, frame: Mapping[str, Any] | None = None) -> int:
    if frame is not None:
        direct_command = _extract_direct_uniad_command(frame)
        if direct_command is not None:
            return direct_command
    if ego_future_trajectory.numel() == 0:
        return 2
    lateral_y = float(ego_future_trajectory[-1, 1])
    if lateral_y < -2.0:
        return 0
    if lateral_y > 2.0:
        return 1
    return 2


def _extract_direct_uniad_command(frame: Mapping[str, Any]) -> int | None:
    for key in ("command", "gt_ego_fut_cmd"):
        if key not in frame:
            continue
        command = np.asarray(frame[key])
        if command.size == 1:
            command_idx = int(command.reshape(-1)[0])
            return command_idx if 0 <= command_idx <= 2 else None
        if command.size >= 3:
            command_idx = int(np.argmax(command.reshape(-1)[:3]))
            return command_idx if 0 <= command_idx <= 2 else None
    return None


def _annotation_token_to_index(annotations: Mapping[str, Any]) -> dict[str, int]:
    token_to_index: dict[str, int] = {}
    boxes = np.asarray(annotations.get("gt_boxes", []))
    track_tokens = list(annotations.get("track_tokens", []))
    instance_tokens = list(annotations.get("instance_tokens", []))
    for idx in range(len(boxes)):
        for token_list in (track_tokens, instance_tokens):
            if idx < len(token_list):
                token = token_list[idx]
                if token is not None and str(token) not in token_to_index:
                    token_to_index[str(token)] = idx
    return token_to_index


def build_obstacle_future_trajectories(
    current_frame: Mapping[str, Any],
    future_frames: Sequence[Mapping[str, Any]],
) -> list[ObstacleTrajectory]:
    annotations = current_frame["anns"]
    boxes = np.asarray(annotations["gt_boxes"], dtype=np.float32)
    names = list(annotations["gt_names"])
    track_tokens = list(annotations.get("track_tokens", []))
    instance_tokens = list(annotations.get("instance_tokens", []))
    trajectories: list[ObstacleTrajectory] = []

    for idx, box in enumerate(boxes):
        token = track_tokens[idx] if idx < len(track_tokens) else instance_tokens[idx] if idx < len(instance_tokens) else None
        if token is None:
            continue
        token = str(token)
        current_global = local_to_global_xy(current_frame, np.asarray(box[:2], dtype=np.float64))
        points = [global_to_local_xy(current_frame, current_global)]
        for future_frame in future_frames:
            future_annotations = future_frame["anns"]
            future_idx = _annotation_token_to_index(future_annotations).get(token)
            if future_idx is None:
                break
            future_box = np.asarray(future_annotations["gt_boxes"][future_idx], dtype=np.float64)
            future_global = local_to_global_xy(future_frame, future_box[:2])
            points.append(global_to_local_xy(current_frame, future_global))
        if len(points) > 1:
            trajectories.append(ObstacleTrajectory(token=token, name=str(names[idx]), points=np.stack(points).astype(np.float32)))
    return trajectories


def build_map_lane_gt(config: NavsimUniADConfig, frame: Mapping[str, Any], map_api: Any) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    masks = rasterize_map_lane_masks(config, frame, map_api)
    return labels, torch.from_numpy(masks[None]).to(torch.long)


def rasterize_map_lane_masks(config: NavsimUniADConfig, frame: Mapping[str, Any], map_api: Any) -> np.ndarray:
    masks = np.zeros((4, *config.bev_hw), dtype=np.uint8)
    if map_api is None:
        return masks

    try:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from shapely import affinity
    except Exception:
        return masks

    layer_names = (
        "LANE",
        "LANE_CONNECTOR",
        "INTERSECTION",
        "DRIVABLE_AREA",
        "ROADBLOCK",
        "ROADBLOCK_CONNECTOR",
        "CARPARK_AREA",
        "CROSSWALK",
    )
    layers = [getattr(SemanticMapLayer, layer_name) for layer_name in layer_names if hasattr(SemanticMapLayer, layer_name)]
    pose = frame_pose(frame)
    radius = config.bev_extent_m * math.sqrt(2.0) / 2.0
    map_objects = _safe_get_proximal_map_objects(map_api, Point2D(float(pose[0]), float(pose[1])), radius, layers)

    images = [Image.new("L", (config.bev_hw[1], config.bev_hw[0]), 0) for _ in range(4)]
    draws = [ImageDraw.Draw(image) for image in images]
    line_width = max(1, int(round(config.bev_hw[1] / config.bev_extent_m)))

    def to_local(geometry: Any) -> Any:
        translated = affinity.affine_transform(geometry, [1, 0, 0, 1, -pose[0], -pose[1]])
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return affinity.affine_transform(translated, [c, s, -s, c, 0, 0])

    lane_layers = [layer for layer in (getattr(SemanticMapLayer, "LANE", None), getattr(SemanticMapLayer, "LANE_CONNECTOR", None)) if layer is not None]
    for layer in lane_layers:
        for map_object in map_objects.get(layer, []):
            baseline = getattr(getattr(map_object, "baseline_path", None), "linestring", None)
            polygon = getattr(map_object, "polygon", None)
            if baseline is not None:
                _draw_local_geometry(draws[0], to_local(baseline), config, fill=False, width=line_width)
            elif polygon is not None:
                _draw_local_geometry(draws[0], to_local(polygon.boundary), config, fill=False, width=line_width)
            if polygon is not None:
                local_polygon = to_local(polygon)
                _draw_local_geometry(draws[2], local_polygon.boundary, config, fill=False, width=line_width)

    crosswalk_layer = getattr(SemanticMapLayer, "CROSSWALK", None)
    if crosswalk_layer is not None:
        for map_object in map_objects.get(crosswalk_layer, []):
            polygon = getattr(map_object, "polygon", None)
            if polygon is not None:
                _draw_local_geometry(draws[1], to_local(polygon), config, fill=True, width=line_width)

    drivable_layers = [
        layer
        for layer in (
            getattr(SemanticMapLayer, "DRIVABLE_AREA", None),
            getattr(SemanticMapLayer, "ROADBLOCK", None),
            getattr(SemanticMapLayer, "ROADBLOCK_CONNECTOR", None),
            getattr(SemanticMapLayer, "INTERSECTION", None),
            getattr(SemanticMapLayer, "LANE", None),
            getattr(SemanticMapLayer, "LANE_CONNECTOR", None),
            getattr(SemanticMapLayer, "CARPARK_AREA", None),
        )
        if layer is not None
    ]
    for layer in drivable_layers:
        for map_object in map_objects.get(layer, []):
            polygon = getattr(map_object, "polygon", None)
            if polygon is not None:
                _draw_local_geometry(draws[3], to_local(polygon), config, fill=True, width=line_width)

    return np.stack([(np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8) for image in images], axis=0)


def _safe_get_proximal_map_objects(map_api: Any, point: Any, radius: float, layers: Sequence[Any]) -> dict[Any, list[Any]]:
    try:
        available_layers = set(map_api.get_available_map_objects())
    except Exception:
        available_layers = set(layers)

    object_map: dict[Any, list[Any]] = {layer: [] for layer in layers}
    for layer in layers:
        if layer not in available_layers:
            continue
        try:
            layer_objects = map_api.get_proximal_map_objects(point=point, radius=radius, layers=[layer])
        except Exception:
            continue
        object_map[layer] = list(layer_objects.get(layer, []))
    return object_map


def _local_xy_to_bev_pixel(config: NavsimUniADConfig, x: float, y: float) -> tuple[float, float]:
    height, width = config.bev_hw
    half_extent = config.bev_extent_m / 2.0
    col = (float(x) + half_extent) / config.bev_extent_m * width
    row = (float(y) + half_extent) / config.bev_extent_m * height
    return col, row


def _draw_local_geometry(draw: ImageDraw.ImageDraw, geometry: Any, config: NavsimUniADConfig, fill: bool, width: int) -> None:
    if geometry is None or getattr(geometry, "is_empty", False):
        return
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        exterior = [_local_xy_to_bev_pixel(config, x, y) for x, y in geometry.exterior.coords]
        if fill and len(exterior) >= 3:
            draw.polygon(exterior, fill=1)
            for interior in geometry.interiors:
                hole = [_local_xy_to_bev_pixel(config, x, y) for x, y in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)
        elif len(exterior) >= 2:
            draw.line(exterior, fill=1, width=width, joint="curve")
        return
    if geom_type == "LineString":
        coords = [_local_xy_to_bev_pixel(config, x, y) for x, y in geometry.coords]
        if len(coords) >= 2:
            draw.line(coords, fill=1, width=width, joint="curve")
        return
    if hasattr(geometry, "geoms"):
        for sub_geometry in geometry.geoms:
            _draw_local_geometry(draw, sub_geometry, config, fill=fill, width=width)


def build_occupancy_gt(
    config: NavsimUniADConfig,
    frame_list: Sequence[Mapping[str, Any]],
    frame_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segmentation = np.zeros((config.occ_num_frames, *config.bev_hw), dtype=np.int64)
    instance = np.zeros_like(segmentation)
    valid = np.zeros((config.occ_num_frames,), dtype=np.bool_)

    frames = frame_list[frame_idx : frame_idx + config.occ_num_frames]
    for out_idx, frame in enumerate(frames):
        occ_payload = _load_occupancy_payload(config, frame)
        if occ_payload is None:
            continue
        seg_array, instance_array = occ_payload
        remaining = config.occ_num_frames - out_idx
        seg_array = _normalise_occ_array(seg_array, remaining, config.bev_hw)
        if seg_array is None:
            continue
        instance_array = _normalise_occ_array(instance_array, remaining, config.bev_hw) if instance_array is not None else None
        num_loaded = min(remaining, seg_array.shape[0])
        segmentation[out_idx : out_idx + num_loaded] = seg_array[:num_loaded]
        if instance_array is not None:
            instance[out_idx : out_idx + num_loaded] = instance_array[:num_loaded]
        valid[out_idx : out_idx + num_loaded] = True
        if num_loaded > 1:
            break

    return (
        torch.from_numpy(segmentation[None]).to(torch.long),
        torch.from_numpy(instance[None]).to(torch.long),
        torch.from_numpy(valid[None]).to(torch.bool),
    )


def _resolve_frame_data_path(config: NavsimUniADConfig, path_value: Any) -> Path | None:
    if path_value is None:
        return None
    raw_path = Path(str(path_value))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend(
            [
                config.dataset_root / raw_path,
                config.dataset_root / str(raw_path).removeprefix("dataset/"),
                config.dataset_root.parent / raw_path,
                config.dataset_root.parent / str(raw_path).removeprefix("dataset/"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_occupancy_payload(config: NavsimUniADConfig, frame: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray | None] | None:
    for key in ("occ_gt_final_path", "flow_gt_final_path"):
        data_path = _resolve_frame_data_path(config, frame.get(key))
        if data_path is None:
            continue
        try:
            loaded = np.load(data_path, allow_pickle=True)
        except Exception:
            continue
        payload = dict(loaded.items()) if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
            payload = payload.item()
        extracted = _extract_occupancy_arrays(payload)
        if extracted is not None:
            return extracted
    return None


def _extract_occupancy_arrays(payload: Any) -> tuple[np.ndarray, np.ndarray | None] | None:
    if isinstance(payload, Mapping):
        segmentation = _first_payload_array(
            payload,
            ("gt_segmentation", "segmentation", "seg_gt", "occupancy", "occ", "occ_gt", "semantic", "vehicle_occupancy"),
        )
        instance = _first_payload_array(payload, ("gt_instance", "instance", "instances", "ins_seg_gt", "instance_ids"))
        if segmentation is None and instance is not None:
            segmentation = (np.asarray(instance) > 0).astype(np.int64)
        if segmentation is not None:
            return np.asarray(segmentation), np.asarray(instance) if instance is not None else None
        return None
    array = np.asarray(payload)
    if array.size == 0:
        return None
    return array, None


def _first_payload_array(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _normalise_occ_array(array: Any, max_frames: int, bev_hw: tuple[int, int]) -> np.ndarray | None:
    occ = np.asarray(array)
    if occ.size == 0:
        return None
    occ = np.squeeze(occ)
    if occ.ndim == 2:
        occ = occ[None]
    elif occ.ndim == 4 and occ.shape[1] == 1:
        occ = occ[:, 0]
    elif occ.ndim == 4 and occ.shape[-1] == 1:
        occ = occ[..., 0]
    if occ.ndim != 3:
        return None
    occ = occ[:max_frames].astype(np.int64, copy=False)
    if occ.shape[-2:] == bev_hw:
        return occ
    resized = np.zeros((occ.shape[0], *bev_hw), dtype=np.int64)
    for idx, frame in enumerate(occ):
        image = Image.fromarray(frame.astype(np.int32), mode="I")
        image = image.resize((bev_hw[1], bev_hw[0]), getattr(Image.Resampling, "NEAREST", Image.NEAREST))
        resized[idx] = np.asarray(image, dtype=np.int64)
    return resized


def load_map_api(config: NavsimUniADConfig, map_name: str) -> Any:
    if not config.maps_root.exists():
        raise FileNotFoundError(f"NAVSIM maps root does not exist: {config.maps_root}")
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    return get_maps_api(str(config.maps_root), config.map_version, map_name)


def annotations_as_boxes(frame: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    annotations = frame["anns"]
    return np.asarray(annotations["gt_boxes"], dtype=np.float32), [str(name) for name in annotations["gt_names"]]
