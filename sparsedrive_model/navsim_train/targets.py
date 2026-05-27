from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from pyquaternion import Quaternion

SPARSEDRIVE_CLASSES = (
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
_CLASS_TO_LABEL = {name: idx for idx, name in enumerate(SPARSEDRIVE_CLASSES)}
# NavSim/NuPlan consolidates all vehicle subtypes into a single "vehicle" label,
# so truck/bus/trailer/construction_vehicle/motorcycle all map to "car".
# Fine-grained vehicle class distinction is intentionally lost.
# "generic_object" and "czone_sign" map to None and are excluded from training.
_NUPLAN_TO_NUSCENES = {
    "vehicle": "car",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "generic_object": None,
    "czone_sign": None,
}


def nuplan_to_nuscenes_label(name: str) -> int | None:
    normalized = str(name).lower()
    mapped = _NUPLAN_TO_NUSCENES.get(normalized, normalized)
    if mapped is None:
        return None
    return _CLASS_TO_LABEL.get(mapped)


def stable_track_id(token: str) -> int:
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _as_ann_dict(frame_annotations: Any) -> Mapping[str, Any]:
    if frame_annotations is None:
        return {"gt_boxes": np.zeros((0, 7), dtype=np.float32), "gt_names": [], "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32), "track_tokens": []}
    if isinstance(frame_annotations, Mapping):
        return frame_annotations
    return {
        "gt_boxes": getattr(frame_annotations, "boxes"),
        "gt_names": getattr(frame_annotations, "names"),
        "gt_velocity_3d": getattr(frame_annotations, "velocity_3d"),
        "instance_tokens": getattr(frame_annotations, "instance_tokens", []),
        "track_tokens": getattr(frame_annotations, "track_tokens"),
    }


def _frame_global_to_lidar(frame: Mapping[str, Any]) -> np.ndarray:
    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, :3] = Quaternion(*frame["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    return np.linalg.inv(ego2global).astype(np.float32)


def _looks_like_local_boxes(boxes: np.ndarray, transform: np.ndarray) -> bool:
    if boxes.size == 0:
        return False
    box_xy = np.linalg.norm(boxes[:, :2], axis=1)
    transform_xy = float(np.linalg.norm(transform[:2, 3]))
    return transform_xy > 1.0e4 and float(np.nanmedian(box_xy)) < 1.0e4


def _transform_boxes(
    boxes_global: np.ndarray,
    velocities_global: np.ndarray,
    global_to_lidar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if boxes_global.size == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    transform = np.eye(4, dtype=np.float32) if _looks_like_local_boxes(boxes_global, global_to_lidar) else global_to_lidar
    centers_h = np.concatenate([boxes_global[:, :3], np.ones((boxes_global.shape[0], 1), dtype=np.float32)], axis=1)
    centers = centers_h @ transform.T
    rot = transform[:3, :3]
    yaw_vec_global = np.stack([np.cos(boxes_global[:, 6]), np.sin(boxes_global[:, 6]), np.zeros(len(boxes_global))], axis=1)
    yaw_vec_lidar = yaw_vec_global @ rot.T
    yaw = np.arctan2(yaw_vec_lidar[:, 1], yaw_vec_lidar[:, 0])
    vel = np.zeros((boxes_global.shape[0], 3), dtype=np.float32)
    if velocities_global is not None and len(velocities_global):
        vel[:, : min(3, velocities_global.shape[1])] = velocities_global[:, : min(3, velocities_global.shape[1])]
    vel_lidar = vel @ rot.T
    length = boxes_global[:, 3]
    width = boxes_global[:, 4]
    height = boxes_global[:, 5]
    encoded = np.stack(
        [
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            width,
            length,
            height,
            yaw,
            vel_lidar[:, 0],
            vel_lidar[:, 1],
        ],
        axis=1,
    ).astype(np.float32)
    return encoded, yaw.astype(np.float32)


def build_gt_bboxes_3d(
    frame_annotations: Any,
    ego2lidar: np.ndarray,
    range_threshold: float = 55.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ann = _as_ann_dict(frame_annotations)
    boxes = np.asarray(ann.get("gt_boxes", ann.get("boxes", np.zeros((0, 7)))), dtype=np.float32)
    names = list(ann.get("gt_names", ann.get("names", [])))
    velocity = np.asarray(ann.get("gt_velocity_3d", ann.get("velocity_3d", np.zeros((len(boxes), 3)))), dtype=np.float32)
    track_tokens = list(ann.get("track_tokens", [str(i) for i in range(len(boxes))]))
    if len(boxes) == 0:
        return torch.zeros((0, 9), dtype=torch.float32), torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)

    encoded, _ = _transform_boxes(boxes, velocity, np.asarray(ego2lidar, dtype=np.float32))
    labels: list[int] = []
    keep: list[int] = []
    for i, name in enumerate(names):
        label = nuplan_to_nuscenes_label(name)
        if label is None:
            continue
        dist = float(np.linalg.norm(encoded[i, :2]))
        if dist > range_threshold:
            continue
        labels.append(label)
        keep.append(i)
    if not keep:
        return torch.zeros((0, 9), dtype=torch.float32), torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
    keep_arr = np.asarray(keep, dtype=np.int64)
    return (
        torch.from_numpy(encoded[keep_arr].astype(np.float32)),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor([stable_track_id(track_tokens[i]) for i in keep], dtype=torch.long),
    )


def _lidar_to_global(frame: Mapping[str, Any]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(*frame["ego2global_rotation"]).rotation_matrix
    mat[:3, 3] = np.asarray(frame["ego2global_translation"], dtype=np.float32)
    return mat


def _filtered_track_centers(frame: Mapping[str, Any], global_to_lidar: np.ndarray, range_threshold: float = 55.0) -> dict[str, np.ndarray]:
    ann = _as_ann_dict(frame.get("anns", frame.get("annotations")))
    boxes = np.asarray(ann.get("gt_boxes", ann.get("boxes", np.zeros((0, 7)))), dtype=np.float32)
    names = list(ann.get("gt_names", ann.get("names", [])))
    velocity = np.asarray(ann.get("gt_velocity_3d", ann.get("velocity_3d", np.zeros((len(boxes), 3)))), dtype=np.float32)
    track_tokens = list(ann.get("track_tokens", []))
    transform = global_to_lidar @ _lidar_to_global(frame) if _looks_like_local_boxes(boxes, global_to_lidar) else global_to_lidar
    encoded, _ = _transform_boxes(boxes, velocity, transform)
    out: dict[str, np.ndarray] = {}
    for i, token in enumerate(track_tokens):
        if i >= len(names) or nuplan_to_nuscenes_label(names[i]) is None:
            continue
        if float(np.linalg.norm(encoded[i, :2])) > range_threshold:
            continue
        out[str(token)] = encoded[i, :2].astype(np.float32)
    return out


def build_agent_futures(
    frame_list: Sequence[Mapping[str, Any]],
    current_index: int,
    current_track_to_box_index: Mapping[str, int],
    fut_ts: int = 12,
    ego_lidar_transform: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(current_track_to_box_index)
    trajs = np.zeros((n, fut_ts, 2), dtype=np.float32)
    masks = np.zeros((n, fut_ts), dtype=np.int64)
    if n == 0:
        return torch.from_numpy(trajs), torch.from_numpy(masks)
    global_to_lidar = np.asarray(ego_lidar_transform, dtype=np.float32) if ego_lidar_transform is not None else _frame_global_to_lidar(frame_list[current_index])
    prev_xy = _filtered_track_centers(frame_list[current_index], global_to_lidar)
    for step in range(fut_ts):
        frame_idx = current_index + step + 1
        if frame_idx >= len(frame_list):
            break
        fut_centers = _filtered_track_centers(frame_list[frame_idx], global_to_lidar, range_threshold=float("inf"))
        for token, row in current_track_to_box_index.items():
            token = str(token)
            if token not in fut_centers or token not in prev_xy:
                continue
            current_xy = fut_centers[token]
            trajs[int(row), step] = current_xy - prev_xy[token]
            masks[int(row), step] = 1
            prev_xy[token] = current_xy
    return torch.from_numpy(trajs), torch.from_numpy(masks)


def _frame_xy_yaw(frame: Mapping[str, Any]) -> np.ndarray:
    trans = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([trans[0], trans[1], yaw], dtype=np.float64)


def _global_points_to_current_ego(current_pose: np.ndarray, poses: np.ndarray) -> np.ndarray:
    theta = -current_pose[2]
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    rel = poses[:, :2] - current_pose[None, :2]
    return rel @ rot.T


def build_ego_futures(
    frame_list: Sequence[Mapping[str, Any]],
    current_index: int,
    ego_fut_ts: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    trajs_abs = np.zeros((ego_fut_ts + 1, 2), dtype=np.float32)
    masks = np.zeros((ego_fut_ts,), dtype=np.int64)
    current_pose = _frame_xy_yaw(frame_list[current_index])
    future_poses = [current_pose]
    for step in range(ego_fut_ts):
        idx = current_index + step + 1
        if idx >= len(frame_list):
            future_poses.append(future_poses[-1])
            continue
        future_poses.append(_frame_xy_yaw(frame_list[idx]))
        masks[step] = 1
    rel_xy = _global_points_to_current_ego(current_pose, np.stack(future_poses, axis=0)).astype(np.float32)
    deltas = rel_xy[1:] - rel_xy[:-1]
    return torch.from_numpy(deltas.astype(np.float32)), torch.from_numpy(masks.astype(np.float32))


def build_ego_status(current_frame_ego_status: Any) -> torch.Tensor:
    if isinstance(current_frame_ego_status, Mapping):
        dyn = np.asarray(current_frame_ego_status.get("ego_dynamic_state", np.zeros(4)), dtype=np.float32)
        velocity = dyn[:2]
        accel = dyn[2:4]
    else:
        velocity = np.asarray(getattr(current_frame_ego_status, "ego_velocity", np.zeros(2)), dtype=np.float32)
        accel = np.asarray(getattr(current_frame_ego_status, "ego_acceleration", np.zeros(2)), dtype=np.float32)
    out = np.zeros(10, dtype=np.float32)
    out[0 : min(2, len(accel))] = accel[:2]
    out[6 : 6 + min(2, len(velocity))] = velocity[:2]
    return torch.from_numpy(out)


def build_gt_ego_fut_cmd(current_frame: Mapping[str, Any], future_ego_trajectory: torch.Tensor | None = None) -> torch.Tensor:
    try:
        from sparsedrive_model.navsim_adapter import _build_gt_ego_fut_cmd

        if future_ego_trajectory is None:
            future_ego_trajectory = torch.zeros((0, 3), dtype=torch.float32)
        return _build_gt_ego_fut_cmd(current_frame, future_ego_trajectory)
    except Exception:
        command = np.asarray(current_frame.get("driving_command", []))
        if command.size:
            idx = int(np.argmax(command))
            mapped = {0: 0, 1: 1, 2: 2, 3: 0}.get(idx, 0)
            out = np.zeros(3, dtype=np.float32)
            out[mapped] = 1.0
            return torch.from_numpy(out)
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
