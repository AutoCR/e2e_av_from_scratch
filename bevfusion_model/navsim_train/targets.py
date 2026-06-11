from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
from pyquaternion import Quaternion

NAVSIM_OBJECT_CLASSES = ("car", "barrier", "bicycle", "pedestrian", "traffic_cone")
NAVSIM_CLASS_TO_LABEL = {name: idx for idx, name in enumerate(NAVSIM_OBJECT_CLASSES)}

_NUPLAN_TO_NAVSIM_CLASS = {
    "vehicle": "car",
    "car": "car",
    "barrier": "barrier",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
    "traffic_cone": "traffic_cone",
    "generic_object": None,
    "czone_sign": None,
}


def navsim_name_to_label(name: str) -> int | None:
    mapped = _NUPLAN_TO_NAVSIM_CLASS.get(str(name).lower())
    if mapped is None:
        return None
    return NAVSIM_CLASS_TO_LABEL[mapped]


def stable_track_id(token: str) -> int:
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _as_ann_dict(frame_annotations: Any) -> Mapping[str, Any]:
    if frame_annotations is None:
        return {
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_names": [],
            "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32),
            "track_tokens": [],
        }
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
    yaw_vec_global = np.stack(
        [np.cos(boxes_global[:, 6]), np.sin(boxes_global[:, 6]), np.zeros(len(boxes_global))],
        axis=1,
    )
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
    velocity = np.asarray(
        ann.get("gt_velocity_3d", ann.get("velocity_3d", np.zeros((len(boxes), 3)))),
        dtype=np.float32,
    )
    track_tokens = list(ann.get("track_tokens", [str(i) for i in range(len(boxes))]))
    if len(boxes) == 0:
        return (
            torch.zeros((0, 9), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )

    encoded, _ = _transform_boxes(boxes, velocity, np.asarray(ego2lidar, dtype=np.float32))
    labels: list[int] = []
    keep: list[int] = []
    for i, name in enumerate(names):
        label = navsim_name_to_label(name)
        if label is None:
            continue
        dist = float(np.linalg.norm(encoded[i, :2]))
        if dist > range_threshold:
            continue
        labels.append(label)
        keep.append(i)
    if not keep:
        return (
            torch.zeros((0, 9), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )

    keep_arr = np.asarray(keep, dtype=np.int64)
    return (
        torch.from_numpy(encoded[keep_arr].astype(np.float32)),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor([stable_track_id(track_tokens[i]) for i in keep], dtype=torch.long),
    )
