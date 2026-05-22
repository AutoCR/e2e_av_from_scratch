from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pyquaternion import Quaternion

from navsim.common.dataclasses import Annotations, Trajectory
from navsim.common.enums import BoundingBoxIndex
from navsim.visualization.bev import (
    add_annotations_to_bev_ax,
    add_map_to_bev_ax,
    add_oriented_box_to_bev_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG

try:
    from .prediction_decode import DecodedSparseDrivePrediction, decode_sparsedrive_outputs
except ImportError:  # Supports importing this module from sparsedrive_model/test.py.
    from prediction_decode import DecodedSparseDrivePrediction, decode_sparsedrive_outputs


PathLike = Union[str, os.PathLike[str]]

PREDICTED_OBSTACLE_CONFIG: dict[str, Any] = {
    "fill_color": "#4e79a7",
    "fill_color_alpha": 0.35,
    "line_color": "#1f4e79",
    "line_color_alpha": 1.0,
    "line_width": 1.0,
    "line_style": "-",
    "zorder": 4,
}
PREDICTED_MAP_CONFIG: dict[str, Any] = {
    "color": "#9467bd",
    "alpha": 0.85,
    "linewidth": 1.2,
    "linestyle": "-",
    "zorder": 2,
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
GT_OBSTACLE_TRAJECTORY_CONFIG: dict[str, Any] = {
    "color": "#2ca02c",
    "alpha": 0.9,
    "linewidth": 1.0,
    "linestyle": "-",
    "marker": ".",
    "markersize": 2.5,
    "zorder": 5,
}


@dataclass(frozen=True)
class GroundTruthObstacleTrajectory:
    """Obstacle center trajectory in the current ego-local NAVSIM BEV frame."""

    token: str
    token_field: str
    name: str
    points: np.ndarray


@dataclass(frozen=True)
class _AnnotationIdentifier:
    field: str
    value: str


def visualize_navsim_inference(
    sample: Any,
    decoded_prediction: DecodedSparseDrivePrediction,
    output_dir: PathLike,
    *,
    filename: Optional[str] = None,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """Render and save a two-panel SparseDrive NAVSIM inference BEV figure."""

    output_path = _figure_output_path(sample, output_dir, filename)
    fig, axes = plt.subplots(2, 1, figsize=(8, 13))

    _plot_prediction_panel(axes[0], sample, decoded_prediction)
    _plot_ground_truth_panel(axes[1], sample)

    for ax in axes:
        _configure_bev_axis(ax)

    fig.suptitle(f"SparseDrive NAVSIM inference: {_scene_token(sample)}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return output_path


def visualize_navsim_sparsedrive_predictions(
    *,
    samples: Sequence[Any],
    outputs: Optional[Sequence[Mapping[str, Any]]] = None,
    decoded_predictions: Optional[Sequence[DecodedSparseDrivePrediction]] = None,
    summaries: Optional[Sequence[Mapping[str, Any]]] = None,
    output_dir: PathLike,
) -> list[Path]:
    """Batch-compatible helper used by sparsedrive_model/test.py."""

    del summaries  # Summaries are not needed for plotting, but accepted for compatibility.
    predictions = _resolve_decoded_predictions(samples, outputs, decoded_predictions)
    return [
        visualize_navsim_inference(sample, prediction, output_dir)
        for sample, prediction in zip(samples, predictions)
    ]


def build_gt_obstacle_future_trajectories(
    current_frame: Mapping[str, Any],
    future_frames: Sequence[Mapping[str, Any]],
) -> list[GroundTruthObstacleTrajectory]:
    """Match current annotations through future frames and return local center paths."""

    current_annotations = _annotations_from_mapping(
        _required_mapping(current_frame, "anns", "current_frame"),
        context="current_frame['anns']",
    )
    current_pose = _frame_pose(current_frame)
    trajectories: list[GroundTruthObstacleTrajectory] = []

    for ann_idx, current_box in enumerate(current_annotations.boxes):
        identifiers = _annotation_identifiers(current_annotations, ann_idx)
        if not identifiers:
            continue

        points = [np.asarray(current_box[:2], dtype=np.float64)]
        missing_future_annotation = False
        matched_identifier = identifiers[0]

        for frame_idx, future_frame in enumerate(future_frames):
            future_annotations = _annotations_from_mapping(
                _required_mapping(future_frame, "anns", f"future_frames[{frame_idx}]"),
                context=f"future_frames[{frame_idx}]['anns']",
            )
            match = _find_matching_annotation(future_annotations, identifiers)
            if match is None:
                missing_future_annotation = True
                break

            future_ann_idx, matched_identifier = match
            future_pose = _frame_pose(future_frame)
            future_global_xy = _local_to_global_xy(
                future_pose,
                future_annotations.boxes[future_ann_idx, :2],
            )
            points.append(_global_to_local_xy(current_pose, future_global_xy))

        if missing_future_annotation or len(points) <= 1:
            continue

        trajectories.append(
            GroundTruthObstacleTrajectory(
                token=matched_identifier.value,
                token_field=matched_identifier.field,
                name=current_annotations.names[ann_idx],
                points=np.stack(points, axis=0).astype(np.float32),
            )
        )

    return trajectories


def _plot_prediction_panel(
    ax: plt.Axes,
    sample: Any,
    decoded_prediction: DecodedSparseDrivePrediction,
) -> None:
    ax.set_title("Prediction")
    _add_ego(ax)
    _plot_predicted_map(ax, decoded_prediction.predicted_map_polylines)
    _plot_prediction_obstacles(ax, decoded_prediction)
    _plot_trajectory_with_navsim_helper(
        ax,
        decoded_prediction.predicted_ego_trajectory,
        TRAJECTORY_CONFIG["agent"],
    )
    ax.text(
        0.01,
        0.02,
        f"scene={_scene_token(sample)}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )


def _plot_ground_truth_panel(ax: plt.Axes, sample: Any) -> None:
    ax.set_title("Ground truth")

    if getattr(sample, "map_api", None) is not None:
        add_map_to_bev_ax(ax, sample.map_api, StateSE2(*np.asarray(sample.ego_pose)[:3]))

    current_annotations = getattr(sample, "current_annotations", None)
    if current_annotations is not None:
        add_annotations_to_bev_ax(
            ax,
            _annotations_from_mapping(current_annotations, context="sample.current_annotations"),
            add_ego=True,
        )
    else:
        _add_ego(ax)

    _plot_trajectory_with_navsim_helper(
        ax,
        _to_numpy(getattr(sample, "future_ego_trajectory"), "sample.future_ego_trajectory"),
        TRAJECTORY_CONFIG["human"],
    )

    current_frame = _required_mapping(sample, "current_frame", "sample")
    future_frames = list(getattr(sample, "future_frames", []))
    for trajectory in build_gt_obstacle_future_trajectories(current_frame, future_frames):
        _plot_xy_path(ax, trajectory.points, **GT_OBSTACLE_TRAJECTORY_CONFIG)


def _plot_prediction_obstacles(
    ax: plt.Axes,
    decoded_prediction: DecodedSparseDrivePrediction,
) -> None:
    boxes = np.asarray(decoded_prediction.predicted_obstacle_boxes)
    if boxes.size == 0:
        return
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"predicted_obstacle_boxes must be shaped [N, >=7], got {boxes.shape}.")

    trajectories = np.asarray(decoded_prediction.predicted_obstacle_trajectories)
    if trajectories.size and (
        trajectories.ndim != 3 or trajectories.shape[0] != boxes.shape[0] or trajectories.shape[-1] != 2
    ):
        raise ValueError(
            "predicted_obstacle_trajectories must be shaped [N, T, 2] "
            f"matching boxes, got {trajectories.shape} for boxes {boxes.shape}."
        )

    for index, box_values in enumerate(boxes):
        if not np.isfinite(box_values[:7]).all():
            raise ValueError(f"predicted obstacle box {index} contains non-finite values.")
        x, y, width, length, height, heading = (
            float(box_values[0]),
            float(box_values[1]),
            max(float(box_values[3]), 0.01),
            max(float(box_values[4]), 0.01),
            max(float(box_values[5]), 0.01),
            float(box_values[6]),
        )
        obstacle_box = OrientedBox(StateSE2(x, y, heading), length, width, height)
        add_oriented_box_to_bev_ax(ax, obstacle_box, PREDICTED_OBSTACLE_CONFIG)

        if trajectories.size:
            path = np.concatenate([np.array([[x, y]], dtype=np.float64), trajectories[index]], axis=0)
            _plot_xy_path(ax, path, **PREDICTED_OBSTACLE_TRAJECTORY_CONFIG)


def _plot_predicted_map(ax: plt.Axes, polylines: Sequence[np.ndarray]) -> None:
    for polyline_idx, polyline in enumerate(polylines):
        points = _ensure_xy_array(polyline, f"predicted_map_polylines[{polyline_idx}]")
        if points.shape[0] < 2:
            continue
        _plot_xy_path(ax, points, **PREDICTED_MAP_CONFIG)


def _plot_trajectory_with_navsim_helper(
    ax: plt.Axes,
    trajectory: Any,
    config: Mapping[str, Any],
) -> None:
    poses = _ensure_pose_array(trajectory, "trajectory")
    if poses.shape[0] == 0:
        return
    add_trajectory_to_bev_ax(
        ax,
        Trajectory(
            poses.astype(np.float32),
            TrajectorySampling(num_poses=poses.shape[0], interval_length=0.5),
        ),
        dict(config),
    )


def _plot_xy_path(ax: plt.Axes, xy: Any, **plot_kwargs: Any) -> None:
    points = _ensure_xy_array(xy, "xy path")
    if points.shape[0] == 0:
        return
    if not np.isfinite(points).all():
        raise ValueError("xy path contains non-finite values.")
    ax.plot(points[:, 1], points[:, 0], **plot_kwargs)


def _add_ego(ax: plt.Axes) -> None:
    add_annotations_to_bev_ax(ax, _empty_annotations(), add_ego=True)


def _empty_annotations() -> Annotations:
    return Annotations(
        boxes=np.zeros((0, BoundingBoxIndex.size()), dtype=np.float32),
        names=[],
        velocity_3d=np.zeros((0, 3), dtype=np.float32),
        instance_tokens=[],
        track_tokens=[],
    )


def _annotations_from_mapping(annotations: Mapping[str, Any], *, context: str) -> Annotations:
    required_keys = (
        "gt_boxes",
        "gt_names",
        "gt_velocity_3d",
        "instance_tokens",
        "track_tokens",
    )
    missing = [key for key in required_keys if key not in annotations]
    if missing:
        raise KeyError(f"{context} is missing annotation key(s): {', '.join(missing)}")

    boxes = np.asarray(annotations["gt_boxes"], dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < BoundingBoxIndex.size():
        raise ValueError(
            f"{context}['gt_boxes'] must be shaped [N, >={BoundingBoxIndex.size()}], got {boxes.shape}."
        )

    names = [str(name) for name in annotations["gt_names"]]
    velocity_3d = np.asarray(annotations["gt_velocity_3d"], dtype=np.float32)
    instance_tokens = [str(token) for token in annotations["instance_tokens"]]
    track_tokens = [str(token) for token in annotations["track_tokens"]]
    lengths = {
        "gt_boxes": boxes.shape[0],
        "gt_names": len(names),
        "gt_velocity_3d": len(velocity_3d),
        "instance_tokens": len(instance_tokens),
        "track_tokens": len(track_tokens),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{context} annotation lengths must match, got {lengths}.")

    return Annotations(
        boxes=boxes,
        names=names,
        velocity_3d=velocity_3d,
        instance_tokens=instance_tokens,
        track_tokens=track_tokens,
    )


def _annotation_identifiers(
    annotations: Annotations,
    index: int,
) -> list[_AnnotationIdentifier]:
    identifiers: list[_AnnotationIdentifier] = []
    for field, tokens in (
        ("track_tokens", annotations.track_tokens),
        ("instance_tokens", annotations.instance_tokens),
    ):
        token = str(tokens[index]).strip()
        if token and token.lower() not in {"none", "nan"}:
            identifiers.append(_AnnotationIdentifier(field=field, value=token))
    return identifiers


def _find_matching_annotation(
    annotations: Annotations,
    identifiers: Sequence[_AnnotationIdentifier],
) -> Optional[tuple[int, _AnnotationIdentifier]]:
    for identifier in identifiers:
        tokens = getattr(annotations, identifier.field)
        for index, token in enumerate(tokens):
            if token == identifier.value:
                return index, identifier
    return None


def _frame_pose(frame: Mapping[str, Any]) -> np.ndarray:
    translation = np.asarray(frame["ego2global_translation"][:2], dtype=np.float64)
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def _rotation_matrix(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )


def _local_to_global_xy(pose: np.ndarray, local_xy: Any) -> np.ndarray:
    xy = np.asarray(local_xy, dtype=np.float64)
    return pose[:2] + (_rotation_matrix(float(pose[2])) @ xy.T).T


def _global_to_local_xy(pose: np.ndarray, global_xy: Any) -> np.ndarray:
    xy = np.asarray(global_xy, dtype=np.float64)
    return (_rotation_matrix(-float(pose[2])) @ (xy - pose[:2]).T).T


def _ensure_xy_array(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value, name).astype(np.float64, copy=False)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must be shaped [N, >=2], got {array.shape}.")
    return array[:, :2]


def _ensure_pose_array(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value, name).astype(np.float64, copy=False)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"{name} must be shaped [N, 2] or [N, 3], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    if array.shape[1] == 2:
        return np.concatenate([array, np.zeros((array.shape[0], 1), dtype=array.dtype)], axis=1)
    return array


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    if value is None:
        raise ValueError(f"{name} is None.")
    return np.asarray(value)


def _required_mapping(source: Any, key: str, context: str) -> Mapping[str, Any]:
    if not isinstance(source, Mapping):
        source = vars(source)
    value = source[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{context}[{key!r}] must be a mapping, got {type(value).__name__}.")
    return value


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
        predictions = [
            decode_sparsedrive_outputs(outputs, sample_index=index)
            for index in range(len(samples))
        ]

    if len(predictions) != len(samples):
        raise ValueError(f"Expected {len(samples)} decoded predictions, got {len(predictions)}.")
    return predictions


def _figure_output_path(sample: Any, output_dir: PathLike, filename: Optional[str]) -> Path:
    directory = Path(os.path.expandvars(os.path.expanduser(os.fspath(output_dir))))
    if filename is None:
        filename = f"{_sanitize_filename(_scene_token(sample))}.png"
    return directory / filename


def _scene_token(sample: Any) -> str:
    current_frame = getattr(sample, "current_frame", None)
    if isinstance(current_frame, Mapping):
        return str(current_frame.get("scene_token") or getattr(sample, "token", "sample"))
    return str(getattr(sample, "token", "sample"))


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "sample"


def _configure_bev_axis(ax: plt.Axes) -> None:
    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])


__all__ = [
    "GroundTruthObstacleTrajectory",
    "build_gt_obstacle_future_trajectories",
    "visualize_navsim_inference",
    "visualize_navsim_sparsedrive_predictions",
]
