"""Decode SparseDrive post-processed predictions for local-frame BEV plotting.

The helpers in this module treat SparseDrive outputs as already being in the
ego-local BEV coordinate frame. They do not transform coordinates or prepend the
current ego pose; consumers can add an origin point if their plotting code needs
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is a project dependency.
    torch = None  # type: ignore[assignment]


DETECTION_SCORE_THRESHOLD = 0.3
MAP_SCORE_THRESHOLD = 0.3
MAX_OBSTACLE_PREDICTIONS = 50
MAX_MAP_POLYLINES = 100
EGO_LOCAL_BEV_FRAME = "ego-local-bev"


@dataclass(frozen=True)
class DecodedSparseDrivePrediction:
    """SparseDrive predictions converted to simple numpy arrays.

    Attributes are in SparseDrive's post-processed ego-local BEV frame:
    - `predicted_ego_trajectory`: selected ego future trajectory, `[T, 2]`
      or `[T, 3]`.
    - `predicted_obstacle_boxes`: detection boxes sorted by confidence,
      `[K, D]`.
    - `predicted_obstacle_trajectories`: best motion mode per selected
      detection, `[K, T, 2]`.
    - `predicted_map_polylines`: selected map vectors as a list of `[P, 2]`
      arrays.
    """

    predicted_ego_trajectory: np.ndarray
    predicted_obstacle_boxes: np.ndarray
    predicted_obstacle_trajectories: np.ndarray
    predicted_map_polylines: list[np.ndarray]
    obstacle_scores: np.ndarray
    obstacle_labels: np.ndarray
    obstacle_instance_ids: np.ndarray
    obstacle_indices: np.ndarray
    obstacle_motion_mode_indices: np.ndarray
    map_scores: np.ndarray
    map_labels: np.ndarray
    map_indices: np.ndarray
    planning_source: str
    planning_command_index: Optional[int] = None
    planning_mode_index: Optional[int] = None
    coordinate_frame: str = EGO_LOCAL_BEV_FRAME

    @property
    def ego_trajectory(self) -> np.ndarray:
        return self.predicted_ego_trajectory

    @property
    def obstacle_boxes(self) -> np.ndarray:
        return self.predicted_obstacle_boxes

    @property
    def obstacle_trajectories(self) -> np.ndarray:
        return self.predicted_obstacle_trajectories

    @property
    def map_polylines(self) -> list[np.ndarray]:
        return self.predicted_map_polylines


def decode_sparsedrive_prediction(
    sample_output: Mapping[str, Any],
    *,
    detection_score_threshold: float = DETECTION_SCORE_THRESHOLD,
    max_obstacles: Optional[int] = MAX_OBSTACLE_PREDICTIONS,
    map_score_threshold: float = MAP_SCORE_THRESHOLD,
    max_map_polylines: Optional[int] = MAX_MAP_POLYLINES,
) -> DecodedSparseDrivePrediction:
    """Decode one SparseDrive stage2 sample into BEV-friendly arrays.

    `sample_output` should be either `out[i]["img_bbox"]` or the surrounding
    `out[i]` dictionary containing an `img_bbox` key. Required heads are
    validated explicitly. Planning uses `final_planning` when available;
    otherwise it selects the command/mode pair with the largest
    `planning_score` value from `planning_score[command, mode]` and returns
    `planning[command, mode]`.
    """

    result = _unwrap_img_bbox(sample_output)
    _require_keys(
        result,
        ("boxes_3d", "scores_3d", "labels_3d", "instance_ids"),
        "detection",
    )
    _require_keys(result, ("vectors", "scores", "labels"), "map")
    _require_keys(result, ("trajs_3d", "trajs_score"), "motion")

    ego_trajectory, planning_source, command_idx, mode_idx = _decode_planning(result)
    (
        obstacle_boxes,
        obstacle_scores,
        obstacle_labels,
        obstacle_instance_ids,
        obstacle_indices,
        obstacle_trajectories,
        motion_mode_indices,
    ) = _decode_obstacles(
        result,
        score_threshold=detection_score_threshold,
        max_obstacles=max_obstacles,
    )
    map_polylines, map_scores, map_labels, map_indices = _decode_map(
        result,
        score_threshold=map_score_threshold,
        max_polylines=max_map_polylines,
    )

    return DecodedSparseDrivePrediction(
        predicted_ego_trajectory=ego_trajectory,
        predicted_obstacle_boxes=obstacle_boxes,
        predicted_obstacle_trajectories=obstacle_trajectories,
        predicted_map_polylines=map_polylines,
        obstacle_scores=obstacle_scores,
        obstacle_labels=obstacle_labels,
        obstacle_instance_ids=obstacle_instance_ids,
        obstacle_indices=obstacle_indices,
        obstacle_motion_mode_indices=motion_mode_indices,
        map_scores=map_scores,
        map_labels=map_labels,
        map_indices=map_indices,
        planning_source=planning_source,
        planning_command_index=command_idx,
        planning_mode_index=mode_idx,
    )


def decode_sparsedrive_outputs(
    outputs: Sequence[Mapping[str, Any]],
    sample_index: int = 0,
    **decode_kwargs: Any,
) -> DecodedSparseDrivePrediction:
    """Decode `outputs[sample_index]["img_bbox"]` from model inference."""

    if sample_index < 0 or sample_index >= len(outputs):
        raise IndexError(
            f"sample_index {sample_index} is out of range for {len(outputs)} outputs."
        )
    return decode_sparsedrive_prediction(outputs[sample_index], **decode_kwargs)


def to_numpy(value: Any, name: str = "value") -> np.ndarray:
    """Convert tensors/array-likes to detached CPU numpy arrays."""

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if value is None:
        raise ValueError(f"{name} is None; expected a tensor or array-like value.")
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to a numpy array.") from exc


def _unwrap_img_bbox(sample_output: Mapping[str, Any]) -> Mapping[str, Any]:
    if "img_bbox" not in sample_output:
        return sample_output
    img_bbox = sample_output["img_bbox"]
    if not isinstance(img_bbox, Mapping):
        raise ValueError(
            "sample_output['img_bbox'] must be a mapping with SparseDrive prediction "
            f"heads; got {type(img_bbox).__name__}."
        )
    return img_bbox


def _require_keys(result: Mapping[str, Any], keys: Sequence[str], head_name: str) -> None:
    missing = [key for key in keys if key not in result]
    if missing:
        formatted = ", ".join(missing)
        raise KeyError(
            f"Missing SparseDrive {head_name} output key(s): {formatted}. "
            "Pass the post-processed per-sample dictionary from out[i]['img_bbox']."
        )


def _decode_planning(
    result: Mapping[str, Any],
) -> tuple[np.ndarray, str, Optional[int], Optional[int]]:
    if "final_planning" in result:
        final_planning = _ensure_trajectory_array(
            to_numpy(result["final_planning"], "final_planning"),
            "final_planning",
        )
        return final_planning, "final_planning", None, None

    _require_keys(result, ("planning_score", "planning"), "planning")
    planning_score = to_numpy(result["planning_score"], "planning_score")
    planning = to_numpy(result["planning"], "planning")

    if planning_score.ndim != 2:
        raise ValueError(
            "planning_score must be shaped [num_commands, num_modes], "
            f"got {planning_score.shape}."
        )
    if planning.ndim != 4:
        raise ValueError(
            "planning must be shaped [num_commands, num_modes, T, 2/3], "
            f"got {planning.shape}."
        )
    if planning.shape[:2] != planning_score.shape:
        raise ValueError(
            "planning first two dimensions must match planning_score; "
            f"got planning {planning.shape} and planning_score {planning_score.shape}."
        )
    if planning.shape[-1] not in (2, 3):
        raise ValueError(
            "planning last dimension must be 2 or 3 coordinates, "
            f"got {planning.shape[-1]}."
        )

    if not np.isfinite(planning_score).any():
        raise ValueError("planning_score contains no finite values to select from.")
    finite_scores = np.where(np.isfinite(planning_score), planning_score, -np.inf)
    command_idx, mode_idx = np.unravel_index(
        int(np.argmax(finite_scores)), planning_score.shape
    )
    selected = _ensure_trajectory_array(
        planning[command_idx, mode_idx],
        f"planning[{command_idx}, {mode_idx}]",
    )
    return selected, "planning_score_argmax", int(command_idx), int(mode_idx)


def _decode_obstacles(
    result: Mapping[str, Any],
    *,
    score_threshold: float,
    max_obstacles: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = to_numpy(result["boxes_3d"], "boxes_3d")
    scores = _ensure_1d(to_numpy(result["scores_3d"], "scores_3d"), "scores_3d")
    labels = _ensure_1d(to_numpy(result["labels_3d"], "labels_3d"), "labels_3d")
    instance_ids = _ensure_1d(
        to_numpy(result["instance_ids"], "instance_ids"), "instance_ids"
    )
    trajs = to_numpy(result["trajs_3d"], "trajs_3d")
    trajs_score = to_numpy(result["trajs_score"], "trajs_score")

    if boxes.ndim != 2:
        raise ValueError(f"boxes_3d must be shaped [N, D], got {boxes.shape}.")
    if trajs.ndim != 4 or trajs.shape[-1] != 2:
        raise ValueError(f"trajs_3d must be shaped [N, modes, T, 2], got {trajs.shape}.")
    if trajs_score.ndim != 2:
        raise ValueError(f"trajs_score must be shaped [N, modes], got {trajs_score.shape}.")

    num_detections = boxes.shape[0]
    for name, array in (
        ("scores_3d", scores),
        ("labels_3d", labels),
        ("instance_ids", instance_ids),
        ("trajs_3d", trajs),
        ("trajs_score", trajs_score),
    ):
        if array.shape[0] != num_detections:
            raise ValueError(
                f"{name} first dimension ({array.shape[0]}) must match boxes_3d "
                f"count ({num_detections})."
            )
    if trajs.shape[1] != trajs_score.shape[1]:
        raise ValueError(
            "trajs_3d mode dimension must match trajs_score; "
            f"got {trajs.shape[1]} and {trajs_score.shape[1]}."
        )

    selected = _select_by_score(
        scores,
        score_threshold=score_threshold,
        max_count=max_obstacles,
        score_name="scores_3d",
    )
    if selected.size == 0:
        empty_trajs = np.empty((0, trajs.shape[2], 2), dtype=trajs.dtype)
        return (
            boxes[:0],
            scores[:0],
            labels[:0],
            instance_ids[:0],
            selected,
            empty_trajs,
            np.empty((0,), dtype=np.int64),
        )

    selected_mode_scores = trajs_score[selected]
    if not np.isfinite(selected_mode_scores).all():
        raise ValueError("trajs_score contains non-finite values for selected detections.")
    motion_mode_indices = np.argmax(selected_mode_scores, axis=1).astype(np.int64)
    selected_trajs = trajs[selected, motion_mode_indices]

    return (
        boxes[selected],
        scores[selected],
        labels[selected],
        instance_ids[selected],
        selected,
        selected_trajs,
        motion_mode_indices,
    )


def _decode_map(
    result: Mapping[str, Any],
    *,
    score_threshold: float,
    max_polylines: Optional[int],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    polylines = _map_vectors_to_list(result["vectors"])
    scores = _ensure_1d(to_numpy(result["scores"], "scores"), "scores")
    labels = _ensure_1d(to_numpy(result["labels"], "labels"), "labels")

    if len(polylines) != scores.shape[0] or len(polylines) != labels.shape[0]:
        raise ValueError(
            "Map vectors, scores, and labels must have matching lengths; "
            f"got {len(polylines)}, {scores.shape[0]}, and {labels.shape[0]}."
        )

    selected = _select_by_score(
        scores,
        score_threshold=score_threshold,
        max_count=max_polylines,
        score_name="map scores",
    )
    selected_polylines = [polylines[int(idx)] for idx in selected]
    return selected_polylines, scores[selected], labels[selected], selected


def _map_vectors_to_list(vectors: Any) -> list[np.ndarray]:
    if isinstance(vectors, (list, tuple)):
        polylines = [
            _ensure_polyline_array(to_numpy(vector, f"vectors[{idx}]"), f"vectors[{idx}]")
            for idx, vector in enumerate(vectors)
        ]
        return polylines

    vector_array = to_numpy(vectors, "vectors")
    if vector_array.ndim != 3 or vector_array.shape[-1] != 2:
        raise ValueError(f"vectors must be shaped [M, P, 2], got {vector_array.shape}.")
    return [
        _ensure_polyline_array(vector_array[idx], f"vectors[{idx}]")
        for idx in range(vector_array.shape[0])
    ]


def _select_by_score(
    scores: np.ndarray,
    *,
    score_threshold: float,
    max_count: Optional[int],
    score_name: str,
) -> np.ndarray:
    if max_count is not None and max_count < 0:
        raise ValueError(f"max_count for {score_name} must be non-negative or None.")
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_name} contains non-finite values.")

    selected = np.flatnonzero(scores >= score_threshold)
    if selected.size == 0 or max_count == 0:
        return np.empty((0,), dtype=np.int64)
    selected = selected[np.argsort(scores[selected])[::-1]]
    if max_count is not None:
        selected = selected[:max_count]
    return selected.astype(np.int64, copy=False)


def _ensure_1d(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    return array


def _ensure_trajectory_array(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2 or array.shape[-1] not in (2, 3):
        raise ValueError(f"{name} must be shaped [T, 2] or [T, 3], got {array.shape}.")
    return array


def _ensure_polyline_array(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2 or array.shape[-1] != 2:
        raise ValueError(f"{name} must be shaped [P, 2], got {array.shape}.")
    return array
