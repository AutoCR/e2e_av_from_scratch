from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from navsim.common.dataclasses import Annotations, Trajectory
from navsim.visualization.bev import (
    add_annotations_to_bev_ax,
    add_map_to_bev_ax,
    add_oriented_box_to_bev_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.config import AGENT_CONFIG, TRAJECTORY_CONFIG
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from uniad.examples.forward_navsim import ForwardNavsimConfig, NavsimUniADRunResult, run_navsim_uniad
from uniad.examples.navsim_uniad_dataset import NavsimUniADConfig, NavsimUniADSample, frame_pose, load_navsim_uniad_sample


NAVSIM_CONFIG = NavsimUniADConfig(
    dataset_root=Path("/Users/chenran/Code/navsim/dataset"),
    split="mini",
)
RUN_CONFIG = ForwardNavsimConfig(
    navsim=NAVSIM_CONFIG,
    output_dir=_REPO_ROOT / "exp" / "uniad_navsim_debug",
    device="cpu",
)


@dataclass(frozen=True)
class VisualizationConfig:
    """Script-local configuration. Edit these values directly; no CLI/env config is used."""

    run_config: ForwardNavsimConfig = field(default_factory=lambda: RUN_CONFIG)
    output_dir: Path = field(default_factory=lambda: RUN_CONFIG.output_dir)
    use_forward_artifact: bool = True
    artifact_path: Path | None = None
    rerun_inference_if_artifact_missing: bool = True
    obstacle_score_threshold: float = 0.2
    max_predicted_obstacles: int = 30
    max_predicted_obstacle_trajs: int = 20
    max_predicted_map_elements: int = 30
    max_gt_obstacle_trajs: int = 30
    bev_extent_m: float | None = None
    xlim: tuple[float, float] = (-35.0, 35.0)
    ylim: tuple[float, float] = (-20.0, 60.0)


VIS_CONFIG = VisualizationConfig()


@dataclass
class NavsimUniADVisualizationData:
    config: ForwardNavsimConfig
    sample: NavsimUniADSample
    predictions: dict[str, np.ndarray]
    metrics: dict[str, float]
    output_shapes: dict[str, Any]
    artifact_path: Path | None
    source: str


class ArtifactUnavailable(RuntimeError):
    pass


PRED_BOX_CONFIG = {
    "fill_color": "#4e79a7",
    "fill_color_alpha": 0.25,
    "line_color": "#1f4e79",
    "line_color_alpha": 1.0,
    "line_width": 1.0,
    "line_style": "-",
    "zorder": 4,
}


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_str(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item() if array.shape == () else array.reshape(-1)[0])


def _bev_extent(data: NavsimUniADVisualizationData, viz_config: VisualizationConfig) -> float:
    return float(viz_config.bev_extent_m or data.config.navsim.bev_extent_m)


def _configure_bev(ax: plt.Axes, viz_config: VisualizationConfig) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(*viz_config.xlim)
    ax.set_ylim(*viz_config.ylim)
    ax.invert_xaxis()
    ax.grid(True, linewidth=0.25, alpha=0.4)
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("forward x [m]")


def _require_xy(name: str, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if xy.size == 0:
        return xy.reshape(0, 2)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"{name} must have shape (N, >=2), got {xy.shape}")
    return xy[:, :2]


def _plot_xy(ax: plt.Axes, xy: np.ndarray, *args: Any, **kwargs: Any) -> None:
    xy = _require_xy("xy", xy)
    if xy.size == 0:
        return
    ax.plot(xy[:, 1], xy[:, 0], *args, **kwargs)


def _trajectory(poses: np.ndarray) -> Trajectory:
    poses = _require_xy("trajectory poses", poses)
    if poses.shape[0] == 0:
        raise ValueError("Cannot build a NAVSIM trajectory from an empty pose array")
    poses = np.concatenate([poses, np.zeros((poses.shape[0], 1), dtype=poses.dtype)], axis=1)
    return Trajectory(poses.astype(np.float32), TrajectorySampling(num_poses=len(poses), interval_length=0.5))


def _annotations_from_frame(frame: Mapping[str, Any]) -> Annotations:
    anns = frame["anns"]
    return Annotations(
        boxes=np.asarray(anns["gt_boxes"], dtype=np.float32),
        names=[str(name) for name in anns["gt_names"]],
        velocity_3d=np.asarray(anns["gt_velocity_3d"], dtype=np.float32),
        instance_tokens=[str(token) for token in anns["instance_tokens"]],
        track_tokens=[str(token) for token in anns["track_tokens"]],
    )


def _planning_xy_from_result(result: dict[str, Any]) -> np.ndarray:
    planning = result["planning"]["result_planning"]["sdc_traj"]
    planning_np = _as_numpy(planning)
    if planning_np.ndim == 3:
        planning_np = planning_np[0]
    return _require_xy("planning.sdc_traj", planning_np).astype(np.float32)


def _predictions_from_run(run: NavsimUniADRunResult) -> dict[str, np.ndarray]:
    result = run.model_result
    predictions: dict[str, np.ndarray] = {"pred_ego_xy": _planning_xy_from_result(result)}

    boxes = result.get("boxes_3d")
    if boxes is not None and hasattr(boxes, "tensor"):
        predictions["pred_boxes_3d"] = _as_numpy(boxes.tensor).astype(np.float32)
    for key in ("scores_3d", "labels_3d", "traj", "traj_scores"):
        if key in result and torch.is_tensor(result[key]):
            predictions[f"pred_{key}"] = _as_numpy(result[key])

    pts_bbox = result.get("pts_bbox")
    if isinstance(pts_bbox, Mapping):
        for key in ("bbox", "labels", "lane", "lane_score", "drivable"):
            if key in pts_bbox and torch.is_tensor(pts_bbox[key]):
                predictions[f"pred_map_{key}"] = _as_numpy(pts_bbox[key])

    occ = result.get("occ")
    if isinstance(occ, Mapping) and "seg_out" in occ and torch.is_tensor(occ["seg_out"]):
        predictions["pred_occ_seg_out"] = _as_numpy(occ["seg_out"])
    return predictions


def _data_from_run(run: NavsimUniADRunResult) -> NavsimUniADVisualizationData:
    return NavsimUniADVisualizationData(
        config=run.config,
        sample=run.sample,
        predictions=_predictions_from_run(run),
        metrics=run.metrics,
        output_shapes=run.output_shapes,
        artifact_path=run.artifact_path,
        source="rerun inference",
    )


def _artifact_path_for_sample(sample: NavsimUniADSample, viz_config: VisualizationConfig) -> Path:
    if viz_config.artifact_path is not None:
        path = viz_config.artifact_path
    else:
        path = viz_config.output_dir / f"{sample.final_input.token}_forward_navsim_artifact.npz"
    if not path.is_file():
        raise ArtifactUnavailable(f"forward artifact not found: {path}")
    return path


def _load_data_from_artifact(viz_config: VisualizationConfig) -> NavsimUniADVisualizationData:
    sample = load_navsim_uniad_sample(viz_config.run_config.navsim)
    artifact_path = _artifact_path_for_sample(sample, viz_config)
    with np.load(artifact_path, allow_pickle=False) as artifact:
        frame_token = _scalar_str(artifact["frame_token"])
        if frame_token != sample.final_input.token:
            raise ArtifactUnavailable(
                f"artifact frame_token={frame_token} does not match configured NAVSIM sample={sample.final_input.token}"
            )
        metric_frame = _scalar_str(artifact["metric_frame"]) if "metric_frame" in artifact.files else ""
        if metric_frame and metric_frame != "current_ego_local_xy":
            raise ValueError(f"Unsupported artifact metric_frame={metric_frame}; expected current_ego_local_xy")

        predictions = {key: np.asarray(artifact[key]) for key in artifact.files if key.startswith("pred_")}
        if "pred_ego_xy" not in predictions:
            raise ValueError(f"Artifact is missing pred_ego_xy: {artifact_path}")
        if "gt_ego_xy" in artifact.files:
            gt_artifact = _require_xy("artifact.gt_ego_xy", np.asarray(artifact["gt_ego_xy"]))
            gt_sample = _require_xy(
                "sample.ego_future_trajectory",
                sample.ego_future_trajectory.detach().cpu().numpy()[: gt_artifact.shape[0], :2],
            )
            if gt_sample.shape != gt_artifact.shape or not np.allclose(gt_sample, gt_artifact, atol=1e-3):
                raise ValueError("Artifact GT trajectory does not match the configured NAVSIM sample")
        metrics = json.loads(_scalar_str(artifact["metrics_json"])) if "metrics_json" in artifact.files else {}
        output_shapes = json.loads(_scalar_str(artifact["output_shapes_json"])) if "output_shapes_json" in artifact.files else {}

    return NavsimUniADVisualizationData(
        config=viz_config.run_config,
        sample=sample,
        predictions=predictions,
        metrics={str(key): float(value) for key, value in metrics.items()},
        output_shapes=output_shapes,
        artifact_path=artifact_path,
        source="forward artifact",
    )


def build_visualization_data(viz_config: VisualizationConfig = VIS_CONFIG) -> NavsimUniADVisualizationData:
    if viz_config.use_forward_artifact:
        try:
            return _load_data_from_artifact(viz_config)
        except ArtifactUnavailable as error:
            if not viz_config.rerun_inference_if_artifact_missing:
                raise
            print(f"Forward artifact unavailable ({error}); rerunning UniAD inference.")
    return _data_from_run(run_navsim_uniad(viz_config.run_config))


def _obstacle_keep_indices(predictions: Mapping[str, np.ndarray], viz_config: VisualizationConfig) -> np.ndarray:
    boxes = predictions.get("pred_boxes_3d")
    if boxes is None or np.asarray(boxes).size == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"pred_boxes_3d must have shape (N, >=7), got {boxes.shape}")
    keep = np.arange(boxes.shape[0], dtype=np.int64)
    scores = predictions.get("pred_scores_3d")
    if scores is not None and np.asarray(scores).size:
        scores = np.asarray(scores).reshape(-1)
        if scores.shape[0] != boxes.shape[0]:
            raise ValueError(f"pred_scores_3d length {scores.shape[0]} does not match boxes {boxes.shape[0]}")
        keep = keep[scores > viz_config.obstacle_score_threshold]
    return keep


def _plot_predicted_boxes(ax: plt.Axes, predictions: Mapping[str, np.ndarray], viz_config: VisualizationConfig) -> None:
    boxes = predictions.get("pred_boxes_3d")
    if boxes is None or np.asarray(boxes).size == 0:
        return
    boxes = np.asarray(boxes, dtype=np.float32)
    keep = _obstacle_keep_indices(predictions, viz_config)[: viz_config.max_predicted_obstacles]
    for idx in keep:
        x, y, _z, length, width, height, yaw = boxes[idx, :7]
        oriented_box = OrientedBox(StateSE2(float(x), float(y), float(yaw)), float(length), float(width), float(max(height, 0.1)))
        add_oriented_box_to_bev_ax(ax, oriented_box, PRED_BOX_CONFIG)
    if keep.size:
        ax.plot([], [], color=PRED_BOX_CONFIG["line_color"], linewidth=1.0, label="pred obstacles")


def _plot_predicted_obstacle_trajs(ax: plt.Axes, predictions: Mapping[str, np.ndarray], viz_config: VisualizationConfig) -> None:
    traj = predictions.get("pred_traj")
    traj_scores = predictions.get("pred_traj_scores")
    if traj is None or traj_scores is None or np.asarray(traj).size == 0:
        return
    traj = np.asarray(traj, dtype=np.float32)
    traj_scores = np.asarray(traj_scores)
    if traj.ndim != 4 or traj.shape[-1] < 2:
        raise ValueError(f"pred_traj must have shape (N, M, T, >=2), got {traj.shape}")
    if traj_scores.ndim != 2 or traj_scores.shape[:2] != traj.shape[:2]:
        raise ValueError(f"pred_traj_scores shape {traj_scores.shape} is incompatible with pred_traj {traj.shape}")

    boxes = predictions.get("pred_boxes_3d")
    boxes_np = np.asarray(boxes, dtype=np.float32) if boxes is not None and np.asarray(boxes).size else None
    keep = _obstacle_keep_indices(predictions, viz_config)
    if keep.size == 0:
        keep = np.arange(traj.shape[0], dtype=np.int64)
    keep = keep[keep < traj.shape[0]][: viz_config.max_predicted_obstacle_trajs]
    mode_idx = traj_scores.argmax(axis=-1)
    plotted = False
    for obj_idx in keep:
        points = traj[obj_idx, mode_idx[obj_idx], :, :2]
        if boxes_np is not None and obj_idx < boxes_np.shape[0]:
            points = points + boxes_np[obj_idx, :2]
        _plot_xy(ax, points, color="#ff7f0e", alpha=0.8, linewidth=0.8, marker=".", markersize=2)
        plotted = True
    if plotted:
        ax.plot([], [], color="#ff7f0e", linewidth=0.8, marker=".", markersize=2, label="pred obstacle futures")


def _reduce_bev_mask(name: str, array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.size == 0:
        return np.zeros((0, 0), dtype=bool)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least two BEV dimensions, got {array.shape}")
    if array.ndim == 2:
        return array != 0
    return np.any(array != 0, axis=tuple(range(array.ndim - 2)))


def _grid_points(mask: np.ndarray, pc_extent: float) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    height, width = mask.shape
    x = (cols + 0.5) / width * pc_extent - pc_extent / 2.0
    y = (rows + 0.5) / height * pc_extent - pc_extent / 2.0
    return np.stack([x, y], axis=-1).astype(np.float32)


def _bev_pixel_to_local_xy(pixel_xy: np.ndarray, bev_hw: tuple[int, int], pc_extent: float) -> np.ndarray:
    pixel_xy = np.asarray(pixel_xy, dtype=np.float32)
    height, width = bev_hw
    x = pixel_xy[..., 0] / width * pc_extent - pc_extent / 2.0
    y = pixel_xy[..., 1] / height * pc_extent - pc_extent / 2.0
    return np.stack([x, y], axis=-1).astype(np.float32)


def _plot_predicted_map(ax: plt.Axes, data: NavsimUniADVisualizationData, viz_config: VisualizationConfig) -> None:
    predictions = data.predictions
    extent = _bev_extent(data, viz_config)
    plotted_mask = False
    for key, color, label, alpha in (
        ("pred_map_drivable", "#9edae5", "pred drivable", 0.18),
        ("pred_map_lane", "#9467bd", "pred lane cells", 0.45),
        ("pred_map_lane_score", "#9467bd", "pred lane scores", 0.35),
    ):
        if key not in predictions:
            continue
        mask = _reduce_bev_mask(key, predictions[key])
        points = _grid_points(mask, extent)
        if points.size:
            ax.scatter(points[:, 1], points[:, 0], s=2, c=color, alpha=alpha, label=label, zorder=2)
            plotted_mask = True

    bbox = predictions.get("pred_map_bbox")
    if bbox is None or np.asarray(bbox).size == 0:
        return
    bbox = np.asarray(bbox, dtype=np.float32)
    if bbox.ndim != 2 or bbox.shape[1] < 4:
        raise ValueError(f"pred_map_bbox must have shape (N, >=4), got {bbox.shape}")
    order = np.arange(bbox.shape[0])
    if bbox.shape[1] >= 5:
        order = order[np.argsort(-bbox[:, 4])]
    order = order[: viz_config.max_predicted_map_elements]
    bev_hw = data.config.navsim.bev_hw
    for draw_idx, idx in enumerate(order):
        x1, y1, x2, y2 = bbox[idx, :4]
        corners_px = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], dtype=np.float32)
        corners_xy = _bev_pixel_to_local_xy(corners_px, bev_hw, extent)
        ax.plot(
            corners_xy[:, 1],
            corners_xy[:, 0],
            color="#9467bd",
            alpha=0.65 if draw_idx < 10 else 0.3,
            linewidth=0.8,
            linestyle="--",
            zorder=2,
        )
    label = "pred map boxes" if not plotted_mask else "pred map box fallback"
    ax.plot([], [], color="#9467bd", linewidth=0.8, linestyle="--", label=label)


def _plot_predicted_occ(ax: plt.Axes, data: NavsimUniADVisualizationData, viz_config: VisualizationConfig) -> None:
    occ = data.predictions.get("pred_occ_seg_out")
    if occ is None or np.asarray(occ).size == 0:
        return
    mask = _reduce_bev_mask("pred_occ_seg_out", occ)
    points = _grid_points(mask, _bev_extent(data, viz_config))
    if points.size:
        ax.scatter(points[:, 1], points[:, 0], s=2, c="#d62728", alpha=0.25, label="pred occ", zorder=3)


def _plot_prediction(ax: plt.Axes, data: NavsimUniADVisualizationData, viz_config: VisualizationConfig) -> None:
    predictions = data.predictions
    ax.set_title("UniAD model output")
    ax.scatter([0], [0], marker="x", color="black", s=35, label="ego", zorder=6)
    _plot_predicted_map(ax, data, viz_config)
    _plot_predicted_occ(ax, data, viz_config)
    _plot_predicted_boxes(ax, predictions, viz_config)
    _plot_predicted_obstacle_trajs(ax, predictions, viz_config)
    pred_traj = _require_xy("pred_ego_xy", predictions["pred_ego_xy"])
    if pred_traj.shape[0] == 0:
        raise ValueError("pred_ego_xy is empty; the model output does not contain an ego trajectory")
    add_trajectory_to_bev_ax(ax, _trajectory(pred_traj), TRAJECTORY_CONFIG["agent"])

    metrics = data.metrics
    metric_text = (
        f"ADE={metrics.get('planning_ade_m', float('nan')):.2f}m "
        f"FDE={metrics.get('planning_fde_m', float('nan')):.2f}m\n"
        f"boxes={metrics.get('predicted_obstacle_count', 0.0):.0f} "
        f"map_cells={metrics.get('predicted_map_cells', 0.0):.0f} "
        f"occ_cells={metrics.get('predicted_occ_cells', 0.0):.0f}"
    )
    ax.text(
        0.01,
        0.02,
        metric_text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.legend(loc="upper right", fontsize=8)


def _plot_ground_truth(ax: plt.Axes, data: NavsimUniADVisualizationData, viz_config: VisualizationConfig) -> None:
    sample = data.sample
    ax.set_title("NAVSIM ground truth")
    if sample.map_api is not None:
        add_map_to_bev_ax(ax, sample.map_api, StateSE2(*frame_pose(sample.current_frame)))
    add_annotations_to_bev_ax(ax, _annotations_from_frame(sample.current_frame), add_ego=True)
    ax.scatter([0], [0], marker="x", color="black", s=35, label="GT ego", zorder=6)
    gt_ego = _require_xy("sample.ego_future_trajectory", sample.ego_future_trajectory.detach().cpu().numpy())
    if gt_ego.shape[0] == 0:
        raise ValueError("sample.ego_future_trajectory is empty; cannot draw GT ego trajectory")
    add_trajectory_to_bev_ax(ax, _trajectory(gt_ego), TRAJECTORY_CONFIG["human"])
    plotted_future = False
    for obstacle_traj in sample.obstacle_future_trajectories[: viz_config.max_gt_obstacle_trajs]:
        _plot_xy(ax, obstacle_traj.points, color="#2ca02c", alpha=0.75, linewidth=0.8, marker=".", markersize=2)
        plotted_future = True
    ax.plot([], [], color=AGENT_CONFIG[TrackedObjectType.VEHICLE]["line_color"], label="GT obstacles")
    if plotted_future:
        ax.plot([], [], color="#2ca02c", linewidth=0.8, marker=".", markersize=2, label="GT obstacle futures")
    ax.legend(loc="upper right", fontsize=8)


def visualize_navsim_uniad(
    data: NavsimUniADVisualizationData,
    output_path: Path | None = None,
    viz_config: VisualizationConfig = VIS_CONFIG,
) -> Path:
    if output_path is None:
        output_path = viz_config.output_dir / f"{data.sample.final_input.token}_uniad_navsim.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    _plot_prediction(axes[0], data, viz_config)
    _plot_ground_truth(axes[1], data, viz_config)
    for ax in axes:
        _configure_bev(ax, viz_config)
    fig.suptitle(f"UniAD NAVSIM {data.config.navsim.split}: {data.sample.final_input.token}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    data = build_visualization_data(VIS_CONFIG)
    output_path = visualize_navsim_uniad(data, viz_config=VIS_CONFIG)
    print(f"Saved UniAD NAVSIM visualization: {output_path}")
    print(f"Visualization source: {data.source}")
    if data.artifact_path is not None:
        print(f"Forward artifact: {data.artifact_path}")


if __name__ == "__main__":
    main()
