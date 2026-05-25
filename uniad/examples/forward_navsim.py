from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import sys
import warnings
from typing import Any, Mapping

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from uniad import build_uniad, load_checkpoint
from uniad.examples.navsim_uniad_dataset import NavsimUniADConfig, NavsimUniADSample, load_navsim_uniad_sample


@dataclass(frozen=True)
class ForwardNavsimConfig:
    navsim: NavsimUniADConfig = field(default_factory=NavsimUniADConfig)
    checkpoint_path: Path = _REPO_ROOT / "model_weights" / "uniad" / "uniad_base_e2e.pth"
    output_dir: Path = _REPO_ROOT / "exp" / "uniad_navsim_debug"
    seed: int = 20240523
    device: str = "cpu"
    use_dcn: bool = True
    dummy_motion_anchors: bool = True
    use_col_optim: bool = False


CONFIG = ForwardNavsimConfig()


@dataclass
class NavsimUniADRunResult:
    config: ForwardNavsimConfig
    sample: NavsimUniADSample
    model_result: dict[str, Any]
    metrics: dict[str, float]
    output_shapes: dict[str, Any]
    input_shapes: dict[str, Any]
    calibration_checks: dict[str, Any]
    checkpoint_load: dict[str, Any]
    artifact_path: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(config: ForwardNavsimConfig) -> torch.device:
    if config.device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(config.device)


def _shape_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return tuple(value.shape)
    if hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        return {"box_tensor": tuple(value.tensor.shape)}
    if isinstance(value, Mapping):
        return {str(key): _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_tree(item) for item in value[:5]]
    return type(value).__name__


def _assert_finite_tree(value: Any, prefix: str = "result") -> None:
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite tensor values in {prefix}")
    elif hasattr(value, "tensor") and torch.is_tensor(value.tensor):
        _assert_finite_tree(value.tensor, f"{prefix}.tensor")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _assert_finite_tree(item, f"{prefix}[{idx}]")


def _input_shape_summary(sample: NavsimUniADSample) -> dict[str, Any]:
    final_input = sample.final_input
    meta = final_input.img_metas[0][0]
    return {
        "history_frame_count": len(sample.frame_inputs),
        "frame_tokens": [frame_input.token for frame_input in sample.frame_inputs],
        "img": tuple(final_input.img.shape),
        "img_arg": [tuple(final_input.img.shape)],
        "img_metas": (len(final_input.img_metas), len(final_input.img_metas[0])),
        "lidar2img": tuple(np.asarray(meta["lidar2img"]).shape),
        "can_bus": tuple(np.asarray(meta["can_bus"]).shape),
        "l2g_t": tuple(final_input.l2g_t.shape),
        "l2g_r_mat": tuple(final_input.l2g_r_mat.shape),
        "timestamp": [tuple(timestamp.shape) for timestamp in final_input.timestamp],
        "gt_lane_labels": [tuple(label.shape) for label in final_input.gt_lane_labels],
        "gt_lane_masks": [tuple(mask.shape) for mask in final_input.gt_lane_masks],
        "gt_segmentation": [tuple(seg.shape) for seg in final_input.gt_segmentation],
        "gt_instance": [tuple(instance.shape) for instance in final_input.gt_instance],
        "gt_occ_img_is_valid": [tuple(valid.shape) for valid in final_input.gt_occ_img_is_valid],
        "sdc_planning": tuple(final_input.sdc_planning.shape),
        "sdc_planning_mask": tuple(final_input.sdc_planning_mask.shape),
        "command": tuple(final_input.command.shape),
    }


def _calibration_summary(sample: NavsimUniADSample) -> dict[str, Any]:
    lidar2img = np.stack([np.asarray(frame_input.img_metas[0][0]["lidar2img"]) for frame_input in sample.frame_inputs])
    can_bus = np.stack([np.asarray(frame_input.img_metas[0][0]["can_bus"]) for frame_input in sample.frame_inputs])
    l2g_t = torch.cat([frame_input.l2g_t.detach().cpu().reshape(1, 3) for frame_input in sample.frame_inputs], dim=0)
    l2g_r_mat = torch.cat([frame_input.l2g_r_mat.detach().cpu().reshape(1, 3, 3) for frame_input in sample.frame_inputs], dim=0)
    timestamps = torch.stack([frame_input.timestamp[0].detach().cpu().reshape(()) for frame_input in sample.frame_inputs])
    determinants = torch.linalg.det(l2g_r_mat)
    summary = {
        "lidar2img_finite": bool(np.isfinite(lidar2img).all()),
        "can_bus_finite": bool(np.isfinite(can_bus).all()),
        "l2g_t_finite": bool(torch.isfinite(l2g_t).all()),
        "l2g_r_mat_finite": bool(torch.isfinite(l2g_r_mat).all()),
        "timestamp_finite": bool(torch.isfinite(timestamps).all()),
        "lidar2img_max_abs": float(np.abs(lidar2img).max()),
        "can_bus_max_abs": float(np.abs(can_bus).max()),
        "l2g_t_max_abs": float(l2g_t.abs().max()),
        "l2g_r_mat_det_min": float(determinants.min()),
        "l2g_r_mat_det_max": float(determinants.max()),
    }
    finite_keys = ("lidar2img_finite", "can_bus_finite", "l2g_t_finite", "l2g_r_mat_finite", "timestamp_finite")
    if not all(bool(summary[key]) for key in finite_keys):
        raise RuntimeError(f"Non-finite NAVSIM calibration/timing values: {summary}")
    return summary


def _predicted_planning_xy(result: dict[str, Any]) -> torch.Tensor:
    planning = result["planning"]["result_planning"]["sdc_traj"].detach().cpu()
    if planning.ndim == 3:
        planning = planning[0]
    if planning.ndim != 2 or planning.shape[-1] < 2:
        raise RuntimeError(f"Unexpected planning trajectory shape: {tuple(planning.shape)}")
    return planning[:, :2]


def _gt_planning_xy(sample: NavsimUniADSample, num_steps: int) -> torch.Tensor:
    if sample.final_input.token != sample.current_frame["token"]:
        raise RuntimeError(
            "Final inference frame does not match the NAVSIM current frame used for local ego GT; "
            f"final={sample.final_input.token} current={sample.current_frame['token']}"
        )
    gt = sample.ego_future_trajectory.detach().cpu()[:num_steps, :2]
    if gt.shape[0] != num_steps:
        raise RuntimeError(f"GT planning horizon {gt.shape[0]} does not match prediction {num_steps}")
    return gt


def _planning_metrics(result: dict[str, Any], sample: NavsimUniADSample) -> dict[str, float]:
    planning = _predicted_planning_xy(result)
    gt = _gt_planning_xy(sample, planning.shape[0])
    displacement = torch.linalg.norm(planning - gt, dim=-1)
    return {
        "planning_ade_m": float(displacement.mean()),
        "planning_fde_m": float(displacement[-1]),
        "planning_max_deviation_m": float(displacement.max()),
    }


def _count_outputs(result: dict[str, Any]) -> dict[str, float]:
    counts: dict[str, float] = {
        "predicted_obstacle_count": 0.0,
        "predicted_map_count": 0.0,
        "predicted_map_cells": 0.0,
        "predicted_occupancy_count": 0.0,
        "predicted_occ_cells": 0.0,
    }
    if "boxes_3d" in result and hasattr(result["boxes_3d"], "tensor"):
        counts["predicted_obstacle_count"] = float(len(result["boxes_3d"]))
    if "pts_bbox" in result and isinstance(result["pts_bbox"], Mapping):
        pts_bbox = result["pts_bbox"]
        if "labels" in pts_bbox and torch.is_tensor(pts_bbox["labels"]):
            counts["predicted_map_count"] = float(pts_bbox["labels"].numel())
        elif "bbox" in pts_bbox and torch.is_tensor(pts_bbox["bbox"]):
            counts["predicted_map_count"] = float(pts_bbox["bbox"].shape[0])
        if "lane" in pts_bbox and torch.is_tensor(pts_bbox["lane"]):
            counts["predicted_map_cells"] = float(torch.count_nonzero(pts_bbox["lane"].detach().cpu()))
    if "occ" in result and "seg_out" in result["occ"] and torch.is_tensor(result["occ"]["seg_out"]):
        occupancy_cells = float(torch.count_nonzero(result["occ"]["seg_out"].detach().cpu()))
        counts["predicted_occupancy_count"] = occupancy_cells
        counts["predicted_occ_cells"] = occupancy_cells
    return counts


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_visualization_artifact(
    config: ForwardNavsimConfig,
    sample: NavsimUniADSample,
    result: dict[str, Any],
    metrics: dict[str, float],
    output_shapes: dict[str, Any],
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = config.output_dir / f"{sample.final_input.token}_forward_navsim_artifact.npz"
    pred_ego_xy = _predicted_planning_xy(result).numpy().astype(np.float32)
    gt_ego_xy = _gt_planning_xy(sample, pred_ego_xy.shape[0]).numpy().astype(np.float32)
    arrays: dict[str, Any] = {
        "sample_token": np.array(sample.token),
        "frame_token": np.array(sample.final_input.token),
        "log_name": np.array(str(sample.current_frame.get("log_name", ""))),
        "log_token": np.array(str(sample.current_frame.get("log_token", ""))),
        "scene_token": np.array(sample.final_input.img_metas[0][0]["scene_token"]),
        "map_name": np.array(sample.map_name),
        "metric_frame": np.array("current_ego_local_xy"),
        "metrics_json": np.array(json.dumps(metrics, sort_keys=True)),
        "output_shapes_json": np.array(json.dumps(output_shapes, default=str, sort_keys=True)),
        "pred_ego_xy": pred_ego_xy,
        "gt_ego_xy": gt_ego_xy,
    }

    boxes = result.get("boxes_3d")
    if boxes is not None and hasattr(boxes, "tensor"):
        arrays["pred_boxes_3d"] = _to_numpy(boxes.tensor).astype(np.float32)
    for key in ("scores_3d", "labels_3d", "traj", "traj_scores"):
        if key in result and torch.is_tensor(result[key]):
            arrays[f"pred_{key}"] = _to_numpy(result[key])

    pts_bbox = result.get("pts_bbox")
    if isinstance(pts_bbox, Mapping):
        for key in ("bbox", "labels", "lane", "lane_score", "drivable"):
            if key in pts_bbox and torch.is_tensor(pts_bbox[key]):
                arrays[f"pred_map_{key}"] = _to_numpy(pts_bbox[key])

    occ = result.get("occ")
    if isinstance(occ, Mapping) and "seg_out" in occ and torch.is_tensor(occ["seg_out"]):
        arrays["pred_occ_seg_out"] = _to_numpy(occ["seg_out"])

    np.savez_compressed(artifact_path, **arrays)
    return artifact_path


def run_navsim_uniad(config: ForwardNavsimConfig = CONFIG) -> NavsimUniADRunResult:
    warnings.filterwarnings("ignore", message=r"The arguments `ffn_?.*`", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"The arguments `feedforward_channels`.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*", category=UserWarning)
    _seed_everything(config.seed)

    if not config.checkpoint_path.is_file():
        raise FileNotFoundError(f"UniAD checkpoint not found: {config.checkpoint_path}")

    sample = load_navsim_uniad_sample(config.navsim)
    device = _device(config)
    model = build_uniad(
        bev_h=config.navsim.bev_hw[0],
        bev_w=config.navsim.bev_hw[1],
        use_dcn=config.use_dcn,
        use_col_optim=config.use_col_optim,
        dummy_motion_anchors=config.dummy_motion_anchors,
    )
    checkpoint_load = load_checkpoint(model, str(config.checkpoint_path), strict=False)
    model.to(device)
    model.eval()
    input_shapes = _input_shape_summary(sample)
    calibration_checks = _calibration_summary(sample)

    final_result: dict[str, Any] | None = None
    with torch.no_grad():
        for frame_input in sample.frame_inputs:
            outputs = model(**frame_input.to_model_kwargs(device))
            if not isinstance(outputs, list) or not outputs:
                raise RuntimeError(f"UniAD forward_test expected a non-empty result list, got {type(outputs).__name__}")
            final_result = outputs[0]

    if final_result is None:
        raise RuntimeError("No UniAD result was produced")
    _assert_finite_tree(final_result)
    metrics = _planning_metrics(final_result, sample)
    metrics.update(_count_outputs(final_result))
    output_shapes = _shape_tree(final_result)
    artifact_path = _save_visualization_artifact(config, sample, final_result, metrics, output_shapes)

    return NavsimUniADRunResult(
        config=config,
        sample=sample,
        model_result=final_result,
        metrics=metrics,
        output_shapes=output_shapes,
        input_shapes=input_shapes,
        calibration_checks=calibration_checks,
        checkpoint_load=checkpoint_load,
        artifact_path=artifact_path,
    )


def _print_run_summary(run: NavsimUniADRunResult) -> None:
    sample = run.sample
    final_input = sample.final_input
    checkpoint = run.checkpoint_load
    print("UniAD NAVSIM forward succeeded.")
    print(f"dataset_root: {run.config.navsim.dataset_root}")
    print(f"log_path: {run.config.navsim.log_path}")
    print(f"sensor_path: {run.config.navsim.sensor_path}")
    print(f"split: {run.config.navsim.split}")
    print(f"log_name: {sample.current_frame.get('log_name', '')}")
    print(f"log_token: {sample.current_frame.get('log_token', '')}")
    print(f"sample_token: {sample.token}")
    print(f"frame_token: {final_input.token}")
    print(f"scene_token: {final_input.img_metas[0][0]['scene_token']}")
    print(f"map: {sample.map_name}")
    print(f"device: {run.config.device}")
    print(f"checkpoint_path: {run.config.checkpoint_path}")
    print(f"input_shapes: {run.input_shapes}")
    print(f"calibration_checks: {run.calibration_checks}")
    print(
        "checkpoint_load: "
        f"missing={len(checkpoint['missing_keys'])} "
        f"unexpected={len(checkpoint['unexpected_keys'])} "
        f"skipped_mismatched={len(checkpoint.get('skipped_mismatched_keys', []))}"
    )
    print("trajectory_metric_frame: current_ego_local_xy")
    print(f"planning_ade_m: {run.metrics['planning_ade_m']:.6f}")
    print(f"planning_fde_m: {run.metrics['planning_fde_m']:.6f}")
    print(f"predicted_obstacle_count: {run.metrics['predicted_obstacle_count']:.0f}")
    print(f"predicted_map_count: {run.metrics['predicted_map_count']:.0f}")
    print(f"predicted_occupancy_count: {run.metrics['predicted_occupancy_count']:.0f}")
    print(f"metrics: {run.metrics}")
    print(f"output_shapes: {run.output_shapes}")
    print(f"artifact_path: {run.artifact_path}")


def main() -> None:
    run = run_navsim_uniad(CONFIG)
    _print_run_summary(run)


if __name__ == "__main__":
    main()
