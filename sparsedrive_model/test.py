from __future__ import annotations

import importlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSEDRIVE_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SPARSEDRIVE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# -----------------------------------------------------------------------------
# Editable mini-inference configuration. This script intentionally has no CLI
# parser; edit these constants when you want to run a different sample set.
# -----------------------------------------------------------------------------
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")
SPLIT = "mini"
CONFIG_NAME = "stage2"  # "stage1" or "stage2"; stage2 enables motion/planning.
CHECKPOINT_PATH = "/Users/chenran/Code/e2e_av_from_scratch/model_weights/sparse_drive/sparsedrive_stage1.pth"
LIMIT = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Valid NAVSIM camera names: CAM_F0, CAM_L0, CAM_L1, CAM_L2, CAM_R0, CAM_R1,
# CAM_R2, CAM_B0. SparseDrive expects exactly six cameras in this order.
CAMERA_ORDER = (
    "CAM_F0",  # front
    "CAM_R0",  # front/right side
    "CAM_L0",  # front/left side
    "CAM_B0",  # rear
    "CAM_L2",  # rear/left side
    "CAM_R2",  # rear/right side
)

OUTPUT_DIR = "sparsedrive_model/outputs/navsim_mini_inference"
VISUALIZE = False
# -----------------------------------------------------------------------------

if TYPE_CHECKING:
    from prediction_decode import DecodedSparseDrivePrediction

CONFIG_MODULES = {
    "stage1": "configs.sparsedrive_small_stage1",
    "stage2": "configs.sparsedrive_small_stage2",
}
STATE_DICT_WRAPPERS = ("state_dict", "model", "model_state_dict", "net")
VISUALIZATION_HELPERS = (
    ("visualize_navsim_inference", "visualize_navsim_sparsedrive_predictions"),
    ("sparsedrive_model.visualize_navsim_inference", "visualize_navsim_sparsedrive_predictions"),
    ("navsim_visualization", "visualize_navsim_sparsedrive_predictions"),
    ("tools.visualization.navsim_visualization", "visualize_navsim_sparsedrive_predictions"),
    ("tools.visualization.visualize_navsim", "visualize_navsim_sparsedrive_predictions"),
)


def _resolve_repo_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(os.fspath(path_like))))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _require_checkpoint_path(path_like: str | os.PathLike[str]) -> Path:
    checkpoint_path = _resolve_repo_path(path_like)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SparseDrive checkpoint not found: {checkpoint_path}. Place the stage2 "
            "weights there or edit CHECKPOINT_PATH in sparsedrive_model/test.py; "
            "this script refuses to run with random weights."
        )
    return checkpoint_path


def _build_model(config_name: str) -> torch.nn.Module:
    normalized = config_name.lower().strip()
    module_name = CONFIG_MODULES.get(normalized)
    if module_name is None:
        valid = ", ".join(sorted(CONFIG_MODULES))
        raise ValueError(f"Unsupported CONFIG_NAME={config_name!r}; expected one of: {valid}.")

    config_module = importlib.import_module(module_name)
    model = config_module.build()
    if hasattr(model, "init_weights"):
        model.init_weights()
    return model


def _torch_load_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch versions do not expose weights_only.
        return torch.load(checkpoint_path, map_location="cpu")


def _looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and any(torch.is_tensor(item) for item in value.values())


def _extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if _looks_like_state_dict(checkpoint):
        return checkpoint, "raw state dict"

    if isinstance(checkpoint, Mapping):
        for wrapper_key in STATE_DICT_WRAPPERS:
            if wrapper_key not in checkpoint:
                continue
            wrapped = checkpoint[wrapper_key]
            if _looks_like_state_dict(wrapped):
                return wrapped, wrapper_key
            if isinstance(wrapped, Mapping):
                for nested_key in STATE_DICT_WRAPPERS:
                    nested = wrapped.get(nested_key)
                    if _looks_like_state_dict(nested):
                        return nested, f"{wrapper_key}.{nested_key}"

    expected = ", ".join(STATE_DICT_WRAPPERS)
    raise ValueError(
        "Checkpoint does not contain a usable SparseDrive state dict. Expected a raw "
        f"state dict or one wrapped by one of: {expected}."
    )


def _strip_common_state_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"State-dict key {key!r} is not a string.")
        cleaned_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[len(prefix) :]
                    changed = True
        cleaned[cleaned_key] = value
    return cleaned


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    state_dict, wrapper_name = _extract_state_dict(checkpoint)
    state_dict = _strip_common_state_prefixes(state_dict)

    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Strict checkpoint load failed for {checkpoint_path}. The checkpoint was "
            f"read from wrapper '{wrapper_name}', but its keys do not match the "
            "selected SparseDrive config. Original error follows:\n"
            f"{exc}"
        ) from exc

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        f"Loaded checkpoint: {checkpoint_path} ({wrapper_name}); "
        f"missing keys={len(missing)}, unexpected keys={len(unexpected)}"
    )


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE requests CUDA, but torch.cuda.is_available() is False.")
    return device


def _load_navsim_samples() -> list[Any]:
    from navsim_adapter import (
        build_navsim_scene_loader,
        load_navsim_sparsedrive_samples,
        normalize_camera_order,
    )

    normalize_camera_order(CAMERA_ORDER)
    if LIMIT is not None and LIMIT <= 0:
        raise ValueError(f"LIMIT must be positive or None, got {LIMIT!r}.")

    scene_loader = build_navsim_scene_loader(
        split=SPLIT,
        camera_order=CAMERA_ORDER,
        openscene_data_root=OPENSCENE_DATA_ROOT,
        nuplan_maps_root=NUPLAN_MAPS_ROOT,
        max_scenes=LIMIT,
    )
    samples = load_navsim_sparsedrive_samples(
        scene_loader,
        max_samples=LIMIT,
        camera_order=CAMERA_ORDER,
        maps_root=NUPLAN_MAPS_ROOT,
        include_map_api=VISUALIZE,
    )
    if not samples:
        raise RuntimeError(f"No NAVSIM samples were loaded for split={SPLIT!r}.")
    return samples


def _sanitize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token)[:120] or "sample"


def _rounded_rows(array: np.ndarray, limit: int = 5, decimals: int = 4) -> list[Any]:
    return np.round(np.asarray(array)[:limit], decimals=decimals).tolist()


def _summary_for_sample(
    sample: Any,
    output: Mapping[str, Any],
    decoded: "DecodedSparseDrivePrediction",
) -> dict[str, Any]:
    img_bbox = output.get("img_bbox", output)
    output_keys = sorted(img_bbox.keys()) if isinstance(img_bbox, Mapping) else []
    gt_cmd = sample.gt_ego_fut_cmd.detach().cpu().numpy()

    return {
        "token": sample.token,
        "map_name": sample.map_name,
        "output_keys": output_keys,
        "gt_ego_fut_cmd": gt_cmd.tolist(),
        "gt_ego_fut_cmd_index": int(np.argmax(gt_cmd)),
        "planning_source": decoded.planning_source,
        "planning_command_index": decoded.planning_command_index,
        "planning_mode_index": decoded.planning_mode_index,
        "ego_trajectory_shape": list(decoded.predicted_ego_trajectory.shape),
        "ego_trajectory": _rounded_rows(decoded.predicted_ego_trajectory, limit=20),
        "num_obstacles": int(decoded.predicted_obstacle_boxes.shape[0]),
        "top_obstacle_scores": _rounded_rows(decoded.obstacle_scores, limit=10),
        "top_obstacle_labels": decoded.obstacle_labels[:10].astype(int, copy=False).tolist(),
        "top_obstacle_boxes": _rounded_rows(decoded.predicted_obstacle_boxes, limit=5),
        "num_map_polylines": int(len(decoded.predicted_map_polylines)),
        "top_map_scores": _rounded_rows(decoded.map_scores, limit=10),
        "top_map_labels": decoded.map_labels[:10].astype(int, copy=False).tolist(),
        "top_map_polyline_lengths": [
            int(polyline.shape[0]) for polyline in decoded.predicted_map_polylines[:10]
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _import_visualization_helper() -> Any:
    failures: list[str] = []
    for module_name, attr_name in VISUALIZATION_HELPERS:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            failures.append(f"{module_name}: {exc}")
            continue
        helper = getattr(module, attr_name, None)
        if callable(helper):
            return helper
        failures.append(f"{module_name}: missing callable {attr_name}")

    expected = ", ".join(f"{module}.{attr}" for module, attr in VISUALIZATION_HELPERS)
    raise RuntimeError(
        "VISUALIZE=True, but no NAVSIM SparseDrive visualization helper was found. "
        f"Expected one of: {expected}. Import failures: {'; '.join(failures)}"
    )


def _maybe_visualize(
    samples: Sequence[Any],
    outputs: Sequence[Mapping[str, Any]],
    decoded_predictions: Sequence["DecodedSparseDrivePrediction"],
    summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    if not VISUALIZE:
        return

    helper = _import_visualization_helper()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    try:
        helper(
            samples=samples,
            outputs=outputs,
            decoded_predictions=decoded_predictions,
            summaries=summaries,
            output_dir=vis_dir,
        )
    except TypeError as exc:
        raise TypeError(
            "Visualization helper must accept keyword arguments: samples, outputs, "
            "decoded_predictions, summaries, and output_dir."
        ) from exc
    print(f"Visualization outputs written under: {vis_dir}")


def main() -> None:
    checkpoint_path = _require_checkpoint_path(CHECKPOINT_PATH)
    output_dir = _resolve_repo_path(OUTPUT_DIR)
    device = _resolve_device(DEVICE)

    print(f"Building SparseDrive {CONFIG_NAME} model on {device}...")
    model = _build_model(CONFIG_NAME)
    _load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    print(f"Loading NAVSIM split={SPLIT!r}, limit={LIMIT}...")
    from navsim_adapter import collate_navsim_sparsedrive_samples, sample_to_device
    from prediction_decode import DecodedSparseDrivePrediction, decode_sparsedrive_outputs

    samples = _load_navsim_samples()
    batch = collate_navsim_sparsedrive_samples(samples)
    batch = sample_to_device(batch, device)

    print(f"Running inference for {len(samples)} sample(s)...")
    with torch.no_grad():
        outputs = model(**batch)

    if not isinstance(outputs, Sequence) or len(outputs) != len(samples):
        raise RuntimeError(
            f"Expected one output per sample ({len(samples)}), got {type(outputs).__name__} "
            f"with length {len(outputs) if isinstance(outputs, Sequence) else 'n/a'}."
        )

    decoded_predictions: list[DecodedSparseDrivePrediction] = []
    summaries: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, output) in enumerate(zip(samples, outputs)):
        decoded = decode_sparsedrive_outputs(outputs, sample_index=index)
        decoded_predictions.append(decoded)
        summary = _summary_for_sample(sample, output, decoded)
        summaries.append(summary)
        sample_path = output_dir / f"{index:03d}_{_sanitize_token(sample.token)}.json"
        _write_json(sample_path, summary)
        print(
            f"[{index}] token={sample.token}: "
            f"obstacles={summary['num_obstacles']}, maps={summary['num_map_polylines']}, "
            f"summary={sample_path}"
        )

    aggregate_path = output_dir / "summary.json"
    _write_json(
        aggregate_path,
        {
            "config_name": CONFIG_NAME,
            "checkpoint_path": str(checkpoint_path),
            "split": SPLIT,
            "limit": LIMIT,
            "device": str(device),
            "camera_order": list(CAMERA_ORDER),
            "num_samples": len(samples),
            "samples": summaries,
        },
    )
    _maybe_visualize(samples, outputs, decoded_predictions, summaries, output_dir)
    print(f"Wrote aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
