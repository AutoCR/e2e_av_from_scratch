from __future__ import annotations

import importlib
import json
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
# Editable nuScenes mini inference configuration. This script intentionally has
# no CLI parser and does not read environment variables.
# -----------------------------------------------------------------------------
NUSCENES_DATA_ROOT = "/Users/chenran/Code/nuscenes/nuscenes"
NUSCENES_VERSION = "v1.0-mini"
CONFIG_NAME = "stage2"  # "stage1" or "stage2"; stage2 enables motion/planning.
CHECKPOINT_PATH_BY_CONFIG = {
    "stage1": "model_weights/sparse_drive/sparsedrive_stage1.pth",
    "stage2": "model_weights/sparse_drive/sparsedrive_stage2.pth",
}
LIMIT = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

OUTPUT_DIR = "sparsedrive_model/outputs/nuscenes_mini_inference"
VISUALIZE = True
# -----------------------------------------------------------------------------

if TYPE_CHECKING:
    from prediction_decode import DecodedSparseDrivePrediction

CONFIG_MODULES = {
    "stage1": "configs.sparsedrive_small_stage1",
    "stage2": "configs.sparsedrive_small_stage2",
}
STATE_DICT_WRAPPERS = ("state_dict", "model", "model_state_dict", "net")
VISUALIZATION_HELPERS = (
    ("visualize_nuscenes_inference", "visualize_nuscenes_sparsedrive_predictions"),
    ("sparsedrive_model.visualize_nuscenes_inference", "visualize_nuscenes_sparsedrive_predictions"),
)


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _checkpoint_path_for_config(config_name: str) -> str:
    normalized = config_name.lower().strip()
    try:
        return CHECKPOINT_PATH_BY_CONFIG[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(CHECKPOINT_PATH_BY_CONFIG))
        raise ValueError(f"Unsupported CONFIG_NAME={config_name!r}; expected one of: {valid}.") from exc


def _require_checkpoint_path(path_like: str | Path) -> Path:
    checkpoint_path = _resolve_repo_path(path_like)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SparseDrive checkpoint not found: {checkpoint_path}. Edit "
            "CHECKPOINT_PATH_BY_CONFIG in sparsedrive_model/test_nuscenes_mini.py "
            "or place matching weights there; this script refuses to run with "
            "random weights."
        )
    return checkpoint_path


def _get_config_module(config_name: str) -> Any:
    normalized = config_name.lower().strip()
    module_name = CONFIG_MODULES.get(normalized)
    if module_name is None:
        valid = ", ".join(sorted(CONFIG_MODULES))
        raise ValueError(f"Unsupported CONFIG_NAME={config_name!r}; expected one of: {valid}.")
    return importlib.import_module(module_name)


def _image_hw_from_config(config_name: str) -> tuple[int, int]:
    config_module = _get_config_module(config_name)
    input_shape = getattr(config_module, "input_shape", None)
    if input_shape is None:
        hyperparams = getattr(config_module, "hyperparams", None)
        if isinstance(hyperparams, Mapping):
            input_shape = hyperparams.get("input_shape")

    from nuscenes_adapter import image_hw_from_sparsedrive_input_shape

    return image_hw_from_sparsedrive_input_shape(input_shape)


def _build_model(config_name: str) -> torch.nn.Module:
    config_module = _get_config_module(config_name)
    model = config_module.build()
    if hasattr(model, "init_weights"):
        model.init_weights()
    return model


def _torch_load_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
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


def _load_nuscenes_samples(image_hw: tuple[int, int]) -> list[Any]:
    from nuscenes_adapter import load_nuscenes_sparsedrive_samples, normalize_camera_order

    normalize_camera_order(CAMERA_ORDER)
    if LIMIT is not None and LIMIT <= 0:
        raise ValueError(f"LIMIT must be positive or None, got {LIMIT!r}.")

    samples = load_nuscenes_sparsedrive_samples(
        dataset_root=NUSCENES_DATA_ROOT,
        version=NUSCENES_VERSION,
        max_samples=LIMIT,
        camera_order=CAMERA_ORDER,
        image_hw=image_hw,
    )
    if not samples:
        raise RuntimeError(f"No nuScenes samples were loaded for version={NUSCENES_VERSION!r}.")
    return samples


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch[key]
    if not torch.is_tensor(value):
        raise TypeError(f"Batch key {key!r} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Batch key {name!r} has shape {tuple(tensor.shape)}, expected {expected_shape}."
        )
    if tensor.dtype != expected_dtype:
        raise TypeError(f"Batch key {name!r} has dtype {tensor.dtype}, expected {expected_dtype}.")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Batch key {name!r} contains non-finite values.")


def _metadata_transform(meta: Mapping[str, Any], key: str, sample_index: int) -> np.ndarray:
    if key not in meta:
        raise RuntimeError(f"img_metas[{sample_index}] is missing required key {key!r}.")

    value = meta[key]
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"img_metas[{sample_index}][{key!r}] must be a numpy.ndarray, "
            f"got {type(value).__name__}."
        )
    if value.shape != (4, 4):
        raise ValueError(
            f"img_metas[{sample_index}][{key!r}] has shape {value.shape}, expected (4, 4)."
        )

    try:
        matrix = value.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"img_metas[{sample_index}][{key!r}] with dtype {value.dtype} "
            "is not convertible to float32."
        ) from exc
    if not np.isfinite(matrix).all():
        raise ValueError(f"img_metas[{sample_index}][{key!r}] contains non-finite values.")
    return matrix


def _validate_model_batch(
    batch: Mapping[str, Any],
    samples: Sequence[Any],
    image_hw: tuple[int, int],
) -> None:
    if not isinstance(batch, Mapping):
        raise TypeError(f"Model batch must be a mapping, got {type(batch).__name__}.")

    required_keys = {
        "img",
        "projection_mat",
        "image_wh",
        "timestamp",
        "img_metas",
        "gt_ego_fut_cmd",
    }
    missing_keys = sorted(required_keys.difference(batch))
    if missing_keys:
        raise RuntimeError(f"Model batch is missing required key(s): {missing_keys}.")

    batch_size = len(samples)
    camera_count = len(CAMERA_ORDER)
    if camera_count != 6:
        raise ValueError(f"SparseDrive nuScenes batch expects 6 cameras, got {camera_count}.")

    try:
        image_h, image_w = (int(image_hw[0]), int(image_hw[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"image_hw must contain (height, width), got {image_hw!r}.") from exc
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"image_hw dimensions must be positive, got {image_hw!r}.")

    img = _require_tensor(batch, "img")
    projection_mat = _require_tensor(batch, "projection_mat")
    image_wh = _require_tensor(batch, "image_wh")
    timestamp = _require_tensor(batch, "timestamp")
    gt_ego_fut_cmd = _require_tensor(batch, "gt_ego_fut_cmd")

    _validate_tensor(
        img,
        name="img",
        expected_shape=(batch_size, camera_count, 3, image_h, image_w),
        expected_dtype=torch.float32,
    )
    _validate_tensor(
        projection_mat,
        name="projection_mat",
        expected_shape=(batch_size, camera_count, 3, 4),
        expected_dtype=torch.float32,
    )
    _validate_tensor(
        image_wh,
        name="image_wh",
        expected_shape=(batch_size, camera_count, 2),
        expected_dtype=torch.float32,
    )
    expected_image_wh = torch.tensor([image_w, image_h], dtype=torch.float32).view(1, 1, 2)
    if not torch.allclose(image_wh, expected_image_wh.expand_as(image_wh), rtol=0.0, atol=0.0):
        raise ValueError(
            "Batch key 'image_wh' must equal [width, height] for every camera; "
            f"expected {[image_w, image_h]}."
        )

    _validate_tensor(
        timestamp,
        name="timestamp",
        expected_shape=(batch_size,),
        expected_dtype=torch.float32,
    )
    _validate_tensor(
        gt_ego_fut_cmd,
        name="gt_ego_fut_cmd",
        expected_shape=(batch_size, 3),
        expected_dtype=torch.float32,
    )
    if not torch.allclose(
        gt_ego_fut_cmd.sum(dim=1),
        torch.ones(batch_size, dtype=torch.float32),
        rtol=1.0e-4,
        atol=1.0e-4,
    ):
        raise ValueError("Batch key 'gt_ego_fut_cmd' rows must sum close to 1.")
    if ((gt_ego_fut_cmd < -1.0e-4) | (gt_ego_fut_cmd > 1.0 + 1.0e-4)).any().item():
        raise ValueError("Batch key 'gt_ego_fut_cmd' must contain one-hot-like values in [0, 1].")

    img_metas = batch["img_metas"]
    if not isinstance(img_metas, list):
        raise TypeError(f"Batch key 'img_metas' must be a list, got {type(img_metas).__name__}.")
    if len(img_metas) != batch_size:
        raise ValueError(f"Batch key 'img_metas' has length {len(img_metas)}, expected {batch_size}.")

    identity = np.eye(4, dtype=np.float32)
    for sample_index, meta in enumerate(img_metas):
        if not isinstance(meta, Mapping):
            raise TypeError(
                f"img_metas[{sample_index}] must be a mapping, got {type(meta).__name__}."
            )
        t_global = _metadata_transform(meta, "T_global", sample_index)
        t_global_inv = _metadata_transform(meta, "T_global_inv", sample_index)
        if not np.allclose(t_global @ t_global_inv, identity, rtol=1.0e-3, atol=1.0e-3):
            raise ValueError(
                f"img_metas[{sample_index}] T_global and T_global_inv do not multiply "
                "close to identity."
            )
        if not np.allclose(t_global_inv @ t_global, identity, rtol=1.0e-3, atol=1.0e-3):
            raise ValueError(
                f"img_metas[{sample_index}] T_global_inv and T_global do not multiply "
                "close to identity."
            )


def _sanitize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token)[:120] or "sample"


def _rounded_rows(array: np.ndarray, limit: int = 5, decimals: int = 4) -> list[Any]:
    return np.round(np.asarray(array)[:limit], decimals=decimals).tolist()


def _summary_for_sample(
    sample: Any,
    output: Mapping[str, Any],
    decoded: "DecodedSparseDrivePrediction",
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    img_bbox = output.get("img_bbox", output)
    output_keys = sorted(img_bbox.keys()) if isinstance(img_bbox, Mapping) else []
    top_level_output_keys = sorted(output.keys()) if isinstance(output, Mapping) else []
    gt_cmd = sample.gt_ego_fut_cmd.detach().cpu().numpy()
    timestamp_seconds = float(sample.timestamp.detach().cpu().item())

    return {
        "token": sample.token,
        "scene_name": sample.scene_name,
        "map_name": sample.map_name,
        "timestamp": timestamp_seconds,
        "raw_sample_timestamp": sample.sample.get("timestamp"),
        "raw_lidar_timestamp": sample.lidar_sample_data.get("timestamp"),
        "dataset_root": NUSCENES_DATA_ROOT,
        "version": NUSCENES_VERSION,
        "config_name": CONFIG_NAME,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "limit": LIMIT,
        "camera_order": list(CAMERA_ORDER),
        "output_keys": output_keys,
        "top_level_output_keys": top_level_output_keys,
        "gt_ego_fut_cmd": gt_cmd.tolist(),
        "gt_ego_fut_cmd_index": int(np.argmax(gt_cmd)),
        "planning": {
            "source": decoded.planning_source,
            "command_index": decoded.planning_command_index,
            "mode_index": decoded.planning_mode_index,
        },
        "ego_trajectory_shape": list(decoded.predicted_ego_trajectory.shape),
        "ego_trajectory_sample": _rounded_rows(decoded.predicted_ego_trajectory, limit=20),
        "num_obstacles": int(decoded.predicted_obstacle_boxes.shape[0]),
        "obstacle_scores": _rounded_rows(decoded.obstacle_scores, limit=10),
        "obstacle_labels": decoded.obstacle_labels[:10].astype(int, copy=False).tolist(),
        "obstacle_boxes": _rounded_rows(decoded.predicted_obstacle_boxes, limit=5),
        "num_map_polylines": int(len(decoded.predicted_map_polylines)),
        "map_scores": _rounded_rows(decoded.map_scores, limit=10),
        "map_labels": decoded.map_labels[:10].astype(int, copy=False).tolist(),
        "map_polyline_lengths": [
            int(polyline.shape[0]) for polyline in decoded.predicted_map_polylines[:10]
        ],
        "map_polylines": [
            _rounded_rows(polyline, limit=5) for polyline in decoded.predicted_map_polylines[:5]
        ],
        "future_sample_tokens": list(sample.future_sample_tokens),
        "future_lidar_origins_shape": list(sample.future_lidar_origins.shape),
        "future_lidar_origins_sample": _rounded_rows(
            sample.future_lidar_origins.detach().cpu().numpy(),
            limit=6,
        ),
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
        "VISUALIZE=True, but no nuScenes SparseDrive visualization helper was found. "
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
    checkpoint_path = _require_checkpoint_path(_checkpoint_path_for_config(CONFIG_NAME))
    output_dir = _resolve_repo_path(OUTPUT_DIR)
    device = _resolve_device(DEVICE)
    image_hw = _image_hw_from_config(CONFIG_NAME)

    print(f"Building SparseDrive {CONFIG_NAME} model on {device}...")
    model = _build_model(CONFIG_NAME)
    _load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    print(
        f"Loading nuScenes root={NUSCENES_DATA_ROOT!r}, version={NUSCENES_VERSION!r}, "
        f"limit={LIMIT}, image_hw={image_hw}..."
    )
    from nuscenes_adapter import collate_nuscenes_sparsedrive_samples, sample_to_device
    from prediction_decode import DecodedSparseDrivePrediction, decode_sparsedrive_outputs

    samples = _load_nuscenes_samples(image_hw)
    batch = collate_nuscenes_sparsedrive_samples(samples)
    _validate_model_batch(batch, samples, image_hw)
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
        summary = _summary_for_sample(
            sample,
            output,
            decoded,
            checkpoint_path=checkpoint_path,
            device=device,
        )
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
            "dataset_root": NUSCENES_DATA_ROOT,
            "version": NUSCENES_VERSION,
            "config_name": CONFIG_NAME,
            "checkpoint_path": str(checkpoint_path),
            "limit": LIMIT,
            "device": str(device),
            "camera_order": list(CAMERA_ORDER),
            "image_hw": list(image_hw),
            "output_dir": str(output_dir),
            "num_samples": len(samples),
            "sample_tokens": [sample.token for sample in samples],
            "samples": summaries,
        },
    )
    _maybe_visualize(samples, outputs, decoded_predictions, summaries, output_dir)
    print(f"Wrote aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
