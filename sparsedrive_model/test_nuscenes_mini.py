from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
LIMIT = 1
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

# Whole-log video output is disabled by default so the existing LIMIT-based
# sample inference path remains unchanged. When enabled by future integration,
# log sampling selects which logs to render; every selected log should include
# every keyframe sample.
VIDEO_OUTPUT = True
VIDEO_FPS = 10
VIDEO_OUTPUT_DIRNAME = "videos"
VIDEO_OUTPUT_FILENAME_TEMPLATE = "{log_index:02d}_{log_name}.mp4"
VIDEO_MAX_LOGS: int | None = 1
VIDEO_LOG_SAMPLING_METHOD = "sequence"  # "sequence" or "random"
VIDEO_RANDOM_SEED: int | None = None
VIDEO_BATCH_SIZE = 1
# -----------------------------------------------------------------------------

if TYPE_CHECKING:
    from prediction_decode import DecodedSparseDrivePrediction


@dataclass(frozen=True)
class SparseDriveInferenceResult:
    samples: tuple[Any, ...]
    outputs: tuple[Mapping[str, Any], ...]
    decoded_predictions: tuple["DecodedSparseDrivePrediction", ...]
    summaries: tuple[dict[str, Any], ...]
    sample_json_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SparseDriveLogInferenceResult:
    log: Any
    log_index: int
    samples: tuple[Any, ...]
    outputs: tuple[Mapping[str, Any], ...]
    decoded_predictions: tuple["DecodedSparseDrivePrediction", ...]
    summaries: tuple[dict[str, Any], ...]
    sample_json_paths: tuple[Path, ...]
    summary_path: Path
    video_path: Path
    video_fps: int = VIDEO_FPS
    video_frame_count: int = 0
    video_frame_width: int = 0
    video_frame_height: int = 0
    frame_layout: str = "horizontal"


STATE_DICT_WRAPPERS = ("state_dict", "model", "model_state_dict", "net")
VISUALIZATION_HELPERS = (
    ("visualize_nuscenes_inference", "visualize_nuscenes_sparsedrive_predictions"),
    ("sparsedrive_model.visualize_nuscenes_inference", "visualize_nuscenes_sparsedrive_predictions"),
)
VIDEO_FRAME_RENDERERS = (
    ("visualize_nuscenes_inference", "render_nuscenes_inference_figure"),
    ("sparsedrive_model.visualize_nuscenes_inference", "render_nuscenes_inference_figure"),
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


def _image_hw_from_config(config_name: str) -> tuple[int, int]:
    from sparsedrive_model.configs.sparsedrive_hyperparams import MODEL_ARCH
    from nuscenes_adapter import image_hw_from_sparsedrive_input_shape

    return image_hw_from_sparsedrive_input_shape(MODEL_ARCH["input_shape"])


def _build_model(config_name: str) -> torch.nn.Module:
    normalized = config_name.lower().strip()
    if normalized == "stage1":
        from sparsedrive_model.configs import build_stage1

        model = build_stage1()
    elif normalized == "stage2":
        from sparsedrive_model.configs import build_stage2

        model = build_stage2()
    else:
        raise ValueError(f"Unsupported CONFIG_NAME={config_name!r}; expected 'stage1' or 'stage2'.")
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


def _adapter_video_log_sampling_method() -> str:
    if not isinstance(VIDEO_LOG_SAMPLING_METHOD, str):
        raise TypeError(
            "VIDEO_LOG_SAMPLING_METHOD must be 'sequence', 'sequential', or 'random', "
            f"got {type(VIDEO_LOG_SAMPLING_METHOD).__name__}."
        )
    method = VIDEO_LOG_SAMPLING_METHOD.strip().lower()
    if method in {"sequence", "sequential"}:
        return "sequential"
    if method == "random":
        return "random"
    raise ValueError(
        "VIDEO_LOG_SAMPLING_METHOD must be 'sequence', 'sequential', or 'random', "
        f"got {VIDEO_LOG_SAMPLING_METHOD!r}."
    )


def _validated_video_fps() -> int:
    if type(VIDEO_FPS) is not int or VIDEO_FPS <= 0:
        raise ValueError(f"VIDEO_FPS must be a positive integer, got {VIDEO_FPS!r}.")
    return VIDEO_FPS


def _validate_video_config() -> None:
    if not isinstance(VIDEO_OUTPUT, bool):
        raise TypeError(f"VIDEO_OUTPUT must be a bool, got {type(VIDEO_OUTPUT).__name__}.")
    if not VIDEO_OUTPUT:
        return

    _validated_video_fps()
    if not isinstance(VIDEO_OUTPUT_DIRNAME, str) or not VIDEO_OUTPUT_DIRNAME.strip():
        raise ValueError(
            f"VIDEO_OUTPUT_DIRNAME must be a non-empty string, got {VIDEO_OUTPUT_DIRNAME!r}."
        )
    if not isinstance(VIDEO_OUTPUT_FILENAME_TEMPLATE, str) or not VIDEO_OUTPUT_FILENAME_TEMPLATE:
        raise ValueError(
            "VIDEO_OUTPUT_FILENAME_TEMPLATE must be a non-empty string, "
            f"got {VIDEO_OUTPUT_FILENAME_TEMPLATE!r}."
        )
    try:
        VIDEO_OUTPUT_FILENAME_TEMPLATE.format(log_index=0, log_name="log", log_token="token")
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            "VIDEO_OUTPUT_FILENAME_TEMPLATE must be compatible with log_index, "
            "log_name, and log_token fields."
        ) from exc
    if VIDEO_MAX_LOGS is not None and (
        not isinstance(VIDEO_MAX_LOGS, int) or VIDEO_MAX_LOGS <= 0
    ):
        raise ValueError(
            f"VIDEO_MAX_LOGS must be a positive integer or None, got {VIDEO_MAX_LOGS!r}."
        )
    _adapter_video_log_sampling_method()
    if VIDEO_RANDOM_SEED is not None and not isinstance(VIDEO_RANDOM_SEED, int):
        raise TypeError(
            f"VIDEO_RANDOM_SEED must be an integer or None, got {type(VIDEO_RANDOM_SEED).__name__}."
        )
    if not isinstance(VIDEO_BATCH_SIZE, int) or VIDEO_BATCH_SIZE <= 0:
        raise ValueError(f"VIDEO_BATCH_SIZE must be a positive integer, got {VIDEO_BATCH_SIZE!r}.")


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


def _load_nuscenes_samples_for_tokens(
    sample_tokens: Sequence[str],
    image_hw: tuple[int, int],
    metadata: Any,
) -> tuple[Any, ...]:
    from nuscenes_adapter import load_nuscenes_sparsedrive_samples, normalize_camera_order

    normalize_camera_order(CAMERA_ORDER)
    tokens = tuple(str(token) for token in sample_tokens)
    if not tokens:
        raise ValueError("At least one sample token is required.")

    samples = tuple(
        load_nuscenes_sparsedrive_samples(
            dataset_root=NUSCENES_DATA_ROOT,
            version=NUSCENES_VERSION,
            tokens=tokens,
            max_samples=None,
            camera_order=CAMERA_ORDER,
            image_hw=image_hw,
            metadata=metadata,
        )
    )
    if len(samples) != len(tokens):
        raise RuntimeError(f"Loaded {len(samples)} samples for {len(tokens)} requested token(s).")
    for index, (sample, expected_token) in enumerate(zip(samples, tokens)):
        if getattr(sample, "token", None) != expected_token:
            raise RuntimeError(
                f"Loaded sample {index} has token {getattr(sample, 'token', None)!r}, "
                f"expected {expected_token!r}."
            )
    return samples


def _load_video_metadata_and_logs() -> tuple[Any, tuple[Any, ...]]:
    from nuscenes_adapter import load_nuscenes_metadata, selected_nuscenes_logs

    metadata = load_nuscenes_metadata(dataset_root=NUSCENES_DATA_ROOT, version=NUSCENES_VERSION)
    logs = tuple(
        selected_nuscenes_logs(
            metadata=metadata,
            max_logs=VIDEO_MAX_LOGS,
            sampling_method=_adapter_video_log_sampling_method(),
            random_seed=VIDEO_RANDOM_SEED,
        )
    )
    if not logs:
        raise RuntimeError("No nuScenes logs were selected for video output.")
    return metadata, logs


def _iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterator[tuple[int, tuple[Any, ...]]]:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    for start in range(0, len(items), chunk_size):
        yield start, tuple(items[start : start + chunk_size])


def _collate_validate_to_device(
    samples: Sequence[Any],
    *,
    image_hw: tuple[int, int],
    device: torch.device,
) -> Mapping[str, Any]:
    from nuscenes_adapter import collate_nuscenes_sparsedrive_samples, sample_to_device

    batch = collate_nuscenes_sparsedrive_samples(samples)
    _validate_model_batch(batch, samples, image_hw)
    return sample_to_device(batch, device)


def _validate_outputs_sequence(
    outputs: Any,
    *,
    expected_count: int,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(outputs, Sequence):
        raise RuntimeError(
            f"Expected one output per sample ({expected_count}), got {type(outputs).__name__}."
        )
    output_tuple = tuple(outputs)
    if len(output_tuple) != expected_count:
        raise RuntimeError(
            f"Expected one output per sample ({expected_count}), got {len(output_tuple)}."
        )
    for index, output in enumerate(output_tuple):
        if not isinstance(output, Mapping):
            raise TypeError(
                f"Model output {index} must be a mapping, got {type(output).__name__}."
            )
    return output_tuple


def _run_model_inference(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    expected_count: int,
) -> tuple[Mapping[str, Any], ...]:
    with torch.no_grad():
        outputs = model(**batch)
    return _validate_outputs_sequence(outputs, expected_count=expected_count)


def _decode_outputs(
    outputs: Sequence[Mapping[str, Any]],
) -> tuple["DecodedSparseDrivePrediction", ...]:
    from prediction_decode import decode_sparsedrive_outputs

    return tuple(
        decode_sparsedrive_outputs(outputs, sample_index=sample_index)
        for sample_index in range(len(outputs))
    )


def _summaries_for_samples(
    samples: Sequence[Any],
    outputs: Sequence[Mapping[str, Any]],
    decoded_predictions: Sequence["DecodedSparseDrivePrediction"],
    *,
    checkpoint_path: Path,
    device: torch.device,
    limit: int | None,
    extra_fields: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    if len(samples) != len(outputs) or len(samples) != len(decoded_predictions):
        raise RuntimeError(
            "samples, outputs, and decoded_predictions must have matching lengths; "
            f"got {len(samples)}, {len(outputs)}, and {len(decoded_predictions)}."
        )
    if extra_fields is not None and len(extra_fields) != len(samples):
        raise RuntimeError(
            f"extra_fields length {len(extra_fields)} does not match samples length {len(samples)}."
        )

    summaries: list[dict[str, Any]] = []
    for index, (sample, output, decoded) in enumerate(zip(samples, outputs, decoded_predictions)):
        summary = _summary_for_sample(
            sample,
            output,
            decoded,
            checkpoint_path=checkpoint_path,
            device=device,
            limit=limit,
        )
        if extra_fields is not None:
            summary.update(dict(extra_fields[index]))
        summaries.append(summary)
    return tuple(summaries)


def _run_inference_for_loaded_samples(
    model: torch.nn.Module,
    samples: Sequence[Any],
    *,
    image_hw: tuple[int, int],
    device: torch.device,
    checkpoint_path: Path,
    limit: int | None,
    extra_summary_fields: Sequence[Mapping[str, Any]] | None = None,
) -> SparseDriveInferenceResult:
    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise ValueError("At least one sample is required for inference.")

    batch = _collate_validate_to_device(sample_tuple, image_hw=image_hw, device=device)
    outputs = _run_model_inference(model, batch, expected_count=len(sample_tuple))
    decoded_predictions = _decode_outputs(outputs)
    summaries = _summaries_for_samples(
        sample_tuple,
        outputs,
        decoded_predictions,
        checkpoint_path=checkpoint_path,
        device=device,
        limit=limit,
        extra_fields=extra_summary_fields,
    )
    return SparseDriveInferenceResult(
        samples=sample_tuple,
        outputs=outputs,
        decoded_predictions=decoded_predictions,
        summaries=summaries,
    )


def _run_inference_for_sample_chunks(
    model: torch.nn.Module,
    samples: Sequence[Any],
    *,
    image_hw: tuple[int, int],
    device: torch.device,
    checkpoint_path: Path,
    chunk_size: int,
    limit: int | None,
    sample_summary_dir: Path | None = None,
) -> SparseDriveInferenceResult:
    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise ValueError("At least one sample is required for chunked inference.")

    all_outputs: list[Mapping[str, Any]] = []
    all_decoded: list["DecodedSparseDrivePrediction"] = []
    all_summaries: list[dict[str, Any]] = []
    all_json_paths: list[Path] = []
    for chunk_start, chunk_samples in _iter_chunks(sample_tuple, chunk_size):
        chunk_result = _run_inference_for_loaded_samples(
            model,
            chunk_samples,
            image_hw=image_hw,
            device=device,
            checkpoint_path=checkpoint_path,
            limit=limit,
        )
        all_outputs.extend(chunk_result.outputs)
        all_decoded.extend(chunk_result.decoded_predictions)
        all_summaries.extend(chunk_result.summaries)
        if sample_summary_dir is not None:
            all_json_paths.extend(
                _write_sample_summaries(
                    chunk_samples,
                    chunk_result.summaries,
                    sample_summary_dir,
                    start_index=chunk_start,
                )
            )

    return SparseDriveInferenceResult(
        samples=sample_tuple,
        outputs=tuple(all_outputs),
        decoded_predictions=tuple(all_decoded),
        summaries=tuple(all_summaries),
        sample_json_paths=tuple(all_json_paths),
    )


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
    limit: int | None = LIMIT,
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
        "limit": limit,
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


def _write_sample_summaries(
    samples: Sequence[Any],
    summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    start_index: int = 0,
) -> tuple[Path, ...]:
    if len(samples) != len(summaries):
        raise RuntimeError(
            f"Cannot write {len(summaries)} summaries for {len(samples)} sample(s)."
        )
    paths: list[Path] = []
    for offset, (sample, summary) in enumerate(zip(samples, summaries)):
        sample_index = start_index + offset
        sample_path = output_dir / f"{sample_index:03d}_{_sanitize_token(sample.token)}.json"
        _write_json(sample_path, summary)
        paths.append(sample_path)
        print(
            f"[{sample_index}] token={sample.token}: "
            f"obstacles={summary['num_obstacles']}, maps={summary['num_map_polylines']}, "
            f"summary={sample_path}"
        )
    return tuple(paths)


def _aggregate_summary_payload(
    samples: Sequence[Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path,
    device: torch.device,
    image_hw: tuple[int, int],
    output_dir: Path,
    limit: int | None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(samples) != len(summaries):
        raise RuntimeError(
            f"Cannot aggregate {len(summaries)} summaries for {len(samples)} sample(s)."
        )
    payload: dict[str, Any] = {
        "dataset_root": NUSCENES_DATA_ROOT,
        "version": NUSCENES_VERSION,
        "config_name": CONFIG_NAME,
        "checkpoint_path": str(checkpoint_path),
        "limit": limit,
        "device": str(device),
        "camera_order": list(CAMERA_ORDER),
        "image_hw": list(image_hw),
        "output_dir": str(output_dir),
        "num_samples": len(samples),
        "sample_tokens": [sample.token for sample in samples],
        "samples": list(summaries),
    }
    if extra_fields:
        payload.update(dict(extra_fields))
    return payload


def _write_aggregate_summary(
    path: Path,
    samples: Sequence[Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path,
    device: torch.device,
    image_hw: tuple[int, int],
    output_dir: Path,
    limit: int | None,
    extra_fields: Mapping[str, Any] | None = None,
) -> Path:
    _write_json(
        path,
        _aggregate_summary_payload(
            samples,
            summaries,
            checkpoint_path=checkpoint_path,
            device=device,
            image_hw=image_hw,
            output_dir=output_dir,
            limit=limit,
            extra_fields=extra_fields,
        ),
    )
    return path


def _log_payload(log: Any) -> dict[str, Any]:
    return {
        "token": log.token,
        "name": log.name,
        "location": log.location,
        "scene_tokens": list(log.scene_tokens),
        "sample_tokens": list(log.sample_tokens),
        "num_samples": len(log.sample_tokens),
    }


def _video_filename_for_log(log: Any, log_index: int) -> str:
    filename = VIDEO_OUTPUT_FILENAME_TEMPLATE.format(
        log_index=log_index,
        log_name=_sanitize_token(log.name),
        log_token=_sanitize_token(log.token),
    )
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"VIDEO_OUTPUT_FILENAME_TEMPLATE produced unsafe video filename {filename!r}."
        )
    return filename


def _video_path_for_log(output_dir: Path, log: Any, log_index: int) -> Path:
    return output_dir / VIDEO_OUTPUT_DIRNAME / _video_filename_for_log(log, log_index)


def _log_summary_dir(output_dir: Path, log: Any, log_index: int) -> Path:
    return output_dir / "logs" / f"{log_index:02d}_{_sanitize_token(log.name)}"


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


def _import_video_frame_renderer() -> Any:
    failures: list[str] = []
    for module_name, attr_name in VIDEO_FRAME_RENDERERS:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            failures.append(f"{module_name}: {exc}")
            continue
        renderer = getattr(module, attr_name, None)
        if callable(renderer):
            return renderer
        failures.append(f"{module_name}: missing callable {attr_name}")

    expected = ", ".join(f"{module}.{attr}" for module, attr in VIDEO_FRAME_RENDERERS)
    raise RuntimeError(
        "VIDEO_OUTPUT=True, but no horizontal nuScenes SparseDrive frame renderer was found. "
        f"Expected one of: {expected}. Import failures: {'; '.join(failures)}"
    )


def _matplotlib_figure_to_bgr_frame(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Rendered video frame has invalid dimensions {width}x{height}.")

    rgba = np.asarray(fig.canvas.buffer_rgba())
    if rgba.shape != (height, width, 4):
        raise RuntimeError(
            "Rendered video frame buffer has unexpected shape "
            f"{rgba.shape}; expected {(height, width, 4)}."
        )
    if rgba.dtype != np.uint8:
        raise TypeError(f"Rendered video frame dtype must be uint8, got {rgba.dtype}.")
    return np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])


def _render_nuscenes_video_frame(
    render_nuscenes_inference_figure: Any,
    sample: Any,
    decoded_prediction: "DecodedSparseDrivePrediction",
    summary: Mapping[str, Any],
) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig = render_nuscenes_inference_figure(
        sample,
        decoded_prediction,
        summary=summary,
        layout="horizontal",
    )
    try:
        return _matplotlib_figure_to_bgr_frame(fig)
    finally:
        plt.close(fig)


def _prepare_video_frame(
    frame: np.ndarray,
    *,
    video_path: Path,
    expected_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Video frame for {video_path} must be a numpy.ndarray.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Video frame for {video_path} must have shape (height, width, 3), "
            f"got {frame.shape}."
        )
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Video frame for {video_path} has invalid dimensions {width}x{height}.")
    if frame.dtype != np.uint8:
        raise TypeError(f"Video frame for {video_path} must have dtype uint8, got {frame.dtype}.")
    frame_hw = (height, width)
    if expected_hw is not None and frame_hw != expected_hw:
        expected_height, expected_width = expected_hw
        raise RuntimeError(
            f"Inconsistent video frame dimensions for {video_path}: got {width}x{height}, "
            f"expected {expected_width}x{expected_height}."
        )
    return np.ascontiguousarray(frame), frame_hw


def _open_video_writer_for_frame(
    video_path: Path,
    frame: np.ndarray,
) -> tuple[Any, tuple[int, int], np.ndarray]:
    import cv2

    prepared_frame, frame_hw = _prepare_video_frame(frame, video_path=video_path)
    height, width = frame_hw
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(_validated_video_fps()),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to open OpenCV MP4 writer for {video_path}.")
    return writer, frame_hw, prepared_frame


def _video_frame_summary_fields(
    *,
    frame_index: int,
    frame_hw: tuple[int, int],
) -> dict[str, Any]:
    height, width = frame_hw
    return {
        "video_frame_index": frame_index,
        "video_frame_width": width,
        "video_frame_height": height,
        "video_frame_size": [width, height],
        "video_fps": _validated_video_fps(),
        "frame_layout": "horizontal",
    }


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


def _run_limit_based_inference(
    model: torch.nn.Module,
    *,
    image_hw: tuple[int, int],
    device: torch.device,
    checkpoint_path: Path,
    output_dir: Path,
) -> SparseDriveInferenceResult:
    print(
        f"Loading nuScenes root={NUSCENES_DATA_ROOT!r}, version={NUSCENES_VERSION!r}, "
        f"limit={LIMIT}, image_hw={image_hw}..."
    )
    samples = tuple(_load_nuscenes_samples(image_hw))
    print(f"Running inference for {len(samples)} sample(s)...")
    result = _run_inference_for_sample_chunks(
        model,
        samples,
        image_hw=image_hw,
        device=device,
        checkpoint_path=checkpoint_path,
        chunk_size=len(samples),
        limit=LIMIT,
        sample_summary_dir=output_dir,
    )
    aggregate_path = _write_aggregate_summary(
        output_dir / "summary.json",
        result.samples,
        result.summaries,
        checkpoint_path=checkpoint_path,
        device=device,
        image_hw=image_hw,
        output_dir=output_dir,
        limit=LIMIT,
    )
    _maybe_visualize(
        result.samples,
        result.outputs,
        result.decoded_predictions,
        result.summaries,
        output_dir,
    )
    print(f"Wrote aggregate summary: {aggregate_path}")
    return result


def _video_summary_extra_fields(
    log: Any,
    *,
    log_index: int,
    video_path: Path,
    chunk_start: int,
    chunk_size: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "video_output": True,
            "log_index": log_index,
            "log_token": log.token,
            "log_name": log.name,
            "log_location": log.location,
            "sample_index_in_log": chunk_start + offset,
            "num_log_samples": len(log.sample_tokens),
            "video_path": str(video_path),
            "video_fps": _validated_video_fps(),
            "frame_layout": "horizontal",
        }
        for offset in range(chunk_size)
    )


def _run_video_log_inference(
    model: torch.nn.Module,
    *,
    metadata: Any,
    log: Any,
    log_index: int,
    image_hw: tuple[int, int],
    device: torch.device,
    checkpoint_path: Path,
    output_dir: Path,
) -> SparseDriveLogInferenceResult:
    sample_tokens = tuple(log.sample_tokens)
    if not sample_tokens:
        raise RuntimeError(f"Selected nuScenes log {log.name!r} has no sample tokens.")

    log_dir = _log_summary_dir(output_dir, log, log_index)
    sample_summary_dir = log_dir / "samples"
    video_path = _video_path_for_log(output_dir, log, log_index)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[log {log_index}] name={log.name!r}, token={log.token}, "
        f"samples={len(sample_tokens)}, batch_size={VIDEO_BATCH_SIZE}"
    )
    all_samples: list[Any] = []
    all_outputs: list[Mapping[str, Any]] = []
    all_decoded: list["DecodedSparseDrivePrediction"] = []
    all_summaries: list[dict[str, Any]] = []
    all_json_paths: list[Path] = []
    render_nuscenes_inference_figure = _import_video_frame_renderer()
    video_writer: Any | None = None
    video_frame_hw: tuple[int, int] | None = None
    video_frame_count = 0

    try:
        for chunk_start, token_chunk in _iter_chunks(sample_tokens, VIDEO_BATCH_SIZE):
            print(
                f"[log {log_index}] loading/inferencing samples "
                f"{chunk_start}..{chunk_start + len(token_chunk) - 1}"
            )
            chunk_samples = _load_nuscenes_samples_for_tokens(token_chunk, image_hw, metadata)
            chunk_result = _run_inference_for_loaded_samples(
                model,
                chunk_samples,
                image_hw=image_hw,
                device=device,
                checkpoint_path=checkpoint_path,
                limit=None,
                extra_summary_fields=_video_summary_extra_fields(
                    log,
                    log_index=log_index,
                    video_path=video_path,
                    chunk_start=chunk_start,
                    chunk_size=len(chunk_samples),
                ),
            )
            for sample, decoded_prediction, summary in zip(
                chunk_result.samples,
                chunk_result.decoded_predictions,
                chunk_result.summaries,
            ):
                frame = _render_nuscenes_video_frame(
                    render_nuscenes_inference_figure,
                    sample,
                    decoded_prediction,
                    summary,
                )
                if video_writer is None:
                    video_writer, video_frame_hw, frame = _open_video_writer_for_frame(
                        video_path,
                        frame,
                    )
                    frame_hw = video_frame_hw
                else:
                    if video_frame_hw is None:
                        raise RuntimeError(f"Video writer for {video_path} has no frame size.")
                    frame, frame_hw = _prepare_video_frame(
                        frame,
                        video_path=video_path,
                        expected_hw=video_frame_hw,
                    )
                video_writer.write(frame)
                summary.update(
                    _video_frame_summary_fields(
                        frame_index=video_frame_count,
                        frame_hw=frame_hw,
                    )
                )
                video_frame_count += 1

            all_samples.extend(chunk_result.samples)
            all_outputs.extend(chunk_result.outputs)
            all_decoded.extend(chunk_result.decoded_predictions)
            all_summaries.extend(chunk_result.summaries)
            all_json_paths.extend(
                _write_sample_summaries(
                    chunk_result.samples,
                    chunk_result.summaries,
                    sample_summary_dir,
                    start_index=chunk_start,
                )
            )
    finally:
        if video_writer is not None:
            video_writer.release()

    if video_frame_count <= 0:
        raise RuntimeError(f"No video frames were written for selected nuScenes log {log.name!r}.")
    if video_frame_count != len(all_samples):
        raise RuntimeError(
            f"Wrote {video_frame_count} video frame(s) for {len(all_samples)} inferred "
            f"sample(s) in log {log.name!r}."
        )
    if video_frame_hw is None:
        raise RuntimeError(f"Video frame dimensions were not established for {video_path}.")
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"OpenCV reported video writes for {video_path}, but no MP4 was created.")

    video_frame_height, video_frame_width = video_frame_hw

    summary_path = _write_aggregate_summary(
        log_dir / "summary.json",
        all_samples,
        all_summaries,
        checkpoint_path=checkpoint_path,
        device=device,
        image_hw=image_hw,
        output_dir=log_dir,
        limit=None,
        extra_fields={
            "video_output": True,
            "log_index": log_index,
            "log": _log_payload(log),
            "video_path": str(video_path),
            "video_filename": video_path.name,
            "video_fps": _validated_video_fps(),
            "video_frame_count": video_frame_count,
            "video_frame_width": video_frame_width,
            "video_frame_height": video_frame_height,
            "video_frame_size": [video_frame_width, video_frame_height],
            "frame_layout": "horizontal",
            "sample_json_paths": [str(path) for path in all_json_paths],
        },
    )
    print(
        f"[log {log_index}] wrote video: {video_path} "
        f"({video_frame_count} frames, {video_frame_width}x{video_frame_height}, "
        f"{_validated_video_fps()} fps)"
    )
    print(f"[log {log_index}] wrote log summary: {summary_path}")

    return SparseDriveLogInferenceResult(
        log=log,
        log_index=log_index,
        samples=tuple(all_samples),
        outputs=tuple(all_outputs),
        decoded_predictions=tuple(all_decoded),
        summaries=tuple(all_summaries),
        sample_json_paths=tuple(all_json_paths),
        summary_path=summary_path,
        video_path=video_path,
        video_fps=_validated_video_fps(),
        video_frame_count=video_frame_count,
        video_frame_width=video_frame_width,
        video_frame_height=video_frame_height,
    )


def _write_video_aggregate_summary(
    output_dir: Path,
    log_results: Sequence[SparseDriveLogInferenceResult],
    *,
    checkpoint_path: Path,
    device: torch.device,
    image_hw: tuple[int, int],
) -> Path:
    all_samples = tuple(sample for result in log_results for sample in result.samples)
    all_summaries = tuple(summary for result in log_results for summary in result.summaries)
    log_payloads = [
        {
            "log_index": result.log_index,
            "log": _log_payload(result.log),
            "num_samples": len(result.samples),
            "sample_tokens": [sample.token for sample in result.samples],
            "summary_path": str(result.summary_path),
            "video_path": str(result.video_path),
            "video_filename": result.video_path.name,
            "video_fps": result.video_fps,
            "video_frame_count": result.video_frame_count,
            "video_frame_width": result.video_frame_width,
            "video_frame_height": result.video_frame_height,
            "video_frame_size": [result.video_frame_width, result.video_frame_height],
            "frame_layout": result.frame_layout,
            "sample_json_paths": [str(path) for path in result.sample_json_paths],
            "samples": list(result.summaries),
        }
        for result in log_results
    ]
    return _write_aggregate_summary(
        output_dir / "summary.json",
        all_samples,
        all_summaries,
        checkpoint_path=checkpoint_path,
        device=device,
        image_hw=image_hw,
        output_dir=output_dir,
        limit=None,
        extra_fields={
            "video_output": True,
            "video_fps": _validated_video_fps(),
            "video_output_dir": str(output_dir / VIDEO_OUTPUT_DIRNAME),
            "frame_layout": "horizontal",
            "num_logs": len(log_results),
            "logs": log_payloads,
        },
    )


def _run_video_output_inference(
    model: torch.nn.Module,
    *,
    image_hw: tuple[int, int],
    device: torch.device,
    checkpoint_path: Path,
    output_dir: Path,
) -> tuple[SparseDriveLogInferenceResult, ...]:
    print(
        f"Loading nuScenes logs root={NUSCENES_DATA_ROOT!r}, version={NUSCENES_VERSION!r}, "
        f"max_logs={VIDEO_MAX_LOGS}, sampling={VIDEO_LOG_SAMPLING_METHOD!r}, image_hw={image_hw}..."
    )
    metadata, logs = _load_video_metadata_and_logs()
    log_results = tuple(
        _run_video_log_inference(
            model,
            metadata=metadata,
            log=log,
            log_index=log_index,
            image_hw=image_hw,
            device=device,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )
        for log_index, log in enumerate(logs)
    )
    aggregate_path = _write_video_aggregate_summary(
        output_dir,
        log_results,
        checkpoint_path=checkpoint_path,
        device=device,
        image_hw=image_hw,
    )
    print(f"Wrote video aggregate summary: {aggregate_path}")
    return log_results


def main() -> None:
    _validate_video_config()
    checkpoint_path = _require_checkpoint_path(_checkpoint_path_for_config(CONFIG_NAME))
    output_dir = _resolve_repo_path(OUTPUT_DIR)
    device = _resolve_device(DEVICE)
    image_hw = _image_hw_from_config(CONFIG_NAME)

    print(f"Building SparseDrive {CONFIG_NAME} model on {device}...")
    model = _build_model(CONFIG_NAME)
    _load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    if VIDEO_OUTPUT:
        _run_video_output_inference(
            model,
            image_hw=image_hw,
            device=device,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )
        return

    _run_limit_based_inference(
        model,
        image_hw=image_hw,
        device=device,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
