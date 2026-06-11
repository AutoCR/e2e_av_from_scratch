from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_config: str) -> torch.device:
    if device_config != "auto":
        return torch.device(device_config)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patch_cuda_calls_for_local_device(device: torch.device) -> None:
    if device.type == "cuda" or getattr(torch.Tensor, "_bevfusion_cuda_patched", False):
        return

    def _cuda_to_local(self, *args, **kwargs):
        non_blocking = bool(kwargs.get("non_blocking", False))
        return self.to(device, non_blocking=non_blocking)

    torch.Tensor.cuda = _cuda_to_local
    torch.Tensor._bevfusion_cuda_patched = True


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def load_scene_filter_fields(
    yaml_path: str | Path,
    log_names_key: str | None = "log_names",
    tokens_key: str | None = "tokens",
) -> tuple[list[str] | None, list[str] | None]:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene filter YAML not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)

    def _extract(key):
        if key is None or key not in data:
            return None
        values = data[key]
        if not isinstance(values, list):
            raise TypeError(f"Expected a list at key {key!r} in {path}, got {type(values).__name__}")
        return [str(value) for value in values]

    return _extract(log_names_key), _extract(tokens_key)


def resolve_split_config(split_config, repo_root=None):
    if isinstance(split_config, str):
        return split_config, None, None

    dir_name = split_config["dir"]

    def _resolve_path(yaml_path_str):
        if yaml_path_str is None:
            return None
        p = Path(yaml_path_str)
        if not p.is_absolute() and repo_root is not None:
            p = Path(repo_root) / p
        return p

    log_names = None
    log_names_yaml = _resolve_path(split_config.get("log_names_yaml"))
    if log_names_yaml is not None:
        log_names_key = split_config.get("log_names_key", "log_names")
        log_names, _ = load_scene_filter_fields(log_names_yaml, log_names_key=log_names_key, tokens_key=None)

    tokens = None
    tokens_yaml = _resolve_path(split_config.get("tokens_yaml"))
    if tokens_yaml is not None:
        tokens_key = split_config.get("tokens_key", "tokens")
        _, tokens = load_scene_filter_fields(tokens_yaml, log_names_key=None, tokens_key=tokens_key)

    return dir_name, log_names, tokens


def init_distributed():
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    timeout_min = int(os.environ.get("NCCL_TIMEOUT_MINUTES", "60"))
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_min))
    return local_rank, world_size


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


class _TeeStream:
    def __init__(self, original, file_handle):
        self._original = original
        self._file = file_handle

    def write(self, data):
        self._original.write(data)
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._original.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)


def install_console_log(output_dir: Path):
    if not is_main_process():
        return None
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(output_dir) / f"console_{ts}.log"
        fh = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _TeeStream(sys.stdout, fh)
        sys.stderr = _TeeStream(sys.stderr, fh)
        print(f"[console-log] mirroring stdout/stderr to {log_path}", flush=True)
        return log_path
    except Exception as exc:
        print(f"[console-log] failed to set up file logging: {exc}", flush=True)
        return None


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
