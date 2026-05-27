"""Utilities for loading NAVSIM scene-filter fields from YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml


def load_scene_filter_fields(
    yaml_path: str | Path,
    log_names_key: str | None = "log_names",
    tokens_key: str | None = "tokens",
) -> tuple[list[str] | None, list[str] | None]:
    """Load log_names and/or tokens from a NAVSIM scene-filter YAML in one pass.

    Args:
        yaml_path: Path to the YAML file.
        log_names_key: Key whose value is the list of log names. Pass ``None`` to skip.
        tokens_key: Key whose value is the list of scene tokens. Pass ``None`` to skip.

    Returns:
        ``(log_names_or_None, tokens_or_None)`` — ``None`` when the key is absent
        or the corresponding key param is ``None``.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene filter YAML not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)

    def _extract(key):
        if key is None or key not in data:
            return None
        names = data[key]
        if not isinstance(names, list):
            raise TypeError(f"Expected a list at key {key!r} in {path}, got {type(names).__name__}")
        return [str(n) for n in names]

    return _extract(log_names_key), _extract(tokens_key)


def load_log_names(yaml_path: str | Path, key: str = "log_names") -> List[str]:
    """Load a list of log names from a NAVSIM scene-filter YAML file.

    Handles two formats:
    - navtrain.yaml / navtest.yaml: top-level ``log_names`` key.
    - default_train_val_test_log_split.yaml: keys ``train_logs``, ``val_logs``, ``test_logs``.

    Args:
        yaml_path: Path to the YAML file.
        key: Top-level key whose value is the list of log name strings.

    Returns:
        List of log name strings.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If ``key`` is not found in the YAML.
        TypeError: If the value at ``key`` is not a list.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene filter YAML not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if key not in data:
        raise KeyError(f"Key {key!r} not found in {path}. Available keys: {list(data.keys())}")
    names = data[key]
    if not isinstance(names, list):
        raise TypeError(f"Expected a list at key {key!r} in {path}, got {type(names).__name__}")
    return [str(n) for n in names]
