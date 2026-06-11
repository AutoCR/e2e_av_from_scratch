from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import navsim.common.dataclasses as navsim_dataclasses
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

PathLike = Union[str, os.PathLike[str]]

DEFAULT_OPENSCENE_DATA_ROOT: Optional[PathLike] = None
DEFAULT_NUPLAN_MAPS_ROOT: Optional[PathLike] = None
DEFAULT_SPLIT = "mini"
DEFAULT_NUM_HISTORY_FRAMES = 4
DEFAULT_NUM_FUTURE_FRAMES = 10

VALID_NAVSIM_CAMERA_NAMES = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)

DEFAULT_CAMERA_ORDER_8 = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_R0",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
)


@dataclass(frozen=True)
class NavSimPathConfig:
    openscene_data_root: Path
    nuplan_maps_root: Path


def _expand_path(path: PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(path))))


def _resolve_required_path(env_name: str, explicit: Optional[PathLike], default: Optional[PathLike]) -> Path:
    value = explicit if explicit is not None else os.environ.get(env_name)
    if value is None:
        value = default
    if value is None or str(value).strip() == "":
        raise RuntimeError(
            f"{env_name} is required. Set the {env_name} environment variable, "
            "pass it to resolve_navsim_paths/build_navsim_scene_loader, or edit "
            f"DEFAULT_{env_name}."
        )
    path = _expand_path(value)
    if not path.exists():
        raise FileNotFoundError(f"{env_name} path does not exist: {path}")
    return path


def resolve_navsim_paths(
    openscene_data_root: Optional[PathLike] = None,
    nuplan_maps_root: Optional[PathLike] = None,
) -> NavSimPathConfig:
    openscene_root = _resolve_required_path(
        "OPENSCENE_DATA_ROOT", openscene_data_root, DEFAULT_OPENSCENE_DATA_ROOT
    )
    maps_root = _resolve_required_path("NUPLAN_MAPS_ROOT", nuplan_maps_root, DEFAULT_NUPLAN_MAPS_ROOT)
    return NavSimPathConfig(openscene_data_root=openscene_root, nuplan_maps_root=maps_root)


def _set_navsim_maps_root(maps_root: Path) -> None:
    os.environ["NUPLAN_MAPS_ROOT"] = str(maps_root)
    navsim_dataclasses.NUPLAN_MAPS_ROOT = str(maps_root)


def _normalize_camera_name(camera_name: str) -> str:
    normalized = camera_name.upper()
    if not normalized.startswith("CAM_"):
        normalized = f"CAM_{normalized}"
    if normalized not in VALID_NAVSIM_CAMERA_NAMES:
        valid = ", ".join(VALID_NAVSIM_CAMERA_NAMES)
        raise ValueError(f"Unknown NAVSIM camera '{camera_name}'. Valid names: {valid}")
    return normalized


def normalize_camera_order(camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER_8) -> tuple[str, ...]:
    normalized = tuple(_normalize_camera_name(camera) for camera in camera_order)
    if not (1 <= len(normalized) <= len(VALID_NAVSIM_CAMERA_NAMES)):
        raise ValueError(
            f"BEVFusion requires between 1 and {len(VALID_NAVSIM_CAMERA_NAMES)} cameras, "
            f"got {len(normalized)}: {normalized}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Camera order must not contain duplicates: {normalized}")
    return normalized


def build_navsim_sensor_config(camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER_8) -> SensorConfig:
    selected = set(normalize_camera_order(camera_order))
    kwargs = {camera.lower(): camera in selected for camera in VALID_NAVSIM_CAMERA_NAMES}
    kwargs["lidar_pc"] = False
    return SensorConfig(**kwargs)


def build_navsim_scene_loader(
    split: str = DEFAULT_SPLIT,
    camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER_8,
    openscene_data_root: Optional[PathLike] = None,
    nuplan_maps_root: Optional[PathLike] = None,
    num_history_frames: int = DEFAULT_NUM_HISTORY_FRAMES,
    num_future_frames: int = DEFAULT_NUM_FUTURE_FRAMES,
    frame_interval: Optional[int] = None,
    has_route: bool = True,
    max_scenes: Optional[int] = None,
    log_names: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
) -> SceneLoader:
    paths = resolve_navsim_paths(openscene_data_root, nuplan_maps_root)
    _set_navsim_maps_root(paths.nuplan_maps_root)

    data_path = paths.openscene_data_root / "navsim_logs" / split
    sensor_path = paths.openscene_data_root / "sensor_blobs" / split
    if not data_path.exists():
        raise FileNotFoundError(f"NAVSIM log path does not exist: {data_path}")
    if not sensor_path.exists():
        raise FileNotFoundError(f"NAVSIM sensor path does not exist: {sensor_path}")

    scene_filter = SceneFilter(
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        frame_interval=frame_interval,
        has_route=has_route,
        max_scenes=max_scenes,
        log_names=list(log_names) if log_names is not None else None,
        tokens=list(tokens) if tokens is not None else None,
        include_synthetic_scenes=False,
    )
    return SceneLoader(
        data_path=data_path,
        original_sensor_path=sensor_path,
        scene_filter=scene_filter,
        sensor_config=build_navsim_sensor_config(camera_order),
    )
