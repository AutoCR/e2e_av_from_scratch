"""
Build a minimal self-contained NAVSIM dataset subset in-repo.

Extracts a tiny subset of NAVSIM data (default 100 train / 10 val / 10 test samples)
into an in-repo folder with trimmed pickles and sensor files, suitable for quick
local development and testing without the full dataset.

CRITICAL INVARIANT: This script ONLY works with num_history_frames=1, num_future_frames=0.
Under these settings, SceneFilter.num_frames == 1, so each token maps to exactly one frame,
and a trimmed pickle (flat list of selected frames) reloads correctly.

Example:
    uv run python bevfusion_model/tools/build_mini_subset.py --train 100 --val 10 --test 10
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "bevfusion_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bevfusion_model.configs.bevfusion_hyperparams import get_runtime_config
from bevfusion_model.navsim_train.runner_utils import resolve_split_config
from bevfusion_model.navsim_train.navsim_adapter import build_navsim_scene_loader


def relative_sensor_paths(frame: dict[str, Any], camera_order: tuple[str, ...]) -> list[str]:
    """
    Extract relative sensor paths from a frame dict.

    For each camera in camera_order, gets the data_path from frame["cams"].
    If lidar_path exists, appends it. Raises KeyError if a camera is missing.

    Args:
        frame: Frame dict with "cams" and optional "lidar_path" keys.
        camera_order: Tuple of 8 camera names.

    Returns:
        List of relative path strings.

    Raises:
        KeyError: If a camera is missing from frame["cams"].
    """
    paths = []
    for cam_name in camera_order:
        if cam_name not in frame["cams"]:
            raise KeyError(
                f"Camera {cam_name!r} missing from frame {frame['token']!r}. "
                f"Available cameras: {list(frame['cams'].keys())}"
            )
        paths.append(frame["cams"][cam_name]["data_path"])

    if frame.get("lidar_path"):
        paths.append(frame["lidar_path"])

    return paths


def copy_sensor_file(
    rel: str,
    src_split_dir: Path,
    dst_split_dir: Path,
    copied: set[str],
    byte_counter: list[int],
) -> None:
    """
    Copy a sensor file from source to destination, deduplicating and tracking size.

    Args:
        rel: Relative path within the split directory.
        src_split_dir: Source sensor_blobs/{dir} directory.
        dst_split_dir: Destination sensor_blobs/{dir} directory.
        copied: Set to track already-copied relative paths (for deduplication).
        byte_counter: 1-element list to accumulate total bytes copied.

    Raises:
        FileNotFoundError: If source file does not exist.
    """
    if rel in copied:
        return

    src = src_split_dir / rel
    dst = dst_split_dir / rel

    if not src.exists():
        raise FileNotFoundError(f"Sensor file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    byte_counter[0] += dst.stat().st_size
    copied.add(rel)


def select_tokens_for_split(
    split_name: str,
    split_config: dict[str, Any],
    n: int,
    src_root: Path,
    maps_root: Path,
    repo_root: Path,
    runtime: dict[str, Any],
) -> tuple[str, list[tuple[str, list[dict[str, Any]]]]]:
    """
    Select n tokens from a split using the scene loader.

    Args:
        split_name: "train", "val", or "test".
        split_config: Config dict from runtime["splits"][split_name].
        n: Maximum number of scenes to select.
        src_root: OpenScene data root.
        maps_root: NuPlan maps root.
        repo_root: Repository root.
        runtime: Runtime config dict.

    Returns:
        Tuple of (dir_name, [(token, frame_list), ...]).
        frame_list is always length 1 due to num_frames==1 invariant.

    Raises:
        AssertionError: If num_frames != 1.
    """
    dir_name, log_names, tokens = resolve_split_config(split_config, repo_root)

    loader = build_navsim_scene_loader(
        split=dir_name,
        camera_order=runtime["camera_order"],
        openscene_data_root=src_root,
        nuplan_maps_root=maps_root,
        num_history_frames=runtime["num_history_frames"],
        num_future_frames=runtime["num_future_frames"],
        frame_interval=1,
        has_route=True,
        max_scenes=n,
        log_names=log_names,
        tokens=tokens,
    )

    assert (
        loader._scene_filter.num_frames == 1
    ), "build_mini_subset only supports num_frames==1 (detection config); trimmed-pickle-as-flat-list would be invalid otherwise."

    return dir_name, [(t, loader.scene_frames_dicts[t]) for t in loader.tokens]


def build_subset(
    out_root: Path,
    counts: dict[str, int],
    src_root: Path,
    maps_root: Path,
    runtime: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """
    Extract and write a NAVSIM subset to out_root.

    Orchestrates sensor file copying, pickle trimming, and manifest generation.

    Args:
        out_root: Output directory for subset.
        counts: Dict with "train", "val", "test" counts.
        src_root: OpenScene data root.
        maps_root: NuPlan maps root.
        runtime: Runtime config dict.
        repo_root: Repository root.

    Returns:
        Summary dict with per-split counts, file stats, and output path.
    """
    copied: set[str] = set()
    byte_counter: list[int] = [0]

    frames_by_dir_log: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    summary: dict[str, Any] = {}

    for split_name in ("train", "val", "test"):
        n = counts[split_name]
        print(f"[{split_name}] Selecting {n} scenes...", flush=True)

        dir_name, token_frames = select_tokens_for_split(
            split_name,
            runtime["splits"][split_name],
            n,
            src_root,
            maps_root,
            repo_root,
            runtime,
        )

        src_sensor_split = src_root / "sensor_blobs" / dir_name
        dst_sensor_split = out_root / "sensor_blobs" / dir_name

        selected = 0
        for token, frame_list in token_frames:
            frame = frame_list[0]
            for rel in relative_sensor_paths(frame, runtime["camera_order"]):
                copy_sensor_file(rel, src_sensor_split, dst_sensor_split, copied, byte_counter)

            frames_by_dir_log[dir_name][frame["log_name"]][token] = frame
            selected += 1

        summary[split_name] = {
            "requested": n,
            "selected": selected,
            "dir": dir_name,
            "tokens": [t for t, _ in token_frames],
            "log_names": sorted({fl[0]["log_name"] for _, fl in token_frames}),
        }

    print(f"[io] Writing trimmed pickles...", flush=True)
    for dir_name, logs in frames_by_dir_log.items():
        logs_out_dir = out_root / "navsim_logs" / dir_name
        logs_out_dir.mkdir(parents=True, exist_ok=True)
        for log_name, token_to_frame in logs.items():
            pkl_path = logs_out_dir / f"{log_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(list(token_to_frame.values()), f)

    print(f"[io] Writing manifest...", flush=True)
    manifest = {
        "source_openscene_data_root": str(src_root),
        "created_counts": {
            "train": summary["train"]["selected"],
            "val": summary["val"]["selected"],
            "test": summary["test"]["selected"],
        },
        "splits": {
            split: {
                "dir": summary[split]["dir"],
                "tokens": summary[split]["tokens"],
                "log_names": summary[split]["log_names"],
            }
            for split in ("train", "val", "test")
        },
    }
    manifest_path = out_root / "subset_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False)

    summary["total_bytes"] = byte_counter[0]
    summary["files_copied"] = len(copied)
    summary["out_root"] = str(out_root)

    return summary


def _human_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit, threshold in [("B", 1), ("KB", 1024), ("MB", 1024**2), ("GB", 1024**3)]:
        if n < threshold * 1024:
            if unit == "B":
                return f"{n}{unit}"
            return f"{n / threshold:.1f}{unit}"
    return f"{n / (1024**4):.1f}TB"


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build a minimal NAVSIM dataset subset for local development."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory (default: bevfusion_model/data/navsim_mini)",
    )
    parser.add_argument("--train", type=int, default=100, help="Number of training samples (default: 100)")
    parser.add_argument("--val", type=int, default=10, help="Number of validation samples (default: 10)")
    parser.add_argument("--test", type=int, default=10, help="Number of test samples (default: 10)")
    parser.add_argument(
        "--openscene-data-root",
        type=Path,
        default=None,
        help="OpenScene data root (default: from runtime config)",
    )
    parser.add_argument(
        "--nuplan-maps-root",
        type=Path,
        default=None,
        help="NuPlan maps root (default: from runtime config)",
    )

    args = parser.parse_args()

    runtime = get_runtime_config()

    out_root = args.out_root or (_REPO_ROOT / "bevfusion_model" / "data" / "navsim_mini")
    src_root = Path(args.openscene_data_root or runtime["openscene_data_root"])
    maps_root = Path(args.nuplan_maps_root or runtime["nuplan_maps_root"])

    print(f"Building subset:", flush=True)
    print(f"  Source: {src_root}", flush=True)
    print(f"  Output: {out_root}", flush=True)
    print(f"  Requested: train={args.train} val={args.val} test={args.test}", flush=True)
    print()

    counts = {"train": args.train, "val": args.val, "test": args.test}
    summary = build_subset(out_root, counts, src_root, maps_root, runtime, _REPO_ROOT)

    print()
    print("=== SUMMARY ===", flush=True)
    for split in ("train", "val", "test"):
        req = summary[split]["requested"]
        sel = summary[split]["selected"]
        status = "OK" if sel == req else f"WARN: selected < requested"
        print(f"{split:6} : {sel:3}/{req:3} {status}", flush=True)

    print(f"Files:  {summary['files_copied']} unique sensor files", flush=True)
    print(f"Size:   {_human_bytes(summary['total_bytes'])}", flush=True)
    print(f"Output: {summary['out_root']}", flush=True)


if __name__ == "__main__":
    main()
