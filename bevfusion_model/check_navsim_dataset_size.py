"""Print NAVSIM BEVFusion dataset sizes and configured token counts.

Run:
    uv run python bevfusion_model/check_navsim_dataset_size.py

Examples:
    uv run python bevfusion_model/check_navsim_dataset_size.py --split train
    uv run python bevfusion_model/check_navsim_dataset_size.py --num-history-frames 4 --num-future-frames 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "bevfusion_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bevfusion_model.configs.bevfusion_hyperparams import get_runtime_config
from bevfusion_model.navsim_dataset import NavSimBEVFusionDataset
from bevfusion_model.navsim_train.runner_utils import resolve_split_config


def _count(values) -> int:
    return 0 if values is None else len(values)


def _preview(values, limit: int = 8) -> str:
    if not values:
        return "[]"
    shown = list(values)[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit})"
    return "[" + ", ".join(str(value) for value in shown) + suffix + "]"


def _build_dataset(config: dict, split_name: str, split_dir: str, log_names, tokens, args) -> NavSimBEVFusionDataset:
    return NavSimBEVFusionDataset(
        split=split_dir,
        openscene_data_root=config["openscene_data_root"],
        nuplan_maps_root=config["nuplan_maps_root"],
        camera_order=config.get("camera_order"),
        image_hw=tuple(config.get("image_hw", (256, 704))),
        test_mode=False,
        max_scenes=args.max_scenes,
        log_names=log_names,
        tokens=tokens,
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
    )


def _report_split(config: dict, split_name: str, args) -> None:
    split_cfg = config["splits"][split_name]
    split_dir, log_names, tokens = resolve_split_config(split_cfg, _REPO_ROOT)

    print(f"\n[{split_name}]")
    print(f"split_dir: {split_dir}")
    print(f"log_names_count: {_count(log_names)}")
    print(f"configured_tokens_count: {_count(tokens)}")
    print(
        "sample_window: "
        f"history={args.num_history_frames} future={args.num_future_frames} "
        f"frames_per_sample={args.num_history_frames + args.num_future_frames}"
    )

    dataset = _build_dataset(config, split_name, split_dir, log_names, tokens, args)
    dataset_tokens = list(dataset.scene_tokens)
    scene_loader_tokens = list(dataset.scene_loader.tokens)

    print(f"dataset_size: {len(dataset)}")
    print(f"dataset_scene_tokens_count: {len(dataset_tokens)}")
    print(f"scene_loader_tokens_count: {len(scene_loader_tokens)}")
    print(f"loaded_original_scene_windows: {len(dataset.scene_loader.scene_frames_dicts)}")

    if tokens is not None:
        configured = set(tokens)
        loaded = set(dataset_tokens)
        missing = sorted(configured - loaded)
        extra = sorted(loaded - configured)
        print(f"loaded_configured_tokens_count: {len(configured & loaded)}")
        print(f"missing_configured_tokens_count: {len(missing)}")
        print(f"extra_loaded_tokens_count: {len(extra)}")
        if args.show_examples:
            print(f"missing_examples: {_preview(missing)}")
            print(f"extra_examples: {_preview(extra)}")


def main() -> None:
    config = get_runtime_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Which runtime split to inspect.",
    )
    parser.add_argument(
        "--num-history-frames",
        type=int,
        default=int(config.get("num_history_frames", 1)),
        help="History frames used to build each NAVSIM sample window.",
    )
    parser.add_argument(
        "--num-future-frames",
        type=int,
        default=int(config.get("num_future_frames", 0)),
        help="Future frames used to build each NAVSIM sample window.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap for quick checks; defaults to no cap.",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print a few missing/extra token examples for each split.",
    )
    args = parser.parse_args()

    print(f"openscene_data_root: {config['openscene_data_root']}")
    print(f"nuplan_maps_root: {config['nuplan_maps_root']}")

    split_names = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split_name in split_names:
        _report_split(config, split_name, args)


if __name__ == "__main__":
    main()
