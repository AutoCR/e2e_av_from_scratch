"""Validate NAVSIM train-log frames against configured training tokens.

This script scans the configured runtime split logs directly and reports:
  - how many raw frame tokens exist in the configured logs;
  - how many configured YAML tokens are present in those logs;
  - how many tokens would be loaded with/without token filtering;
  - how many tokens would be loaded with/without the route filter.

Run:
    uv run python bevfusion_model/validate_navsim_train_logs.py

Useful comparisons:
    uv run python bevfusion_model/validate_navsim_train_logs.py --split train
    uv run python bevfusion_model/validate_navsim_train_logs.py --split train --show-examples
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "bevfusion_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bevfusion_model.configs.bevfusion_hyperparams import get_runtime_config
from bevfusion_model.navsim_train.runner_utils import resolve_split_config


def _preview(values, limit: int = 10) -> str:
    values = list(values)
    if not values:
        return "[]"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit})"
    return "[" + ", ".join(str(value) for value in shown) + suffix + "]"


def _load_log(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        return pickle.load(f)


def _window_tokens(
    frames: list[dict[str, Any]],
    num_history_frames: int,
    num_future_frames: int,
    frame_interval: int,
) -> tuple[list[str], list[str]]:
    num_frames = num_history_frames + num_future_frames
    current_index = num_history_frames - 1
    all_tokens: list[str] = []
    route_tokens: list[str] = []

    for start in range(0, len(frames), frame_interval):
        frame_list = frames[start : start + num_frames]
        if len(frame_list) < num_frames:
            continue
        current_frame = frame_list[current_index]
        token = str(current_frame["token"])
        all_tokens.append(token)
        if len(current_frame.get("roadblock_ids", [])) > 0:
            route_tokens.append(token)

    return all_tokens, route_tokens


def _count_intersection(left: set[str], right: set[str] | None) -> int:
    return len(left) if right is None else len(left & right)


def main() -> None:
    config = get_runtime_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--num-history-frames",
        type=int,
        default=int(config.get("num_history_frames", 1)),
    )
    parser.add_argument(
        "--num-future-frames",
        type=int,
        default=int(config.get("num_future_frames", 0)),
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Stride used to generate sample windows.",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print example missing/filtered tokens.",
    )
    args = parser.parse_args()

    split_cfg = config["splits"][args.split]
    split_dir, log_names, configured_tokens = resolve_split_config(split_cfg, _REPO_ROOT)
    configured_token_set = set(configured_tokens) if configured_tokens is not None else None
    current_has_route = bool(split_cfg.get("has_route", config.get("has_route", True)))

    data_path = Path(config["openscene_data_root"]) / "navsim_logs" / split_dir
    if not data_path.exists():
        raise FileNotFoundError(f"NAVSIM log path does not exist: {data_path}")

    all_log_files = sorted(data_path.glob("*.pkl"))
    configured_log_set = set(log_names) if log_names is not None else None
    selected_log_files = [
        path for path in all_log_files
        if configured_log_set is None or path.stem in configured_log_set
    ]
    existing_log_names = {path.stem for path in selected_log_files}
    missing_logs = sorted(configured_log_set - existing_log_names) if configured_log_set is not None else []

    raw_frame_count = 0
    raw_token_counter: Counter[str] = Counter()
    window_token_counter: Counter[str] = Counter()
    route_window_token_counter: Counter[str] = Counter()

    for log_path in tqdm(selected_log_files, desc=f"Scanning {args.split} logs"):
        frames = _load_log(log_path)
        raw_frame_count += len(frames)
        raw_token_counter.update(str(frame["token"]) for frame in frames)
        window_tokens, route_window_tokens = _window_tokens(
            frames,
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_interval=args.frame_interval,
        )
        window_token_counter.update(window_tokens)
        route_window_token_counter.update(route_window_tokens)

    raw_tokens = set(raw_token_counter)
    window_tokens = set(window_token_counter)
    route_window_tokens = set(route_window_token_counter)
    duplicate_raw_tokens = sorted(token for token, count in raw_token_counter.items() if count > 1)
    duplicate_window_tokens = sorted(token for token, count in window_token_counter.items() if count > 1)

    print("\nConfig")
    print(f"split: {args.split}")
    print(f"split_dir: {split_dir}")
    print(f"navsim_logs_path: {data_path}")
    print(f"configured_log_names_count: {0 if log_names is None else len(log_names)}")
    print(f"existing_selected_log_files_count: {len(selected_log_files)}")
    print(f"missing_configured_log_files_count: {len(missing_logs)}")
    print(f"configured_tokens_count: {0 if configured_tokens is None else len(configured_tokens)}")
    print(f"current_config_tokens_filter_enabled: {configured_tokens is not None}")
    print(f"current_config_has_route: {current_has_route}")
    print(
        "sample_window: "
        f"history={args.num_history_frames} future={args.num_future_frames} "
        f"frames_per_sample={args.num_history_frames + args.num_future_frames} "
        f"frame_interval={args.frame_interval}"
    )

    print("\nRaw Configured Logs")
    print(f"raw_frame_count: {raw_frame_count}")
    print(f"raw_unique_frame_tokens_count: {len(raw_tokens)}")
    print(f"duplicate_raw_frame_tokens_count: {len(duplicate_raw_tokens)}")
    print(f"window_candidate_unique_tokens_count: {len(window_tokens)}")
    print(f"route_window_candidate_unique_tokens_count: {len(route_window_tokens)}")
    print(f"route_filtered_out_window_tokens_count: {len(window_tokens - route_window_tokens)}")
    print(f"duplicate_window_candidate_tokens_count: {len(duplicate_window_tokens)}")

    print("\nExpected Dataset Sizes From These Logs")
    print(f"tokens=None, has_route=False: {len(window_tokens)}")
    print(f"tokens=None, has_route=True:  {len(route_window_tokens)}")
    current_token_pool = configured_token_set if configured_token_set is not None else window_tokens
    current_route_pool = route_window_tokens if current_has_route else window_tokens
    current_dataset_tokens = current_route_pool & current_token_pool
    print(f"current config:             {len(current_dataset_tokens)}")
    if configured_token_set is not None:
        print(f"tokens=configured, has_route=False: {_count_intersection(window_tokens, configured_token_set)}")
        print(f"tokens=configured, has_route=True:  {_count_intersection(route_window_tokens, configured_token_set)}")

        configured_present_in_raw = configured_token_set & raw_tokens
        configured_present_in_windows = configured_token_set & window_tokens
        configured_present_with_route = configured_token_set & route_window_tokens
        missing_from_raw = sorted(configured_token_set - raw_tokens)
        missing_from_windows = sorted(configured_token_set - window_tokens)
        missing_due_to_route = sorted(configured_present_in_windows - route_window_tokens)
        raw_not_configured = sorted(raw_tokens - configured_token_set)
        windows_not_configured = sorted(window_tokens - configured_token_set)

        print("\nConfigured Token Validation")
        print(f"configured_tokens_present_in_raw_logs_count: {len(configured_present_in_raw)}")
        print(f"configured_tokens_missing_from_raw_logs_count: {len(missing_from_raw)}")
        print(f"configured_tokens_present_as_window_candidates_count: {len(configured_present_in_windows)}")
        print(f"configured_tokens_missing_as_window_candidates_count: {len(missing_from_windows)}")
        print(f"configured_tokens_present_with_route_count: {len(configured_present_with_route)}")
        print(f"configured_tokens_filtered_out_by_route_count: {len(missing_due_to_route)}")
        print(f"raw_log_tokens_not_in_configured_tokens_count: {len(raw_not_configured)}")
        print(f"window_candidate_tokens_not_in_configured_tokens_count: {len(windows_not_configured)}")

        current_training_dataset_tokens = current_dataset_tokens
        only_configured_in_dataset = current_training_dataset_tokens <= configured_token_set
        all_configured_log_windows_in_configured_dataset = window_tokens <= configured_token_set
        print("\nAnswers")
        print(f"current_training_dataset_token_count: {len(current_training_dataset_tokens)}")
        print(
            "current_training_dataset_contains_only_configured_tokens: "
            f"{only_configured_in_dataset}"
        )
        print(
            "all_window_candidates_from_configured_logs_are_in_configured_tokens: "
            f"{all_configured_log_windows_in_configured_dataset}"
        )
    else:
        missing_from_raw = []
        missing_from_windows = []
        missing_due_to_route = []
        raw_not_configured = []
        windows_not_configured = []
        print("\nAnswers")
        print(f"current_training_dataset_token_count: {len(current_dataset_tokens)}")
        print("current_training_dataset_contains_only_configured_tokens: no token filter configured")

    if args.show_examples:
        print("\nExamples")
        print(f"missing_configured_log_files: {_preview(missing_logs)}")
        print(f"duplicate_raw_frame_tokens: {_preview(duplicate_raw_tokens)}")
        print(f"duplicate_window_candidate_tokens: {_preview(duplicate_window_tokens)}")
        if configured_token_set is not None:
            print(f"configured_tokens_missing_from_raw_logs: {_preview(missing_from_raw)}")
            print(f"configured_tokens_missing_as_window_candidates: {_preview(missing_from_windows)}")
            print(f"configured_tokens_filtered_out_by_route: {_preview(missing_due_to_route)}")
            print(f"raw_log_tokens_not_in_configured_tokens: {_preview(raw_not_configured)}")
            print(f"window_candidate_tokens_not_in_configured_tokens: {_preview(windows_not_configured)}")


if __name__ == "__main__":
    main()
