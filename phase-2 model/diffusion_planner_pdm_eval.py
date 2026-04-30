import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffusion_planner import DiffusionPlanner, ObservationNormalizer, cfg as MODEL_CFG
from diffusion_planner_eval_dataset import DiffusionPlannerEvalDataset

from navsim.common.dataclasses import PDMResults, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NAVSIM_CONFIG_DIR = Path(__file__).resolve().parents[1] / "navsim/planning/script/config/pdm_scoring"


def create_scene_aggregators(all_mappings, full_score_df, proposal_sampling):
    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        all_updates.append(aggregator.aggregate_scores())

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])
    return full_score_df


def compute_final_scores(pdm_score_df):
    df = pdm_score_df.reset_index()
    assert not df["two_frame_extended_comfort"].isna().any(), \
        "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())

    weighted_metrics[:, WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = two_frame_scores
    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * (weighted_sum / total_weight)
    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)
    return df


def load_pdm_cfg(split: str):
    with initialize_config_dir(config_dir=str(NAVSIM_CONFIG_DIR.resolve()), version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[f"train_test_split={split}", "worker=single_machine_thread_pool"],
        )
    return cfg


def batchify(features, device):
    return {k: v.to(device).unsqueeze(0) for k, v in features.items()}


def score_token(token, frame_type_hint, *, scene_loader, dataset, processor, model, observation_normalizer, simulator, scorer, traffic_policy, device):
    """Score a single token and return a DataFrame row."""
    try:
        metric_cache = processor.compute_metric_cache(
            NavSimScenario(
                scene_loader.get_scene_from_token(token),
                map_root=os.environ["NUPLAN_MAPS_ROOT"],
                map_version="nuplan-maps-v1.0",
            )
        )

        frame_list = dataset.get_frame_list(token)

        neighbor_tokens = dataset._select_neighbor_tokens(frame_list)
        features = dataset._build_diffusion_planner_inputs(frame_list, neighbor_tokens)

        features_batch = batchify(features, device)
        features_batch = observation_normalizer(features_batch)

        with torch.no_grad():
            _, decoder_out = model(features_batch)

        pred = decoder_out["prediction"][0, 0]
        assert pred.shape[0] >= simulator.proposal_sampling.num_poses, \
            f"Prediction length {pred.shape[0]} < num_poses {simulator.proposal_sampling.num_poses}"

        num_poses = simulator.proposal_sampling.num_poses
        heading = torch.atan2(pred[:num_poses, 3], pred[:num_poses, 2])
        poses = torch.stack([pred[:num_poses, 0], pred[:num_poses, 1], heading], dim=-1).cpu().numpy()

        trajectory = Trajectory(poses=poses, trajectory_sampling=simulator.proposal_sampling)

        score_row, ego_simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_policy,
        )

        score_row["valid"] = True
        score_row["log_name"] = metric_cache.log_name
        score_row["frame_type"] = metric_cache.scene_type
        score_row["start_time"] = metric_cache.timepoint.time_s
        end_pose = StateSE2(
            x=trajectory.poses[-1, 0],
            y=trajectory.poses[-1, 1],
            heading=trajectory.poses[-1, 2],
        )
        absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
        score_row["endpoint_x"] = absolute_endpoint.x
        score_row["endpoint_y"] = absolute_endpoint.y
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [ego_simulated_states]

    except Exception:
        logger.warning(f"----------- Model failed for token {token}:")
        traceback.print_exc()
        score_row = pd.DataFrame([PDMResults.get_empty_results()])
        score_row["valid"] = False

    score_row["token"] = token
    return score_row


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiffusionPlanner with PDM Score")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--split", type=str, default="navtest_two_stage", help="Data split to evaluate")
    parser.add_argument("--output-dir", type=str, default="./pdm_eval_output", help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tokens (for testing)")
    parser.add_argument(
        "--weights",
        type=str,
        default="ema",
        choices=["ema", "model"],
        help="Which weights to load from a training snapshot (default: ema)",
    )
    args = parser.parse_args()

    if "NUPLAN_MAPS_ROOT" not in os.environ:
        raise RuntimeError("NUPLAN_MAPS_ROOT environment variable not set")
    if "OPENSCENE_DATA_ROOT" not in os.environ:
        raise RuntimeError("OPENSCENE_DATA_ROOT environment variable not set")

    logger.info(f"Loading config for split: {args.split}")
    cfg = load_pdm_cfg(args.split)

    logger.info(f"Building scene loader and objects")
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    traffic_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    processor = MetricCacheProcessor(cache_path=None, force_feature_computation=True, proposal_sampling=simulator.proposal_sampling)

    logger.info(f"Building dataset")
    dataset = DiffusionPlannerEvalDataset(scene_loader=scene_loader, cfg=MODEL_CFG)

    logger.info(f"Building and loading model from {args.ckpt}")
    device = args.device
    model = DiffusionPlanner(MODEL_CFG).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)

    if isinstance(ckpt, dict) and ("ema_state_dict" in ckpt or "model" in ckpt):
        if args.weights == "ema" and "ema_state_dict" in ckpt:
            state_dict = ckpt["ema_state_dict"]
            logger.info("Loading EMA weights from training snapshot")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            logger.info("Loading raw model weights from training snapshot")
        else:
            raise KeyError(f"Could not find requested weights in checkpoint: keys={list(ckpt.keys())}")
    else:
        state_dict = ckpt

    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)

    observation_normalizer = ObservationNormalizer(
        {
            k: {kk: torch.tensor(vv, dtype=torch.float32, device=device) for kk, vv in v.items()}
            for k, v in MODEL_CFG["observation_normalizer"].items()
        }
    )

    logger.info(f"Starting evaluation on split: {args.split}")
    rows = []
    tokens_stage_one = list(scene_loader.tokens_stage_one)
    tokens_stage_two = list(scene_loader.reactive_tokens_stage_two)

    if args.limit is not None:
        tokens_stage_one = tokens_stage_one[: args.limit]
        tokens_stage_two = tokens_stage_two[: args.limit]

    logger.info(f"Stage 1: {len(tokens_stage_one)} tokens, Stage 2: {len(tokens_stage_two)} tokens")

    for token in tqdm(tokens_stage_one, desc="stage 1"):
        row = score_token(
            token,
            "stage_one",
            scene_loader=scene_loader,
            dataset=dataset,
            processor=processor,
            model=model,
            observation_normalizer=observation_normalizer,
            simulator=simulator,
            scorer=scorer,
            traffic_policy=traffic_policy,
            device=device,
        )
        rows.append(row)

    for token in tqdm(tokens_stage_two, desc="stage 2"):
        row = score_token(
            token,
            "stage_two",
            scene_loader=scene_loader,
            dataset=dataset,
            processor=processor,
            model=model,
            observation_normalizer=observation_normalizer,
            simulator=simulator,
            scorer=scorer,
            traffic_policy=traffic_policy,
            device=device,
        )
        rows.append(row)

    pdm_score_df = pd.concat(rows, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_scores_path = output_dir / "raw_scores.csv"
    pdm_score_df.to_csv(raw_scores_path, index=False)
    logger.info(f"Saved raw scores to {raw_scores_path}")

    try:
        raw_mapping = cfg.train_test_split.reactive_all_mapping
        all_mappings = {}
        scored_tokens = set(scene_loader.tokens)

        for orig_token, prev_token, two_stage_pairs in raw_mapping:
            if prev_token in scored_tokens or orig_token in scored_tokens:
                all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]

        logger.info(f"Aggregating scores with {len(all_mappings)} mappings")
        pdm_score_df = create_scene_aggregators(all_mappings, pdm_score_df, instantiate(cfg.simulator.proposal_sampling))
        pdm_score_df = compute_final_scores(pdm_score_df)

        final_scores_path = output_dir / "pdm_scores.csv"
        pdm_score_df.to_csv(final_scores_path, index=False)
        logger.info(f"Saved aggregated scores to {final_scores_path}")

    except Exception:
        logger.warning("----------- Failed to calculate aggregation, skipping:")
        traceback.print_exc()

    num_valid = pdm_score_df["valid"].sum()
    num_failed = len(pdm_score_df) - num_valid
    logger.info(f"Evaluation complete: {num_valid} valid, {num_failed} failed tokens")

    if "score" in pdm_score_df.columns:
        final_score = pdm_score_df[pdm_score_df["valid"]]["score"].mean()
        logger.info(f"Final PDM Score: {final_score:.4f}")


if __name__ == "__main__":
    main()
