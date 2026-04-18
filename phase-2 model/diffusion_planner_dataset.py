import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyogrio
import torch
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

from navsim.common.enums import BoundingBoxIndex

try:
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
except ModuleNotFoundError:
    SceneLoader = Any

    def normalize_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    class StateSE2:
        def __init__(self, x: float, y: float, heading: float):
            self.x = x
            self.y = y
            self.heading = heading

        @property
        def point(self):
            return None

    class _TrackedObjectType:
        VEHICLE = "vehicle"
        PEDESTRIAN = "pedestrian"
        BICYCLE = "bicycle"
        TRAFFIC_CONE = "traffic_cone"
        BARRIER = "barrier"
        CZONE_SIGN = "czone_sign"
        GENERIC_OBJECT = "generic_object"
        EGO = "ego"

    class _SemanticMapLayer:
        LANE = "LANE"
        LANE_CONNECTOR = "LANE_CONNECTOR"
        INTERSECTION = "INTERSECTION"
        WALKWAYS = "WALKWAYS"

    TrackedObjectType = _TrackedObjectType
    SemanticMapLayer = _SemanticMapLayer
    OrientedBox = None

    def get_maps_api(*args, **kwargs):
        raise ModuleNotFoundError("nuplan-devkit is required for DiffusionPlannerDataset.")


class DiffusionPlannerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        max_len=None,
        random_sample=False,
        trajectory_num_poses=None,
        max_ped_bike=5,
        cfg: dict = None,
    ):
        if cfg is None:
            raise ValueError("`cfg` must be provided for DiffusionPlannerDataset.")

        required_cfg_keys = [
            "agent_num",
            "predicted_neighbor_num",
            "static_objects_num",
            "lane_num",
            "route_num",
            "lane_len",
            "time_len",
            "future_len",
        ]
        missing_cfg_keys = [key for key in required_cfg_keys if key not in cfg]
        if missing_cfg_keys:
            raise KeyError(f"Missing required cfg keys for DiffusionPlannerDataset: {missing_cfg_keys}")

        self.scene_loader = scene_loader
        self.trajectory_num_poses = trajectory_num_poses

        if random_sample:
            if max_len is None:
                max_len = len(scene_loader.tokens)
            max_len = min(max_len, len(scene_loader.tokens))
            self.tokens = np.random.choice(scene_loader.tokens, size=max_len, replace=False).tolist()
        else:
            max_len = min(max_len, len(scene_loader.tokens)) if max_len is not None else len(scene_loader.tokens)
            self.tokens = scene_loader.tokens[:max_len]

        self.max_len = len(self.tokens)
        self.current_index = self.scene_loader._scene_filter.num_history_frames - 1

        self.agent_num = cfg['agent_num']
        self.predicted_neighbor_num = cfg['predicted_neighbor_num']
        self.static_objects_num = cfg['static_objects_num']
        self.lane_num = cfg['lane_num']
        self.route_num = cfg['route_num']
        self.lane_len = cfg['lane_len']
        self.time_len = cfg['time_len']
        self.future_len = cfg['future_len']
        self.max_ped_bike = max_ped_bike
        self.map_radius = 100.0
        self.model_interval = 0.1

        self.bev_pixel_width = 256
        self.bev_pixel_height = 128
        self.bev_pixel_size = 0.25
        self.num_bev_classes = 7
        self.bev_radius = 32.0

        self.dynamic_tracked_object_types = {
            "vehicle": TrackedObjectType.VEHICLE,
            "pedestrian": TrackedObjectType.PEDESTRIAN,
            "bicycle": TrackedObjectType.BICYCLE,
        }
        self.static_tracked_object_types = {
            "traffic_cone": TrackedObjectType.TRAFFIC_CONE,
            "barrier": TrackedObjectType.BARRIER,
            "czone_sign": TrackedObjectType.CZONE_SIGN,
            "generic_object": TrackedObjectType.GENERIC_OBJECT,
        }
        self.tracked_object_types = {
            **self.dynamic_tracked_object_types,
            **self.static_tracked_object_types,
            "ego": TrackedObjectType.EGO,
        }
        self.bev_semantic_classes = {
            1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),
            2: ("polygon", [SemanticMapLayer.WALKWAYS]),
            3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),
            4: (
                "box",
                [
                    TrackedObjectType.CZONE_SIGN,
                    TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT,
                ],
            ),
            5: ("box", [TrackedObjectType.VEHICLE]),
            6: ("box", [TrackedObjectType.PEDESTRIAN]),
        }

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        token = self.tokens[idx]
        frame_list = self.scene_loader.scene_frames_dicts[token]
        selected_neighbor_tokens = self._select_neighbor_tokens(frame_list)

        features = self._build_diffusion_planner_inputs(frame_list, selected_neighbor_tokens)
        targets = {
            "ego_future_gt": self._build_ego_future(frame_list),
            "neighbors_future_gt": self._build_neighbors_future(frame_list, selected_neighbor_tokens),
            "neighbor_future_mask": self._build_neighbor_future_mask(frame_list, selected_neighbor_tokens),
            "trajectory": self._build_future_trajectory(frame_list),
        }
        return token, features, targets

    def _maps_root(self) -> str:
        maps_root = os.getenv("NUPLAN_MAPS_ROOT")
        if maps_root is None:
            raise RuntimeError("NUPLAN_MAPS_ROOT is not set.")
        return maps_root

    def _map_api(self, map_name: str):
        return get_maps_api(self._maps_root(), "nuplan-maps-v1.0", map_name)

    def _frame_pose(self, frame_dict) -> np.ndarray:
        translation = np.asarray(frame_dict["ego2global_translation"][:2], dtype=np.float64)
        yaw = Quaternion(*frame_dict["ego2global_rotation"]).yaw_pitch_roll[0]
        return np.array([translation[0], translation[1], yaw], dtype=np.float64)

    def _relative_times(self, frame_list_slice, reference_timestamp: int) -> np.ndarray:
        timestamps = np.asarray([frame["timestamp"] for frame in frame_list_slice], dtype=np.int64)
        return (timestamps - reference_timestamp).astype(np.float64) * 1e-6

    def _history_target_times(self) -> np.ndarray:
        return np.linspace(-(self.time_len - 1) * self.model_interval, 0.0, self.time_len, dtype=np.float64)

    def _future_target_times(self) -> np.ndarray:
        return np.linspace(self.model_interval, self.future_len * self.model_interval, self.future_len, dtype=np.float64)

    def _rotation_matrix(self, angle: float) -> np.ndarray:
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64)

    def _to_relative_poses(self, origin_pose: np.ndarray, global_poses: np.ndarray) -> np.ndarray:
        theta = -origin_pose[2]
        rotation = self._rotation_matrix(theta)
        relative = global_poses - origin_pose[None]
        relative[:, :2] = relative[:, :2] @ rotation.T
        relative[:, 2] = np.arctan2(np.sin(relative[:, 2]), np.cos(relative[:, 2]))
        return relative

    def _global_to_local_xy(self, origin_pose: np.ndarray, global_xy: np.ndarray) -> np.ndarray:
        return self._rotation_matrix(-origin_pose[2]) @ (global_xy - origin_pose[:2])

    def _local_to_global_xy(self, frame_pose: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
        return frame_pose[:2] + self._rotation_matrix(frame_pose[2]) @ local_xy

    def _global_to_local_vec(self, origin_pose: np.ndarray, global_vec: np.ndarray) -> np.ndarray:
        return self._rotation_matrix(-origin_pose[2]) @ global_vec

    def _local_to_global_vec(self, frame_pose: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
        return self._rotation_matrix(frame_pose[2]) @ local_vec

    def _interp_feature(self, sample_times: np.ndarray, sample_values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
        sample_times = np.asarray(sample_times, dtype=np.float64)
        sample_values = np.asarray(sample_values, dtype=np.float64)
        if sample_values.ndim == 1:
            sample_values = sample_values[:, None]

        result = np.zeros((len(target_times), sample_values.shape[-1]), dtype=np.float64)
        for dim in range(sample_values.shape[-1]):
            result[:, dim] = np.interp(
                target_times,
                sample_times,
                sample_values[:, dim],
                left=sample_values[0, dim],
                right=sample_values[-1, dim],
            )
        return result

    def _interp_heading(self, sample_times: np.ndarray, headings: np.ndarray, target_times: np.ndarray) -> np.ndarray:
        headings = np.asarray(headings, dtype=np.float64)
        headings_unwrapped = np.unwrap(headings)
        interp = np.interp(
            target_times,
            sample_times,
            headings_unwrapped,
            left=headings_unwrapped[0],
            right=headings_unwrapped[-1],
        )
        return np.arctan2(np.sin(interp), np.cos(interp))

    def _fill_history_states(self, values: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filled = values.copy()
        valid = valid_mask.copy()
        if not valid.any():
            return filled, valid

        last_valid_idx = None
        for idx in range(len(valid) - 1, -1, -1):
            if valid[idx]:
                last_valid_idx = idx
            elif last_valid_idx is not None:
                filled[idx] = filled[last_valid_idx]
                valid[idx] = True
        return filled, valid

    def _history_time_grid(self, frame_list) -> tuple[np.ndarray, np.ndarray]:
        history_frames = frame_list[: self.current_index + 1]
        reference_timestamp = frame_list[self.current_index]["timestamp"]
        return self._relative_times(history_frames, reference_timestamp), self._history_target_times()

    def _future_time_grid(self, frame_list) -> tuple[np.ndarray, np.ndarray]:
        current_frame = frame_list[self.current_index]
        future_frames = frame_list[self.current_index + 1 :]
        reference_timestamp = current_frame["timestamp"]
        future_times = self._relative_times([current_frame] + future_frames, reference_timestamp)
        return future_times, self._future_target_times()

    def _build_future_trajectory(self, frame_list) -> torch.Tensor:
        current_pose = self._frame_pose(frame_list[self.current_index])
        future_frames = frame_list[self.current_index + 1 :]
        if len(future_frames) == 0:
            return torch.zeros((self.future_len, 3), dtype=torch.float32)

        coarse_times, target_times = self._future_time_grid(frame_list)
        coarse_poses = np.stack([current_pose] + [self._frame_pose(frame_dict) for frame_dict in future_frames], axis=0)
        coarse_relative = self._to_relative_poses(current_pose, coarse_poses)

        xy = self._interp_feature(coarse_times, coarse_relative[:, :2], target_times)
        heading = self._interp_heading(coarse_times, coarse_relative[:, 2], target_times)
        future_relative_poses = np.concatenate([xy, heading[:, None]], axis=-1).astype(np.float32)
        return torch.from_numpy(future_relative_poses)

    def _iter_annotations(self, frame_dict):
        boxes = np.asarray(frame_dict["anns"]["gt_boxes"], dtype=np.float32)
        names = np.asarray(frame_dict["anns"]["gt_names"])
        velocities = np.asarray(frame_dict["anns"]["gt_velocity_3d"], dtype=np.float32)
        track_tokens = np.asarray(frame_dict["anns"]["track_tokens"])
        for box, name, velocity, track_token in zip(boxes, names, velocities, track_tokens):
            yield box, str(name), velocity, str(track_token)

    def _agent_type_one_hot(self, tracked_type: TrackedObjectType) -> np.ndarray:
        if tracked_type == TrackedObjectType.VEHICLE:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if tracked_type == TrackedObjectType.PEDESTRIAN:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _static_type_one_hot(self, tracked_type: TrackedObjectType) -> np.ndarray:
        if tracked_type == TrackedObjectType.CZONE_SIGN:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if tracked_type == TrackedObjectType.BARRIER:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if tracked_type == TrackedObjectType.TRAFFIC_CONE:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _select_neighbor_tokens(self, frame_list):
        current_frame = frame_list[self.current_index]
        candidates = []
        for box, name, _, track_token in self._iter_annotations(current_frame):
            tracked_type = self.dynamic_tracked_object_types.get(name)
            if tracked_type is None:
                continue
            distance = float(np.linalg.norm(box[BoundingBoxIndex.POINT2D]))
            candidates.append((distance, tracked_type, track_token))

        candidates.sort(key=lambda item: item[0])
        ped_bike = [
            item for item in candidates if item[1] in (TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE)
        ]
        vehicles = [item for item in candidates if item[1] == TrackedObjectType.VEHICLE]

        if len(candidates) <= self.agent_num:
            selected = candidates[: self.agent_num]
        else:
            selected = ped_bike[: self.max_ped_bike] + vehicles
            remaining_slots = self.agent_num - len(selected)
            if remaining_slots > 0:
                selected += ped_bike[self.max_ped_bike : self.max_ped_bike + remaining_slots]
            selected = sorted(selected, key=lambda item: item[0])[: self.agent_num]

        return [track_token for _, _, track_token in selected]

    def _convert_box_to_current_frame(self, box: np.ndarray, velocity: np.ndarray, frame_pose: np.ndarray, current_pose: np.ndarray):
        box_xy_global = self._local_to_global_xy(
            frame_pose,
            np.array([box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]], dtype=np.float64),
        )
        box_heading_global = normalize_angle(frame_pose[2] + float(box[BoundingBoxIndex.HEADING]))
        velocity_global = self._local_to_global_vec(frame_pose, np.nan_to_num(np.asarray(velocity[:2], dtype=np.float64)))

        box_xy_local = self._global_to_local_xy(current_pose, box_xy_global)
        box_heading_local = normalize_angle(box_heading_global - current_pose[2])
        velocity_local = self._global_to_local_vec(current_pose, velocity_global)
        return box_xy_local, box_heading_local, velocity_local

    def _build_ego_current_state(self, frame_list) -> torch.Tensor:
        current_pose = self._frame_pose(frame_list[self.current_index])
        current_frame = frame_list[self.current_index]
        current_dynamic = np.asarray(current_frame["ego_dynamic_state"], dtype=np.float64)
        current_velocity = self._global_to_local_vec(
            current_pose, self._local_to_global_vec(current_pose, current_dynamic[:2])
        )
        current_accel = self._global_to_local_vec(
            current_pose, self._local_to_global_vec(current_pose, current_dynamic[2:4])
        )

        history_times, target_times = self._history_time_grid(frame_list)
        history_poses = np.stack([self._frame_pose(frame_dict) for frame_dict in frame_list[: self.current_index + 1]], axis=0)
        history_relative = self._to_relative_poses(current_pose, history_poses)
        interp_heading = self._interp_heading(history_times, history_relative[:, 2], target_times)
        yaw_rate = 0.0 if len(interp_heading) < 2 else normalize_angle(interp_heading[-1] - interp_heading[-2]) / self.model_interval

        if abs(current_velocity[0]) < 0.2:
            steering_angle = 0.0
            yaw_rate = 0.0
        else:
            wheel_base = 3.089
            steering_angle = np.arctan(yaw_rate * wheel_base / abs(current_velocity[0]))
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

        ego_current_state = np.zeros((10,), dtype=np.float32)
        ego_current_state[0] = 0.0
        ego_current_state[1] = 0.0
        ego_current_state[2] = 1.0
        ego_current_state[3] = 0.0
        ego_current_state[4:6] = current_velocity.astype(np.float32)
        ego_current_state[6:8] = current_accel.astype(np.float32)
        ego_current_state[8] = float(steering_angle)
        ego_current_state[9] = float(yaw_rate)
        return torch.from_numpy(ego_current_state)

    def _build_neighbor_agents_past(self, frame_list, selected_neighbor_tokens) -> torch.Tensor:
        current_pose = self._frame_pose(frame_list[self.current_index])
        history_frames = frame_list[: self.current_index + 1]
        token_to_index = {track_token: idx for idx, track_token in enumerate(selected_neighbor_tokens)}
        coarse_neighbors = np.zeros((self.agent_num, len(history_frames), 8), dtype=np.float64)
        coarse_valid = np.zeros((self.agent_num, len(history_frames)), dtype=bool)
        type_features = np.zeros((self.agent_num, 3), dtype=np.float32)

        for t, frame_dict in enumerate(history_frames):
            frame_pose = self._frame_pose(frame_dict)
            for box, name, velocity, track_token in self._iter_annotations(frame_dict):
                tracked_type = self.dynamic_tracked_object_types.get(name)
                if tracked_type is None or track_token not in token_to_index:
                    continue

                agent_idx = token_to_index[track_token]
                xy_local, heading_local, velocity_local = self._convert_box_to_current_frame(
                    box, velocity, frame_pose, current_pose
                )
                coarse_neighbors[agent_idx, t, 0:2] = xy_local
                coarse_neighbors[agent_idx, t, 2] = heading_local
                coarse_neighbors[agent_idx, t, 3:5] = velocity_local
                coarse_neighbors[agent_idx, t, 5] = float(box[BoundingBoxIndex.WIDTH])
                coarse_neighbors[agent_idx, t, 6] = float(box[BoundingBoxIndex.LENGTH])
                coarse_neighbors[agent_idx, t, 7] = 1.0
                coarse_valid[agent_idx, t] = True
                type_features[agent_idx] = self._agent_type_one_hot(tracked_type)

        history_times, target_times = self._history_time_grid(frame_list)
        neighbors = np.zeros((self.agent_num, self.time_len, 11), dtype=np.float32)
        for agent_idx in range(self.agent_num):
            filled_values, filled_valid = self._fill_history_states(coarse_neighbors[agent_idx], coarse_valid[agent_idx])
            if not filled_valid.any():
                continue

            interp_xy = self._interp_feature(history_times, filled_values[:, 0:2], target_times)
            interp_heading = self._interp_heading(history_times, filled_values[:, 2], target_times)
            interp_velocity = self._interp_feature(history_times, filled_values[:, 3:5], target_times)
            interp_size = self._interp_feature(history_times, filled_values[:, 5:7], target_times)
            interp_valid = self._interp_feature(history_times, filled_values[:, 7], target_times)[:, 0]

            neighbors[agent_idx, :, 0:2] = interp_xy.astype(np.float32)
            neighbors[agent_idx, :, 2] = np.cos(interp_heading).astype(np.float32)
            neighbors[agent_idx, :, 3] = np.sin(interp_heading).astype(np.float32)
            neighbors[agent_idx, :, 4:6] = interp_velocity.astype(np.float32)
            neighbors[agent_idx, :, 6:8] = interp_size.astype(np.float32)
            neighbors[agent_idx, :, 8:11] = type_features[agent_idx]
            neighbors[agent_idx, interp_valid <= 0.0] = 0.0

        return torch.from_numpy(neighbors)

    def _build_neighbors_future_with_mask(self, frame_list, selected_neighbor_tokens):
        current_pose = self._frame_pose(frame_list[self.current_index])
        future_frames = frame_list[self.current_index + 1 :]
        coarse_times, target_times = self._future_time_grid(frame_list)
        neighbors_future = np.zeros((self.predicted_neighbor_num, self.future_len, 4), dtype=np.float32)
        neighbor_future_mask = np.ones((self.predicted_neighbor_num, self.future_len), dtype=bool)
        token_to_index = {
            track_token: idx for idx, track_token in enumerate(selected_neighbor_tokens[: self.predicted_neighbor_num])
        }
        coarse_future = np.zeros((self.predicted_neighbor_num, len(coarse_times), 3), dtype=np.float64)
        coarse_valid = np.zeros((self.predicted_neighbor_num, len(coarse_times)), dtype=bool)

        current_frame = frame_list[self.current_index]
        for box, name, velocity, track_token in self._iter_annotations(current_frame):
            if track_token not in token_to_index or name not in self.dynamic_tracked_object_types:
                continue
            agent_idx = token_to_index[track_token]
            xy_local, heading_local, _ = self._convert_box_to_current_frame(box, velocity, current_pose, current_pose)
            coarse_future[agent_idx, 0, 0:2] = xy_local
            coarse_future[agent_idx, 0, 2] = heading_local
            coarse_valid[agent_idx, 0] = True

        for t, frame_dict in enumerate(future_frames, start=1):
            frame_pose = self._frame_pose(frame_dict)
            for box, name, velocity, track_token in self._iter_annotations(frame_dict):
                if track_token not in token_to_index or name not in self.dynamic_tracked_object_types:
                    continue
                agent_idx = token_to_index[track_token]
                xy_local, heading_local, _ = self._convert_box_to_current_frame(box, velocity, frame_pose, current_pose)
                coarse_future[agent_idx, t, 0:2] = xy_local
                coarse_future[agent_idx, t, 2] = heading_local
                coarse_valid[agent_idx, t] = True

        for agent_idx in range(self.predicted_neighbor_num):
            valid_indices = np.where(coarse_valid[agent_idx])[0]
            if len(valid_indices) < 2:
                continue

            sample_times = coarse_times[valid_indices]
            target_valid = (target_times >= sample_times[0]) & (target_times <= sample_times[-1])
            if not np.any(target_valid):
                continue

            interp_xy = self._interp_feature(sample_times, coarse_future[agent_idx, valid_indices, 0:2], target_times[target_valid])
            interp_heading = self._interp_heading(sample_times, coarse_future[agent_idx, valid_indices, 2], target_times[target_valid])
            neighbors_future[agent_idx, target_valid, 0:2] = interp_xy.astype(np.float32)
            neighbors_future[agent_idx, target_valid, 2] = np.cos(interp_heading).astype(np.float32)
            neighbors_future[agent_idx, target_valid, 3] = np.sin(interp_heading).astype(np.float32)
            neighbor_future_mask[agent_idx, target_valid] = False

        return torch.from_numpy(neighbors_future), torch.from_numpy(neighbor_future_mask)

    def _build_neighbors_future(self, frame_list, selected_neighbor_tokens) -> torch.Tensor:
        neighbors_future, _ = self._build_neighbors_future_with_mask(frame_list, selected_neighbor_tokens)
        return neighbors_future

    def _build_neighbor_future_mask(self, frame_list, selected_neighbor_tokens) -> torch.Tensor:
        _, neighbor_future_mask = self._build_neighbors_future_with_mask(frame_list, selected_neighbor_tokens)
        return neighbor_future_mask

    def _build_ego_future(self, frame_list) -> torch.Tensor:
        current_pose = self._frame_pose(frame_list[self.current_index])
        future_frames = frame_list[self.current_index + 1 :]
        if len(future_frames) == 0:
            return torch.zeros((self.future_len, 4), dtype=torch.float32)

        coarse_times, target_times = self._future_time_grid(frame_list)
        coarse_poses = np.stack([current_pose] + [self._frame_pose(frame_dict) for frame_dict in future_frames], axis=0)
        coarse_relative = self._to_relative_poses(current_pose, coarse_poses)

        interp_xy = self._interp_feature(coarse_times, coarse_relative[:, :2], target_times)
        interp_heading = self._interp_heading(coarse_times, coarse_relative[:, 2], target_times)

        ego_future = np.zeros((self.future_len, 4), dtype=np.float32)
        ego_future[:, 0:2] = interp_xy.astype(np.float32)
        ego_future[:, 2] = np.cos(interp_heading).astype(np.float32)
        ego_future[:, 3] = np.sin(interp_heading).astype(np.float32)
        return torch.from_numpy(ego_future)

    def _build_static_objects(self, frame_list) -> torch.Tensor:
        current_frame = frame_list[self.current_index]
        static_objects = []
        for box, name, _, _ in self._iter_annotations(current_frame):
            tracked_type = self.static_tracked_object_types.get(name)
            if tracked_type is None:
                continue

            static_feature = np.zeros((10,), dtype=np.float32)
            heading = float(box[BoundingBoxIndex.HEADING])
            static_feature[0] = float(box[BoundingBoxIndex.X])
            static_feature[1] = float(box[BoundingBoxIndex.Y])
            static_feature[2] = np.cos(heading)
            static_feature[3] = np.sin(heading)
            static_feature[4] = float(box[BoundingBoxIndex.WIDTH])
            static_feature[5] = float(box[BoundingBoxIndex.LENGTH])
            static_feature[6:10] = self._static_type_one_hot(tracked_type)
            static_objects.append(static_feature)

        static_tensor = np.zeros((self.static_objects_num, 10), dtype=np.float32)
        if static_objects:
            static_objects = np.stack(static_objects, axis=0)
            distances = np.linalg.norm(static_objects[:, :2], axis=-1)
            keep = np.argsort(distances)[: self.static_objects_num]
            static_tensor[: len(keep)] = static_objects[keep]

        return torch.from_numpy(static_tensor)

    def _sample_path(self, discrete_path, num_points: int) -> np.ndarray:
        points = np.array([[state.x, state.y] for state in discrete_path], dtype=np.float64)
        if len(points) == 0:
            return np.zeros((num_points, 2), dtype=np.float64)
        if len(points) == 1:
            return np.repeat(points, num_points, axis=0)

        line = LineString(points)
        distances = np.linspace(0.0, line.length, num_points, dtype=np.float64)
        return np.array([line.interpolate(distance).coords[0] for distance in distances], dtype=np.float64)

    def _global_points_to_local(self, origin_pose: np.ndarray, points: np.ndarray) -> np.ndarray:
        shifted = points - origin_pose[None, :2]
        return shifted @ self._rotation_matrix(-origin_pose[2]).T

    def _traffic_light_one_hot(self, map_object, layer, traffic_lights) -> np.ndarray:
        if layer != SemanticMapLayer.LANE_CONNECTOR:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        traffic_light_lookup = {str(lane_connector_id): bool(is_red) for lane_connector_id, is_red in traffic_lights}
        state = traffic_light_lookup.get(str(map_object.id))
        if state is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if state:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _encode_lane(self, map_object, layer, origin_pose: np.ndarray, traffic_lights) -> np.ndarray:
        center = self._sample_path(map_object.baseline_path.discrete_path, self.lane_len)
        left = self._sample_path(map_object.left_boundary.discrete_path, self.lane_len)
        right = self._sample_path(map_object.right_boundary.discrete_path, self.lane_len)

        center = self._global_points_to_local(origin_pose, center)
        left = self._global_points_to_local(origin_pose, left)
        right = self._global_points_to_local(origin_pose, right)

        if np.linalg.norm(left[-1] - center[0]) < np.linalg.norm(left[0] - center[0]):
            left = left[::-1]
        if np.linalg.norm(right[-1] - center[0]) < np.linalg.norm(right[0] - center[0]):
            right = right[::-1]

        direction = np.zeros_like(center)
        direction[:-1] = center[1:] - center[:-1]

        lane_feature = np.zeros((self.lane_len, 12), dtype=np.float32)
        lane_feature[:, 0:2] = center.astype(np.float32)
        lane_feature[:, 2:4] = direction.astype(np.float32)
        lane_feature[:, 4:6] = (left - center).astype(np.float32)
        lane_feature[:, 6:8] = (right - center).astype(np.float32)
        lane_feature[:, 8:12] = self._traffic_light_one_hot(map_object, layer, traffic_lights)
        return lane_feature

    def _build_lane_features(self, frame_list):
        current_frame = frame_list[self.current_index]
        current_pose = self._frame_pose(current_frame)
        map_api = self._map_api(current_frame["map_location"])
        route_roadblock_ids = list(dict.fromkeys(str(r) for r in current_frame["roadblock_ids"]))

        proximal = map_api.get_proximal_map_objects(
            point=StateSE2(*current_pose).point,
            radius=self.map_radius,
            layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        )

        lane_records = []
        for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
            for map_object in proximal[layer]:
                lane_feature = self._encode_lane(map_object, layer, current_pose, current_frame["traffic_lights"])
                lane_distance = float(np.linalg.norm(lane_feature[:, :2], axis=-1).min())
                speed_limit = float(map_object.speed_limit_mps) if map_object.speed_limit_mps is not None else 0.0
                has_speed_limit = map_object.speed_limit_mps is not None
                roadblock_id = str(map_object.get_roadblock_id())
                lane_records.append((lane_distance, lane_feature, speed_limit, has_speed_limit, roadblock_id))

        lane_records.sort(key=lambda item: item[0])
        lane_records = lane_records[: self.lane_num]

        extracted_roadblocks = {record[4] for record in lane_records}
        pruned_route_roadblocks: list[str] = []
        route_started = False
        for roadblock_id in route_roadblock_ids:
            if roadblock_id in extracted_roadblocks:
                pruned_route_roadblocks.append(roadblock_id)
                route_started = True
            elif route_started:
                break
        pruned_route_roadblocks_set = set(pruned_route_roadblocks)

        lanes = np.zeros((self.lane_num, self.lane_len, 12), dtype=np.float32)
        lanes_speed_limit = np.zeros((self.lane_num, 1), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((self.lane_num, 1), dtype=bool)
        route_lanes = np.zeros((self.route_num, self.lane_len, 12), dtype=np.float32)
        route_lanes_speed_limit = np.zeros((self.route_num, 1), dtype=np.float32)
        route_lanes_has_speed_limit = np.zeros((self.route_num, 1), dtype=bool)

        route_idx = 0
        for idx, (_, lane_feature, speed_limit, has_speed_limit, roadblock_id) in enumerate(lane_records):
            lanes[idx] = lane_feature
            lanes_speed_limit[idx, 0] = speed_limit
            lanes_has_speed_limit[idx, 0] = has_speed_limit
            if roadblock_id in pruned_route_roadblocks_set and route_idx < self.route_num:
                route_lanes[route_idx] = lane_feature
                route_lanes_speed_limit[route_idx, 0] = speed_limit
                route_lanes_has_speed_limit[route_idx, 0] = has_speed_limit
                route_idx += 1

        return {
            "lanes": torch.from_numpy(lanes),
            "lanes_speed_limit": torch.from_numpy(lanes_speed_limit),
            "lanes_has_speed_limit": torch.from_numpy(lanes_has_speed_limit),
            "route_lanes": torch.from_numpy(route_lanes),
            "route_lanes_speed_limit": torch.from_numpy(route_lanes_speed_limit),
            "route_lanes_has_speed_limit": torch.from_numpy(route_lanes_has_speed_limit),
        }

    def _build_diffusion_planner_inputs(self, frame_list, selected_neighbor_tokens):
        inputs = {
            "ego_current_state": self._build_ego_current_state(frame_list),
            "neighbor_agents_past": self._build_neighbor_agents_past(frame_list, selected_neighbor_tokens),
            "static_objects": self._build_static_objects(frame_list),
        }
        inputs.update(self._build_lane_features(frame_list))
        return inputs

    def _geometry_local_coords(self, geometry, origin: StateSE2):
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        return affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

    def _coords_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        pixel_center = np.array([[0.0, self.bev_pixel_width / 2.0]])
        coords_idx = (coords / self.bev_pixel_size) + pixel_center
        return coords_idx.astype(np.int32)

    def _compute_map_polygon_mask(self, map_api, ego_pose: StateSE2, layers) -> np.ndarray:
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self.bev_radius, layers=layers)
        mask = np.zeros((self.bev_pixel_width, self.bev_pixel_height), dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(mask, [exterior], color=255)
        return np.rot90(mask)[::-1] > 0

    def _compute_map_linestring_mask(self, map_api, ego_pose: StateSE2, layers) -> np.ndarray:
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self.bev_radius, layers=layers)
        mask = np.zeros((self.bev_pixel_width, self.bev_pixel_height), dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(mask, [points], isClosed=False, color=255, thickness=2)
        return np.rot90(mask)[::-1] > 0

    def _compute_box_mask(self, frame_dict, tracked_types) -> np.ndarray:
        mask = np.zeros((self.bev_pixel_width, self.bev_pixel_height), dtype=np.uint8)
        boxes = np.asarray(frame_dict["anns"]["gt_boxes"], dtype=np.float32)
        names = np.asarray(frame_dict["anns"]["gt_names"])

        for name, box in zip(names, boxes):
            tracked_type = self.tracked_object_types.get(str(name))
            if tracked_type not in tracked_types:
                continue

            x = float(box[BoundingBoxIndex.X])
            y = float(box[BoundingBoxIndex.Y])
            heading = float(box[BoundingBoxIndex.HEADING])
            length = float(box[BoundingBoxIndex.LENGTH])
            width = float(box[BoundingBoxIndex.WIDTH])
            height = float(box[BoundingBoxIndex.HEIGHT])

            oriented_box = OrientedBox(StateSE2(x, y, heading), length, width, height)
            exterior = np.array(oriented_box.geometry.exterior.coords).reshape((-1, 1, 2))
            exterior = self._coords_to_pixel(exterior)
            cv2.fillPoly(mask, [exterior], color=255)

        return np.rot90(mask)[::-1] > 0

    def _build_bev_semantic_map(self, frame_dict) -> torch.Tensor:
        bev_map = np.zeros((self.bev_pixel_height, self.bev_pixel_width), dtype=np.int64)
        ego_pose_arr = self._frame_pose(frame_dict)
        ego_pose = StateSE2(*ego_pose_arr)
        map_api = self._map_api(frame_dict["map_location"])

        for label, (entity_type, layers) in self.bev_semantic_classes.items():
            if entity_type == "polygon":
                mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                mask = self._compute_box_mask(frame_dict, layers)
            bev_map[mask] = label

        return torch.from_numpy(bev_map)


class _NuplanMapStore:
    def __init__(self, maps_root: Path):
        self.maps_root = Path(maps_root)
        self._cache: dict[str, dict[str, Any]] = {}

    def _map_gpkg_path(self, map_name: str) -> Path:
        map_dir = self.maps_root / map_name
        if not map_dir.exists():
            raise FileNotFoundError(f"Map directory not found for {map_name}: {map_dir}")

        versions = sorted(path for path in map_dir.iterdir() if path.is_dir())
        if not versions:
            raise FileNotFoundError(f"No map version folder found under {map_dir}")

        gpkg_path = versions[0] / "map.gpkg"
        if not gpkg_path.exists():
            raise FileNotFoundError(f"Map geopackage not found: {gpkg_path}")
        return gpkg_path

    def _read_layer(self, gpkg_path: Path, layer: str, crs) -> Any:
        gdf = pyogrio.read_dataframe(gpkg_path, layer=layer, fid_as_index=True)
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        return gdf

    def _load_map(self, map_name: str) -> dict[str, Any]:
        if map_name in self._cache:
            return self._cache[map_name]

        gpkg_path = self._map_gpkg_path(map_name)
        baseline_paths = pyogrio.read_dataframe(gpkg_path, layer="baseline_paths", fid_as_index=True)
        projected_crs = baseline_paths.estimate_utm_crs()
        baseline_paths = baseline_paths.to_crs(projected_crs)
        boundaries = self._read_layer(gpkg_path, "boundaries", projected_crs)
        lanes = self._read_layer(gpkg_path, "lanes_polygons", projected_crs)
        lane_connectors = self._read_layer(gpkg_path, "lane_connectors", projected_crs)
        lane_group_connectors = self._read_layer(gpkg_path, "lane_group_connectors", projected_crs)

        lane_baselines = baseline_paths[baseline_paths["lane_fid"].notna()].copy()
        lane_baselines["lane_fid"] = lane_baselines["lane_fid"].astype(np.int64)
        lane_baselines = lane_baselines.drop_duplicates(subset=["lane_fid"]).set_index("lane_fid", drop=False)

        connector_baselines = baseline_paths[baseline_paths["lane_connector_fid"].notna()].copy()
        connector_baselines["lane_connector_fid"] = connector_baselines["lane_connector_fid"].astype(np.int64)
        connector_baselines = connector_baselines.drop_duplicates(subset=["lane_connector_fid"]).set_index(
            "lane_connector_fid", drop=False
        )

        map_data = {
            "boundaries": boundaries,
            "lanes": lanes,
            "lane_connectors": lane_connectors,
            "lane_group_connectors": lane_group_connectors,
            "lane_baselines": lane_baselines,
            "connector_baselines": connector_baselines,
        }
        self._cache[map_name] = map_data
        return map_data

    def query_lane_records(self, map_name: str, origin_xy: np.ndarray, radius: float) -> list[dict[str, Any]]:
        map_data = self._load_map(map_name)
        query_geom = Point(float(origin_xy[0]), float(origin_xy[1]))
        minx, miny, maxx, maxy = query_geom.buffer(radius).bounds

        lane_candidates = map_data["lane_baselines"].cx[minx:maxx, miny:maxy]
        connector_candidates = map_data["connector_baselines"].cx[minx:maxx, miny:maxy]

        records: list[dict[str, Any]] = []
        for lane_fid, baseline in lane_candidates.iterrows():
            if lane_fid not in map_data["lanes"].index:
                continue
            lane = map_data["lanes"].loc[lane_fid]
            left_fid = int(lane["left_boundary_fid"])
            right_fid = int(lane["right_boundary_fid"])
            if left_fid not in map_data["boundaries"].index or right_fid not in map_data["boundaries"].index:
                continue

            roadblock_id = str(int(lane["lane_group_fid"]))
            speed_limit = float(lane["speed_limit_mps"]) if not np.isnan(lane["speed_limit_mps"]) else 0.0
            records.append(
                {
                    "lane_id": int(lane_fid),
                    "is_connector": False,
                    "roadblock_id": roadblock_id,
                    "baseline": baseline.geometry,
                    "left_boundary": map_data["boundaries"].loc[left_fid].geometry,
                    "right_boundary": map_data["boundaries"].loc[right_fid].geometry,
                    "speed_limit": speed_limit,
                    "has_speed_limit": not np.isnan(lane["speed_limit_mps"]),
                    "distance": float(baseline.geometry.distance(query_geom)),
                }
            )

        for connector_fid, baseline in connector_candidates.iterrows():
            if connector_fid not in map_data["lane_connectors"].index:
                continue

            lane_connector = map_data["lane_connectors"].loc[connector_fid]
            lane_group_connector_fid = int(lane_connector["lane_group_connector_fid"])
            if lane_group_connector_fid not in map_data["lane_group_connectors"].index:
                continue

            lane_group_connector = map_data["lane_group_connectors"].loc[lane_group_connector_fid]
            left_fid = int(lane_group_connector["left_boundary_fid"])
            right_fid = int(lane_group_connector["right_boundary_fid"])
            if left_fid not in map_data["boundaries"].index or right_fid not in map_data["boundaries"].index:
                continue

            speed_limit_value = lane_connector["speed_limit_mps"]
            roadblock_id = str(lane_group_connector_fid)
            records.append(
                {
                    "lane_id": int(connector_fid),
                    "is_connector": True,
                    "roadblock_id": roadblock_id,
                    "baseline": baseline.geometry,
                    "left_boundary": map_data["boundaries"].loc[left_fid].geometry,
                    "right_boundary": map_data["boundaries"].loc[right_fid].geometry,
                    "speed_limit": float(speed_limit_value) if not np.isnan(speed_limit_value) else 0.0,
                    "has_speed_limit": not np.isnan(speed_limit_value),
                    "distance": float(baseline.geometry.distance(query_geom)),
                }
            )

        return records


class DiffusionPlannerNuplanDataset(DiffusionPlannerDataset):
    def __init__(
        self,
        data_path,
        cfg: dict,
        maps_root=None,
        max_len=None,
        random_sample=False,
        log_names=None,
        tokens=None,
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=None,
        max_ped_bike=5,
    ):
        if cfg is None:
            raise ValueError("`cfg` must be provided for DiffusionPlannerNuplanDataset.")

        required_cfg_keys = [
            "agent_num",
            "predicted_neighbor_num",
            "static_objects_num",
            "lane_num",
            "route_num",
            "lane_len",
            "time_len",
            "future_len",
        ]
        missing_cfg_keys = [key for key in required_cfg_keys if key not in cfg]
        if missing_cfg_keys:
            raise KeyError(f"Missing required cfg keys for DiffusionPlannerNuplanDataset: {missing_cfg_keys}")

        self.data_path = Path(data_path)
        self.maps_root = Path(maps_root) if maps_root is not None else Path(os.environ["NUPLAN_MAPS_ROOT"])
        self.map_store = _NuplanMapStore(self.maps_root)

        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        self.frame_interval = frame_interval or (num_history_frames + num_future_frames)
        self.current_index = num_history_frames - 1

        self.scene_frames_dicts = self._load_scene_frames(tokens=tokens, log_names=log_names)
        all_tokens = list(self.scene_frames_dicts.keys())
        if random_sample:
            max_len = len(all_tokens) if max_len is None else min(max_len, len(all_tokens))
            self.tokens = np.random.choice(all_tokens, size=max_len, replace=False).tolist()
        else:
            max_len = len(all_tokens) if max_len is None else min(max_len, len(all_tokens))
            self.tokens = all_tokens[:max_len]
        self.max_len = len(self.tokens)

        self.agent_num = cfg["agent_num"]
        self.predicted_neighbor_num = cfg["predicted_neighbor_num"]
        self.static_objects_num = cfg["static_objects_num"]
        self.lane_num = cfg["lane_num"]
        self.route_num = cfg["route_num"]
        self.lane_len = cfg["lane_len"]
        self.time_len = cfg["time_len"]
        self.future_len = cfg["future_len"]
        self.max_ped_bike = max_ped_bike
        self.map_radius = 100.0
        self.model_interval = 0.1

        self.dynamic_tracked_object_types = {
            "vehicle": TrackedObjectType.VEHICLE,
            "pedestrian": TrackedObjectType.PEDESTRIAN,
            "bicycle": TrackedObjectType.BICYCLE,
        }
        self.static_tracked_object_types = {
            "traffic_cone": TrackedObjectType.TRAFFIC_CONE,
            "barrier": TrackedObjectType.BARRIER,
            "czone_sign": TrackedObjectType.CZONE_SIGN,
            "generic_object": TrackedObjectType.GENERIC_OBJECT,
        }

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        token = self.tokens[idx]
        frame_list = self.scene_frames_dicts[token]
        selected_neighbor_tokens = self._select_neighbor_tokens(frame_list)

        features = self._build_diffusion_planner_inputs(frame_list, selected_neighbor_tokens)
        targets = {
            "ego_future_gt": self._build_ego_future(frame_list),
            "neighbors_future_gt": self._build_neighbors_future(frame_list, selected_neighbor_tokens),
            "neighbor_future_mask": self._build_neighbor_future_mask(frame_list, selected_neighbor_tokens),
            "trajectory": self._build_future_trajectory(frame_list),
        }
        return token, features, targets

    def _load_scene_frames(self, tokens=None, log_names=None) -> dict[str, list[dict[str, Any]]]:
        windows: dict[str, list[dict[str, Any]]] = {}
        log_files = sorted(self.data_path.glob("*.pkl"))
        if log_names is not None:
            allowed = {name.replace(".pkl", "") for name in log_names}
            log_files = [log_file for log_file in log_files if log_file.stem in allowed]

        token_filter = set(tokens) if tokens is not None else None
        num_frames = self.num_history_frames + self.num_future_frames
        for log_file in log_files:
            with open(log_file, "rb") as fp:
                scene_dict_list = pickle.load(fp)

            for start_idx in range(0, len(scene_dict_list), self.frame_interval):
                frame_list = scene_dict_list[start_idx : start_idx + num_frames]
                if len(frame_list) < num_frames:
                    continue
                if len(frame_list[self.current_index]["roadblock_ids"]) == 0:
                    continue

                token = frame_list[self.current_index]["token"]
                if token_filter is not None and token not in token_filter:
                    continue
                windows[token] = frame_list

        return windows

    def _sample_geometry(self, geometry, num_points: int) -> np.ndarray:
        if geometry.is_empty:
            return np.zeros((num_points, 2), dtype=np.float64)

        if geometry.geom_type == "LineString":
            line = geometry
        else:
            line = LineString(geometry.exterior.coords)

        if line.length == 0:
            point = np.asarray(line.coords[0], dtype=np.float64)
            return np.repeat(point[None], num_points, axis=0)

        distances = np.linspace(0.0, line.length, num_points, dtype=np.float64)
        return np.asarray([line.interpolate(distance).coords[0] for distance in distances], dtype=np.float64)

    def _encode_map_record(self, map_record: dict[str, Any], origin_pose: np.ndarray, traffic_lights) -> np.ndarray:
        center = self._sample_geometry(map_record["baseline"], self.lane_len)
        left = self._sample_geometry(map_record["left_boundary"], self.lane_len)
        right = self._sample_geometry(map_record["right_boundary"], self.lane_len)

        center = self._global_points_to_local(origin_pose, center)
        left = self._global_points_to_local(origin_pose, left)
        right = self._global_points_to_local(origin_pose, right)

        if np.linalg.norm(left[-1] - center[0]) < np.linalg.norm(left[0] - center[0]):
            left = left[::-1]
        if np.linalg.norm(right[-1] - center[0]) < np.linalg.norm(right[0] - center[0]):
            right = right[::-1]

        direction = np.zeros_like(center)
        direction[:-1] = center[1:] - center[:-1]

        lane_feature = np.zeros((self.lane_len, 12), dtype=np.float32)
        lane_feature[:, 0:2] = center.astype(np.float32)
        lane_feature[:, 2:4] = direction.astype(np.float32)
        lane_feature[:, 4:6] = (left - center).astype(np.float32)
        lane_feature[:, 6:8] = (right - center).astype(np.float32)

        if map_record["is_connector"]:
            traffic_light_lookup = {str(connector_id): bool(is_red) for connector_id, is_red in traffic_lights}
            state = traffic_light_lookup.get(str(map_record["lane_id"]))
            if state is None:
                lane_feature[:, 8:12] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            elif state:
                lane_feature[:, 8:12] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            else:
                lane_feature[:, 8:12] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            lane_feature[:, 8:12] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        return lane_feature

    def _build_lane_features(self, frame_list):
        current_frame = frame_list[self.current_index]
        current_pose = self._frame_pose(current_frame)
        route_roadblock_ids = list(dict.fromkeys(str(roadblock_id) for roadblock_id in current_frame["roadblock_ids"]))

        lane_records = self.map_store.query_lane_records(
            current_frame["map_location"],
            origin_xy=current_pose[:2],
            radius=self.map_radius,
        )
        lane_records = sorted(lane_records, key=lambda r: r["distance"])[: self.lane_num]

        encoded_lanes = []
        for map_record in lane_records:
            lane_feature = self._encode_map_record(map_record, current_pose, current_frame["traffic_lights"])
            encoded_lanes.append(
                (
                    lane_feature,
                    float(map_record["speed_limit"]),
                    bool(map_record["has_speed_limit"]),
                    map_record["roadblock_id"],
                )
            )

        extracted_roadblocks = {record[3] for record in encoded_lanes}
        pruned_route_roadblocks: list[str] = []
        route_started = False
        for roadblock_id in route_roadblock_ids:
            if roadblock_id in extracted_roadblocks:
                pruned_route_roadblocks.append(roadblock_id)
                route_started = True
            elif route_started:
                break
        pruned_route_roadblocks_set = set(pruned_route_roadblocks)

        lanes = np.zeros((self.lane_num, self.lane_len, 12), dtype=np.float32)
        lanes_speed_limit = np.zeros((self.lane_num, 1), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((self.lane_num, 1), dtype=bool)
        route_lanes = np.zeros((self.route_num, self.lane_len, 12), dtype=np.float32)
        route_lanes_speed_limit = np.zeros((self.route_num, 1), dtype=np.float32)
        route_lanes_has_speed_limit = np.zeros((self.route_num, 1), dtype=bool)

        route_idx = 0
        for idx, (lane_feature, speed_limit, has_speed_limit, roadblock_id) in enumerate(encoded_lanes):
            lanes[idx] = lane_feature
            lanes_speed_limit[idx, 0] = speed_limit
            lanes_has_speed_limit[idx, 0] = has_speed_limit
            if roadblock_id in pruned_route_roadblocks_set and route_idx < self.route_num:
                route_lanes[route_idx] = lane_feature
                route_lanes_speed_limit[route_idx, 0] = speed_limit
                route_lanes_has_speed_limit[route_idx, 0] = has_speed_limit
                route_idx += 1

        return {
            "lanes": torch.from_numpy(lanes),
            "lanes_speed_limit": torch.from_numpy(lanes_speed_limit),
            "lanes_has_speed_limit": torch.from_numpy(lanes_has_speed_limit),
            "route_lanes": torch.from_numpy(route_lanes),
            "route_lanes_speed_limit": torch.from_numpy(route_lanes_speed_limit),
            "route_lanes_has_speed_limit": torch.from_numpy(route_lanes_has_speed_limit),
        }
