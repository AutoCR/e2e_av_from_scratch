import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString, Polygon

from navsim.common.dataloader import SceneLoader
from navsim.common.enums import BoundingBoxIndex
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        max_len=None,
        random_sample=False,
        trajectory_num_poses=None,
    ):
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

        self.camera_size = (1024, 256)
        self.lidar_range = (-32.0, 32.0, -32.0, 32.0)
        self.pixels_per_meter = 4.0
        self.hist_max_per_pixel = 5
        self.max_height_lidar = 100.0
        self.lidar_split_height = 0.2
        self.num_bounding_boxes = 30

        self.bev_pixel_width = 256
        self.bev_pixel_height = 128
        self.bev_pixel_size = 0.25
        self.num_bev_classes = 7
        self.bev_radius = 32.0

        self.tracked_object_types = {
            "vehicle": TrackedObjectType.VEHICLE,
            "pedestrian": TrackedObjectType.PEDESTRIAN,
            "bicycle": TrackedObjectType.BICYCLE,
            "traffic_cone": TrackedObjectType.TRAFFIC_CONE,
            "barrier": TrackedObjectType.BARRIER,
            "czone_sign": TrackedObjectType.CZONE_SIGN,
            "generic_object": TrackedObjectType.GENERIC_OBJECT,
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
        current_frame = frame_list[self.current_index]

        features = {
            "camera_feature": self._build_camera_feature(current_frame),
            "lidar_feature": self._build_lidar_feature(current_frame),
            "status_feature": self._build_status_feature(current_frame),
        }
        targets = {
            "trajectory": self._build_future_trajectory(frame_list),
            "agent_states": self._build_agent_states(current_frame),
            "agent_labels": self._build_agent_labels(current_frame),
            "bev_semantic_map": self._build_bev_semantic_map(current_frame),
        }
        return features, targets

    def _sensor_root(self) -> Path:
        return Path(self.scene_loader._original_sensor_path)

    def _maps_root(self) -> str:
        maps_root = os.getenv("NUPLAN_MAPS_ROOT")
        if maps_root is None:
            raise RuntimeError("NUPLAN_MAPS_ROOT is not set.")
        return maps_root

    def _map_api(self, map_name: str):
        return get_maps_api(self._maps_root(), "nuplan-maps-v1.0", map_name)

    def _load_image(self, relative_path: str) -> np.ndarray:
        image_path = self._sensor_root() / relative_path
        return np.array(Image.open(image_path))

    def _build_camera_feature(self, frame_dict) -> torch.Tensor:
        cameras = frame_dict["cams"]

        left = self._load_image(cameras["CAM_L0"]["data_path"])[28:-28, 416:-416]
        front = self._load_image(cameras["CAM_F0"]["data_path"])[28:-28]
        right = self._load_image(cameras["CAM_R0"]["data_path"])[28:-28, 416:-416]

        stitched = np.concatenate([left, front, right], axis=1)
        resized = cv2.resize(stitched, self.camera_size)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).to(torch.float32) / 255.0
        return tensor

    def _load_lidar_points(self, relative_path: str) -> np.ndarray:
        lidar_path = self._sensor_root() / relative_path
        with open(lidar_path, "rb") as fp:
            lidar_pc = LidarPointCloud.from_buffer(BytesIO(fp.read()), "pcd").points
        return lidar_pc[:3].T

    def _splat_points(self, point_cloud: np.ndarray) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.lidar_range
        xbins = np.linspace(min_x, max_x, int((max_x - min_x) * self.pixels_per_meter) + 1)
        ybins = np.linspace(min_y, max_y, int((max_y - min_y) * self.pixels_per_meter) + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist = np.clip(hist, 0, self.hist_max_per_pixel)
        return hist / self.hist_max_per_pixel

    def _build_lidar_feature(self, frame_dict) -> torch.Tensor:
        lidar_pc = self._load_lidar_points(frame_dict["lidar_path"])
        lidar_pc = lidar_pc[lidar_pc[:, 2] < self.max_height_lidar]
        above = lidar_pc[lidar_pc[:, 2] > self.lidar_split_height]
        overhead = self._splat_points(above)
        feature = overhead[None].astype(np.float32)
        return torch.from_numpy(feature)

    def _build_status_feature(self, frame_dict) -> torch.Tensor:
        driving_command = np.asarray(frame_dict["driving_command"], dtype=np.float32)
        ego_dynamic_state = np.asarray(frame_dict["ego_dynamic_state"], dtype=np.float32)
        ego_velocity = ego_dynamic_state[:2]
        ego_acceleration = ego_dynamic_state[2:]
        status = np.concatenate([driving_command, ego_velocity, ego_acceleration], axis=0)
        return torch.from_numpy(status)

    def _frame_pose(self, frame_dict) -> np.ndarray:
        translation = np.asarray(frame_dict["ego2global_translation"][:2], dtype=np.float64)
        yaw = Quaternion(*frame_dict["ego2global_rotation"]).yaw_pitch_roll[0]
        return np.array([translation[0], translation[1], yaw], dtype=np.float64)

    def _to_relative_poses(self, origin_pose: np.ndarray, global_poses: np.ndarray) -> np.ndarray:
        theta = -origin_pose[2]
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float64,
        )
        relative = global_poses - origin_pose[None]
        relative[:, :2] = relative[:, :2] @ rotation.T
        relative[:, 2] = np.arctan2(np.sin(relative[:, 2]), np.cos(relative[:, 2]))
        return relative

    def _build_future_trajectory(self, frame_list) -> torch.Tensor:
        current_pose = self._frame_pose(frame_list[self.current_index])
        future_frames = frame_list[self.current_index + 1 :]

        if self.trajectory_num_poses is not None:
            future_frames = future_frames[: self.trajectory_num_poses]

        future_global_poses = np.stack([self._frame_pose(frame_dict) for frame_dict in future_frames], axis=0)
        future_relative_poses = self._to_relative_poses(current_pose, future_global_poses).astype(np.float32)
        return torch.from_numpy(future_relative_poses)

    def _xy_in_lidar(self, x: float, y: float) -> bool:
        min_x, max_x, min_y, max_y = self.lidar_range
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _build_agent_targets(self, frame_dict):
        boxes = np.asarray(frame_dict["anns"]["gt_boxes"], dtype=np.float32)
        names = np.asarray(frame_dict["anns"]["gt_names"])

        selected = []
        for box, name in zip(boxes, names):
            x = float(box[BoundingBoxIndex.X])
            y = float(box[BoundingBoxIndex.Y])
            if name == "vehicle" and self._xy_in_lidar(x, y):
                selected.append(
                    np.array(
                        [
                            box[BoundingBoxIndex.X],
                            box[BoundingBoxIndex.Y],
                            box[BoundingBoxIndex.HEADING],
                            box[BoundingBoxIndex.LENGTH],
                            box[BoundingBoxIndex.WIDTH],
                        ],
                        dtype=np.float32,
                    )
                )

        if selected:
            selected = np.stack(selected, axis=0)
            distances = np.linalg.norm(selected[:, :2], axis=-1)
            selected = selected[np.argsort(distances)[: self.num_bounding_boxes]]
        else:
            selected = np.zeros((0, 5), dtype=np.float32)

        agent_states = np.zeros((self.num_bounding_boxes, 5), dtype=np.float32)
        agent_labels = np.zeros((self.num_bounding_boxes,), dtype=bool)

        num_valid = len(selected)
        if num_valid > 0:
            agent_states[:num_valid] = selected
            agent_labels[:num_valid] = True

        return torch.from_numpy(agent_states), torch.from_numpy(agent_labels)

    def _build_agent_states(self, frame_dict) -> torch.Tensor:
        agent_states, _ = self._build_agent_targets(frame_dict)
        return agent_states

    def _build_agent_labels(self, frame_dict) -> torch.Tensor:
        _, agent_labels = self._build_agent_targets(frame_dict)
        return agent_labels

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


class DatasetV2(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        max_len=None,
        random_sample=False,
        config: TransfuserConfig = None,
        trajectory_sampling: TrajectorySampling = None,
    ):
        self.scene_loader = scene_loader
        self.config = config or TransfuserConfig()
        self.trajectory_sampling = trajectory_sampling or TrajectorySampling(time_horizon=4, interval_length=0.5)

        if random_sample:
            if max_len is None:
                max_len = len(scene_loader.tokens)
            max_len = min(max_len, len(scene_loader.tokens))
            self.tokens = np.random.choice(scene_loader.tokens, size=max_len, replace=False).tolist()
        else:
            max_len = min(max_len, len(scene_loader.tokens)) if max_len is not None else len(scene_loader.tokens)
            self.tokens = scene_loader.tokens[:max_len]

        self.max_len = len(self.tokens)
        self.feature_builder = TransfuserFeatureBuilder(self.config)
        self.target_builder = TransfuserTargetBuilder(self.trajectory_sampling, self.config)

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        token = self.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        features = self.feature_builder.compute_features(agent_input)
        targets = self.target_builder.compute_targets(scene)
        return features, targets
