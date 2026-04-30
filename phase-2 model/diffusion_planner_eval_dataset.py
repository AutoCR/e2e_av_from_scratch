"""Eval-only dataset that handles synthetic (stage-two) tokens.

The training-time DiffusionPlannerDataset reads raw log frame dicts from
SceneLoader.scene_frames_dicts, which only indexes original-log tokens.
Synthetic tokens live in SceneLoader.synthetic_scenes and are exposed as
Scene.frames (list of Frame dataclasses) via get_scene_from_token.

This subclass adds `get_frame_list(token)` which returns a list of raw-log-
schema dicts for either token type, so `_build_diffusion_planner_inputs`
can consume both without changes.
"""

from __future__ import annotations

from typing import List

import numpy as np
from pyquaternion import Quaternion

from diffusion_planner_dataset import DiffusionPlannerDataset


class DiffusionPlannerEvalDataset(DiffusionPlannerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._synthetic_frame_cache: dict[str, List[dict]] = {}

    def get_frame_list(self, token: str) -> List[dict]:
        scene_frames_dicts = self.scene_loader.scene_frames_dicts
        if token in scene_frames_dicts:
            return scene_frames_dicts[token]

        if token in self._synthetic_frame_cache:
            return self._synthetic_frame_cache[token]

        scene = self.scene_loader.get_scene_from_token(token)
        frame_list = [
            self._frame_to_log_dict(frame, scene.scene_metadata) for frame in scene.frames
        ]
        self._synthetic_frame_cache[token] = frame_list
        return frame_list

    @staticmethod
    def _frame_to_log_dict(frame, scene_metadata) -> dict:
        ego_status = frame.ego_status
        assert ego_status.in_global_frame, (
            "Expected synthetic Frame.ego_status.in_global_frame=True; "
            "conversion assumes ego_pose is global."
        )

        ego_pose = np.asarray(ego_status.ego_pose, dtype=np.float64)
        yaw = float(ego_pose[2])
        quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw).elements  # [w, x, y, z]

        ego_velocity = np.asarray(ego_status.ego_velocity, dtype=np.float64)
        ego_acceleration = np.asarray(ego_status.ego_acceleration, dtype=np.float64)
        ego_dynamic_state = np.concatenate([ego_velocity[:2], ego_acceleration[:2]])

        anns = frame.annotations
        return {
            "token": frame.token,
            "timestamp": frame.timestamp,
            "roadblock_ids": list(frame.roadblock_ids),
            "traffic_lights": list(frame.traffic_lights),
            "map_location": scene_metadata.map_name,
            "log_name": scene_metadata.log_name,
            "ego2global_translation": [float(ego_pose[0]), float(ego_pose[1]), 0.0],
            "ego2global_rotation": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "ego_dynamic_state": ego_dynamic_state,
            "anns": {
                "gt_boxes": np.asarray(anns.boxes, dtype=np.float32),
                "gt_names": list(anns.names),
                "gt_velocity_3d": np.asarray(anns.velocity_3d, dtype=np.float32),
                "track_tokens": list(anns.track_tokens),
            },
        }
