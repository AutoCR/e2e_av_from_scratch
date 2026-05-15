import os
import pickle
from pathlib import Path
from typing import Any, List

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Point, LineString
import pyogrio


def get_ego_pose(frame: dict) -> np.ndarray:
    """Return [x, y, yaw] of the ego vehicle in global frame."""
    translation = np.asarray(frame["ego2global_translation"][:2], dtype=np.float64)
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def rotation_matrix(angle: float) -> np.ndarray:
    """2D rotation matrix for given angle."""
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]], dtype=np.float64)


def local_to_global_xy(pose: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
    """Transform 2D point(s) from ego-local frame to global frame.
    pose: [x, y, yaw]; local_xy: (..., 2)"""
    return pose[:2] + (rotation_matrix(pose[2]) @ local_xy.T).T


def global_to_local_xy(pose: np.ndarray, global_xy: np.ndarray) -> np.ndarray:
    """Transform 2D point(s) from global frame to ego-local frame.
    pose: [x, y, yaw]; global_xy: (..., 2)"""
    return (rotation_matrix(-pose[2]) @ (global_xy - pose[:2]).T).T


def load_sequences(data_path: str, split: str) -> List[List[dict]]:
    """Load all .pkl files from navsim_logs/{split}/.
    Returns list of frame lists (each inner list is one log sequence)."""
    log_dir = Path(data_path) / "navsim_logs" / split
    sequences = []
    from tqdm import tqdm
    for pkl_path in tqdm(sorted(log_dir.glob("*.pkl")), desc="loading sequences"):
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        sequences.append(frames)
    return sequences


class NuplanMapStore:
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
