from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

from sparsedrive_model.navsim_adapter import (
    DEFAULT_CAMERA_ORDER_8,
    DEFAULT_MAP_VERSION,
    build_navsim_scene_loader,
    normalize_camera_order,
)
from sparsedrive_model.navsim_train.augment import (
    DATA_AUG_CONF,
    BBoxRotation,
    NormalizeMultiviewImage,
    PhotoMetricDistortionMultiViewImage,
    ResizeCropFlipImage,
)
from sparsedrive_model.navsim_train.depth import MultiScaleDepthMapGenerator
from sparsedrive_model.navsim_train.targets import (
    _as_ann_dict,
    _frame_global_to_lidar,
    _transform_boxes,
    build_agent_futures,
    build_ego_futures,
    build_ego_status,
    build_gt_bboxes_3d,
    build_gt_ego_fut_cmd,
    nuplan_to_nuscenes_label,
)

TRAIN_KEYS = (
    "img",
    "timestamp",
    "projection_mat",
    "image_wh",
    "gt_depth",
    "focal",
    "gt_bboxes_3d",
    "gt_labels_3d",
    "gt_map_labels",
    "gt_map_pts",
    "gt_agent_fut_trajs",
    "gt_agent_fut_masks",
    "gt_ego_fut_trajs",
    "gt_ego_fut_masks",
    "gt_ego_fut_cmd",
    "ego_status",
)
TEST_KEYS = ("img", "timestamp", "projection_mat", "image_wh", "ego_status", "gt_ego_fut_cmd")
VARIABLE_KEYS = {
    "gt_bboxes_3d",
    "gt_labels_3d",
    "instance_id",
    "gt_agent_fut_trajs",
    "gt_agent_fut_masks",
    "gt_map_labels",
    "gt_map_pts",
}
FIXED_STACK_KEYS = {
    "img",
    "projection_mat",
    "image_wh",
    "timestamp",
    "focal",
    "gt_ego_fut_trajs",
    "gt_ego_fut_masks",
    "gt_ego_fut_cmd",
    "ego_status",
}


def _timestamp_seconds(timestamp: Any) -> float:
    value = float(timestamp)
    abs_value = abs(value)
    if abs_value > 1.0e17:
        return value / 1.0e9
    if abs_value > 1.0e12:
        return value / 1.0e6
    return value


def _ego_to_global_matrix(frame: Mapping[str, Any]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(*frame["ego2global_rotation"]).rotation_matrix
    mat[:3, 3] = np.asarray(frame["ego2global_translation"], dtype=np.float32)
    return mat


def _camera_projection(camera_info: Mapping[str, Any], intrinsics: np.ndarray) -> np.ndarray:
    sensor2lidar_rotation = np.asarray(camera_info["sensor2lidar_rotation"], dtype=np.float64)
    sensor2lidar_translation = np.asarray(camera_info["sensor2lidar_translation"], dtype=np.float64).reshape(3)
    lidar2sensor_rotation = sensor2lidar_rotation.T
    lidar2sensor_translation = -lidar2sensor_rotation @ sensor2lidar_translation
    lidar2sensor = np.concatenate([lidar2sensor_rotation, lidar2sensor_translation[:, None]], axis=1)
    return (intrinsics @ lidar2sensor).astype(np.float32)


def _load_lidar_points(frame: Mapping[str, Any], sensor_root: Path) -> torch.Tensor:
    lidar_path = frame.get("lidar_path")
    if lidar_path is None:
        return torch.zeros((0, 5), dtype=torch.float32)
    full_path = sensor_root / lidar_path
    try:
        from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

        with open(full_path, "rb") as fp:
            import io

            pc = LidarPointCloud.from_buffer(io.BytesIO(fp.read()), "pcd").points
        arr = np.asarray(pc, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] in {3, 4, 5, 6}:
            arr = arr.T
        if arr.shape[1] < 5:
            arr = np.pad(arr, ((0, 0), (0, 5 - arr.shape[1])), mode="constant")
        return torch.from_numpy(arr[:, :5].astype(np.float32))
    except Exception:
        return torch.zeros((0, 5), dtype=torch.float32)


def _current_track_mapping(frame: Mapping[str, Any], global_to_lidar: np.ndarray, range_threshold: float = 55.0) -> dict[str, int]:
    ann = _as_ann_dict(frame.get("anns", frame.get("annotations")))
    boxes = np.asarray(ann.get("gt_boxes", ann.get("boxes", np.zeros((0, 7)))), dtype=np.float32)
    names = list(ann.get("gt_names", ann.get("names", [])))
    velocity = np.asarray(ann.get("gt_velocity_3d", ann.get("velocity_3d", np.zeros((len(boxes), 3)))), dtype=np.float32)
    track_tokens = list(ann.get("track_tokens", []))
    encoded, _ = _transform_boxes(boxes, velocity, global_to_lidar)
    mapping: dict[str, int] = {}
    row = 0
    for i, token in enumerate(track_tokens):
        if i >= len(names) or nuplan_to_nuscenes_label(names[i]) is None:
            continue
        if float(np.linalg.norm(encoded[i, :2])) > range_threshold:
            continue
        mapping[str(token)] = row
        row += 1
    return mapping


def _future_ego_positions(frame_list: Sequence[Mapping[str, Any]], current_index: int, num_future: int) -> torch.Tensor:
    current = frame_list[current_index]
    global_to_ego = _frame_global_to_lidar(current)
    pts: list[np.ndarray] = []
    for step in range(num_future):
        idx = current_index + step + 1
        if idx >= len(frame_list):
            break
        trans = np.asarray(frame_list[idx]["ego2global_translation"], dtype=np.float32)
        pt = global_to_ego @ np.array([trans[0], trans[1], trans[2], 1.0], dtype=np.float32)
        pts.append(pt[:3])
    if not pts:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(pts).astype(np.float32))


def _empty_map_targets() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((0,), dtype=torch.long), torch.zeros((0, 20, 2), dtype=torch.float32)


def _build_map_targets(map_api: Any, map_name: str | None, frame: Mapping[str, Any], global_to_lidar: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from sparsedrive_model.navsim_train import map_vectorize  # type: ignore
    except ImportError:
        return _empty_map_targets()
    try:
        if hasattr(map_vectorize, "build_map_targets"):
            labels, pts = map_vectorize.build_map_targets(map_api, map_name, frame, global_to_lidar)
        elif hasattr(map_vectorize, "vectorize_map"):
            labels, pts = map_vectorize.vectorize_map(map_api, map_name, frame, global_to_lidar)
        else:
            return _empty_map_targets()
        return torch.as_tensor(labels, dtype=torch.long), torch.as_tensor(pts, dtype=torch.float32)
    except Exception:
        return _empty_map_targets()


class NavSimSparseDriveDataset(Dataset):
    def __init__(
        self,
        split: str,
        openscene_data_root: str | os.PathLike[str],
        nuplan_maps_root: str | os.PathLike[str],
        camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER_8,
        num_history_frames: int = 4,
        num_future_frames: int = 12,
        image_hw: tuple[int, int] = (256, 704),
        test_mode: bool = False,
        max_scenes: int | None = None,
        log_names: Optional[List[str]] = None,
        tokens: Optional[List[str]] = None,
        augment: bool = True,
        with_seq_flag: bool = True,
        sequences_split_num: int | str = 2,
    ) -> None:
        self.split = split
        self.openscene_data_root = Path(openscene_data_root)
        self.nuplan_maps_root = Path(nuplan_maps_root)
        self.camera_order = normalize_camera_order(camera_order)
        self.num_history_frames = int(num_history_frames)
        self.num_future_frames = int(num_future_frames)
        self.image_hw = tuple(int(x) for x in image_hw)
        self.test_mode = bool(test_mode)
        self.log_names = log_names
        self.tokens = tokens
        self.augment_enabled = bool(augment) and not self.test_mode
        self.with_seq_flag = bool(with_seq_flag)
        self.sequences_split_num = sequences_split_num
        self.epoch = 0
        self._aug_cache: dict[tuple[int, int, tuple[int, int]], dict[str, Any]] = {}
        self._map_api_cache: dict[str, Any] = {}

        self.scene_loader = build_navsim_scene_loader(
            split=split,
            camera_order=self.camera_order,
            openscene_data_root=self.openscene_data_root,
            nuplan_maps_root=self.nuplan_maps_root,
            num_history_frames=self.num_history_frames,
            num_future_frames=self.num_future_frames,
            max_scenes=max_scenes,
            log_names=log_names,
            tokens=tokens,
        )
        self.scene_tokens = list(self.scene_loader.tokens)
        self._subseq_for_index, self._subseq_to_indices = self._build_subseq_index()
        self.resize_aug = ResizeCropFlipImage(DATA_AUG_CONF, test_mode=self.test_mode)
        self.photo_aug = PhotoMetricDistortionMultiViewImage()
        self.bbox_rot = BBoxRotation()
        self.normalize = NormalizeMultiviewImage()
        self.depth_generator = MultiScaleDepthMapGenerator(downsample=(4, 8, 16))

    def __len__(self) -> int:
        return len(self.scene_tokens)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._aug_cache.clear()

    def current_subseq_id(self, idx: int) -> int:
        return self._subseq_for_index[int(idx)]

    def _build_subseq_index(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        groups: dict[str, list[int]] = {}
        for idx, token in enumerate(self.scene_tokens):
            frames = self.scene_loader.scene_frames_dicts[token]
            log_name = str(frames[0].get("log_name", "default"))
            groups.setdefault(log_name, []).append(idx)
        subseq_for: dict[int, int] = {}
        subseq_to_indices: dict[int, list[int]] = {}
        next_subseq = 0
        split_num = self.sequences_split_num
        for _, indices in groups.items():
            if not self.with_seq_flag or split_num == 1:
                chunks = [indices]
            elif split_num == "all":
                chunks = [[i] for i in indices]
            else:
                step = max(1, int(math.ceil(len(indices) / int(split_num))))
                chunks = [indices[start : start + step] for start in range(0, len(indices), step)]
            for chunk in chunks:
                if not chunk:
                    continue
                subseq_to_indices[next_subseq] = chunk
                for idx in chunk:
                    subseq_for[idx] = next_subseq
                next_subseq += 1
        return subseq_for, subseq_to_indices

    def _get_aug_params(self, idx: int, raw_hw: tuple[int, int]) -> dict[str, Any]:
        if not self.augment_enabled:
            rng = np.random.default_rng(0)
            return self.resize_aug.sample_params(rng, raw_hw)
        subseq = self.current_subseq_id(idx)
        key = (self.epoch, subseq, raw_hw)
        if key not in self._aug_cache:
            rng = np.random.default_rng(10_000 * self.epoch + subseq)
            params = self.resize_aug.sample_params(rng, raw_hw)
            params["photo"] = self.photo_aug.sample_params(rng)
            self._aug_cache[key] = params
        return self._aug_cache[key]

    def _load_map_api(self, map_name: str | None) -> Any:
        if not map_name:
            return None
        if map_name not in self._map_api_cache:
            try:
                from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

                self._map_api_cache[map_name] = get_maps_api(str(self.nuplan_maps_root), DEFAULT_MAP_VERSION, map_name)
            except Exception:
                self._map_api_cache[map_name] = None
        return self._map_api_cache[map_name]

    def _load_images_and_calibration(
        self,
        idx: int,
        current_frame: Mapping[str, Any],
        boxes_for_rotation: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor_root = Path(self.scene_loader._original_sensor_path)
        cameras = current_frame["cams"]
        imgs: list[torch.Tensor] = []
        projection: list[torch.Tensor] = []
        image_wh: list[torch.Tensor] = []
        focal: list[float] = []
        aug_params: dict[str, Any] | None = None
        rot_mat = np.eye(4, dtype=np.float32)
        for cam_idx, camera_name in enumerate(self.camera_order):
            camera_info = cameras[camera_name]
            image_path = sensor_root / camera_info["data_path"]
            with Image.open(image_path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            bgr = rgb[..., ::-1]
            raw_hw = bgr.shape[:2]
            if aug_params is None:
                aug_params = self._get_aug_params(idx, (int(raw_hw[0]), int(raw_hw[1])))
            intrinsics = np.asarray(camera_info["cam_intrinsic"], dtype=np.float32)[:3, :3]
            img, intrinsics, _, aug4 = self.resize_aug.apply(bgr, intrinsics, boxes_for_rotation, aug_params)
            if self.augment_enabled:
                img = self.photo_aug.apply(img, aug_params["photo"])
            img, intrinsics, _, _ = self.normalize.apply(img, intrinsics, None, {})
            _, _, _, rot_mat = self.bbox_rot.apply(img, intrinsics, None, aug_params)
            base_proj = _camera_projection(camera_info, intrinsics)
            if abs(float(aug_params.get("rotate_3d", 0.0))) > 1e-9:
                base_proj = base_proj @ np.linalg.inv(rot_mat).astype(np.float32)
            imgs.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float())
            projection.append(torch.from_numpy(base_proj.astype(np.float32)))
            image_wh.append(torch.tensor([self.image_hw[1], self.image_hw[0]], dtype=torch.float32))
            focal.append(float(intrinsics[0, 0]))
        return torch.stack(imgs), torch.stack(projection), torch.stack(image_wh), torch.tensor(focal, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token = self.scene_tokens[int(index)]
        frame_list = self.scene_loader.scene_frames_dicts[token]
        current_index = self.num_history_frames - 1
        current_frame = frame_list[current_index]
        global_to_lidar = _frame_global_to_lidar(current_frame)
        lidar_to_global = _ego_to_global_matrix(current_frame)
        timestamp = torch.tensor(_timestamp_seconds(current_frame["timestamp"]), dtype=torch.float32)

        img, projection_mat, image_wh, focal = self._load_images_and_calibration(index, current_frame)
        future_ego_positions = _future_ego_positions(frame_list, current_index, self.num_future_frames)
        gt_ego_fut_cmd = build_gt_ego_fut_cmd(current_frame, future_ego_positions)
        ego_status = build_ego_status(current_frame)
        img_metas = {
            "T_global": lidar_to_global.astype(np.float32),
            "T_global_inv": global_to_lidar.astype(np.float32),
            "timestamp": float(timestamp.item()),
        }

        data: dict[str, Any] = {
            "img": img,
            "timestamp": timestamp,
            "projection_mat": projection_mat,
            "image_wh": image_wh,
            "ego_status": ego_status,
            "gt_ego_fut_cmd": gt_ego_fut_cmd,
            "img_metas": img_metas,
        }
        if self.test_mode:
            return data

        ann = current_frame.get("anns", current_frame.get("annotations"))
        gt_bboxes_3d, gt_labels_3d, instance_id = build_gt_bboxes_3d(ann, global_to_lidar, range_threshold=55.0)
        track_mapping = _current_track_mapping(current_frame, global_to_lidar, range_threshold=55.0)
        gt_agent_fut_trajs, gt_agent_fut_masks = build_agent_futures(
            frame_list, current_index, track_mapping, fut_ts=12, ego_lidar_transform=global_to_lidar
        )
        if gt_agent_fut_trajs.shape[0] != gt_bboxes_3d.shape[0]:
            gt_agent_fut_trajs = torch.zeros((gt_bboxes_3d.shape[0], 12, 2), dtype=torch.float32)
            gt_agent_fut_masks = torch.zeros((gt_bboxes_3d.shape[0], 12), dtype=torch.float32)
        gt_ego_fut_trajs, gt_ego_fut_masks = build_ego_futures(frame_list, current_index, ego_fut_ts=6)
        lidar_points = _load_lidar_points(current_frame, Path(self.scene_loader._original_sensor_path))
        gt_depth = self.depth_generator(lidar_points, projection_mat, image_wh)
        map_api = self._load_map_api(current_frame.get("map_location"))
        gt_map_labels, gt_map_pts = _build_map_targets(map_api, current_frame.get("map_location"), current_frame, global_to_lidar)
        img_metas["instance_id"] = instance_id.detach().cpu().numpy()

        data.update(
            {
                "gt_depth": gt_depth,
                "focal": focal,
                "gt_bboxes_3d": gt_bboxes_3d,
                "gt_labels_3d": gt_labels_3d,
                "instance_id": instance_id,
                "gt_map_labels": gt_map_labels,
                "gt_map_pts": gt_map_pts,
                "gt_agent_fut_trajs": gt_agent_fut_trajs.float(),
                "gt_agent_fut_masks": gt_agent_fut_masks.float(),
                "gt_ego_fut_trajs": gt_ego_fut_trajs.float(),
                "gt_ego_fut_masks": gt_ego_fut_masks.float(),
            }
        )
        return data


class SequenceSplitSampler(Sampler[int]):
    def __init__(self, dataset: NavSimSparseDriveDataset, seed: int = 0) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def current_subseq_id(self, idx: int) -> int:
        return self.dataset.current_subseq_id(idx)

    def __iter__(self) -> Iterator[int]:
        subseq_ids = list(self.dataset._subseq_to_indices.keys())
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(subseq_ids)
        for subseq_id in subseq_ids:
            yield from self.dataset._subseq_to_indices[subseq_id]

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    out: dict[str, Any] = {}
    keys = set().union(*(sample.keys() for sample in batch))
    for key in keys:
        values = [sample[key] for sample in batch if key in sample]
        if key == "img_metas":
            out[key] = list(values)
        elif key == "gt_depth":
            num_scales = len(values[0]) if values else 0
            out[key] = [torch.stack([sample_depths[i] for sample_depths in values], dim=0) for i in range(num_scales)]
        elif key in VARIABLE_KEYS:
            out[key] = list(values)
        elif key in FIXED_STACK_KEYS and all(torch.is_tensor(v) for v in values):
            out[key] = torch.stack([v.reshape(()) if v.ndim == 0 else v for v in values], dim=0)
        elif all(torch.is_tensor(v) and v.shape == values[0].shape for v in values):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = list(values)
    return out


def build_dataloader(
    dataset: NavSimSparseDriveDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    collate_fn: Any | None = None,
) -> DataLoader:
    batch_collate_fn = collate_fn if collate_fn is not None else globals()["collate_fn"]
    if shuffle and not dataset.test_mode and dataset.with_seq_flag:
        sampler: Sampler[int] = SequenceSplitSampler(dataset, seed=seed)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=batch_collate_fn)
    sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=batch_collate_fn)


def _describe_value(value: Any) -> str:
    if torch.is_tensor(value):
        return f"shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, list):
        inner = ""
        if value and torch.is_tensor(value[0]):
            inner = f" first_shape={tuple(value[0].shape)} first_dtype={value[0].dtype}"
        elif value and isinstance(value[0], list):
            inner = f" nested_len={len(value[0])}"
        return f"list len={len(value)}{inner}"
    if isinstance(value, dict):
        return f"dict keys={sorted(value.keys())}"
    return type(value).__name__


if __name__ == "__main__":
    os.environ.setdefault("OPENSCENE_DATA_ROOT", "/Users/chenran/Code/navsim/dataset")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", "/Users/chenran/Code/navsim/dataset/maps")
    root = os.environ["OPENSCENE_DATA_ROOT"]
    maps = os.environ["NUPLAN_MAPS_ROOT"]
    dataset = NavSimSparseDriveDataset(
        split="mini",
        openscene_data_root=root,
        nuplan_maps_root=maps,
        augment=True,
        test_mode=False,
        max_scenes=2,
    )
    print("train_len", len(dataset))
    sample0 = dataset[0]
    for key in sorted(sample0.keys()):
        print("train", key, _describe_value(sample0[key]))
    batch = collate_fn([dataset[0], dataset[1]])
    for key in sorted(batch.keys()):
        print("batch", key, _describe_value(batch[key]))
    test_dataset = NavSimSparseDriveDataset(
        split="mini",
        openscene_data_root=root,
        nuplan_maps_root=maps,
        augment=True,
        test_mode=True,
        max_scenes=2,
    )
    test_sample = test_dataset[0]
    print("test_keys", sorted(test_sample.keys()))
    for key in sorted(test_sample.keys()):
        print("test", key, _describe_value(test_sample[key]))
