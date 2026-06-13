"""
BEVFusion dataset for NAVSIM (5-class: car, barrier, bicycle, pedestrian, traffic_cone).

Loads NAVSIM/OpenScene dataset and emits BEVFusion-shaped training inputs:
  - img: (N, 3, H, W) float32, ImageNet-normalized, 8 cameras
  - points: (P, 5) [x, y, z, intensity, rel_timestamp] in LiDAR frame
  - camera2ego, lidar2camera, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix: (N, 4, 4)
  - lidar2ego: (4, 4) identity (LiDAR==ego in NAVSIM)
  - lidar_aug_matrix: (4, 4) identity
  - gt_bboxes_3d: (G, 9) [cx,cy,cz,w,l,h,yaw,vx,vy] in LiDAR frame
  - gt_labels_3d: (G,) long in [0,4] (5-class NAVSIM order)
  - metas: dict with 'token'
"""

from __future__ import annotations

import math
import sys
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.utils.data
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bevfusion_model.navsim_train.navsim_adapter import (
    DEFAULT_CAMERA_ORDER_8,
    normalize_camera_order,
    build_navsim_scene_loader,
)
from bevfusion_model.navsim_train.targets import (
    build_gt_bboxes_3d,
    _frame_global_to_lidar,
)
from bevfusion_model.navsim_train.data_utils import load_lidar_points


DEFAULT_IMAGE_HW = (256, 704)
DEFAULT_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _resize_crop_image(
    image: Image.Image,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
) -> tuple[Image.Image, np.ndarray]:
    """Deterministic resize+crop to target H,W with affine recording."""
    target_h, target_w = image_hw
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized_w = max(target_w, int(math.ceil(src_w * scale)))
    resized_h = max(target_h, int(math.ceil(src_h * scale)))
    crop_left = (resized_w - target_w) // 2
    crop_top = (resized_h - target_h) // 2

    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = image.resize((resized_w, resized_h), resampling)
    cropped = resized.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))

    # 3x3 affine for image transform: scale + crop translation
    image_transform = np.array(
        [[scale, 0.0, -float(crop_left)], [0.0, scale, -float(crop_top)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cropped, image_transform @ intrinsics


def _image_to_tensor(
    image: Image.Image,
    mean: np.ndarray = DEFAULT_IMAGE_MEAN,
    std: np.ndarray = DEFAULT_IMAGE_STD,
) -> torch.Tensor:
    """Normalize RGB image to (3, H, W) float32 tensor."""
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    mean_np = mean.reshape(1, 1, 3)
    std_np = std.reshape(1, 1, 3)
    image_np = (image_np - mean_np) / std_np
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().to(torch.float32)


class NavSimBEVFusionDataset(torch.utils.data.Dataset):
    """NAVSIM dataset for BEVFusion camera+LiDAR 3D detection training."""

    def __init__(
        self,
        split: str,
        openscene_data_root: str | os.PathLike[str],
        nuplan_maps_root: str | os.PathLike[str],
        camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER_8,
        image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
        test_mode: bool = False,
        max_scenes: Optional[int] = None,
        log_names: Optional[Sequence[str]] = None,
        tokens: Optional[Sequence[str]] = None,
        num_history_frames: int = 1,
        num_future_frames: int = 0,
    ):
        """
        Args:
            split: dataset split (e.g., "trainval", "mini")
            openscene_data_root: root directory of OpenScene dataset
            nuplan_maps_root: root directory of NuPlan maps
            camera_order: sequence of camera names (default: all 8 NAVSIM cameras)
            image_hw: (height, width) for image resizing
            test_mode: if True, omit GT annotations
            max_scenes: limit number of scenes (for smoke testing)
            log_names: filter by log names
            tokens: filter by scene tokens
            num_history_frames: number of history frames in each NAVSIM sample window
            num_future_frames: number of future frames in each NAVSIM sample window
        """
        self.camera_order = normalize_camera_order(camera_order)
        self.image_hw = image_hw
        self.test_mode = test_mode

        self.scene_loader = build_navsim_scene_loader(
            split=split,
            camera_order=self.camera_order,
            openscene_data_root=openscene_data_root,
            nuplan_maps_root=nuplan_maps_root,
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
            frame_interval=1,
            max_scenes=max_scenes,
            log_names=log_names,
            tokens=tokens,
        )

        self.scene_tokens = list(self.scene_loader.tokens)
        self.sensor_root = Path(self.scene_loader._original_sensor_path)
        self.num_history_frames = num_history_frames

    def __len__(self) -> int:
        return len(self.scene_tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token = self.scene_tokens[index]
        frame_list = self.scene_loader.scene_frames_dicts[token]
        current_index = self.num_history_frames - 1
        current_frame = frame_list[current_index]

        # Load camera images and build transforms
        cameras = current_frame["cams"]
        img_list = []
        camera2ego_list = []
        lidar2camera_list = []
        lidar2image_list = []
        camera_intrinsics_list = []
        camera2lidar_list = []
        img_aug_matrix_list = []

        for camera_name in self.camera_order:
            if camera_name not in cameras:
                raise KeyError(f"Token {token}: missing camera {camera_name}")

            cam = cameras[camera_name]
            image_path = self.sensor_root / cam["data_path"]
            if not image_path.exists():
                raise FileNotFoundError(f"Camera image not found: {image_path}")

            # Load image and apply resize+crop
            with Image.open(image_path) as raw_image:
                rgb_image = raw_image.convert("RGB")
                K3 = np.asarray(cam["cam_intrinsic"], dtype=np.float64)[:3, :3]
                resized_image, adjusted_K3 = _resize_crop_image(rgb_image, K3, self.image_hw)

            # Normalize image
            img_tensor = _image_to_tensor(resized_image, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
            img_list.append(img_tensor)

            # Build camera transforms (LiDAR == ego in NAVSIM)
            sensor2lidar_rotation = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
            sensor2lidar_translation = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64).reshape(3)

            camera2lidar_4x4 = np.eye(4, dtype=np.float64)
            camera2lidar_4x4[:3, :3] = sensor2lidar_rotation
            camera2lidar_4x4[:3, 3] = sensor2lidar_translation
            camera2lidar_4x4 = camera2lidar_4x4.astype(np.float32)

            camera2ego_4x4 = camera2lidar_4x4.copy()
            lidar2camera_4x4 = np.linalg.inv(camera2lidar_4x4).astype(np.float32)

            # Build camera intrinsics matrix (4x4)
            camera_intrinsics_4x4 = np.eye(4, dtype=np.float32)
            camera_intrinsics_4x4[:3, :3] = adjusted_K3.astype(np.float32)

            # Build image augmentation matrix (4x4): encode resize+crop affine
            img_aug_3x3 = np.eye(3, dtype=np.float32)
            src_w, src_h = rgb_image.size
            scale = max(self.image_hw[1] / src_w, self.image_hw[0] / src_h)
            resized_w = max(self.image_hw[1], int(math.ceil(src_w * scale)))
            resized_h = max(self.image_hw[0], int(math.ceil(src_h * scale)))
            crop_left = (resized_w - self.image_hw[1]) // 2
            crop_top = (resized_h - self.image_hw[0]) // 2
            img_aug_3x3[0, 0] = scale
            img_aug_3x3[1, 1] = scale
            img_aug_3x3[0, 2] = -float(crop_left)
            img_aug_3x3[1, 2] = -float(crop_top)
            img_aug_matrix_4x4 = np.eye(4, dtype=np.float32)
            img_aug_matrix_4x4[:3, :3] = img_aug_3x3

            # lidar2image = K @ lidar2camera
            lidar2image_4x4 = camera_intrinsics_4x4 @ lidar2camera_4x4

            camera2ego_list.append(torch.from_numpy(camera2ego_4x4))
            lidar2camera_list.append(torch.from_numpy(lidar2camera_4x4))
            lidar2image_list.append(torch.from_numpy(lidar2image_4x4))
            camera_intrinsics_list.append(torch.from_numpy(camera_intrinsics_4x4))
            camera2lidar_list.append(torch.from_numpy(camera2lidar_4x4))
            img_aug_matrix_list.append(torch.from_numpy(img_aug_matrix_4x4))

        img = torch.stack(img_list, dim=0)
        camera2ego = torch.stack(camera2ego_list, dim=0)
        lidar2camera = torch.stack(lidar2camera_list, dim=0)
        lidar2image = torch.stack(lidar2image_list, dim=0)
        camera_intrinsics = torch.stack(camera_intrinsics_list, dim=0)
        camera2lidar = torch.stack(camera2lidar_list, dim=0)
        img_aug_matrix = torch.stack(img_aug_matrix_list, dim=0)

        # Load LiDAR points (single keyframe sweep)
        pts = load_lidar_points(current_frame, self.sensor_root)
        pts[:, 4] = 0.0  # Set rel_timestamp = 0 for keyframe

        # Build GT (skip if test_mode)
        if self.test_mode:
            gt_bboxes_3d = torch.zeros((0, 9), dtype=torch.float32)
            gt_labels_3d = torch.zeros((0,), dtype=torch.long)
        else:
            global_to_lidar = _frame_global_to_lidar(current_frame)
            ann = current_frame.get("anns", current_frame.get("annotations"))
            gt_bboxes_3d, gt_labels_3d, _ = build_gt_bboxes_3d(ann, global_to_lidar, range_threshold=55.0)

        # LiDAR and identity transforms
        lidar2ego = torch.eye(4, dtype=torch.float32)
        lidar_aug_matrix = torch.eye(4, dtype=torch.float32)

        return {
            "token": token,
            "img": img,
            "points": pts,
            "camera2ego": camera2ego,
            "lidar2ego": lidar2ego,
            "lidar2camera": lidar2camera,
            "lidar2image": lidar2image,
            "camera_intrinsics": camera_intrinsics,
            "camera2lidar": camera2lidar,
            "img_aug_matrix": img_aug_matrix,
            "lidar_aug_matrix": lidar_aug_matrix,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "metas": {"token": token},
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch: stack fixed-size tensors, keep variable-length as lists."""
    if not batch:
        raise ValueError("collate_fn requires at least one sample")

    result = {}

    # Stack fixed-size tensors
    for key in [
        "img",
        "camera2ego",
        "lidar2ego",
        "lidar2camera",
        "lidar2image",
        "camera_intrinsics",
        "camera2lidar",
        "img_aug_matrix",
        "lidar_aug_matrix",
    ]:
        result[key] = torch.stack([sample[key] for sample in batch], dim=0)

    # Keep variable-length as lists
    result["points"] = [sample["points"] for sample in batch]
    result["gt_bboxes_3d"] = [sample["gt_bboxes_3d"] for sample in batch]
    result["gt_labels_3d"] = [sample["gt_labels_3d"] for sample in batch]
    result["metas"] = [sample["metas"] for sample in batch]

    return result


def build_dataloader(
    dataset: NavSimBEVFusionDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    collate_fn_override: Optional[Any] = None,
    sampler: Optional[Any] = None,
) -> torch.utils.data.DataLoader:
    """Build DataLoader for BEVFusion NAVSIM dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(shuffle and sampler is None),
        collate_fn=collate_fn_override or collate_fn,
        sampler=sampler,
    )


if __name__ == "__main__":
    # Smoke test
    from bevfusion_model.configs.bevfusion_hyperparams import RUNTIME_CONFIG

    runtime = RUNTIME_CONFIG
    openscene_root = runtime["openscene_data_root"]
    maps_root = runtime["nuplan_maps_root"]
    splits = runtime["splits"]

    # Try to load real NAVSIM data if available
    try:
        # Use "trainval" split (the training/validation split directory)
        split_dir = splits["train"]["dir"]

        ds = NavSimBEVFusionDataset(
            split=split_dir,
            openscene_data_root=openscene_root,
            nuplan_maps_root=maps_root,
            max_scenes=2,
        )

        print(f"Dataset OK: {len(ds)} scenes available")

        # Load a sample
        s = ds[0]
        print(f"  img shape: {s['img'].shape}")
        assert s["img"].shape[1:] == (3, 256, 704), f"Expected img shape (N, 3, 256, 704), got {s['img'].shape}"

        # Check camera transform matrices
        for k in ["camera2ego", "lidar2camera", "lidar2image", "camera_intrinsics", "camera2lidar", "img_aug_matrix"]:
            assert s[k].shape[-2:] == (4, 4), f"Expected {k} shape (..., 4, 4), got {s[k].shape}"
            assert s[k].shape[0] == s["img"].shape[0], f"Mismatch in first dim for {k}: {s[k].shape[0]} vs img {s['img'].shape[0]}"

        # Check LiDAR
        assert s["points"].shape[1] == 5, f"Expected points shape (P, 5), got {s['points'].shape}"
        assert float(s["points"][:, 4].abs().max()) == 0.0, "Expected rel_timestamp=0 for keyframe"

        # Check GT
        assert s["gt_bboxes_3d"].shape[1] == 9, f"Expected boxes shape (G, 9), got {s['gt_bboxes_3d'].shape}"
        if s["gt_labels_3d"].numel() > 0:
            assert int(s["gt_labels_3d"].max()) <= 4, f"Expected labels in [0,4], got max={s['gt_labels_3d'].max()}"

        # Test collation
        b = collate_fn([ds[0], ds[1]])
        assert b["img"].shape[0] == 2, f"Expected batch size 2, got {b['img'].shape[0]}"
        assert isinstance(b["points"], list) and len(b["points"]) == 2, "Expected points as list of length 2"
        assert isinstance(b["metas"], list) and len(b["metas"]) == 2, "Expected metas as list of length 2"

        print(f"✓ dataset OK: img {s['img'].shape}, gts {s['gt_bboxes_3d'].shape}, labels<5 ok")
        print("✓ collation OK: batch img shape", b["img"].shape, "points list len", len(b["points"]))

    except (FileNotFoundError, RuntimeError) as e:
        print(f"NAVSIM data not available locally: {e}")
        print("\nTesting collate_fn with dummy data...")

        # Smoke test collate_fn with hand-built dicts
        dummy_samples = []
        for i in range(2):
            dummy_samples.append({
                "token": f"token_{i}",
                "img": torch.randn(8, 3, 256, 704, dtype=torch.float32),
                "points": torch.randn(100, 5, dtype=torch.float32),
                "camera2ego": torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(8, 4, 4).clone(),
                "lidar2ego": torch.eye(4, dtype=torch.float32),
                "lidar2camera": torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(8, 4, 4).clone(),
                "lidar2image": torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(8, 4, 4).clone(),
                "camera_intrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(8, 4, 4).clone(),
                "camera2lidar": torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(8, 4, 4).clone(),
                "img_aug_matrix": torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(8, 4, 4).clone(),
                "lidar_aug_matrix": torch.eye(4, dtype=torch.float32),
                "gt_bboxes_3d": torch.randn(5, 9, dtype=torch.float32),
                "gt_labels_3d": torch.randint(0, 5, (5,), dtype=torch.long),
                "metas": {"token": f"token_{i}"},
            })

        b = collate_fn(dummy_samples)
        assert b["img"].shape == (2, 8, 3, 256, 704), f"Expected batch img shape, got {b['img'].shape}"
        assert isinstance(b["points"], list) and len(b["points"]) == 2
        assert isinstance(b["metas"], list) and len(b["metas"]) == 2
        print("✓ collate_fn OK with dummy data: img shape", b["img"].shape, "points list len", len(b["points"]))
        print("\nModule imports OK, run on server with NAVSIM data available")
