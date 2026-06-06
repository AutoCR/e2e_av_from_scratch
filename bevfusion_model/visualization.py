"""Visualization utilities for BEVFusion nuScenes inference.

Renders two figures per sample:
  * a 6-camera grid with predicted 3D boxes projected into each image, and
  * a top-down LiDAR BEV figure with the predicted boxes drawn as footprints.

Kept dependency-free apart from numpy / OpenCV so it can be reused outside the
test harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# nuScenes 10-class detection labels (TransFusionHead output order).
OBJECT_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# Per-class BGR colors (OpenCV convention).
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 200, 100),
    2: (0, 150, 255),
    3: (0, 100, 200),
    4: (100, 0, 255),
    5: (255, 255, 0),
    6: (255, 100, 0),
    7: (255, 0, 100),
    8: (255, 0, 0),
    9: (200, 200, 200),
}

DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# ImageNet normalization used by the nuScenes adapter.
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default score threshold for drawing a box.
DEFAULT_SCORE_THRESH = 0.02

# LiDAR BEV rendering defaults.
BEV_RANGE_M = 54.0       # half-extent of the BEV window in meters (matches pc_range)
BEV_IMG_SIZE = 800       # output BEV image is BEV_IMG_SIZE x BEV_IMG_SIZE pixels


# ──────────────────────────────────────────────────────────────────────────────
# 3D box geometry
# ──────────────────────────────────────────────────────────────────────────────

def get_box_corners(box: np.ndarray) -> np.ndarray:
    """Generate 8 corners of a 3D box in the lidar frame.

    Args:
        box: [x, y, z, l, w, h, yaw, ...] where z is the bottom-center height.

    Returns:
        corners: (8, 3) corners in the lidar frame. Indices 0-3 are the top
        face, 4-7 the bottom face (matching the image-projection edge list).
    """
    cx, cy, bottom_z, l, w, h, yaw = box[:7]
    cz = bottom_z + h / 2.0
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    hl, hw, hh = l / 2, w / 2, h / 2

    corners = np.array([
        [ hl,  hw,  hh], [ hl, -hw,  hh], [-hl, -hw,  hh], [-hl,  hw,  hh],
        [ hl,  hw, -hh], [ hl, -hw, -hh], [-hl, -hw, -hh], [-hl,  hw, -hh],
    ], dtype=np.float32)

    rot = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ], dtype=np.float32)

    corners = (rot @ corners.T).T
    corners += np.array([cx, cy, cz], dtype=np.float32)
    return corners


def project_corners_to_image(
    corners: np.ndarray,
    lidar2image: np.ndarray,
    img_aug_matrix: np.ndarray | None = None,
):
    """Project 3D corners to 2D image coordinates.

    Args:
        corners: (8, 3) in lidar frame.
        lidar2image: (4, 4) projection matrix (no image augmentation baked in).
        img_aug_matrix: optional (4, 4) test-time image resize+crop transform.
            It operates in 2D *pixel* space and must therefore be applied
            *after* the perspective divide — not pre-multiplied onto
            ``lidar2image``, which would incorrectly scale the crop offset by
            depth. This mirrors how the model's view transform consumes
            ``post_trans``/``post_rots`` (see ``view_transform.py``).

    Returns:
        uv: (8, 2) image pixel coordinates.
        depth: (8,) depth values.
    """
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])
    proj = (lidar2image @ corners_h.T).T

    depth = proj[:, 2:3].clip(min=0.01)
    uv = proj[:, :2] / depth

    if img_aug_matrix is not None:
        # Apply the 2D resize (2x2 block) + crop translation in pixel space.
        rot = img_aug_matrix[:2, :2]
        trans = img_aug_matrix[:2, 3]
        uv = uv @ rot.T + trans

    return uv, proj[:, 2]


def _draw_box_edges(img, uv, depth, color, thickness: int = 2) -> None:
    """Draw the 12 edges of a 3D box on an image."""
    import cv2

    H, W = img.shape[:2]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        if depth[i] > 0 and depth[j] > 0:
            pt1 = tuple(uv[i].astype(int))
            pt2 = tuple(uv[j].astype(int))
            if (-50 <= pt1[0] <= W + 50 and -50 <= pt1[1] <= H + 50 and
                    -50 <= pt2[0] <= W + 50 and -50 <= pt2[1] <= H + 50):
                cv2.line(img, pt1, pt2, color, thickness)


# ──────────────────────────────────────────────────────────────────────────────
# Output decoding helper
# ──────────────────────────────────────────────────────────────────────────────

def _decode_arrays(output: dict[str, Any]):
    """Pull boxes/scores/labels out of a model output as numpy arrays."""
    boxes = output["boxes_3d"].numpy() if len(output["boxes_3d"]) > 0 else np.empty((0, 7))
    scores = output["scores_3d"].numpy() if len(output["scores_3d"]) > 0 else np.empty((0,))
    labels = output["labels_3d"].numpy() if len(output["labels_3d"]) > 0 else np.empty((0,), dtype=int)
    return boxes, scores, labels


# ──────────────────────────────────────────────────────────────────────────────
# Camera-grid visualization
# ──────────────────────────────────────────────────────────────────────────────

def render_camera_grid(
    sample: Any,
    output: dict[str, Any],
    camera_order: tuple[str, ...] = DEFAULT_CAMERA_ORDER,
    score_thresh: float = DEFAULT_SCORE_THRESH,
) -> np.ndarray:
    """Render the 6-camera grid with predicted 3D boxes projected in.

    Returns a BGR uint8 image (the stitched 2x3 grid).
    """
    import cv2

    boxes, scores, labels = _decode_arrays(output)

    imgs = sample.img.numpy()
    lidar2image = sample.lidar2image.numpy()
    img_aug_matrix = sample.img_aug_matrix.numpy()

    cam_imgs = []
    for cam_idx in range(imgs.shape[0]):
        img = imgs[cam_idx].transpose(1, 2, 0)
        img = (img * _IMG_STD + _IMG_MEAN).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if len(boxes) > 0:
            for box, score, label in zip(boxes, scores, labels):
                if score > score_thresh:
                    corners = get_box_corners(box)
                    uv, depth = project_corners_to_image(
                        corners, lidar2image[cam_idx], img_aug_matrix[cam_idx]
                    )

                    color = CLASS_COLORS.get(int(label), (128, 128, 128))
                    _draw_box_edges(img_bgr, uv, depth, color, thickness=2)

                    center_uv = uv.mean(axis=0).astype(int)
                    if 0 <= center_uv[0] < img_bgr.shape[1] and 0 <= center_uv[1] < img_bgr.shape[0]:
                        cv2.putText(
                            img_bgr, f"{score:.2f}", tuple(center_uv),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                        )

        cam_name = camera_order[cam_idx] if cam_idx < len(camera_order) else f"cam{cam_idx}"
        cv2.putText(img_bgr, cam_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cam_imgs.append(img_bgr)

    if len(cam_imgs) >= 6:
        row1 = np.concatenate(cam_imgs[:3], axis=1)
        row2 = np.concatenate(cam_imgs[3:6], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
    else:
        grid = np.concatenate(cam_imgs, axis=1)

    n_det = len(scores)
    max_score = float(scores.max()) if len(scores) > 0 else 0.0
    n_shown = int((scores > score_thresh).sum()) if len(scores) > 0 else 0
    cv2.putText(
        grid,
        f"Detections: {n_det} total, {n_shown} shown (>{score_thresh}), max={max_score:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# LiDAR BEV visualization
# ──────────────────────────────────────────────────────────────────────────────

def _lidar_xy_to_bev_px(x: np.ndarray, y: np.ndarray, range_m: float, img_size: int):
    """Map lidar (x, y) coordinates to BEV pixel (col, row).

    The BEV image is a top-down view with the ego vehicle at the center.
    Lidar +x (forward) points up in the image and lidar +y (left) points left,
    matching the conventional nuScenes BEV layout.

    Returns (cols, rows) as float arrays.
    """
    scale = img_size / (2.0 * range_m)
    # +x forward -> up (smaller row); +y left -> left (smaller col).
    cols = img_size / 2.0 - y * scale
    rows = img_size / 2.0 - x * scale
    return cols, rows


def render_lidar_bev(
    sample: Any,
    output: dict[str, Any],
    range_m: float = BEV_RANGE_M,
    img_size: int = BEV_IMG_SIZE,
    score_thresh: float = DEFAULT_SCORE_THRESH,
) -> np.ndarray:
    """Render a top-down LiDAR BEV figure with predicted box footprints.

    Points are scattered onto a square BEV canvas covering
    [-range_m, range_m] in both x and y, colored by height. Each predicted
    box above ``score_thresh`` is drawn as its BEV footprint (rotated
    rectangle) plus a short heading line indicating the +x (front) direction.

    Returns a BGR uint8 image of shape (img_size, img_size, 3).
    """
    import cv2

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # ── scatter the point cloud ────────────────────────────────────────────
    points = sample.points.numpy()
    if points.shape[0] > 0:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        in_range = (np.abs(x) <= range_m) & (np.abs(y) <= range_m)
        x, y, z = x[in_range], y[in_range], z[in_range]

        cols, rows = _lidar_xy_to_bev_px(x, y, range_m, img_size)
        c = cols.astype(np.int32)
        r = rows.astype(np.int32)
        valid = (c >= 0) & (c < img_size) & (r >= 0) & (r < img_size)
        c, r, z = c[valid], r[valid], z[valid]

        # Color by height: low (blue) -> high (red), via a JET colormap.
        z_lo, z_hi = -5.0, 3.0
        z_norm = np.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0)
        cmap = cv2.applyColorMap((z_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
        colors = cmap.reshape(-1, 3)
        canvas[r, c] = colors

    # ── grid + ego marker ──────────────────────────────────────────────────
    center = img_size // 2
    grid_color = (60, 60, 60)
    for ring_m in range(10, int(range_m) + 1, 10):
        radius_px = int(ring_m * img_size / (2.0 * range_m))
        cv2.circle(canvas, (center, center), radius_px, grid_color, 1)
    cv2.line(canvas, (center, 0), (center, img_size), grid_color, 1)
    cv2.line(canvas, (0, center), (img_size, center), grid_color, 1)
    # Ego vehicle: small triangle pointing forward (up).
    cv2.drawMarker(canvas, (center, center), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)

    # ── predicted box footprints ───────────────────────────────────────────
    boxes, scores, labels = _decode_arrays(output)
    n_shown = 0
    for box, score, label in zip(boxes, scores, labels):
        if score <= score_thresh:
            continue
        n_shown += 1
        corners = get_box_corners(box)              # (8, 3)
        bottom = corners[4:8]                        # 4 bottom-face corners
        cols, rows = _lidar_xy_to_bev_px(bottom[:, 0], bottom[:, 1], range_m, img_size)
        poly = np.stack([cols, rows], axis=1).astype(np.int32)

        color = CLASS_COLORS.get(int(label), (128, 128, 128))
        cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=1)

        # Heading line: box center -> midpoint of the front edge (corners 4 & 5).
        cx, cy = box[0], box[1]
        fx, fy = bottom[:2, 0].mean(), bottom[:2, 1].mean()
        c0, r0 = _lidar_xy_to_bev_px(np.array([cx]), np.array([cy]), range_m, img_size)
        c1, r1 = _lidar_xy_to_bev_px(np.array([fx]), np.array([fy]), range_m, img_size)
        cv2.line(canvas, (int(c0[0]), int(r0[0])), (int(c1[0]), int(r1[0])), color, 1)

    cv2.putText(
        canvas,
        f"LiDAR BEV ({int(2 * range_m)}m) - {n_shown} boxes (>{score_thresh})",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def visualize_sample(
    sample: Any,
    output: dict[str, Any],
    out_dir: Path,
    sample_idx: int,
    camera_order: tuple[str, ...] = DEFAULT_CAMERA_ORDER,
    score_thresh: float = DEFAULT_SCORE_THRESH,
) -> None:
    """Render and save the camera-grid and LiDAR BEV figures for one sample."""
    try:
        import cv2
    except ImportError:
        print("  cv2 not available, skipping visualization")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_idx:04d}_{sample.token[:8]}"

    grid = render_camera_grid(sample, output, camera_order=camera_order, score_thresh=score_thresh)
    cam_path = out_dir / f"{stem}_cameras.jpg"
    cv2.imwrite(str(cam_path), grid)
    print(f"  Saved camera visualization: {cam_path}")

    bev = render_lidar_bev(sample, output, score_thresh=score_thresh)
    bev_path = out_dir / f"{stem}_bev.jpg"
    cv2.imwrite(str(bev_path), bev)
    print(f"  Saved LiDAR BEV visualization: {bev_path}")


__all__ = [
    "OBJECT_CLASSES",
    "CLASS_COLORS",
    "DEFAULT_CAMERA_ORDER",
    "DEFAULT_SCORE_THRESH",
    "get_box_corners",
    "project_corners_to_image",
    "render_camera_grid",
    "render_lidar_bev",
    "visualize_sample",
]
