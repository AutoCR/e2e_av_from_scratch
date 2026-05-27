from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

DATA_AUG_CONF = {
    "resize_lim": (0.40, 0.47),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "rand_flip": True,
    "rot3d_range": (0.0, 0.0),
}
IMG_NORM_CFG = {
    "mean": np.array([123.675, 116.28, 103.53], dtype=np.float32),
    "std": np.array([58.395, 57.12, 57.375], dtype=np.float32),
    "to_rgb": True,
}


def _rng_uniform(rng: Any, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _rng_int(rng: Any, high: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(high))
    return int(rng.randint(high))


@dataclass
class ResizeCropFlipImage:
    data_aug_conf: Mapping[str, Any] | None = None
    test_mode: bool = False

    def __post_init__(self) -> None:
        self.data_aug_conf = dict(DATA_AUG_CONF if self.data_aug_conf is None else self.data_aug_conf)

    def sample_params(self, rng: Any, image_hw: Sequence[int]) -> dict[str, Any]:
        src_h, src_w = int(image_hw[0]), int(image_hw[1])
        f_h, f_w = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = _rng_uniform(rng, *self.data_aug_conf["resize_lim"])
            new_w, new_h = int(src_w * resize), int(src_h * resize)
            crop_h = int((1.0 - _rng_uniform(rng, *self.data_aug_conf["bot_pct_lim"])) * new_h) - f_h
            crop_w = int(_rng_uniform(rng, 0, max(0, new_w - f_w)))
            flip = bool(self.data_aug_conf.get("rand_flip", False) and _rng_int(rng, 2))
            rotate = _rng_uniform(rng, *self.data_aug_conf["rot_lim"])
            rotate_3d = _rng_uniform(rng, *self.data_aug_conf["rot3d_range"])
        else:
            resize = max(f_h / src_h, f_w / src_w)
            new_w, new_h = int(src_w * resize), int(src_h * resize)
            crop_h = int((1.0 - float(np.mean(self.data_aug_conf["bot_pct_lim"]))) * new_h) - f_h
            crop_w = int(max(0, new_w - f_w) / 2)
            flip = False
            rotate = 0.0
            rotate_3d = 0.0
        return {
            "resize": resize,
            "resize_dims": (new_w, new_h),
            "crop": (crop_w, crop_h, crop_w + f_w, crop_h + f_h),
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }

    def apply(
        self,
        img_np: np.ndarray,
        intrinsics: np.ndarray,
        boxes: np.ndarray | None,
        params: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        img = np.asarray(img_np)
        origin_dtype = img.dtype
        pil_img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        pil_img = pil_img.resize(tuple(params["resize_dims"]), getattr(getattr(Image, "Resampling", Image), "BILINEAR"))
        pil_img = pil_img.crop(tuple(params["crop"]))
        if params.get("flip", False):
            pil_img = pil_img.transpose(method=Image.FLIP_LEFT_RIGHT)
        pil_img = pil_img.rotate(float(params.get("rotate", 0.0)))
        out = np.asarray(pil_img).astype(np.float32)
        if origin_dtype != np.uint8:
            out = out.astype(np.float32)

        mat3 = np.eye(3, dtype=np.float32)
        mat3[:2, :2] *= float(params["resize"])
        mat3[:2, 2] -= np.array(params["crop"][:2], dtype=np.float32)
        crop = params["crop"]
        if params.get("flip", False):
            flip_matrix = np.array([[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            mat3 = flip_matrix @ mat3
        rotate = np.deg2rad(float(params.get("rotate", 0.0)))
        rot_matrix = np.array(
            [[np.cos(rotate), np.sin(rotate), 0], [-np.sin(rotate), np.cos(rotate), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]], dtype=np.float32) / 2.0
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        mat3 = rot_matrix @ mat3
        aug4 = np.eye(4, dtype=np.float32)
        aug4[:3, :3] = mat3
        new_intrinsics = mat3 @ np.asarray(intrinsics, dtype=np.float32)
        return out, new_intrinsics.astype(np.float32), boxes, aug4


@dataclass
class BBoxRotation:
    def sample_params(self, rng: Any) -> dict[str, float]:
        return {"rotate_3d": 0.0}

    def apply(
        self,
        img_np_or_tensor: np.ndarray,
        intrinsics: np.ndarray,
        boxes: np.ndarray | None,
        params: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        angle = float(params.get("rotate_3d", 0.0))
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        if boxes is not None and len(boxes):
            rot_mat_t = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
            boxes = boxes.copy()
            boxes[:, :3] = boxes[:, :3] @ rot_mat_t
            boxes[:, 6] += angle
            if boxes.shape[1] > 7:
                vel_dims = boxes[:, 7:].shape[-1]
                boxes[:, 7:] = boxes[:, 7:] @ rot_mat_t[:vel_dims, :vel_dims]
        return img_np_or_tensor, intrinsics, boxes, rot_mat


def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    rgb = np.clip(img, 0, 255).astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g)
    bmask = mask & (maxc == b)
    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6
    h[gmask] = (b[gmask] - r[gmask]) / delta[gmask] + 2
    h[bmask] = (r[bmask] - g[bmask]) / delta[bmask] + 4
    h *= 60.0
    s = np.zeros_like(maxc)
    nonzero = maxc > 1e-6
    s[nonzero] = delta[nonzero] / maxc[nonzero]
    return np.stack([h, s, maxc], axis=-1).astype(np.float32)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = (hsv[..., 0] % 360.0) / 60.0
    s = np.clip(hsv[..., 1], 0.0, None)
    v = np.clip(hsv[..., 2], 0.0, None)
    c = v * s
    x = c * (1.0 - np.abs(h % 2.0 - 1.0))
    m = v - c
    z = np.zeros_like(h)
    out = np.zeros(hsv.shape, dtype=np.float32)
    conds = [(0 <= h) & (h < 1), (1 <= h) & (h < 2), (2 <= h) & (h < 3), (3 <= h) & (h < 4), (4 <= h) & (h < 5), (5 <= h) & (h < 6)]
    vals = [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)]
    for cond, val in zip(conds, vals):
        out[..., 0][cond], out[..., 1][cond], out[..., 2][cond] = val[0][cond], val[1][cond], val[2][cond]
    return (out + m[..., None]) * 255.0


@dataclass
class PhotoMetricDistortionMultiViewImage:
    brightness_delta: float = 32.0
    contrast_range: tuple[float, float] = (0.5, 1.5)
    saturation_range: tuple[float, float] = (0.5, 1.5)
    hue_delta: float = 18.0

    def sample_params(self, rng: Any) -> dict[str, Any]:
        return {
            "brightness": bool(_rng_int(rng, 2)),
            "brightness_delta": _rng_uniform(rng, -self.brightness_delta, self.brightness_delta),
            "mode": _rng_int(rng, 2),
            "contrast_first": bool(_rng_int(rng, 2)),
            "contrast_first_alpha": _rng_uniform(rng, *self.contrast_range),
            "saturation": bool(_rng_int(rng, 2)),
            "saturation_alpha": _rng_uniform(rng, *self.saturation_range),
            "hue": bool(_rng_int(rng, 2)),
            "hue_delta": _rng_uniform(rng, -self.hue_delta, self.hue_delta),
            "contrast_last": bool(_rng_int(rng, 2)),
            "contrast_last_alpha": _rng_uniform(rng, *self.contrast_range),
            "swap": bool(_rng_int(rng, 2)),
            "permutation": np.asarray(rng.permutation(3), dtype=np.int64),
        }

    def apply(self, img_np: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        img = np.asarray(img_np, dtype=np.float32).copy()
        if params.get("brightness", False):
            img += float(params["brightness_delta"])
        if int(params.get("mode", 0)) == 1 and params.get("contrast_first", False):
            img *= float(params["contrast_first_alpha"])
        # Treat incoming arrays as BGR like mmcv; convert through RGB helpers.
        hsv = _rgb_to_hsv(img[..., ::-1])
        if params.get("saturation", False):
            hsv[..., 1] *= float(params["saturation_alpha"])
        if params.get("hue", False):
            hsv[..., 0] = (hsv[..., 0] + float(params["hue_delta"])) % 360.0
        img = _hsv_to_rgb(hsv)[..., ::-1]
        if int(params.get("mode", 0)) == 0 and params.get("contrast_last", False):
            img *= float(params["contrast_last_alpha"])
        if params.get("swap", False):
            img = img[..., np.asarray(params["permutation"], dtype=np.int64)]
        return img.astype(np.float32)


@dataclass
class NormalizeMultiviewImage:
    mean: Sequence[float] = tuple(IMG_NORM_CFG["mean"].tolist())
    std: Sequence[float] = tuple(IMG_NORM_CFG["std"].tolist())
    to_rgb: bool = True

    def sample_params(self, rng: Any) -> dict[str, Any]:
        return {}

    def apply(
        self,
        img_np_or_tensor: np.ndarray,
        intrinsics: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
        img = np.asarray(img_np_or_tensor, dtype=np.float32)
        if self.to_rgb:
            img = img[..., ::-1]
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
        return img.astype(np.float32), intrinsics, boxes, np.eye(4, dtype=np.float32)
