"""CUDA-accelerated 3D IoU op (ported from BEVFusion / OpenPCDet).

This module provides 3D IoU computation with a CUDA-accelerated BEV overlap kernel
and a pure-torch 3D wrapper. On platforms without CUDA, it falls back to CPU zero-tensors.
"""

import os

import torch

__all__ = [
    "boxes_overlap_bev",
    "xywhr2xyxyr",
    "bbox_overlaps_3d",
    "IOU3D_CUDA_AVAILABLE",
    "load_iou3d_ext",
]

_IOU3D_EXT = None


def _cuda_buildable() -> bool:
    """Return True only when a CUDA toolchain + device are usable."""
    return torch.cuda.is_available() and torch.version.cuda is not None


def load_iou3d_ext():
    """JIT-compile and return the iou3d_ext module.

    Compilation is attempted only when CUDA is available. The compiled module
    is cached for the process lifetime. Returns None if CUDA is unavailable
    or compilation fails (caller should fall back to pure path).
    """
    global _IOU3D_EXT
    if _IOU3D_EXT is not None:
        return _IOU3D_EXT
    if not _cuda_buildable():
        return None

    from torch.utils.cpp_extension import load

    src_dir = os.path.join(os.path.dirname(__file__), "src")
    try:
        _IOU3D_EXT = load(
            name="iou3d_ext",
            sources=[
                os.path.join(src_dir, "iou3d.cpp"),
                os.path.join(src_dir, "iou3d_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[iou3d] CUDA extension build failed, using pure fallback: {exc}")
        _IOU3D_EXT = None
    return _IOU3D_EXT


# Whether the CUDA fast path can be used at all on this platform.
IOU3D_CUDA_AVAILABLE = _cuda_buildable()


def boxes_overlap_bev(boxes_a, boxes_b):
    """Compute BEV box overlap area using CUDA kernel.

    Args:
        boxes_a: (N, 5) tensor in XYXYR format [x1, y1, x2, y2, yaw].
        boxes_b: (M, 5) tensor in XYXYR format.

    Returns:
        (N, M) tensor of intersection areas (not IoU).
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    # The CUDA kernel reads data_ptr<float>; cast so fp16 inputs (e.g. under
    # autocast) cannot raise a dtype error mid-training.
    boxes_a = boxes_a.float().contiguous()
    boxes_b = boxes_b.float().contiguous()
    ans = boxes_a.new_zeros((N, M))

    ext = load_iou3d_ext()
    if ext is None:
        return ans

    ext.boxes_overlap_bev_gpu(boxes_a, boxes_b, ans)
    return ans


def xywhr2xyxyr(boxes_xywhr):
    """Convert boxes from XYWHR to XYXYR format.

    Args:
        boxes_xywhr: (N, 5) tensor [x, y, w, h, yaw].

    Returns:
        (N, 5) tensor [x1, y1, x2, y2, yaw] where (x1,y1) is bottom-left
        and (x2,y2) is top-right in the rotated frame.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def bbox_overlaps_3d(boxes1, boxes2):
    """Compute 3D IoU between two sets of boxes (pure-torch, CUDA kernel for BEV).

    Args:
        boxes1: (N, 7) tensor [x, y, z, w, l, h, yaw] where z is box bottom.
        boxes2: (M, 7) tensor [x, y, z, w, l, h, yaw].

    Returns:
        (N, M) tensor of 3D IoU values. On CPU or if CUDA unavailable,
        returns zeros of the correct shape.
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    # If CUDA not available or boxes on CPU, return zeros.
    if not IOU3D_CUDA_AVAILABLE or not boxes1.is_cuda:
        return boxes1.new_zeros((N, M))

    # Extract components: [x, y, z, w, l, h, yaw]
    x1, y1, z1, w1, l1, h1, yaw1 = (
        boxes1[:, 0],
        boxes1[:, 1],
        boxes1[:, 2],
        boxes1[:, 3],
        boxes1[:, 4],
        boxes1[:, 5],
        boxes1[:, 6],
    )
    x2, y2, z2, w2, l2, h2, yaw2 = (
        boxes2[:, 0],
        boxes2[:, 1],
        boxes2[:, 2],
        boxes2[:, 3],
        boxes2[:, 4],
        boxes2[:, 5],
        boxes2[:, 6],
    )

    # BEV: construct [x, y, w, l, yaw] and convert to XYXYR
    bev1_xywhr = torch.stack([x1, y1, w1, l1, yaw1], dim=1)
    bev2_xywhr = torch.stack([x2, y2, w2, l2, yaw2], dim=1)

    bev1_xyxyr = xywhr2xyxyr(bev1_xywhr)
    bev2_xyxyr = xywhr2xyxyr(bev2_xywhr)

    # Call CUDA kernel for BEV overlap (intersection area)
    overlaps_bev = boxes_overlap_bev(bev1_xyxyr, bev2_xyxyr)

    # Height overlap: z is bottom, top = z + h
    # z1_min, z1_max shape (N, 1)
    z1_min = z1[:, None]
    z1_max = z1[:, None] + h1[:, None]
    # z2_min, z2_max shape (1, M)
    z2_min = z2[None, :]
    z2_max = z2[None, :] + h2[None, :]

    # Pairwise height overlap
    max_of_min = torch.max(z1_min, z2_min)
    min_of_max = torch.min(z1_max, z2_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3D overlap = BEV overlap * height overlap
    overlaps_3d = overlaps_bev * overlaps_h

    # Volumes
    vol1 = (w1 * l1 * h1)[:, None]
    vol2 = (w2 * l2 * h2)[None, :]

    # 3D IoU
    iou3d = overlaps_3d / torch.clamp(vol1 + vol2 - overlaps_3d, min=1e-8)

    return iou3d


if __name__ == "__main__":
    # Self-test: xywhr2xyxyr
    result = xywhr2xyxyr(torch.tensor([[0.0, 0.0, 2.0, 4.0, 0.0]]))
    expected = torch.tensor([[-1.0, -2.0, 1.0, 2.0, 0.0]])
    assert torch.allclose(result, expected), f"xywhr2xyxyr test failed: {result}"
    print("xywhr2xyxyr test passed")

    # Self-test: bbox_overlaps_3d CPU fallback
    boxes1 = torch.randn(2, 7)
    boxes2 = torch.randn(3, 7)
    iou = bbox_overlaps_3d(boxes1, boxes2)
    assert iou.shape == (2, 3), f"Shape mismatch: {iou.shape}"
    assert torch.allclose(iou, torch.zeros(2, 3)), "CPU fallback should return zeros"
    print("bbox_overlaps_3d CPU fallback test passed")

    print("All tests passed!")
