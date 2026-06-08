"""CUDA-accelerated BEV pooling op (ported from MIT HAN Lab BEVFusion).

This is the fast path used in place of the pure-PyTorch ``bev_pool_pure``
scatter when a CUDA device is available. The C++/CUDA sources under ``src/``
are compiled lazily (JIT) on first use via ``torch.utils.cpp_extension.load``.

On platforms without CUDA (e.g. macOS ARM64) the extension is never built and
``BEV_POOL_CUDA_AVAILABLE`` stays ``False``; callers fall back to the pure
PyTorch implementation.

Coordinate / shape convention (identical to ``bev_pool_pure``):
    feats:  (n, c)       per-frustum-point features
    coords: (n, 4)       integer grid coords as (x, y, z, batch)
    output: (B, D, H, W, C)  -- D is the Z (height) dimension

The kernel sums all feature vectors that fall into the same (batch, z, x, y)
cell, i.e. it is the accumulate-style scatter that ``index_put_(...,
accumulate=True)`` performs in the pure path.
"""

import os

import torch

__all__ = ["bev_pool", "BEV_POOL_CUDA_AVAILABLE", "load_bev_pool_ext"]

_BEV_POOL_EXT = None


def _cuda_buildable() -> bool:
    """Return True only when a CUDA toolchain + device are usable."""
    return torch.cuda.is_available() and torch.version.cuda is not None


def load_bev_pool_ext():
    """JIT-compile and return the ``bev_pool_ext`` module.

    Compilation is attempted only when CUDA is available. The compiled module
    is cached for the process lifetime. Returns ``None`` if CUDA is
    unavailable or compilation fails (caller should fall back to pure path).
    """
    global _BEV_POOL_EXT
    if _BEV_POOL_EXT is not None:
        return _BEV_POOL_EXT
    if not _cuda_buildable():
        return None

    from torch.utils.cpp_extension import load

    src_dir = os.path.join(os.path.dirname(__file__), "src")
    try:
        _BEV_POOL_EXT = load(
            name="bev_pool_ext",
            sources=[
                os.path.join(src_dir, "bev_pool_cpu.cpp"),
                os.path.join(src_dir, "bev_pool_cuda.cu"),
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
        print(f"[bev_pool] CUDA extension build failed, using pure fallback: {exc}")
        _BEV_POOL_EXT = None
    return _BEV_POOL_EXT


# Whether the CUDA fast path can be used at all on this platform.
BEV_POOL_CUDA_AVAILABLE = _cuda_buildable()


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        ext = load_bev_pool_ext()
        assert ext is not None, "bev_pool_ext not available"

        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ext = load_bev_pool_ext()
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    """Pool per-point features into a BEV grid via the CUDA kernel.

    Args:
        feats: (n, C) float features per frustum point.
        coords: (n, 4) integer grid coords as (x, y, z, batch).
        B, D, H, W: output batch, Z(depth), and BEV spatial dims.

    Returns:
        (B, C, D, H, W) BEV features (Z dim kept; caller collapses it).
    """
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
