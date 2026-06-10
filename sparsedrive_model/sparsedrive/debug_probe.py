"""Lightweight, env-gated numerical debug probe for SparseDrive training.

The recurring "gradient explosion" in from-scratch NAVSIM training is a BACKWARD
Jacobian blowup: the loss stays finite (~22) while grad_norm reaches 1e14. That
means a forward-only NaN check is insufficient -- the offending value is a
near-zero camera-frame depth `z` feeding the `x/z` perspective division in
DeformableFeatureAggregation.project_points, whose backward scales as x/z^2 and
explodes even though the forward output is finite.

This module instruments that hot path with near-zero overhead when OFF. Enable
with environment variables (set them before launching torchrun):

    SD_DEBUG=1   # forward probes: log z-depth distribution into project_points,
                 # and a one-line min/max/has-nan summary for tagged tensors.
    SD_DEBUG=2   # everything in 1, PLUS torch.autograd.detect_anomaly() around
                 # the training forward+backward (the definitive backward-op
                 # locator; ~2-3x slower, use for a short repro run).

All output goes to stdout via print(flush=True) so it interleaves into the
console log (tee'd to a file). Each line is prefixed "[SD_DEBUG]" for grepping.
Only rank 0 prints by default (set SD_DEBUG_ALL_RANKS=1 to print on every rank).
"""

from __future__ import annotations

import os

import torch

_STEP = 0  # global optimizer-step counter, set by the runner each iteration.


def debug_level() -> int:
    """0 = off, 1 = forward probes, 2 = + autograd anomaly detection."""
    try:
        return int(os.environ.get("SD_DEBUG", "0"))
    except ValueError:
        return 0


def probe_enabled() -> bool:
    return debug_level() >= 1


def anomaly_enabled() -> bool:
    return debug_level() >= 2


def _should_print() -> bool:
    if not probe_enabled():
        return False
    if os.environ.get("SD_DEBUG_ALL_RANKS", "0") == "1":
        return True
    # Default: rank 0 only, to keep the log readable under DDP.
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return rank == "0"


def set_step(step: int) -> None:
    global _STEP
    _STEP = int(step)


def _emit(msg: str) -> None:
    print(f"[SD_DEBUG][iter={_STEP}] {msg}", flush=True)


def log_tensor(name: str, t: torch.Tensor) -> None:
    """One-line finiteness/range summary for a tagged tensor (no-op when off)."""
    if not _should_print():
        return
    with torch.no_grad():
        det = t.detach()
        n_nan = int(torch.isnan(det).sum())
        n_inf = int(torch.isinf(det).sum())
        finite = det[torch.isfinite(det)]
        if finite.numel():
            lo = float(finite.min())
            hi = float(finite.max())
            amax = float(finite.abs().max())
        else:
            lo = hi = amax = float("nan")
        flag = "  <<< NON-FINITE" if (n_nan or n_inf) else ""
        _emit(
            f"{name}: shape={tuple(det.shape)} min={lo:.4g} max={hi:.4g} "
            f"absmax={amax:.4g} n_nan={n_nan} n_inf={n_inf}{flag}"
        )


def probe_projection_depth(z: torch.Tensor, floor: float) -> None:
    """Log the camera-frame depth distribution feeding the perspective divide.

    `z` is points_2d[..., 2:3] (the homogeneous depth) BEFORE clamping. A keypoint
    with z near/below the clamp floor is exactly what drives the x/z^2 backward
    explosion, so this is the leading indicator. Reports how many keypoints are in
    the danger zone; if any are <= 0 (behind the camera) the divide direction is
    already pathological.
    """
    if not _should_print():
        return
    # The 2026-06-10 run showed ~50% of keypoints are behind the camera on EVERY
    # step (normal geometry with 3 front cameras: anchors behind the ego have
    # z<0 for all cams), so "any z below floor" fired 6 lines per iteration and
    # produced a 49k-line log. The genuinely dangerous population is the
    # NEAR-PLANE band 0 < z < 0.1 (on-image with a huge 1/z^2 Jacobian);
    # behind-camera points project far off-image and get zero sampling gradient.
    # Report the band every iteration it is non-empty, plus a rate-limited
    # heartbeat of the full distribution for context.
    with torch.no_grad():
        zf = z.detach().reshape(-1)
        n = zf.numel()
        finite = zf[torch.isfinite(zf)]
        n_nonfinite = n - finite.numel()
        if finite.numel():
            zmin = float(finite.min())
            zmax = float(finite.max())
        else:
            zmin = zmax = float("nan")
        n_nonpos = int((zf <= 0).sum())
        n_danger = int(((zf > 0) & (zf < 0.1)).sum())
        # Healthy baseline (measured on the 2026-06-10 resumed run): ~400-600 of
        # 421k keypoints sit in the 0<z<0.1 band at ALL times -- anchors near the
        # ego whose keypoints pass close to the camera positions. Printing on any
        # non-empty band spams 6 lines/iter, so only alert on a genuine surge
        # (several times baseline) or non-finite depths; otherwise fold the count
        # into a periodic heartbeat so drift remains visible in the log.
        if n_danger >= 2000 or n_nonfinite:
            _emit(
                f"project_points depth: n={n} zmin={zmin:.4g} zmax={zmax:.4g} "
                f"n(0<z<0.1)={n_danger} n(z<=0)={n_nonpos} "
                f"n_nonfinite={n_nonfinite}  <<< near-plane SURGE (1/z^2 cliff)"
            )
        elif _STEP % 100 == 1:
            _emit(
                f"project_points depth heartbeat: n={n} zmin={zmin:.4g} "
                f"zmax={zmax:.4g} n(z<=0)={n_nonpos} n(0<z<0.1)={n_danger}"
            )


def first_nonfinite_param(model) -> None:
    """Print the FIRST parameter (by name) whose grad is non-finite, with path.

    Complements top_grad_norms (which ranks by magnitude): on a true NaN/Inf this
    pinpoints the earliest offender in module order, which is usually closest to
    the source op.
    """
    if not _should_print():
        return
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n_nan = int(torch.isnan(g).sum())
        n_inf = int(torch.isinf(g).sum())
        if n_nan or n_inf:
            _emit(
                f"FIRST non-finite grad: {name} "
                f"shape={tuple(g.shape)} n_nan={n_nan} n_inf={n_inf}"
            )
            return
    _emit("no non-finite grad found (explosion is large-but-finite, not NaN/Inf)")
