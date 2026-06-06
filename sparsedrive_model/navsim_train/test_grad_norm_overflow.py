"""CPU-only regression tests for the grad-norm overflow / diagnostic bug.

Run:  uv run python sparsedrive_model/navsim_train/test_grad_norm_overflow.py

Background
----------
On an 8-GPU run, training was fine for ~78600 iters, then EVERY iteration printed
"largest finite grad-norm parameters:" with tiny, healthy-looking grads while the
progress bar showed grad_norm=33.689. Root cause (reproduced here without a GPU or
the dataset):

1. A few det-head parameters develop large-but-FINITE grads (~1e18). fp32
   ``clip_grad_norm_`` squares-and-sums them; the global sum-of-squares exceeds the
   fp32 max (3.4e38) and saturates to +inf even though no grad ELEMENT is non-finite.
2. The runner reads that inf as "NaN/inf gradients", and because ``clip_grad_norm_``
   scaled every grad by max_norm/inf = 0, the per-parameter diagnostic that runs
   afterward sees all-zero (finite) grads -> finds no offender -> prints the
   misleading "largest finite grad-norm" list.
3. The displayed grad_norm was stale (gated on a successful step).

The fix computes the total norm in fp64 (overflow ceiling ~1.8e308), so a large but
finite gradient yields a finite norm and is clipped & stepped normally. A genuine
NaN/Inf grad element still produces a non-finite norm and is still caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
for _p in (str(_THIS.parents[2]), str(_THIS.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sparsedrive_model.navsim_train.optim import clip_grad_norm, compute_grad_norm


class _Model(torch.nn.Module):
    """Minimal container so clip_grad_norm(model, ...) works; grads set manually."""

    def __init__(self, shapes):
        super().__init__()
        self.ps = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(s)) for s in shapes]
        )

    def set_grads(self, fill):
        for p in self.ps:
            p.grad = torch.full_like(p, fill)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


def test_large_finite_grads_dont_overflow():
    """The exact reported regime: per-param norm finite, fp32 global sum overflows."""
    print("test_large_finite_grads_dont_overflow")
    # 60 params, each grad element 6e17 over 100 elems -> per-param norm 6e18 (finite),
    # global sum-of-squares = 60*(6e18)^2 = 2.16e39 > fp32 max -> fp32 reduction = inf.
    shapes = [(100,)] * 60
    fill = 6e17

    # 1) Confirm the OLD behavior really overflows (this is the bug).
    m_old = _Model(shapes)
    m_old.set_grads(fill)
    old_norm = torch.nn.utils.clip_grad_norm_(m_old.parameters(), max_norm=1.0, norm_type=2.0)
    _assert(not torch.isfinite(old_norm), f"fp32 clip_grad_norm_ overflows to non-finite (got {float(old_norm)})")
    _assert(
        all(bool(torch.isfinite(p.grad).all()) for p in m_old.parameters()),
        "every per-param grad is finite -> old diagnostic would find no offender (the misfire)",
    )

    # 2) The NEW fp64 norm stays finite and clips correctly.
    m_new = _Model(shapes)
    m_new.set_grads(fill)
    new_norm = clip_grad_norm(m_new, max_norm=1.0, norm_type=2.0)
    _assert(torch.isfinite(new_norm), f"fp64 clip_grad_norm stays finite (got {float(new_norm):.3e})")
    # Expected total norm = sqrt(60) * 6e18.
    expected = (60 ** 0.5) * 6e18
    rel_err = abs(float(new_norm) - expected) / expected
    _assert(rel_err < 1e-3, f"fp64 norm matches analytic value (rel_err={rel_err:.2e})")
    # After clipping to max_norm=1.0, the global norm of the clipped grads is ~1.0.
    clipped_global = compute_grad_norm(list(m_new.parameters()), 2.0)
    _assert(abs(float(clipped_global) - 1.0) < 1e-3, f"clipped grads have global norm ~1.0 (got {float(clipped_global):.4f})")
    _assert(
        all(bool(torch.isfinite(p.grad).all()) for p in m_new.parameters()),
        "clipped grads remain finite and non-zero (not annihilated by inf-scale)",
    )
    _assert(float(m_new.ps[0].grad.abs().max()) > 0, "clipped grads are non-zero (the step can proceed)")


def test_normal_grads_unchanged():
    """Healthy small grads below max_norm are left alone (no spurious clipping)."""
    print("test_normal_grads_unchanged")
    m = _Model([(10,)] * 3)
    m.set_grads(0.01)  # global norm = sqrt(3*10)*0.01 ~ 0.0548 < 1.0
    before = m.ps[0].grad.clone()
    norm = clip_grad_norm(m, max_norm=1.0, norm_type=2.0)
    _assert(torch.isfinite(norm) and float(norm) < 1.0, f"small grad norm is finite and < max_norm (got {float(norm):.4f})")
    _assert(torch.equal(m.ps[0].grad, before), "grads below max_norm are not modified")


def test_real_nan_still_detected():
    """A genuine NaN grad element still yields a non-finite norm (real explosion)."""
    print("test_real_nan_still_detected")
    m = _Model([(5,)] * 2)
    m.set_grads(0.1)
    m.ps[0].grad[0] = float("nan")
    norm = clip_grad_norm(m, max_norm=1.0, norm_type=2.0)
    _assert(not torch.isfinite(norm), "NaN grad element -> non-finite norm (still caught by the guard)")
    # On a non-finite norm we deliberately DO NOT clip, so the offending grad is
    # left intact for the per-parameter diagnostic to find.
    _assert(bool(torch.isnan(m.ps[0].grad).any()), "non-finite path leaves grads intact for the diagnostic")


def test_real_inf_still_detected():
    print("test_real_inf_still_detected")
    m = _Model([(5,)] * 2)
    m.set_grads(0.1)
    m.ps[1].grad[2] = float("inf")
    norm = clip_grad_norm(m, max_norm=1.0, norm_type=2.0)
    _assert(not torch.isfinite(norm), "Inf grad element -> non-finite norm (still caught by the guard)")


def test_returns_on_grad_device():
    """Regression: the fp64 accumulator must live on the grads' device.

    Under DDP each rank's grads are on a distinct cuda:N. An accumulator built on
    CPU made ``total += param_norm`` raise "Expected all tensors to be on the same
    device, but found at least two devices, cuda:N and cpu". On a CUDA box this
    runs the real check; on CPU it at least pins the contract that the returned
    norm is on the same device as the grads.
    """
    print("test_returns_on_grad_device")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    for dev in devices:
        m = _Model([(10,)] * 3).to(dev)
        m.set_grads(0.5)
        norm = compute_grad_norm(list(m.parameters()), 2.0)
        _assert(
            norm.device == m.ps[0].grad.device,
            f"[{dev}] norm returned on grads' device ({norm.device})",
        )
        # The clip path must also not raise across devices.
        clipped = clip_grad_norm(m, max_norm=1.0, norm_type=2.0)
        _assert(torch.isfinite(clipped), f"[{dev}] clip_grad_norm runs without device mismatch")


if __name__ == "__main__":
    test_large_finite_grads_dont_overflow()
    test_normal_grads_unchanged()
    test_real_nan_still_detected()
    test_real_inf_still_detected()
    test_returns_on_grad_device()
    print("\nAll grad-norm overflow regression tests passed.")
