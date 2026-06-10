"""CPU-only regression test for the refinement-center gradient-explosion fix.

Run:  uv run python sparsedrive_model/navsim_train/test_refine_center_clamp.py

Background
----------
SparseDrive stage-1 training on NAVSIM ran healthy for ~10% of iters, then
EVERY optimizer step was skipped: the pre-clip grad norm exploded to 4.9e7 ..
1.5e17 (finite but enormous) and tripped the explosion guard (skip_norm=25000).

The frozen kmeans anchor (det/map_anchor_grad=False) was NOT the cause this
time. The cause is ``SparseBox3DRefinementModule``: it adds an *unbounded*
trainable MLP delta to the box centre X/Y/Z at every decoder layer
(detection3d_blocks.py). That centre flows into ``project_points`` whose
backward is proportional to 1/z^2 (blocks.py:237). When the refined centre
drifts so a keypoint projects near the camera plane, the 1/z^2 backward blows
the grad norm into the 1e7-1e17 range.

The fix clamps the refined centre to the metric scene range
(|x|,|y|<=100 m, |z|<=10 m) right after the anchor-add. Real NAVSIM objects are
well inside that box, so valid detections are untouched; only runaway drift is
capped, which removes the projection cliff and lets training proceed.

This test reproduces the explosion through the real project_points math and
proves the clamp tames the gradient while leaving in-range centres exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
for _p in (str(_THIS.parents[2]), str(_THIS.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sparsedrive_model.sparsedrive.blocks import DeformableFeatureAggregation as DFA


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


def _identity_projection(bs, num_cam):
    """A projection matrix that maps (x, y, z, 1) -> image (x, y, z): z stays the
    camera-frame depth, so project_points divides x,y by z exactly as in the
    real pipeline. This isolates the 1/z^2 backward without a real calibration."""
    mat = torch.zeros(bs, num_cam, 4, 4)
    for i in range(4):
        mat[:, :, i, i] = 1.0
    return mat


def _project_center_z(center_z, drift):
    """Run a single keypoint (at planar x=y=1) with depth = center_z + drift
    through project_points and return (points_2d, grad wrt center_z)."""
    bs, num_anchor, num_cam = 1, 1, 1
    z = (center_z + drift).reshape(1, 1, 1)
    key_points = torch.stack(
        [torch.ones_like(z), torch.ones_like(z), z], dim=-1
    )  # (bs, num_anchor, num_pts=1, 3)
    proj = _identity_projection(bs, num_cam)
    pts2d = DFA.project_points(key_points, proj)  # (bs, num_cam, num_anchor, num_pts, 2)
    loss = pts2d.pow(2).sum()
    loss.backward()
    return pts2d.detach(), center_z.grad.detach().clone()


def test_unclamped_center_explodes_grad():
    """Without the clamp, a centre drifting toward the camera plane (z->0.1)
    produces a 1/z^2 gradient cliff: the grad magnitude is enormous."""
    print("test_unclamped_center_explodes_grad")
    # Drift the centre so depth lands right at the z-floor (0.1 m).
    center = torch.tensor(0.1, requires_grad=True)
    _, grad = _project_center_z(center, drift=torch.tensor(0.0))
    # d/dz (1/z)^2-style term at z=0.1 is O(1/z^3) ~ 1e3+; the point is it is huge.
    _assert(
        float(grad.abs()) > 1e3,
        f"near the z-floor the unclamped centre grad is huge (got {float(grad.abs()):.3e})",
    )


def test_clamp_keeps_inrange_center_exact():
    """The clamp range (|z|<=10) must leave valid centres bit-exact: a real box
    at z=1.5 m passes through clamp(-10, 10) unchanged with gradient 1.0."""
    print("test_clamp_keeps_inrange_center_exact")
    z = torch.tensor(1.5, requires_grad=True)
    clamped = z.clamp(min=-10.0, max=10.0)
    _assert(float(clamped) == 1.5, "in-range centre passes the clamp unchanged")
    clamped.backward()
    _assert(float(z.grad) == 1.0, "in-range centre keeps gradient 1.0 (no distortion)")


def test_clamp_blocks_runaway_center_grad():
    """A runaway centre (z far outside the range) is saturated by the clamp, so
    its gradient is zero -> it cannot feed the 1/z^2 projection cliff. This is
    exactly what stops the explosion from a drifted refinement output."""
    print("test_clamp_blocks_runaway_center_grad")
    z = torch.tensor(500.0, requires_grad=True)  # absurd refined depth
    clamped = z.clamp(min=-10.0, max=10.0)
    _assert(float(clamped) == 10.0, "runaway centre is saturated to the clamp bound")
    # Feed the *clamped* (bounded) value into projection: depth is now 10 m, far
    # from the z-floor, so the projection backward is small and finite.
    z2 = clamped.detach().clone().requires_grad_(True)
    _, grad = _project_center_z(z2, drift=torch.tensor(0.0))
    _assert(
        float(grad.abs()) < 1.0,
        f"after clamping, the projection grad is small (got {float(grad.abs()):.3e})",
    )
    # And the clamp itself zeroes the gradient to the runaway parameter.
    clamped.backward()
    _assert(float(z.grad) == 0.0, "saturated clamp stops grad flow to the runaway centre")


def test_refinement_module_applies_clamp():
    """End-to-end: SparseBox3DRefinementModule must bound its output centre.

    Force the MLP to emit a huge centre and confirm the module clamps it to the
    metric range, regardless of the (frozen) anchor prior."""
    print("test_refinement_module_applies_clamp")
    from sparsedrive_model.sparsedrive.detection3d_blocks import SparseBox3DRefinementModule
    from sparsedrive_model.sparsedrive.box3d import X, Y, Z

    torch.manual_seed(0)
    mod = SparseBox3DRefinementModule(embed_dims=32, output_dim=11, with_cls_branch=False)
    # Drive the final Linear to produce a large positive output so the centre
    # would explode without the clamp.
    with torch.no_grad():
        for m in mod.layers.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
                m.bias.fill_(1e3)
    bs, num_anchor = 2, 4
    feat = torch.randn(bs, num_anchor, 32)
    anchor = torch.zeros(bs, num_anchor, 11)
    anchor_embed = torch.zeros(bs, num_anchor, 32)
    output, _, _ = mod(feat, anchor, anchor_embed, return_cls=False)
    _assert(
        float(output[..., [X, Y]].abs().max()) <= 100.0 + 1e-4,
        f"refined x,y are clamped to <=100 m (got {float(output[..., [X, Y]].abs().max()):.3e})",
    )
    _assert(
        float(output[..., Z].abs().max()) <= 10.0 + 1e-4,
        f"refined z is clamped to <=10 m (got {float(output[..., Z].abs().max()):.3e})",
    )


if __name__ == "__main__":
    test_unclamped_center_explodes_grad()
    test_clamp_keeps_inrange_center_exact()
    test_clamp_blocks_runaway_center_grad()
    test_refinement_module_applies_clamp()
    print("\nAll refinement-center clamp regression tests passed.")
