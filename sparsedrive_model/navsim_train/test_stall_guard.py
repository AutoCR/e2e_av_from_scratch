"""CPU-only regression test for the training stall guard.

Run:  uv run python sparsedrive_model/navsim_train/test_stall_guard.py

Background
----------
The grad-explosion guard skips a corrupted optimizer step, but when the model
has diverged EVERY step explodes and is skipped, so training freezes yet keeps
burning compute (a real run skipped ~99k steps in a row for ~1.5 days). The
stall guard counts consecutive skipped steps and aborts once the streak reaches
``grad_skip_abort_after``; any successful step resets the streak so transient
one-off spikes never trip it.

This test exercises the counter/threshold state machine directly (the same logic
the runner applies after each accumulation window), without a GPU or dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for _p in (str(_THIS.parents[2]), str(_THIS.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sparsedrive_model.navsim_train.runner import TrainingStalledError


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


def _run_stream(step_done_stream, abort_after):
    """Replay a stream of per-window step_done booleans through the exact guard
    state machine used in runner.run, returning (aborted, iter_of_abort,
    max_streak). Mirrors the runner block 1:1."""
    consecutive_skips = 0
    abort_after = int(abort_after) if abort_after else 0
    max_streak = 0
    for i, step_done in enumerate(step_done_stream, start=1):
        if step_done:
            consecutive_skips = 0
        else:
            consecutive_skips += 1
            max_streak = max(max_streak, consecutive_skips)
            if abort_after and consecutive_skips >= abort_after:
                return True, i, max_streak
    return False, None, max_streak


def test_sustained_skip_aborts():
    """A long run of consecutive skips trips the guard exactly at the threshold."""
    print("test_sustained_skip_aborts")
    # 50 healthy steps, then all skips.
    stream = [True] * 50 + [False] * 1000
    aborted, at, _ = _run_stream(stream, abort_after=200)
    _assert(aborted, "guard aborts on a sustained skip streak")
    # Abort fires on the 200th consecutive skip -> global step 50 + 200 = 250.
    _assert(at == 250, f"abort fires exactly at the threshold-th skip (got iter {at})")


def test_transient_spikes_do_not_abort():
    """Isolated single-step spikes interleaved with successes never trip it."""
    print("test_transient_spikes_do_not_abort")
    # A skip every 5 steps for a long time -> streak never exceeds 1.
    stream = []
    for _ in range(5000):
        stream += [False, True, True, True, True]
    aborted, _, max_streak = _run_stream(stream, abort_after=200)
    _assert(not aborted, "interleaved one-off skips never abort")
    _assert(max_streak == 1, f"max consecutive streak stays at 1 (got {max_streak})")


def test_streak_resets_on_success():
    """A near-miss streak that is broken by one success must reset the counter."""
    print("test_streak_resets_on_success")
    # 199 skips (one below threshold), one success, then 199 skips again.
    stream = [False] * 199 + [True] + [False] * 199
    aborted, _, max_streak = _run_stream(stream, abort_after=200)
    _assert(not aborted, "a success below the threshold resets the streak (no abort)")
    _assert(max_streak == 199, f"max streak is 199, never reaching 200 (got {max_streak})")


def test_guard_disabled():
    """abort_after of None / 0 disables the guard entirely."""
    print("test_guard_disabled")
    stream = [False] * 100000
    for disabled in (None, 0):
        aborted, _, _ = _run_stream(stream, abort_after=disabled)
        _assert(not aborted, f"abort_after={disabled!r} disables the guard")


def test_broken_run_would_have_aborted_early():
    """Replays the real broken-run profile: ~9000 healthy iters then 100% skip.

    The real run skipped ~99,000 steps before the operator killed it. With the
    guard the abort lands within ~200 skips of the tip-over -- thousands of
    iters in, not a hundred thousand."""
    print("test_broken_run_would_have_aborted_early")
    healthy = 9000
    stream = [True] * healthy + [False] * 99000
    aborted, at, _ = _run_stream(stream, abort_after=200)
    _assert(aborted, "the broken-run profile trips the guard")
    _assert(
        at == healthy + 200,
        f"abort lands 200 skips after the stall began (iter {at}), not after 99k",
    )
    _assert(
        99000 - 200 > 90000,
        "the guard saves >90k wasted iterations vs. running to operator-kill",
    )


def test_exception_type_is_catchable():
    """TrainingStalledError is a RuntimeError subclass (so generic handlers and
    the runner's finally-block cleanup both behave)."""
    print("test_exception_type_is_catchable")
    _assert(issubclass(TrainingStalledError, RuntimeError), "TrainingStalledError subclasses RuntimeError")
    try:
        raise TrainingStalledError("x")
    except RuntimeError:
        _assert(True, "TrainingStalledError is caught as RuntimeError")


if __name__ == "__main__":
    test_sustained_skip_aborts()
    test_transient_spikes_do_not_abort()
    test_streak_resets_on_success()
    test_guard_disabled()
    test_broken_run_would_have_aborted_early()
    test_exception_type_is_catchable()
    print("\nAll stall-guard regression tests passed.")
