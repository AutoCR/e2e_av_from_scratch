# BEVFusion/NAVSIM Training Campaign — Plan

Companion to `bevfusion_model/navsim_train/CLAUDE.md` (the operating manual). This is the
*what and why* of the campaign; the CLAUDE.md is the *how to operate it day to day*.

## Problem statement

The prior 100-epoch run trained BEVFusion **from random init** at lr=1e-4 and barely
converged: `loss_heatmap` flat at ~2.9 and `loss_bbox` only crept 14→8.5, plateauing by
~epoch 30 while the cosine LR decayed to near-zero for the remaining 70 epochs. Root cause
is **no warm-start + too-low LR for from-scratch + too many epochs**, not dataset size
(85k train samples is adequate).

Confirmed feasible: the official `model_weights/bevfusion/bevfusion-det.pth` **can**
warm-start this repo despite NAVSIM using 8 cameras vs nuScenes' 6 — the camera count is
folded into the batch dimension (`img.view(B*N, C, H, W)`) and never appears in any weight
shape. Only the head's 10-class output layers mismatch the 5-class NAVSIM head and must be
dropped on load.

## Decisions (from user)

- **GPUs:** `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` (6 of 8; power limit).
- **Validation gate:** short smoke run first, then full run.
- **Autonomy:** fully autonomous corrections (infra + recipe); record every change.
- **Success metric:** detection/PDM eval — but eval is currently disabled (OOMs) and no
  detection metric is implemented; interim signal is loss-eval + health checks. See
  CLAUDE.md §6.

## Phases

### Phase A — Warm-start + recipe fix (code change)
1. Add a `pretrained_from` config key (distinct from `resume_from`). At model-build time:
   load `bevfusion-det.pth` with `strict=False`, **filter out tensors whose shape doesn't
   match** the 5-class model (the head heatmap/class output layers), and log loaded vs
   skipped keys. Start at **iter 0**.
2. Recipe: peak **lr 1e-4 → 2e-4**, **num_epochs 100 → ~36** (revisit after seeing the
   curve), keep warmup ~500–1000 iters, keep grad clip 35.
3. Commit + changelog + push + pull (CLAUDE.md §3).

### Phase B — Smoke validation (no commit unless it reveals a bug)
1. Launch a short run (~300–500 iters) on 6 GPUs.
2. **Pass criteria:** warm-start log shows the camera+lidar+decoder keys loaded and only the
   head class layers skipped; no shape errors; `loss_heatmap` already trending **below the
   2.9 plateau**; no NaN; 6 GPUs busy; ~stable step time.
3. If it fails, diagnose (CLAUDE.md §5), fix, re-smoke. Only proceed when green.

### Phase B.5 — Survivability setup (before the long run; CLAUDE.md §1.5)
The full run is multi-day and unattended, so set up resilience to all three failure domains
*before* launching it:
1. **Detach training from any session** — launch via tmux or setsid+nohup so this laptop or
   the Claude session dying never kills it (§1.5 A/B).
2. **Survive a remote reboot** — commit an idempotent `autostart_train.sh` wrapper and
   install a `@reboot` cron on the remote (no sudo) that relaunches-if-dead and resumes from
   `last.pth` (§1.5 C). This is a tracked change (changelog + commit).
3. **Tighten checkpoint cadence if desired** — `ckpt_epoch_interval` controls the worst-case
   lost-progress window (currently ~1 epoch ≈ 1 h). Always resume from `last.pth`.

### Phase C — Full run + babysitting
1. Launch the full ~36-epoch run **detached** (tmux/setsid — §1.5), 6 GPUs.
2. Monitor with **`/loop`** on a ~20–30 min cadence (CLAUDE.md §2.5). The loop only
   *monitors*; training is decoupled, so the loop stopping never stops training. Autonomously
   correct per §5, recording every change.
3. Prune checkpoints to keep disk < ~80%.
4. Watch for the real win: heatmap loss < ~1, bbox well below 8.5, grad_norm under clip,
   cosine LR not flat-lining early.

### Phase D — Eval (best-effort, flagged)
1. Re-enable loss-eval guarded against OOM (rank-0 only, small `eval_max_batches`,
   empty_cache around it). Validate it doesn't kill the run.
2. A true detection metric (mAP/NDS or PDM) is a separate, larger task — surface to the
   user before building it. Until then, loss-eval + health checks are the acceptance proxy.

## Risks / watch-list
- **Rig geometry:** warm-started camera branch was tuned to the nuScenes rig; NAVSIM's
  8-cam geometry differs. Weights still load and transfer, but don't freeze the camera
  branch — let it adapt. Verify NAVSIM `camera2lidar`/intrinsics are correct (wrong
  extrinsics scatter features to wrong BEV cells regardless of warm-start).
- **OOM at batch 4 under DDP:** fall back to batch 3 (CLAUDE.md §1), never allocator flags.
- **Multi-day unattended:** the changelog + commits are the only audit trail — keep them
  current.
- **Session/laptop death ≠ training death:** only if training is launched detached (§1.5).
  A foreground SSH launch is the #1 way to lose the run — never do it.
- **`/loop` needs a host:** a laptop-run loop only ticks while the laptop is on. Training
  survives regardless (§1.5), but unattended *correction* needs the monitor running
  somewhere always-on — laptop on, a `/schedule` cloud agent, or a remote liveness cron.

## Definition of done
- Full run completes (or is deliberately stopped at convergence) with healthy curves.
- Every modification is committed on `feat_bevfusion` with a matching changelog entry.
- Final state summarized to the user: best checkpoint, final losses, eval status, and any
  open items (notably: real detection metric not yet implemented).
