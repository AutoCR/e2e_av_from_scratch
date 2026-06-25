# BEVFusion/NAVSIM Training — Change Log

Newest entries on top. One entry per behavioural change to training (code, config, or
launch). Format defined in `bevfusion_model/navsim_train/CLAUDE.md` §3.1. Every entry
corresponds to a git commit on `feat_bevfusion`.

<!-- New entries go directly below this line. -->

## 2026-06-25 — Fix smoke-run failures: DDP OOM + lidar warm-start layout
- **Commit:** <this commit>
- **Why:** First smoke run on the 6× RTX 3090 server surfaced two issues:
  (1) **OOM at the first backward** with `total_batch_size=4` under DDP (~1.8 GiB free/rank)
      — exactly the failure the existing config comment warned about.
  (2) The warm-start dropped **21 `encoders.lidar.backbone.*` sparse-conv weights** as
      shape-mismatches, so the LiDAR branch was training from scratch — undercutting the
      warm-start. Cause: the official checkpoint stores spconv weights as `[kD,kH,kW,in,out]`
      but real spconv-cu120 registers `.weight` as `[out,kD,kH,kW,in]` (same numel, different
      order), so the plain shape-equality filter dropped them.
- **What:**
  - `bevfusion_hyperparams.py`: `total_batch_size` 4→3 (updated the comment with the
    server-confirmed OOM evidence).
  - `runner.py` warm-start loop: for lidar-backbone conv `.weight` keys that match the model
    in numel but not shape, try a small set of candidate 5-D permutes and accept the FIRST
    that matches the model param shape EXACTLY; otherwise drop as before. Self-validating —
    a wrong layout guess yields no match and the key is simply skipped, never corrupted.
    Log now reports how many keys loaded via permute.
- **Effect / how to verify:** Re-run smoke. Warm-start print should now read
  `loaded N keys (21 via spconv-layout permute), dropped 5 (...)` — i.e. the 21 lidar convs
  reconciled and only the 5 class-dependent head tensors dropped. No OOM at batch 3; 6 GPUs
  engage; heatmap loss should fall below the ~2.9 plateau.
- **Restart:** Smoke re-run, then full run at iter 0 (still a fresh campaign).

## 2026-06-24 — Warm-start from official checkpoint + recipe fix + autostart wrapper
- **Commit:** <this commit>
- **Why:** The prior 100-epoch from-scratch run barely converged (heatmap loss flat
  ~2.9, bbox plateaued ~8.5 by epoch 30, then 70 epochs wasted on a near-zero cosine LR).
  Root cause: no warm-start + too-low LR (1e-4) for from-scratch + too many epochs.
- **What:**
  - `runner.py`: new warm-start block after `model.to(device)` (before DDP wrap). Loads an
    external checkpoint, keeps only keys present in the model with matching shape, drops the
    rest, loads `strict=False`, logs loaded/dropped/missing on rank 0. Guarded by
    `pretrained_from and not resume_from` (resume wins). Path resolved vs `_REPO_ROOT`.
  - `bevfusion_hyperparams.py`: `lr` 1e-4→2e-4; `num_epochs` 100→36; `resume_from`→None;
    added `pretrained_from` = absolute path to `model_weights/bevfusion/bevfusion-det.pth`.
  - `autostart_train.sh` (new): idempotent `/bin/sh` wrapper for a `@reboot` cron — sets
    cron-safe env (uv abs path, CUDA), skips if already running, launches 6-GPU DDP detached.
- **Effect / how to verify:** Verified offline that exactly 578/583 keys load identically and
  only 5 class-dependent head tensors mismatch (10→5 classes) and are dropped. Startup will
  print `Warm-start ...: loaded 578 keys, dropped 5, 5 missing (reinitialized)`. Success
  signal: `loss_heatmap` should drop below the ~2.9 plateau within the first epoch (vs flat
  from scratch). Smoke run validates before the full run.
- **Restart:** Fresh run at iter 0 (this is a new campaign, not a resume).

## 2026-06-24 — Harness bootstrap (docs only, no training change)
- **Commit:** <fill on commit>
- **Why:** Set up an autonomous, auditable, multi-day training campaign per user request.
- **What:** Added `bevfusion_model/navsim_train/CLAUDE.md` (operating manual),
  `CHANGELOG_TRAINING.md` (this file), and `bevfusion_model/navsim_train/TRAINING_PLAN.md`
  (the campaign plan). No code or config touched; training behaviour unchanged.
- **Effect / how to verify:** None on training. Docs only.
- **Restart:** No restart.
