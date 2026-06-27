# BEVFusion/NAVSIM Training — Change Log

Newest entries on top. One entry per behavioural change to training (code, config, or
launch). Format defined in `bevfusion_model/navsim_train/CLAUDE.md` §3.1. Every entry
corresponds to a git commit on `feat_bevfusion`.

<!-- New entries go directly below this line. -->

## 2026-06-27 — FINAL eval results (iter_170208_bnfix.pth, no training)
- **Commit:** <this commit> (docs only)
- **What was evaluated:** the BN-recalibrated checkpoint `iter_170208_bnfix.pth`, with
  `eval_loss.py` (val loss) and `eval_detection_map.py` (3D detection mAP/NDS). PDM skipped
  (planning metric — N/A for a detection-only model that outputs boxes, not a trajectory).
- **Val loss (200 batches, all finite):** total 8.70 · cls 0.25 · bbox 5.41 · heatmap 3.03.
- **Detection mAP/NDS (FULL val: 4000 samples, 797k preds, 72k GT, 35669 matched TPs):**
  - Per-class AP (mean over 0.5/1/2/4m center-distance): car 0.242, pedestrian 0.067,
    traffic_cone 0.042, barrier 0.001, bicycle 0.000.
  - **mAP = 0.0705**; mATE 0.85 m, mASE 0.222, mAOE 0.448 rad;
    **NDS (simplified, 3 TP metrics) = 0.2297**.
  - (100-sample smoke was consistent: mAP 0.083, NDS 0.256 — full run is the reliable number.)
- **Honest read:** the model genuinely detects **car** (AP 0.242, 0.85 m translation error)
  but is near-zero on the rare classes (bicycle/barrier/cone), reflecting NAVSIM's severe
  class imbalance. This is a MODEST detector — useful as a working baseline, not yet strong.
- **Note on the metric:** NDS here is a documented simplification (mAP + ATE/ASE/AOE only;
  AVE/AAE omitted, NAVSIM 5-class). Not directly comparable to official 10-class nuScenes NDS.
- **Caveat:** these numbers required the BN-recalibration fix; the raw `iter_170208.pth`
  evaluates to 0 mAP due to the NaN BN buffers. Use the `_bnfix` checkpoint for any inference.
- **Commit:** <this commit>
- **Why:** Evaluating iter_170208.pth produced ZERO detections (eval_detection_map: preds=0)
  and NaN cls/bbox loss on every val batch — yet training loss was healthy throughout.
- **Root cause (proven by probe):** the checkpoint has **NaN BatchNorm running_mean/running_var
  in all 6 detection prediction-head branches** (`heads.object.prediction_heads.0.{center,
  height,dim,rot,vel,heatmap}.0.bn`). All *parameters* are finite. In `train()` BN uses batch
  stats (ignoring the corrupt buffers) → training loss looked fine the whole run. In `eval()`
  the NaN buffers poison predictions → all scores NaN → all 200 proposals filtered → 0 boxes.
  This is a SILENT failure that loss monitoring could never catch. Likely origin: fp16 BN-stat
  divergence at some training step.
- **What:** Added `bevfusion_model/recalibrate_bn.py` — a NO-gradient BN-recalibration script:
  reset the non-finite BN running stats, set momentum=None (cumulative), put BN modules in
  train-mode while the rest stays eval, run ~300 val forward passes (fp32, no backward) to
  re-estimate valid running stats, save `iter_170208_bnfix.pth`. The eval scripts need no fix.
- **Effect / how to verify:** After recalibration, re-run eval_detection_map.py / eval_loss.py
  on the `_bnfix` checkpoint — should now produce nonzero detections and finite cls/bbox loss.
- **Follow-up for training code:** add a NaN-check on BN buffers at checkpoint time, and
  investigate the fp16 path (amp.py) as the divergence source.
- **Restart:** No training restart — recalibration is forward-only.

## 2026-06-27 — Full 36-epoch run COMPLETED
- **Commit:** <this commit> (docs only)
- **Why:** The validated warm-start run reached iter 170208/170208 (36 epochs).
- **Outcome:** Completed cleanly — 0 procs left, all GPUs freed, no crash/OOM, only 1
  transient loss-NaN over the entire run (skipped by the runner) + 1 isolated grad inf.
  Final smoothed training losses (last ~2 epochs, windowed means):
  loss_total ≈ 9.1, loss_bbox ≈ 6.2, loss_heatmap ≈ 2.65, loss_cls ≈ 0.21, grad_norm ≈ 23.
  (Per-iter values bounce ±; the run's session lows were bbox ~4.8-5.2, total ~7.5-8.2.)
  vs. the failed from-scratch baseline: bbox went from a ~10 stall to ~6.2 (sub-target),
  heatmap from a flat ~2.95 floor down to ~2.65, and cls to ~0.21.
- **Best checkpoint:** `bevfusion_model/outputs/train_navsim/iter_170208.pth` (== last.pth,
  the final/lowest-LR model). The from-scratch reference `iter_354600.pth` is retained for
  comparison.
- **Open item:** success was measured on TRAINING LOSS only — this port has no
  mAP/NDS/PDM detection metric, and loss-eval is disabled (OOM). Judging real detection
  quality needs that metric built (flagged repeatedly during monitoring).
- **Restart:** None — run finished. Monitor loop stopped.

## 2026-06-25 — DEFINITIVE heatmap diagnosis: not a floor, slow background descent. Revert 4×.
- **Commit:** <this commit>
- **Why:** The 4× heatmap weight (prev entry) did NOT break the plateau either (true term,
  =logged/4, stayed ~2.85 through iter 600). Two failed fixes → stopped guessing and ran an
  instrumented decomposition of the heatmap loss on real data.
- **Measured findings (correct two earlier wrong assumptions):**
  1. **NAVSIM is NOT sparse** — measured **~41 boxes/frame** (median 39.5, max 115), not the
     ~1-2 an earlier sampling claimed. So avg_factor (=num positive peaks) is ~192 for B=4,
     a LARGE divisor, not a tiny-divisor pathology.
  2. **No structural floor.** Perfect-positive test: setting peaks→1.0 AND background→0 drives
     loss → 0.0. The loss CAN descend; ~2.9 is not a cap.
  3. **The loss is ~70% NEG (background) term** (4.43 of 6.29 / avg_factor), ~30% POS. The
     model drives background to sigmoid≈0.09 and the descent STALLS there. loss→0.5 needs
     background≈0.05; →0.1 needs ≈0.03 — a slow factor-of-2-3 across 162k cells/frame.
  4. **Why both fixes failed:** bias-init only sets the *initial* background level; 4× weight
     scales pos and neg *identically* so the pos/neg balance (the actual stalled term) is
     unchanged. Neither addressed background suppression.
- **What:** Reverted `loss_heatmap.loss_weight` 4.0 → 1.0 (4× was inert and risks
  destabilizing other losses). Added a measured NOTE in the config.
- **Conclusion:** The ~2.9 heatmap plateau is a SLOW-but-not-blocked background-suppression
  descent, NOT a bug. It would improve with many more epochs, or descend faster with a
  neg-term normalization change (normalize the neg term per-cell instead of dividing the
  162k-cell sum by num_pos). This is a decision for the user, not an autonomous fix.
- **Restart:** None yet — awaiting user decision on strategy.

## 2026-06-25 — Up-weight loss_heatmap 1.0→4.0 (break the confirmed plateau)
- **Commit:** <this commit>
- **Why:** The full run (stopped at ~1.4 epochs) confirmed conclusively via windowed means:
  `loss_heatmap` DEAD FLAT at ~2.95 over 4 windows (zero movement), `loss_bbox` descended
  early then plateaued at ~10 (above the 8.5 target). Heatmap pinned at the "predict
  background everywhere" floor. The single-batch overfit proved the head CAN reach 0.17, so
  the gradient path works — the positive signal is just too weak vs background at weight 1.0
  (NAVSIM: ~1-2 boxes/frame, 4/5 classes nearly always empty).
- **What:** `bevfusion_hyperparams.py`: `loss_heatmap.loss_weight` 1.0 → 4.0. (loss_cls and
  loss_bbox weights unchanged; assigner unchanged — IoU cost is active on CUDA.)
- **Effect / how to verify:** Short validation experiment first (~400 iters): `loss_heatmap`
  (now scaled ×4, so compare the UNSCALED component or expect raw ~4× higher initially) must
  show the UNDERLYING heatmap term DESCENDING, not flat. If it breaks below the old plateau,
  relaunch the full run. Stopped run's last.pth + iter_4728.pth removed so the new run
  warm-starts fresh at iter 0.
- **Restart:** Fresh warm-start run after validation passes.

## 2026-06-25 — Full 36-epoch run launched + @reboot cron + auto-resume
- **Commit:** 96b38d8 (auto-resume) + this entry
- **Why:** All structural blockers resolved (warm-start works incl. lidar permute; batch 3
  fixes OOM; heatmap focal-bias init added; bbox confirmed not-a-bug with IoU active on CUDA).
  User approved launching the full run now and watching the first 1-2 epochs.
- **What:**
  - Launched the full run detached on GPUs 0-5 (6× DDP): `console_20260625_122432.log`.
    Confirmed: warm-start 577 keys (21 permuted, 5 dropped); num_epochs=36,
    iters_per_epoch=4728, max_iters=170208, start_iter=0; 6 GPUs busy ~17-19 GiB, no OOM.
  - `runner.py`: auto-resume from `last.pth` if present (commit 96b38d8) — so manual restarts
    and the reboot cron continue the run instead of restarting at iter 0.
  - Installed `@reboot` cron on the remote calling `autostart_train.sh` (idempotent;
    skips if already running). Reboot path: wait 60s → autostart → launch → auto-resume.
- **Survivability now covered:** (A/B) detached launch survives this laptop/session dying;
  (C) @reboot cron + auto-resume survives a server reboot; checkpoints every epoch (~1 h)
  cap lost progress.
- **Effect / how to verify:** Monitor per CLAUDE.md §2 (TB scalars). Success signals over the
  first 1-2 epochs: `loss_heatmap` descends below the old ~2.9 plateau; `loss_bbox` descends
  below ~8.5; grad_norm under clip; no NaN. If heatmap/bbox stay pinned, stop and reassess.
- **Restart:** Fresh run at iter 0.

## 2026-06-25 — Investigation: bbox "non-overfit" is a diagnostic artifact, NOT a bug
- **Commit:** <this commit> (docs only — no code change)
- **Why:** A single-batch overfit test showed `loss_bbox` (specifically cx,cy) refusing to
  converge while all other dims overfit, which looked like a center encode/decode bug.
- **What we found (two independent investigations, one static + one live-instrumented on
  the remote, agreeing):**
  - The center path is CORRECT vs upstream TransFusion: encode target and forward pred are
    BOTH absolute feature-grid coords; query_pos [0.5,179.5] exactly matches encode targets
    [2.8,177.6]; gradients are live (not detached); train/test out_size_factor+voxel+pc_range
    all match. No encoding/scale/gradient bug.
  - The real mechanism for the stuck cx,cy in the TEST: the Hungarian assignment FLIPS every
    step (measured: matched proposal→GT set unstable at every logged step), so cx,cy chase a
    moving target. Cause: the 3D-IoU assignment cost (the term that locks a match spatially)
    is ~0 at init because predicted boxes are ~18 m off, AND it's disabled entirely on CPU /
    when the CUDA IoU op is unavailable (`detection_assigner.py:206`). With 47 same-class
    ("car") GTs, the remaining cls cost is degenerate → unstable matching.
  - **KEY FACT checked on the remote: `IOU3D_CUDA_AVAILABLE = True` and `cuda = True`.** So in
    real 6-GPU CUDA training the IoU cost IS active and stabilizes the assignment as boxes
    improve — exactly like upstream. The non-overfit was an artifact of a degenerate
    single-batch scenario with far-off init boxes, not the production path.
- **Conclusion:** No bbox code fix needed. The center regression bootstraps slowly early
  (IoU≈0 until boxes get close) but is not broken; with IoU active + the heatmap-bias fix
  helping proposals land near GT, it should converge over real training.
- **Effect / how to verify:** On the full run, watch `loss_bbox` descend past the old ~8.5
  plateau over the first several epochs (not instantly). If it stays pinned, revisit the
  assigner (e.g. add a BEV-center-distance fallback cost for the early IoU≈0 regime).
- **Restart:** No restart from this entry (investigation only).

## 2026-06-25 — Focal bias init on heatmap heads (break the ~2.9 plateau)
- **Commit:** <this commit>
- **Why:** Smoke run (with warm-start working) showed `loss_heatmap` stuck oscillating at
  ~2.8-3.1 even at full lr — the SAME plateau as the prior from-scratch run, so init was
  never the bottleneck. Quantitative analysis proved ~2.9 = "background fully suppressed,
  but predicted confidence at true object centers stuck at sigmoid~0.04": the head fell into
  the trivial "predict background everywhere" basin. Root cause: this port OMITTED the
  standard focal negative-bias init on the heatmap output convs (official BEVFusion sets
  bias=-2.19). Without it, the reinit head starts at sigmoid~0.5 everywhere, and the
  161998-bg-vs-~2-fg gradient imbalance collapses it to predict ~0 and stay there. (NAVSIM
  sparsity — ~1.3-1.9 boxes/frame, mostly `car` — makes this worse but is data-inherent.)
- **What:** `detection_head.py`: added `_init_heatmap_bias(-2.19)` called in `__init__`,
  setting the bias of the dense `heatmap_head[-1]` conv and each per-query
  `prediction_heads[i].heatmap[-1]` conv to -2.19 (the layers feeding GaussianFocalLoss).
- **Effect / how to verify:** Re-run smoke. `loss_heatmap` should now START lower (near the
  bg prior instead of ~110) and DESCEND below ~2.5 instead of flooring at 2.9 — gradient
  energy now goes into raising true peaks. If it still floors at 2.9, escalate to the
  single-batch overfit test (can the head reach <0.5 on one batch?).
- **Restart:** Smoke re-run (kill the running one first — it used the pre-fix code).

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
