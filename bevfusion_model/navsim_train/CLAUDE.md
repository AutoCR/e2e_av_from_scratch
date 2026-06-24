# CLAUDE.md — BEVFusion/NAVSIM Training Harness

This file governs how Claude Code drives the **long-running, autonomous** BEVFusion
detection training on the remote GPU server. It is the operating manual for the whole
training campaign: how to launch, monitor, correct, record, and ship changes.

> Scope: this file applies whenever the task is "train / babysit / fix the BEVFusion
> NAVSIM detection run". It does **not** override the repo-root or `bevfusion_model/`
> CLAUDE.md; it adds to them.

---

## 0. The one rule that matters

**You edit code on the LOCAL machine, commit it to git, push, then `git pull` on the
remote server. You NEVER hand-edit code on the remote.** The remote working tree must
stay a clean checkout of an pushed commit (plus untracked outputs). Every behavioural
change to training is a commit with a matching entry in the change log
(`bevfusion_model/navsim_train/CHANGELOG_TRAINING.md`).

If you ever find yourself running `Edit`/`Write` against a path under
`pnc@server:~/Code/e2e_av_from_scratch/...` for a *tracked* file — stop. That is a bug.

---

## 1. Environment & invariants

| Thing | Value |
|---|---|
| Remote host | `pnc@172.16.110.212` port `23230` (passwordless SSH key works; password `senior_pnc` is the fallback) |
| Remote repo | `~/Code/e2e_av_from_scratch` |
| Branch | `feat_bevfusion` (local and remote MUST track the same branch) |
| Origin | `git@github.com:AutoCR/e2e_av_from_scratch.git` |
| GPUs | 8× RTX 3090 (24 GB). **Use only 6** per power limit → `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` |
| Python | `uv run python ...` (never bare `python`; never install deps without explicit user OK) |
| Output dir | `bevfusion_model/outputs/train_navsim/` (checkpoints `iter_*.pth`, `last.pth`, TB under `tb/`) |
| Dataset | NAVSIM: train=85109, val=18179, test=12146 samples. Single-frame detection. |
| Data roots | `OPENSCENE=/prediction_database/navsim`, `NUPLAN_MAPS=/prediction_database/nuplan/dataset/maps` |

Per-rank batch 4 → global batch 24 on 6 GPUs → **3546 iters/epoch**. Keep these in mind
when reading `iter_*` numbers: `epoch = iter / 3546`.

### Allocator / OOM facts (do not relearn the hard way)
- Stay on the **default** CUDA caching allocator. `expandable_segments` is rejected on
  this platform; `cudaMallocAsync` leaks under DDP. (See header of `train_navsim.py`.)
- Per-rank batch 4 OOMs under DDP at the backward spike on some configs; **batch 3 is the
  safe fallback** (leaves ~3 GiB headroom). If you hit OOM, first drop `total_batch_size`
  4 → 3, not allocator flags.
- **Eval (`_try_loss_eval`) is disabled** because it OOMs even with its own OOM guard.
  Re-enabling eval is a tracked work item (§6), not something to flip on casually.

---

## 1.5 Survivability — training must outlive BOTH this laptop and any monitor

There are **three independent failure domains**. Training must survive all of them. Do not
conflate them — the fixes are different.

| If this dies… | Training should… | Mechanism |
|---|---|---|
| **A. This local laptop** (reboot, sleep, network drop) | keep running, untouched | Training runs on the *remote* and must NOT be a child of the SSH session that launched it. |
| **B. This Claude Code session** (closed, crashed, context lost) | keep running, untouched | Same as A — the monitor is decoupled from the training process. A monitor dying never touches training. |
| **C. The remote server itself** (reboot, power loss) | auto-restart and resume from `last.pth` | `@reboot` cron on the remote (no sudo needed). |

### A & B — decouple training from any session (the critical rule)
The launch command in §4 **must** detach training from the SSH/login session, or killing
your terminal kills training. Two acceptable methods, in order of preference:

1. **tmux (preferred — lets you reattach and read the live console):**
   ```bash
   ssh -p 23230 pnc@172.16.110.212
   tmux new -s bevtrain        # or: tmux attach -t bevtrain
   cd ~/Code/e2e_av_from_scratch
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run torchrun --nproc_per_node=6 \
     bevfusion_model/train_navsim.py 2>&1 | tee bevfusion_model/outputs/train_navsim/console_$(date +%Y%m%d_%H%M%S).log
   # detach with Ctrl-b d — training keeps running after you disconnect.
   ```
   The tmux server is owned by `pnc`, not by your SSH connection, so it survives the SSH
   drop and this laptop rebooting. Reattach any time from a fresh session.

2. **nohup + setsid (no reattach, but fully detached):**
   ```bash
   ssh -p 23230 pnc@172.16.110.212 'cd ~/Code/e2e_av_from_scratch && \
     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 setsid nohup uv run torchrun --nproc_per_node=6 \
     bevfusion_model/train_navsim.py \
     > bevfusion_model/outputs/train_navsim/console_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &'
   ```
   `setsid` + `</dev/null` fully orphans it from the SSH session so the connection closing
   can't SIGHUP it.

> **Never** launch training in the foreground of an interactive SSH call from Claude. If
> the session that ran the command ends, an un-detached child dies. This is the #1 way to
> accidentally lose a multi-day run.

### C — survive a remote reboot (auto-resume)
After Phase A makes `resume_from` point at `last.pth` (always-overwritten latest), install a
**guarded `@reboot` cron** that relaunches only if no training is already running and a
checkpoint exists. A small idempotent wrapper script (committed under
`bevfusion_model/navsim_train/`, e.g. `autostart_train.sh`) should:
1. `pgrep -f train_navsim.py` → exit if already running (don't double-launch).
2. Confirm `last.pth` exists and `resume_from` is set to it.
3. `cd` to the repo and launch the §4 detached command, logging to a timestamped console
   file and an `autostart.log`.

Install (no sudo): `crontab -e` →
```
@reboot sleep 60 && /home/pnc/Code/e2e_av_from_scratch/bevfusion_model/navsim_train/autostart_train.sh >> /home/pnc/Code/e2e_av_from_scratch/bevfusion_model/outputs/train_navsim/autostart.log 2>&1
```
The `sleep 60` lets the GPU/driver and filesystems settle after boot. Writing this wrapper +
cron is a **tracked change** (changelog + commit). The wrapper is a real tool, so it lives
in git and is pulled to the remote like any other code.

> **Linger caveat (only if you choose user-systemd instead of cron):** `Linger=no` on this
> box, so a `systemctl --user` service stops when the last session closes. Enabling it needs
> `sudo loginctl enable-linger pnc`, and there is **no passwordless sudo** — you'd have to
> ask the user to run that one command. `@reboot` cron needs no sudo, so **prefer cron.**

### Checkpoint cadence is the real safety net
None of the above matters without frequent checkpoints. The runner saves `iter_*.pth` +
overwrites `last.pth` every `ckpt_epoch_interval` (currently 1 epoch ≈ 3546 iters ≈ ~1 h on
6 GPUs). A crash/reboot therefore costs at most ~1 epoch. If you want a tighter floor, lower
`ckpt_epoch_interval` to e.g. 0.25 (4 ckpts/epoch) — but mind disk (§2.4, each is ~370 MB).
Always resume from `last.pth` unless it is the corrupt/NaN one, in which case use the latest
good `iter_*.pth`.

---

## 2. Standard operating loop (every monitoring tick)

Run this sequence each time you check in. Keep it cheap — it is mostly read-only SSH.

1. **Is it alive?** `ssh ... 'pgrep -af train_navsim.py | head'` and
   `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv`.
   - 6 python procs + ~6 GPUs busy → healthy. 0 procs → crashed (§4). <6 GPUs busy → a
     rank died (§4).
2. **Progress + loss.** Read the latest TB events file, not the console log. The console
   log is tqdm carriage-return spam (an 80 MB file is normal and near-ungreppable). Parse
   TB scalars (`train/loss_total`, `train/loss_cls`, `train/loss_heatmap`,
   `train/loss_bbox`, `train/grad_norm`, `train/lr`) with the standalone parser in §7.
3. **Health checks** (the real signal — total loss alone hides everything):
   - `loss_heatmap` must **fall below ~1** within the first few epochs. If it sticks near
     ~2.9 (the from-scratch plateau) the model is not localizing → §5.
   - `loss_bbox` should fall well below the old ~8.5 plateau.
   - `grad_norm` should sit comfortably under the clip (35), not ride it. Sustained
     clipping or NaN → §5.
   - `lr` should follow the cosine schedule you expect for the configured epoch count.
   - **No NaN/Inf** in any scalar. NaN loss → §5 (stop, don't let it burn GPU-days).
4. **Disk.** `ssh ... 'df -h ~/Code; du -sh ~/Code/e2e_av_from_scratch/bevfusion_model/outputs/train_navsim'`.
   Each checkpoint is ~370 MB; a 100-epoch run wrote 100+ of them. **Prune old `iter_*.pth`**
   (keep `last.pth` + every Nth) if the disk crosses ~80%.
5. **Record** anything notable in the change log; if you changed code, it must already be
   committed+pushed+pulled (§3).

Pace the loop with the scheduler: a healthy run only needs a check every ~20–30 min. Use
`ScheduleWakeup` with a long fallback; do not poll every minute.

### 2.5 How to run the monitor with `/loop`

`/loop` is what makes the §2 tick *recurring and unattended* from a Claude session. Key
mental model: **`/loop` monitors; it does not host the training.** Training lives on the
remote (§1.5). The loop is just a recurring check-in that reads remote state over SSH and
intervenes when needed. If the loop stops, training keeps going — you only lose monitoring,
not the run.

**Start it** (fixed cadence, recommended for steady-state):
```
/loop 20m Run the BEVFusion training monitor tick from bevfusion_model/navsim_train/CLAUDE.md §2: check liveness, parse latest TB scalars, run the health checks, prune checkpoints if disk >80%, and autonomously correct per §5 (recording every change per §3). If training is finished or the curves are healthy and converged, report and stop.
```
Or **self-paced** (let the model choose the next delay based on what it sees — e.g. tighten
to a few minutes right after a restart, relax to 30 min when healthy):
```
/loop Monitor the BEVFusion training per bevfusion_model/navsim_train/CLAUDE.md §2; pace yourself, longer when healthy, shorter just after any intervention.
```

**Cadence guidance for the loop:** 20–30 min in steady state. Tighten to ~5 min for the
first few ticks after a (re)launch or any correction, to confirm the fix took. Don't poll
sub-minute — checkpoints are hourly, so nothing changes that fast.

**When the loop session itself dies** (laptop reboot, Claude closed): training is
unaffected (§1.5 A/B). To resume *monitoring*, just open a new Claude session in this repo
and re-issue the `/loop` command above — it re-reads the same CLAUDE.md and picks up the
live remote state. Nothing about the loop is stateful on the training side; it's safe to
start, stop, and restart the monitor at will.

**The loop must stay idempotent and non-destructive:** every tick re-derives state from the
remote (pgrep + TB + df). It must never assume what it did last tick. Before any restart it
checks `pgrep -f train_navsim.py` first so two overlapping ticks can't double-launch.

**For true zero-human autonomy** (survives the laptop being off for days), `/loop` from a
laptop is not enough — the laptop must be on for the monitor to tick. The *training* still
survives via §1.5, but unattended *correction* would need the monitor to run somewhere
always-on. Options, in order: (a) keep the laptop on and `/loop` running; (b) run the
monitor as a scheduled cloud agent (`/schedule`) so it ticks independent of the laptop;
(c) put a minimal liveness-watchdog cron on the remote itself (relaunch-if-dead), which
covers crash-recovery but not recipe-level corrections. The remote `@reboot` cron (§1.5 C)
already covers reboot-recovery regardless of the monitor.

---

## 3. The change protocol (MANDATORY for every modification)

Whenever you change *anything* that affects training (code, config, launch command):

1. **Edit locally** with `Edit`/`Write` in `/home/chenran/Code/AutoCR/e2e_av_from_scratch`.
2. **Append a change-log entry** to `bevfusion_model/navsim_train/CHANGELOG_TRAINING.md`
   using the template in §3.1. One entry per change, newest at top.
3. **Commit** on `feat_bevfusion` with a clear message ending in the Co-Authored-By line.
   The commit should include both the code change and the changelog entry.
4. **Push**: `git push origin feat_bevfusion`.
5. **Pull on remote**: `ssh ... 'cd ~/Code/e2e_av_from_scratch && git pull --ff-only'`.
   If the remote pull is not fast-forward (someone touched a tracked file there), STOP and
   reconcile — never `git reset --hard` away unknown remote changes without inspecting them.
6. **Restart training** only if the change must take effect now (most do). Resume from the
   right checkpoint (§4) so no progress is lost.

Never skip steps 2–4. "A quick fix" with no changelog entry and no commit is forbidden —
this run is multi-day and unattended; the changelog is the only audit trail.

### 3.1 Changelog entry template
```
## <YYYY-MM-DD HH:MM> — <one-line summary>
- **Commit:** <short sha>
- **Why:** <what symptom / goal triggered this>
- **What:** <files + the precise change>
- **Effect / how to verify:** <metric or behaviour that should change, and how you'll confirm>
- **Restart:** <resumed from which checkpoint, or "no restart">
```

---

## 4. Launch & resume

### Launch (6-GPU DDP) — ALWAYS detached, see §1.5
Use the **tmux** (preferred) or **setsid+nohup** launch from §1.5. Both detach training
from the SSH session so this laptop or this Claude session dying cannot kill it. Never run
training in the foreground of an interactive SSH call. After launch, confirm 6 procs come
up (`pgrep -af train_navsim.py`) and TB starts writing before walking away.

### Resume after any stop
- All training params live in `bevfusion_model/configs/bevfusion_hyperparams.py`.
- Set `RUNTIME_CONFIG["resume_from"]` to the latest good `iter_*.pth` (or `last.pth`).
- The runner tracks progress in **samples_seen**, so resuming under the same 6-GPU/batch-4
  setup continues the cosine schedule correctly. It loads model+optimizer+scaler but
  **recomputes the LR from `samples_seen`** (scheduler state is intentionally not restored).
- The startup line prints `Resumed from ...: samples_seen=N -> start iter X/Y`. Verify X is
  where you expect before walking away.

### Warm-start (distinct from resume — see the Plan)
Warm-starting from the official `model_weights/bevfusion/bevfusion-det.pth` is a code
change (a `pretrained_from` path that loads at build time and drops the 10-class head
output layers), NOT the `resume_from` mechanism. It starts at iter 0. Until that code
exists, do not try to point `resume_from` at the official checkpoint — it will silently
mis-load the head.

---

## 5. Exception playbook (you are FULLY AUTONOMOUS — act, then record)

You may correct both infra and recipe issues without asking. **Every correction gets a
changelog entry + commit (if code) before or immediately after acting.** Bias toward
preserving GPU-time and never letting a divergent run burn for hours.

| Symptom | Diagnosis | Action |
|---|---|---|
| 0 python procs, no new TB points | Crash (OOM, NCCL timeout, node hiccup, exception) | Read tail of console log for the traceback. Fix root cause if code; else just resume from `last.pth`. Record. |
| <6 GPUs busy, run "hung" | One rank died; NCCL watchdog will eventually abort | Kill all 6 procs (`pkill -f train_navsim.py`), resume from `last.pth`. If recurrent, drop batch 4→3. |
| CUDA OOM in traceback | Memory pressure | `total_batch_size` 4→3 first. Do NOT touch allocator flags (proven dead ends). Resume. |
| `loss=nan`/`inf` | LR too high, bad batch, fp16 overflow | Stop immediately. Lower peak LR (e.g. halve), and/or check `fp16_loss_scale`. Resume from the last pre-NaN checkpoint, not the NaN one. |
| `grad_norm` pinned at clip (35) for long | LR/init mismatch | If from-scratch and heatmap stuck → this is the warm-start problem; see Plan. Otherwise consider lowering LR. |
| `loss_heatmap` flat ~2.9 across epochs | Model not localizing (the original bug) | This is the known from-scratch failure. The fix is **warm-start + lr=2e-4 + fewer epochs**, per the Plan — apply it. |
| Loss flat but not NaN, lr already tiny | Cosine over-long; late epochs wasted | Don't waste days. Stop, shorten `num_epochs`, restart or just end if converged. |
| Disk >85% | Too many checkpoints | Prune `iter_*.pth` (keep `last.pth` + sparse milestones). Record what was deleted. |
| Remote `git pull` not fast-forward | Remote tree diverged | Inspect with `git status`/`git log`; reconcile deliberately. Never blind `reset --hard`. |

After any restart, re-run the §2 health checks within the next tick to confirm the fix took.

---

## 6. Success criteria & open work items

**User's stated success metric: detection/PDM evaluation.** Important caveat the harness
must respect: this port has **no mAP/NDS/PDM detection metric implemented** — the only eval
that exists is **loss-eval** on val/test, and it is **disabled because it OOMs**. So
"pass eval" currently has no green check to hit. Tracked items, in order:

1. **Warm-start + recipe fix** (the actual cause of slow convergence) — see Plan §A.
2. **Re-enable eval without OOM** — `_try_loss_eval` already guards OOM but was disabled;
   re-enabling needs memory released around eval (rank-0 only, smaller `eval_max_batches`,
   `torch.cuda.empty_cache()` + `no_grad` already present). Validate it doesn't kill the run.
3. **A real detection metric** (mAP/NDS or PDM) is a larger task — flag to the user before
   building; loss-eval is the interim proxy. Until then, the operational success signal is
   the §2 health checks (heatmap < ~1, bbox well below 8.5, stable grad_norm, no NaN).

Do not silently treat "loss went down" as "eval passed". State plainly which signal you
have.

---

## 7. Tooling notes

- **TB parsing without tensorboard installed:** tensorboard isn't available locally. Parse
  the TFRecord-framed events file with a small standalone protobuf reader (Event →
  Summary → Value{tag, simple_value}). Keep such a helper under
  `bevfusion_model/tools/` if you write one, and commit it (it's a real tool, log it).
- **Console log:** dominated by tqdm `\r` updates; `grep` over it is slow and noisy. Prefer
  TB scalars. To read a traceback, `tail -c 200000` the latest `console_*.log` and look past
  the progress bars.
- **`check_navsim_dataset_size.py`** and **`validate_navsim_train_logs.py`** already exist
  for sanity-checking the dataset and log integrity — use them rather than reinventing.
- **Don't** run foreground `sleep` to wait; use `ScheduleWakeup` (long fallback, 1200s+) or
  a background `Monitor` until-loop.

---

## 8. Quick reference

```bash
# health
ssh -p 23230 pnc@172.16.110.212 'pgrep -af train_navsim.py; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
# disk
ssh -p 23230 pnc@172.16.110.212 'du -sh ~/Code/e2e_av_from_scratch/bevfusion_model/outputs/train_navsim; df -h ~/Code'
# pull latest TB events locally (run dir name varies — list tb/ first)
scp -P 23230 'pnc@172.16.110.212:~/Code/e2e_av_from_scratch/bevfusion_model/outputs/train_navsim/tb/<run>/events.out.tfevents.*' /tmp/
# kill + resume
ssh -p 23230 pnc@172.16.110.212 'pkill -f train_navsim.py'   # then set resume_from, relaunch (§4)
```
