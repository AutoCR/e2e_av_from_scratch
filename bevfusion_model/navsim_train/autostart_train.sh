#!/bin/sh
# =============================================================================
# autostart_train.sh
# -----------------------------------------------------------------------------
# WHAT THIS DOES:
#   Idempotent wrapper to (re)launch the BEVFusion 6-GPU DDP training detached.
#   Intended to be invoked by a @reboot cron job on the remote host
#   (pnc@172.16.110.212) and also usable for manual relaunches.
#
# IDEMPOTENCY:
#   If a training process (matched by `pgrep -f train_navsim.py`) is already
#   running, this script logs a "already running, skipping" line and exits 0
#   without launching a second instance. This makes it safe to call repeatedly
#   (e.g. on every reboot or from a watchdog).
#
# RESUME BEHAVIOR:
#   This script does NOT set or control resume. Whether training warm-starts
#   fresh or resumes from the latest checkpoint is decided entirely by
#   RUNTIME_CONFIG["resume_from"] inside the training config. The script just
#   launches train_navsim.py; the config decides resume vs. fresh start.
#
# ENVIRONMENT:
#   Runs under /bin/sh with a minimal cron environment (no zsh/bash rc files
#   are sourced). All required PATH / LD_LIBRARY_PATH / CUDA env is set
#   explicitly below. uv is not on the default PATH, so its absolute path is
#   used.
#
# INTENDED CRON LINE (on the remote host):
#   @reboot sleep 60 && /home/pnc/Code/e2e_av_from_scratch/bevfusion_model/navsim_train/autostart_train.sh >> /home/pnc/Code/e2e_av_from_scratch/bevfusion_model/outputs/train_navsim/autostart.log 2>&1
# =============================================================================

set -e

# --- Explicit environment (cron gives us almost nothing) ---------------------
export HOME=/home/pnc
export PATH="/usr/local/cuda_pnc/bin:/home/pnc/.local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda_pnc/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

REPO=/home/pnc/Code/e2e_av_from_scratch
OUT="$REPO/bevfusion_model/outputs/train_navsim"
UV=/home/pnc/.local/bin/uv

# Timestamped logger helper.
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*"
}

# --- Idempotency guard -------------------------------------------------------
# pgrep returns non-zero when nothing matches; guard it so `set -e` is happy.
if pgrep -f train_navsim.py >/dev/null 2>&1; then
    log "train_navsim.py already running, skipping launch."
    exit 0
fi

# --- Safety checks -----------------------------------------------------------
cd "$REPO" || { log "ERROR: cannot cd into REPO=$REPO"; exit 1; }

if [ ! -x "$UV" ]; then
    log "ERROR: uv binary not found or not executable at $UV"
    exit 1
fi

# Ensure the output dir exists so the log file can be written.
mkdir -p "$OUT" || { log "ERROR: cannot create OUT=$OUT"; exit 1; }

# --- Launch detached ---------------------------------------------------------
LOG="$OUT/console_$(date +%Y%m%d_%H%M%S).log"

setsid nohup "$UV" run torchrun --nproc_per_node=6 \
    bevfusion_model/train_navsim.py > "$LOG" 2>&1 < /dev/null &

log "launched, pid=$!, log=$LOG"
exit 0
