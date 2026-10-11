#!/bin/bash
# The measured grid: strong scaling at a fixed 524K-token global batch, 2-minute budget per run.
# Explorer note (2026-10-10): the gpu partition's QOS caps every job at ONE GPU (per-user total 4) and
# the multigpu partition is closed to this account, so the default is two A100s on gpu-short
# (2 GPUs per job and per user, 2 h wall). Override at submit time (e.g. --gres=gpu:h200:2) and set
# AR_MAX_WS=4 on a cluster that grants four GPUs to one job; the ws4 rows then run instead of ws2_ckpt.
# Micro-batch (2026-10-11): at the default 128 x 2048 tokens per rank the eager rows OOM on 80 GB A100s
# (72.7 GB in use when an 8 GiB allocation failed; compile peaked at 44 GB, checkpointing at 38 GB), so the
# grid runs AR_DEVICE_BATCH=64 (1 GPU: 64 x 4 accumulation, 2 GPUs: 64 x 2; same 524K global batch).
#SBATCH --job-name=ar-grid
#SBATCH --partition=gpu-short
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/interpretability/shehral.m/autoresearch/logs/%x-%j.out
set -o pipefail
source "${AR_REPO:-/projects/interpretability/shehral.m/autoresearch/repo}/scripts/multigpu/common.sh"
cd "$AR_REPO"
# Race guard: when two copies of this job are queued on different GPU types, the first to start
# claims the lock, cancels its pending siblings (same job name, same user) and the rest exit at once.
if [ -n "${AR_LOCK:-}" ]; then
  mkdir -p "$AR_ROOT/locks"
  if ! mkdir "$AR_ROOT/locks/$AR_LOCK" 2>/dev/null; then ar_log "lock $AR_LOCK already claimed; exiting"; exit 0; fi
  squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -o %i | grep -vx "$SLURM_JOB_ID" | xargs -r scancel 2>/dev/null || true
  ar_log "claimed lock $AR_LOCK on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -4
NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
export AR_TIME_BUDGET="${AR_TIME_BUDGET:-120}" AR_RESULTS_DIR=results/multigpu AR_ATTN="${AR_ATTN:-sdpa}"
export AR_DEVICE_BATCH="${AR_DEVICE_BATCH:-64}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
AR_MAX_WS="${AR_MAX_WS:-$NGPU}"
ar_log "grid with AR_ATTN=$AR_ATTN, device batch $AR_DEVICE_BATCH, budget ${AR_TIME_BUDGET}s per run, $NGPU GPUs visible, max world size $AR_MAX_WS"
PY="$AR_VENV/bin/python"; TR="$AR_VENV/bin/python -m torch.distributed.run --standalone"
run() { tag=$1; shift; ar_log "=== $tag ==="; AR_TAG=$tag "$@" 2>&1 | tail -60 || ar_log "FAILED $tag"; }
run single_compile1 env AR_COMPILE=1 $PY train.py
run single_compile0 env AR_COMPILE=0 $PY train.py
run ddp_ws2         env AR_DIST=ddp  AR_COMPILE=0 $TR --nproc_per_node=2 train.py
if [ "$AR_MAX_WS" -ge 4 ]; then
  run ddp_ws4       env AR_DIST=ddp  AR_COMPILE=0 $TR --nproc_per_node=4 train.py
  run ddp_ws4_ckpt  env AR_DIST=ddp  AR_COMPILE=0 AR_ACT_CKPT=1 $TR --nproc_per_node=4 train.py
  run fsdp_ws4      env AR_DIST=fsdp AR_COMPILE=0 $TR --nproc_per_node=4 train.py
  run fsdp_ws4_ckpt env AR_DIST=fsdp AR_COMPILE=0 AR_ACT_CKPT=1 $TR --nproc_per_node=4 train.py
else
  run ddp_ws2_ckpt  env AR_DIST=ddp  AR_COMPILE=0 AR_ACT_CKPT=1 $TR --nproc_per_node=2 train.py
  run fsdp_ws2      env AR_DIST=fsdp AR_COMPILE=0 $TR --nproc_per_node=2 train.py
  run fsdp_ws2_ckpt env AR_DIST=fsdp AR_COMPILE=0 AR_ACT_CKPT=1 $TR --nproc_per_node=2 train.py
fi
$PY scripts/multigpu/summarize.py results/multigpu docs/multigpu_results.md && cat docs/multigpu_results.md
ar_log "grid done"
