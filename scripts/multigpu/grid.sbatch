#!/bin/bash
# The measured grid: strong scaling at a fixed 524K-token global batch, 2-minute budget per run.
#SBATCH --job-name=ar-grid
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=200G
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
export AR_TIME_BUDGET="${AR_TIME_BUDGET:-120}" AR_RESULTS_DIR=results/multigpu AR_ATTN="${AR_ATTN:-sdpa}"
ar_log "grid with AR_ATTN=$AR_ATTN, budget ${AR_TIME_BUDGET}s per run"
PY="$AR_VENV/bin/python"; TR="$AR_VENV/bin/python -m torch.distributed.run --standalone"
run() { tag=$1; shift; ar_log "=== $tag ==="; AR_TAG=$tag "$@" 2>&1 | tail -40 || ar_log "FAILED $tag"; }
run single_compile1 env AR_COMPILE=1 $PY train.py
run single_compile0 env AR_COMPILE=0 $PY train.py
run ddp_ws2         env AR_DIST=ddp  AR_COMPILE=0 $TR --nproc_per_node=2 train.py
run ddp_ws4         env AR_DIST=ddp  AR_COMPILE=0 $TR --nproc_per_node=4 train.py
run ddp_ws4_ckpt    env AR_DIST=ddp  AR_COMPILE=0 AR_ACT_CKPT=1 $TR --nproc_per_node=4 train.py
run fsdp_ws4        env AR_DIST=fsdp AR_COMPILE=0 $TR --nproc_per_node=4 train.py
run fsdp_ws4_ckpt   env AR_DIST=fsdp AR_COMPILE=0 AR_ACT_CKPT=1 $TR --nproc_per_node=4 train.py
$PY scripts/multigpu/summarize.py results/multigpu docs/multigpu_results.md && cat docs/multigpu_results.md
ar_log "grid done"
