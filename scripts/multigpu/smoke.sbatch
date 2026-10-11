#!/bin/bash
# GPU smoke on 2 H200s: contained probes first (torch+CUDA, module import with SDPA, FA3 attempt),
# then the four distributed configs on random data (AR_SMOKE=1). ~5 minutes.
#SBATCH --job-name=ar-smoke
#SBATCH --partition=gpu-short
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=00:30:00
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
PY="$AR_VENV/bin/python"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -2
probe() { local what=$1; shift; ar_log "--- probe: $what"; "$@" 2>&1 | tail -6; ar_log "    rc=${PIPESTATUS[0]}"; }
probe "torch + CUDA matmul" "$PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print('matmul ok', float((x@x).sum()))"
probe "platform_utils import, SDPA forced" env AR_ATTN=sdpa "$PY" -c "import platform_utils as p; print('backend', p.get_attention_backend_name())"
probe "platform_utils import, FA3 attempt (informational)" env AR_ATTN=auto "$PY" -c "import platform_utils as p; print('backend', p.get_attention_backend_name())"
export AR_SMOKE=1 AR_TIME_BUDGET=5 AR_WARMUP_STEPS=2 AR_COMPILE=0 AR_RESULTS_DIR=results/smoke AR_ATTN="${AR_ATTN:-sdpa}"
ar_log "distributed smokes with AR_ATTN=$AR_ATTN"
rc=0
for cfg in "ddp 0" "ddp 1" "fsdp 0" "fsdp 1"; do
  read -r d c <<< "$cfg"
  ar_log "=== smoke $d act_ckpt=$c ==="
  AR_DIST=$d AR_ACT_CKPT=$c AR_TAG="smoke_${d}_ckpt${c}" "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 train.py 2>&1 | grep -v "^\[rank" | tail -25 || { rc=1; ar_log "FAILED $cfg"; }
done
ar_log "smoke rc=$rc"; exit $rc
