#!/bin/bash
# NCCL smoke on 2 H200s with random data (AR_SMOKE=1): DDP, DDP+ckpt, FSDP2, FSDP2+ckpt. ~5 minutes.
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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -2
export AR_SMOKE=1 AR_TIME_BUDGET=5 AR_WARMUP_STEPS=2 AR_COMPILE=0 AR_RESULTS_DIR=results/smoke
rc=0
for cfg in "ddp 0" "ddp 1" "fsdp 0" "fsdp 1"; do
  read -r d c <<< "$cfg"; set -- "$d" "$c"
  ar_log "=== smoke $1 act_ckpt=$2 ==="
  AR_DIST=$1 AR_ACT_CKPT=$2 AR_TAG="smoke_$1_ckpt$2" "$AR_VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 train.py 2>&1 | tail -40 || { rc=1; ar_log "FAILED $cfg"; }
done
ar_log "smoke rc=$rc"; exit $rc
