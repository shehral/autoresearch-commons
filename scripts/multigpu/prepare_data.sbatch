#!/bin/bash
# Download climbmix shards and train the tokenizer (CPU, proxy on). ~30 shards is enough for a
# 4-GPU run at a 2-minute budget without cycling the data more than once.
#SBATCH --job-name=ar-prepare
#SBATCH --partition=short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/interpretability/shehral.m/autoresearch/logs/%x-%j.out
set -eo pipefail
source "${AR_REPO:-/projects/interpretability/shehral.m/autoresearch/repo}/scripts/multigpu/common.sh"
cd "$AR_REPO"
ar_log "prepare: ${AR_NUM_SHARDS:-30} shards -> $AR_CACHE"
"$AR_VENV/bin/python" prepare.py --num-shards "${AR_NUM_SHARDS:-30}" --download-workers 8
du -sh "$AR_CACHE"/data "$AR_CACHE"/tokenizer
ar_log "prepare done"
