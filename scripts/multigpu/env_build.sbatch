#!/bin/bash
# Build the benchmark venv on top of the cu128 torch env (no torch download).
#SBATCH --job-name=ar-env
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/interpretability/shehral.m/autoresearch/logs/%x-%j.out
set -eo pipefail
source "${AR_REPO:-/projects/interpretability/shehral.m/autoresearch/repo}/scripts/multigpu/common.sh"
ar_log "base python: $AR_BASE_PY"
[ -x "$AR_VENV/bin/python" ] || "$AR_BASE_PY" -m venv --system-site-packages "$AR_VENV"
"$AR_VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$AR_VENV/bin/python" -m pip install rustbpe tiktoken pyarrow requests kernels pytest 2>&1 | tail -3
"$AR_VENV/bin/python" - <<'PY'
import torch, pyarrow, tiktoken, rustbpe
print("torch", torch.__version__, "cuda", torch.version.cuda, "| pyarrow", pyarrow.__version__)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
print("fsdp2 + checkpoint_wrapper import ok")
PY
"$AR_VENV/bin/python" -c "import torch.distributed.run; print(\"torch.distributed.run ok\")"
ar_log "env ready: $AR_VENV"
