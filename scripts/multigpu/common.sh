# Shared settings for the multi-GPU benchmark on Northeastern Explorer (sourced by the sbatch files).
AR_ROOT="${AR_ROOT:-/projects/interpretability/shehral.m/autoresearch}"
AR_REPO="$AR_ROOT/repo"
AR_VENV="$AR_ROOT/.venv"
AR_BASE_PY="${AR_BASE_PY:-/projects/interpretability/shehral.m/conda_envs/monitors-cu128/bin/python}"  # torch 2.11.0+cu128 (driver 570)
AR_CACHE="/scratch/shehral.m/autoresearch-cache"   # prepare.py data + tokenizer (scratch is purged monthly)
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$AR_ROOT/logs" "$AR_CACHE"
# prepare.py writes to ~/.cache/autoresearch; point it at scratch.
mkdir -p ~/.cache && { [ -L ~/.cache/autoresearch ] || { rm -rf ~/.cache/autoresearch 2>/dev/null; ln -s "$AR_CACHE" ~/.cache/autoresearch; }; }
ar_log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }
