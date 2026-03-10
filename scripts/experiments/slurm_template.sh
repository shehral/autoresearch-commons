#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=autoresearch_CONDITION_SEED
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# === EDIT THESE ===
CONDITION="condB"
SEED="1"
AGENT_NUM="1"  # For multi-agent conditions
# ==================

# Load modules (Explorer HPC)
module load explorer 2>/dev/null || true

# Activate environment
export PATH="$HOME/.local/bin:$PATH"

# Create logs directory
mkdir -p logs

# Navigate to the right directory
REPO_ROOT="$HOME/autoresearch-commons"
cd "$REPO_ROOT"

# Load the prompt template for this condition
PROMPT_FILE="scripts/experiments/prompts/${CONDITION}.txt"
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt template not found: $PROMPT_FILE"
    exit 1
fi
PROMPT=$(cat "$PROMPT_FILE")
# Substitute placeholders
PROMPT="${PROMPT//\{SEED\}/$SEED}"
PROMPT="${PROMPT//\{AGENT_NUM\}/$AGENT_NUM}"

echo "=== Starting $CONDITION seed $SEED agent $AGENT_NUM ==="
echo "Prompt file: $PROMPT_FILE"
date

# Launch Claude Code in non-interactive mode
claude --dangerously-skip-permissions -p "$PROMPT"

echo "=== Finished ==="
date
