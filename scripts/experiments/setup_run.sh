#!/usr/bin/env bash
set -euo pipefail

# Usage: setup_run.sh <condition> <seed> [--resume]
# Conditions: condA, condB, condC, condD, condE, condF, condG
#             ablation_G-cov, ablation_G-lin, ablation_G-dir, ablation_G-syn, ablation_G-struct

CONDITION="${1:?Usage: setup_run.sh <condition> <seed> [--resume]}"
SEED="${2:?Usage: setup_run.sh <condition> <seed> [--resume]}"
RESUME="${3:-}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECTS_DIR="${EXPERIMENT_BASE_DIR:-$(dirname "$REPO_ROOT")/experiment_runs}"

# Load condition config (bash 3.2 compatible)
source "$(dirname "$0")/condition_config.sh"
get_config "$CONDITION"

RUN_DIR="$PROJECTS_DIR/${CONDITION}_seed${SEED}"

echo "=== Setting up $CONDITION seed $SEED ==="
echo "Agents: $NUM_AGENTS | Experiments/agent: $NUM_EXPERIMENTS | Shared KB: $IS_SHARED | Director: $HAS_DIRECTOR"

# Create run directory
mkdir -p "$RUN_DIR"

# Helper: reset knowledge base
reset_kb() {
    local dir="$1"
    rm -f "$dir/knowledge/cards/"*.json
    echo '{"version":1,"last_updated":null,"cards":[]}' > "$dir/knowledge/index.json"
    echo '{"version":1,"last_updated":null,"experiments":[]}' > "$dir/knowledge/queue.json"
    rm -f "$dir/knowledge/synthesis/"*.md
}

if [ "$NUM_AGENTS" -eq 1 ]; then
    # Single agent: just create a branch
    cd "$REPO_ROOT"
    BRANCH="experiment/${CONDITION}_seed${SEED}"
    git checkout master
    git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

    if [ "$RESUME" != "--resume" ]; then
        reset_kb "$REPO_ROOT"
        git add -A && git commit -m "reset knowledge base for ${CONDITION}_seed${SEED}" --allow-empty
    fi

    # For condition A: copy the stripped program.md
    if [ "$CONDITION" = "condA" ]; then
        cp "$REPO_ROOT/scripts/experiments/condA_program.md" "$REPO_ROOT/program.md"
        git add program.md && git commit -m "use stripped program.md for condA (no knowledge base)"
    fi

    echo "Ready. Working directory: $REPO_ROOT (branch: $BRANCH)"

elif [ "$IS_SHARED" = "false" ]; then
    # Isolated multi-agent: create N separate repo copies
    for i in $(seq 1 "$NUM_AGENTS"); do
        AGENT_DIR="$RUN_DIR/agent${i}"
        if [ ! -d "$AGENT_DIR" ]; then
            git clone "$REPO_ROOT" "$AGENT_DIR"
            cd "$AGENT_DIR"
            git checkout -b "experiment/${CONDITION}_seed${SEED}_agent${i}"
        else
            cd "$AGENT_DIR"
        fi
        if [ "$RESUME" != "--resume" ]; then
            reset_kb "$AGENT_DIR"
            git add -A && git commit -m "reset KB for ${CONDITION}_seed${SEED}_agent${i}" --allow-empty
        fi
        echo "Agent $i: $AGENT_DIR"
    done

else
    # Shared multi-agent: main repo + worktrees
    cd "$REPO_ROOT"
    BRANCH="experiment/${CONDITION}_seed${SEED}"
    git checkout master
    git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

    if [ "$RESUME" != "--resume" ]; then
        reset_kb "$REPO_ROOT"
        git add -A && git commit -m "reset KB for ${CONDITION}_seed${SEED}" --allow-empty
    fi

    # Create worktrees for agents 2..N
    for i in $(seq 2 "$NUM_AGENTS"); do
        WT_DIR="$RUN_DIR/agent${i}"
        if [ ! -d "$WT_DIR" ]; then
            git worktree add "$WT_DIR" "$BRANCH"
        fi
        echo "Agent $i worktree: $WT_DIR"
    done

    echo "Agent 1: $REPO_ROOT (branch: $BRANCH)"

    # Apply ablation if this is an ablation condition
    case "$CONDITION" in
        ablation_*)
            ABLATION_NAME="${CONDITION#ablation_}"
            ABLATION_SCRIPT="$REPO_ROOT/scripts/experiments/ablations/apply_ablation.sh"
            if [ -x "$ABLATION_SCRIPT" ]; then
                echo "Applying ablation: $ABLATION_NAME"
                "$ABLATION_SCRIPT" "$ABLATION_NAME" "$REPO_ROOT"
                git add -A && git commit -m "apply $ABLATION_NAME ablation" --allow-empty
            else
                echo "WARNING: Ablation script not found: $ABLATION_SCRIPT"
            fi
            ;;
    esac

    # Populate director queue if enabled
    if [ "$HAS_DIRECTOR" = "true" ]; then
        echo "Populating director queue..."
        uv run director.py run --max-experiments 10
        git add -A && git commit -m "populate initial queue for ${CONDITION}_seed${SEED}" --allow-empty
    fi
fi

# Write metadata.json
cat > "$RUN_DIR/metadata.json" << METAEOF
{
  "condition": "$CONDITION",
  "seed": $SEED,
  "agents": $NUM_AGENTS,
  "experiments_per_agent": $NUM_EXPERIMENTS,
  "shared_kb": $IS_SHARED,
  "director": $HAS_DIRECTOR,
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "end_time": null,
  "gpu": null,
  "notes": ""
}
METAEOF

echo "=== Setup complete. Metadata: $RUN_DIR/metadata.json ==="
