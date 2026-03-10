#!/usr/bin/env bash
set -euo pipefail

# Usage: archive_run.sh <condition> <seed>
# Archives experiment results (cards, synthesis, metadata) to a central results directory.

CONDITION="${1:?Usage: archive_run.sh <condition> <seed>}"
SEED="${2:?Usage: archive_run.sh <condition> <seed>}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECTS_DIR="${EXPERIMENT_BASE_DIR:-$(dirname "$REPO_ROOT")/experiment_runs}"
RUN_DIR="$PROJECTS_DIR/${CONDITION}_seed${SEED}"
RESULTS_DIR="${RESULTS_BASE_DIR:-$(dirname "$REPO_ROOT")/results}/${CONDITION}_seed${SEED}"

mkdir -p "$RESULTS_DIR/cards" "$RESULTS_DIR/synthesis"

echo "=== Archiving $CONDITION seed $SEED ==="

# Condition configs
declare -A AGENTS=(
    [condA]=1 [condB]=1 [condC]=2 [condD]=2 [condE]=2 [condF]=4 [condG]=4
    [ablation_G-cov]=4 [ablation_G-lin]=4 [ablation_G-dir]=4 [ablation_G-syn]=4 [ablation_G-struct]=4
)
declare -A SHARED_KB=(
    [condA]=false [condB]=false [condC]=false [condD]=true [condE]=true [condF]=false [condG]=true
    [ablation_G-cov]=true [ablation_G-lin]=true [ablation_G-dir]=true [ablation_G-syn]=true [ablation_G-struct]=true
)

NUM_AGENTS="${AGENTS[$CONDITION]}"
IS_SHARED="${SHARED_KB[$CONDITION]}"

if [[ "$NUM_AGENTS" -eq 1 ]]; then
    # Single agent: copy from main repo
    cp -r "$REPO_ROOT/knowledge/cards/"*.json "$RESULTS_DIR/cards/" 2>/dev/null || true
    cp -r "$REPO_ROOT/knowledge/synthesis/"*.md "$RESULTS_DIR/synthesis/" 2>/dev/null || true
    cp "$REPO_ROOT/knowledge/index.json" "$RESULTS_DIR/" 2>/dev/null || true
    cp "$REPO_ROOT/results.tsv" "$RESULTS_DIR/" 2>/dev/null || true

elif [[ "$IS_SHARED" == "false" ]]; then
    # Isolated: merge cards from all agents, tagging with agent ID
    for i in $(seq 1 "$NUM_AGENTS"); do
        AGENT_DIR="$RUN_DIR/agent${i}"
        if [[ -d "$AGENT_DIR/knowledge/cards" ]]; then
            for card in "$AGENT_DIR/knowledge/cards/"*.json; do
                [[ -f "$card" ]] || continue
                cp "$card" "$RESULTS_DIR/cards/"
            done
        fi
        # Merge results.tsv (skip header for agents 2+)
        if [[ -f "$AGENT_DIR/results.tsv" ]]; then
            if [[ $i -eq 1 ]]; then
                cp "$AGENT_DIR/results.tsv" "$RESULTS_DIR/results.tsv"
            else
                tail -n +2 "$AGENT_DIR/results.tsv" >> "$RESULTS_DIR/results.tsv"
            fi
        fi
    done

else
    # Shared: cards are all in the main repo's knowledge/
    cp -r "$REPO_ROOT/knowledge/cards/"*.json "$RESULTS_DIR/cards/" 2>/dev/null || true
    cp -r "$REPO_ROOT/knowledge/synthesis/"*.md "$RESULTS_DIR/synthesis/" 2>/dev/null || true
    cp "$REPO_ROOT/knowledge/index.json" "$RESULTS_DIR/" 2>/dev/null || true
    cp "$REPO_ROOT/knowledge/queue.json" "$RESULTS_DIR/" 2>/dev/null || true
    cp "$REPO_ROOT/results.tsv" "$RESULTS_DIR/" 2>/dev/null || true
fi

# Count archived cards
CARD_COUNT=$(find "$RESULTS_DIR/cards/" -name "*.json" -o -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')

# Copy and update metadata
if [[ -f "$RUN_DIR/metadata.json" ]]; then
    cp "$RUN_DIR/metadata.json" "$RESULTS_DIR/metadata.json"
fi

# Update end_time and total_experiments in metadata
python3 -c "
import json, datetime, pathlib
meta_path = pathlib.Path('$RESULTS_DIR/metadata.json')
if meta_path.exists():
    m = json.loads(meta_path.read_text())
    m['end_time'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    m['total_experiments'] = $CARD_COUNT
    meta_path.write_text(json.dumps(m, indent=2))
"

echo "=== Archive complete ==="
echo "Location: $RESULTS_DIR"
echo "Cards archived: $CARD_COUNT"
echo "Metadata: $RESULTS_DIR/metadata.json"
