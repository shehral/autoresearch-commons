#!/usr/bin/env bash
set -euo pipefail

# Usage: check_progress.sh <condition> <seed>
# Shows experiment progress: card count, best val_bpb, ETA.

CONDITION="${1:?Usage: check_progress.sh <condition> <seed>}"
SEED="${2:?Usage: check_progress.sh <condition> <seed>}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECTS_DIR="${EXPERIMENT_BASE_DIR:-$(dirname "$REPO_ROOT")/experiment_runs}"
RUN_DIR="$PROJECTS_DIR/${CONDITION}_seed${SEED}"

# Load condition config (bash 3.2 compatible)
source "$(dirname "$0")/condition_config.sh"
get_config "$CONDITION"

TOTAL_TARGET=$((NUM_AGENTS * NUM_EXPERIMENTS))

echo "=== Progress: $CONDITION seed $SEED ==="
echo "Target: $TOTAL_TARGET experiments ($NUM_AGENTS agents x $NUM_EXPERIMENTS each)"
echo ""

# Count cards and find best val_bpb
KNOWLEDGE_DIR="$REPO_ROOT/knowledge"
if [ "$NUM_AGENTS" -gt 1 ] && [ "$IS_SHARED" = "false" ]; then
    # Isolated: count across all agent directories
    TOTAL_CARDS=0
    for i in $(seq 1 "$NUM_AGENTS"); do
        AGENT_DIR="$RUN_DIR/agent${i}"
        if [ -d "$AGENT_DIR/knowledge/cards" ]; then
            AGENT_CARDS=$(find "$AGENT_DIR/knowledge/cards" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
            echo "Agent $i: $AGENT_CARDS cards"
            TOTAL_CARDS=$((TOTAL_CARDS + AGENT_CARDS))
        fi
    done
    echo ""
    echo "Total cards: $TOTAL_CARDS / $TOTAL_TARGET"
else
    TOTAL_CARDS=$(find "$KNOWLEDGE_DIR/cards" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "Cards completed: $TOTAL_CARDS / $TOTAL_TARGET"
fi

# Find best val_bpb across all cards
python3 -c "
import json, glob, os

card_dirs = []
num_agents = $NUM_AGENTS
is_shared = '$IS_SHARED'
if num_agents > 1 and is_shared == 'false':
    for i in range(1, num_agents + 1):
        d = '$RUN_DIR/agent' + str(i) + '/knowledge/cards'
        if os.path.isdir(d):
            card_dirs.append(d)
else:
    card_dirs.append('$KNOWLEDGE_DIR/cards')

best_bpb = float('inf')
latest_bpb = None
latest_ts = ''

for d in card_dirs:
    for f in glob.glob(os.path.join(d, '*.json')):
        with open(f) as fh:
            card = json.load(fh)
        if card.get('status') == 'retracted':
            continue
        bpb = card.get('results', {}).get('val_bpb')
        if bpb is not None and bpb > 0:
            if bpb < best_bpb:
                best_bpb = bpb
            ts = card.get('timestamp', '')
            if ts > latest_ts:
                latest_ts = ts
                latest_bpb = bpb

if best_bpb < float('inf'):
    print(f'Best val_bpb:   {best_bpb:.6f}')
if latest_bpb is not None:
    print(f'Latest val_bpb: {latest_bpb:.6f}')
else:
    print('No experiments recorded yet.')
" 2>/dev/null || echo "Could not parse cards."

# ETA
if [ "$TOTAL_CARDS" -gt 0 ]; then
    REMAINING=$((TOTAL_TARGET - TOTAL_CARDS))
    # ~6 min per experiment (5 min training + overhead)
    ETA_MINUTES=$((REMAINING * 6))
    echo ""
    echo "Remaining: $REMAINING experiments (~$((ETA_MINUTES / 60))h $((ETA_MINUTES % 60))m at ~6 min/exp)"
fi
