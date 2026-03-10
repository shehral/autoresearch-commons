#!/usr/bin/env bash
set -euo pipefail

# Usage: apply_ablation.sh <G-cov|G-lin|G-dir|G-syn|G-struct> [repo_dir]
# Applies an ablation variant by patching commons.py in-place.
# Each ablation removes one component from the full condition G system.

ABLATION="${1:?Usage: apply_ablation.sh <G-cov|G-lin|G-dir|G-syn|G-struct> [repo_dir]}"
REPO_DIR="${2:-.}"

case "$ABLATION" in
    G-cov)
        # Remove coverage map from read-brief output
        python3 -c "
with open('$REPO_DIR/commons.py', 'r') as f:
    content = f.read()
# Make get_coverage_map return empty dict immediately
content = content.replace(
    'def get_coverage_map(knowledge_dir: Path | str) -> dict:',
    'def get_coverage_map(knowledge_dir: Path | str) -> dict:\n    return {}  # ABLATION: coverage map disabled\n\ndef _original_get_coverage_map(knowledge_dir: Path | str) -> dict:'
)
with open('$REPO_DIR/commons.py', 'w') as f:
    f.write(content)
"
        echo "Applied G-cov ablation: coverage map disabled in read-brief"
        ;;

    G-lin)
        # Disable experiment lineage (prior_knowledge_used always empty)
        python3 -c "
with open('$REPO_DIR/commons.py', 'r') as f:
    content = f.read()
content = content.replace(
    '\"prior_knowledge_used\": prior_knowledge_used or [],',
    '\"prior_knowledge_used\": [],  # ABLATION: lineage disabled'
)
with open('$REPO_DIR/commons.py', 'w') as f:
    f.write(content)
"
        echo "Applied G-lin ablation: experiment lineage disabled"
        ;;

    G-dir)
        # Disable director -- empty the queue file
        echo '{"version":1,"last_updated":null,"experiments":[]}' > "$REPO_DIR/knowledge/queue.json"
        echo "Applied G-dir ablation: director disabled (empty queue)"
        ;;

    G-syn)
        # Make synthesize and update-meta no-ops
        python3 -c "
with open('$REPO_DIR/commons.py', 'r') as f:
    content = f.read()
content = content.replace(
    'def generate_session_synthesis(knowledge_dir: Path | str, session_tag: str) -> str:',
    'def generate_session_synthesis(knowledge_dir: Path | str, session_tag: str) -> str:\n    return \"\"  # ABLATION: synthesis disabled\n\ndef _original_generate_session_synthesis(knowledge_dir: Path | str, session_tag: str) -> str:'
)
content = content.replace(
    'def update_meta_synthesis(knowledge_dir: Path | str) -> str:',
    'def update_meta_synthesis(knowledge_dir: Path | str) -> str:\n    return \"\"  # ABLATION: meta-synthesis disabled\n\ndef _original_update_meta_synthesis(knowledge_dir: Path | str) -> str:'
)
with open('$REPO_DIR/commons.py', 'w') as f:
    f.write(content)
"
        echo "Applied G-syn ablation: synthesis reports disabled"
        ;;

    G-struct)
        # Replace structured JSON cards with free-text logs
        python3 -c "
with open('$REPO_DIR/commons.py', 'r') as f:
    content = f.read()
# Change card filename from .json to .txt
content = content.replace(
    'def _card_filename(timestamp: str, commit_id: str) -> str:',
    'def _card_filename(timestamp: str, commit_id: str) -> str:\n    return f\"{timestamp.replace(\\\":\\\", \\\"\\\").replace(\\\"-\\\", \\\"\\\")}_{commit_id[:7]}.txt\"  # ABLATION: free-text\n\ndef _original_card_filename(timestamp: str, commit_id: str) -> str:'
)
# Change json write to plain text
content = content.replace(
    '_atomic_write_json(card_path, card)',
    '''# ABLATION: write free-text instead of JSON
    text = f\"\"\"Experiment: {card['id']}
Time: {card['timestamp']}
Commit: {card['parent_commit']}
Hypothesis: {card['hypothesis']}
Status: {card['status']}
Results: val_bpb={card['results'].get('val_bpb', 'n/a')}, delta={card['results'].get('delta', 'n/a')}
Lesson: {card['lesson']}
Tags: {', '.join(card['tags'])}
\"\"\"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(text)'''
)
# Make load_cards also read .txt files
content = content.replace(
    'for fpath in cards_dir.glob(\"*.json\"):',
    'for fpath in list(cards_dir.glob(\"*.json\")) + list(cards_dir.glob(\"*.txt\")):  # ABLATION: also read .txt'
)
with open('$REPO_DIR/commons.py', 'w') as f:
    f.write(content)
"
        echo "Applied G-struct ablation: cards written as free-text (not JSON)"
        ;;

    *)
        echo "Unknown ablation: $ABLATION"
        echo "Valid: G-cov, G-lin, G-dir, G-syn, G-struct"
        exit 1
        ;;
esac
