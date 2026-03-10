# Ablation Variants

Each ablation removes one component from condition G (full system: 4 agents + shared KB + director) to measure its individual contribution.

## Ablations

| Variant | Component Removed | What Changes |
|---------|-------------------|--------------|
| G-cov | Coverage map | `get_coverage_map()` returns empty dict; agents can't see tag distribution |
| G-lin | Experiment lineage | `prior_knowledge_used` always empty; no citation chains between cards |
| G-dir | Director queue | Queue file emptied; no strategic experiment prioritization |
| G-syn | Synthesis reports | `synthesize` and `update-meta` are no-ops; no periodic summaries |
| G-struct | Structured cards | Cards written as `.txt` free-text instead of `.json`; schema lost |

## Usage

```bash
# Apply an ablation to a repo directory
./apply_ablation.sh G-cov /path/to/repo

# setup_run.sh applies automatically for ablation_ conditions
./setup_run.sh ablation_G-cov 1
```

## How It Works

`apply_ablation.sh` patches `commons.py` in-place using targeted Python string replacements. Each patch inserts an early return or format change, preserving the original function as `_original_<name>` for reference. The patches are intentionally minimal to avoid breaking the rest of the knowledge base protocol.
