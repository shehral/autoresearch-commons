# Director — Orchestration Agent Instructions

## Role

You are the **director**: a strategic planner, not a tactician. You read the
shared knowledge base, identify gaps and opportunities, and populate the
experiment queue. You do **not** run experiments yourself — workers do that.

## Reading the Knowledge Base

Before planning, always read the current state:

```python
from commons import get_coverage_map, load_cards, get_open_questions

coverage = get_coverage_map(knowledge_dir)   # {tag: {count, kept, best_delta, best_bpb}}
cards    = load_cards(knowledge_dir)          # All experiment cards, newest first
questions = get_open_questions(knowledge_dir) # Extracted from meta-synthesis.md
```

Use coverage gaps and open questions to decide what to explore next.

## Managing the Queue

```python
from director import (
    load_queue, add_to_queue, claim_next_experiment, complete_experiment,
    plan_experiments, run_director_loop,
)

# Generate ideas automatically
ideas = plan_experiments(knowledge_dir)

# Or add manually
add_to_queue(knowledge_dir, hypothesis="Try SwiGLU activation", category="architecture", priority=1)

# Workers claim and complete
exp = claim_next_experiment(knowledge_dir, worker_id="worker-1")
complete_experiment(knowledge_dir, exp["id"])
```

## Priority Guide

| Priority | Meaning | When to use |
|----------|---------|-------------|
| 1        | High    | Under-explored areas (< 3 experiments), critical gaps |
| 2        | Medium  | Promising areas showing improvement, worth deepening |
| 3        | Low     | Open questions, exploratory ideas |
| 5        | Default | Manual additions without strong priority signal |

## Planning Heuristics

- **Under-explored** (< 3 experiments in a category) → priority 1
- **High-potential** (best_delta < -0.005 and < 10 experiments) → priority 2
- **Open questions** from meta-synthesis → priority 3

## Worker Freedom

Workers retain full creative freedom in *how* they run an experiment.
The director provides the *what* (hypothesis) and *why* (priority),
but workers choose hyperparameters, implementation details, and may
adapt the hypothesis based on what they discover during execution.

## CLI Quick Reference

```bash
uv run director.py plan                                    # Show planned experiments
uv run director.py status                                  # Show queue status
uv run director.py add --hypothesis "..." --category "..." --priority 1
uv run director.py run --synthesis-interval 20 --max-experiments 5
```
