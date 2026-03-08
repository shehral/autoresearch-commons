# autoresearch-commons

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that adds **collaborative knowledge sharing** between autonomous research agents.

The original autoresearch gives an AI agent a real LLM training setup and lets it experiment autonomously overnight. This fork adds the missing piece: a shared knowledge base so agents **learn from each other's experiments** — across sessions, across hardware, across people.

> *"The next step for autoresearch is that it has to be asynchronously massively collaborative for agents (think SETI@home style). The goal is not to emulate a single PhD student, it's to emulate a research community of them."* — @karpathy, March 2026

## What's new

This fork adds a **knowledge protocol**, **multi-platform support**, and an **orchestration layer** on top of the original autoresearch:

| Layer | Files | Purpose |
|-------|-------|---------|
| **Platform** | `platform_utils.py` | CUDA / MPS / CPU auto-detection, attention backends, memory tracking |
| **Knowledge** | `commons.py`, `knowledge/` | Experiment cards, synthesis, coverage maps, progress plots |
| **Orchestration** | `director.py`, `director.md` | Experiment queue, strategy planning, multi-agent coordination |
| **Agent** | `program.md` | Worker agent instructions with knowledge integration |

Everything from the original is preserved: `prepare.py` (untouched), `train.py` (same structure, now platform-portable), the 5-minute time budget, and val_bpb as the sole metric.

### Key features beyond the original

- **Experiment lineage** — cards track `prior_knowledge_used`, creating a knowledge graph of what inspired each experiment
- **Card validation** — schema validation at creation time (non-empty fields, numeric results, valid status)
- **Card retraction** — soft-delete mechanism to invalidate bad experiments without removing history
- **Atomic writes** — crash-safe JSON operations via `tempfile` + `os.replace()`
- **File locking** — `fcntl.LOCK_EX` on queue operations for multi-agent safety
- **Stale claim timeout** — experiments stuck `in_progress` for >15 minutes are automatically released
- **Hypothesis deduplication** — director prevents duplicate `(category, priority)` pairs
- **Auto-recording** — `train.py` automatically creates an experiment card with full config snapshot after each run
- **Queue integration** — agents see pending experiments from the director queue in `read-brief`
- **MPS peak memory tracking** — background thread polls memory at 100ms intervals (MPS lacks native peak tracking)
- **Progress visualization** — scatter plot with running-best line for val_bpb trends
- **145 tests** across knowledge base, director, and platform layers

For a comprehensive guide covering single-agent mode, multi-agent coordination, the knowledge protocol, and cross-machine collaboration, see the **[detailed walkthrough](docs/walkthrough.md)**.

## Quick start

**Requirements:** NVIDIA GPU, Apple Silicon Mac, or CPU. Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# For CUDA users with Flash Attention 3:
uv sync --extra cuda

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

## Running an agent (single-agent mode)

Same as the original, but now agents read/write to the knowledge base:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The modified `program.md` adds two steps to the experiment loop:
1. **Before experimenting:** `uv run commons.py read-brief` — read what's been tried, see queued experiments
2. **After each experiment:** `uv run commons.py write-card ...` — record what was learned

## Running the director (multi-agent mode)

The director plans experiment strategy from the knowledge base:

```bash
# See what experiments the knowledge base suggests
uv run director.py plan

# Add a custom experiment idea
uv run director.py add --hypothesis "Try SwiGLU activation" --category architecture --priority 1

# Check queue status
uv run director.py status
```

## Knowledge base CLI

### Recording experiments

```bash
# Record an experiment result with full metadata
uv run commons.py write-card \
  --commit abc1234 \
  --hypothesis "Halve batch size" \
  --result 0.986 \
  --delta -0.012 \
  --status keep \
  --lesson "More steps in fixed time budget" \
  --tags "batch_size,optimization" \
  --prior-cards card_001,card_002 \
  --estimated-flops 1.2e15 \
  --num-params 124000000
```

Valid statuses: `keep`, `revert`, `inconclusive`, `crash`.

### Reading knowledge

```bash
# Read knowledge summary (what agents see before planning)
uv run commons.py read-brief

# Read full meta-synthesis
uv run commons.py read-meta

# View experiment coverage by category
uv run commons.py coverage
```

### Synthesis and visualization

```bash
# Generate session synthesis report
uv run commons.py synthesize --session mar8

# Update rolling meta-synthesis
uv run commons.py update-meta

# Generate progress chart (val_bpb scatter + running best)
uv run commons.py plot
```

### Maintenance

```bash
# Retract a bad experiment (soft-delete, preserves history)
uv run commons.py retract --id card_003 --reason "Baseline was misconfigured"

# Regenerate the index
uv run commons.py update-index
```

## Project structure

```
prepare.py          — constants, data prep + runtime utilities (do not modify)
train.py            — model, optimizer, training loop (agent modifies this)
platform_utils.py   — CUDA/MPS/CPU auto-detection, attention, memory tracking
commons.py          — knowledge base interface (library + CLI)
director.py         — experiment orchestration and queue management
program.md          — worker agent instructions
director.md         — director agent instructions
tests/
  test_commons.py   — knowledge base tests (84 tests)
  test_director.py  — director and queue tests (27 tests)
  test_platform.py  — platform abstraction tests (34 tests)
knowledge/
  cards/            — one JSON per experiment (the raw data)
  synthesis/        — session reports + meta-synthesis (the summaries)
  index.json        — fast-query experiment index
  queue.json        — experiment queue (director-managed)
```

## How the knowledge protocol works

Every experiment produces an **experiment card** (JSON):
- What was tried (hypothesis) and what changed (config diff)
- The result (val_bpb, delta, peak memory, estimated flops, parameter count)
- What was learned (lesson) and categorical tags
- Platform info (so agents know which findings are hardware-specific)
- Prior knowledge used (which earlier cards inspired this experiment)

Cards are **validated** at creation time — empty hypotheses, invalid statuses, and non-numeric results are rejected. Bad cards can be **retracted** without deleting history.

Periodically, cards are **synthesized** into session reports and a rolling **meta-synthesis** that includes:
- Confirmed findings and dead ends
- An **experiment coverage map** showing what areas are saturated vs. under-explored
- Open questions for future experiments

The next agent reads the meta-synthesis before planning, so it doesn't repeat dead ends and focuses on high-value directions.

## Multi-agent safety

When multiple agents share the same knowledge base:
- **Atomic writes** prevent corrupted JSON from partial writes or crashes
- **File locking** (`fcntl.LOCK_EX`) serializes queue operations
- **Stale claim timeout** (15 min) releases experiments abandoned by crashed agents
- **Hypothesis dedup** prevents the director from suggesting duplicate experiments

## Platform support

This fork auto-detects your hardware:
- **NVIDIA GPU (CUDA):** Flash Attention 3, torch.compile, bfloat16 autocast, native peak memory tracking
- **Apple Silicon (MPS):** SDPA attention with sliding window mask, no compile, background-thread peak memory tracking
- **CPU:** SDPA attention, bfloat16 autocast, slowest but works anywhere

## Design philosophy

- **The knowledge protocol is the contribution.** Platform ports let more people run experiments. The knowledge protocol lets agents *learn from each other*. That's the real multiplier.
- **Separation of concerns.** The protocol (knowledge format) is git-native and works standalone. The orchestration (director) is an optional convenience layer.
- **Minimal upstream diff.** We add files, not complexity. `train.py` gets one import swap. Everything else is additive.
- **Multi-agent by default.** Atomic writes, file locking, and stale claim timeouts mean agents can safely share a knowledge base without coordination.

## Testing

```bash
# Run all 145 tests
uv run pytest

# Run specific test files
uv run pytest tests/test_commons.py -v
uv run pytest tests/test_director.py -v
uv run pytest tests/test_platform.py -v
```

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — MPS adaptation patterns
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — MLX reference
- [Claude Code](https://claude.com/claude-code) — planning, development, and testing

## License

MIT — see [LICENSE](LICENSE).
