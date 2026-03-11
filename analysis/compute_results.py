#!/usr/bin/env python3
"""Compute all paper metrics from archived experiment cards.

Reads cards from results/<condition>_seed<N>/cards/ and computes:
- Best val_bpb per condition (mean +/- std over seeds)
- Learning trajectory (running min over experiment count)
- Redundancy rate (cross-agent overlap in tag + config)
- Coverage entropy (Shannon entropy over tag distribution)
- Knowledge transfer events (cross-agent citations)
- Category distribution per condition
- Ablation deltas vs full G

Usage:
    python analysis/compute_results.py --results-dir /path/to/results --output analysis/results.json
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


# ─── Card loading ─────────────────────────────────────────

def load_cards(results_dir: Path, condition: str, seed: int) -> list[dict]:
    """Load all experiment cards for a condition+seed."""
    card_dir = results_dir / f"{condition}_seed{seed}" / "cards"
    cards = []
    if not card_dir.exists():
        return cards

    for fpath in sorted(card_dir.iterdir()):
        if fpath.suffix == ".json":
            with open(fpath) as f:
                cards.append(json.load(f))
        elif fpath.suffix == ".txt":
            # Free-text cards from G-struct ablation — parse what we can
            cards.append(_parse_freetext_card(fpath))

    # Sort by timestamp
    cards.sort(key=lambda c: c.get("timestamp", ""))
    return cards


def _parse_freetext_card(fpath: Path) -> dict:
    """Best-effort parse of free-text ablation card."""
    text = fpath.read_text()
    card = {"id": fpath.stem, "tags": [], "prior_knowledge_used": [], "results": {}, "status": "unknown"}
    for line in text.strip().split("\n"):
        if line.startswith("Experiment:"):
            card["id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Hypothesis:"):
            card["hypothesis"] = line.split(":", 1)[1].strip()
        elif line.startswith("Status:"):
            card["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("Results:"):
            parts = line.split(":", 1)[1].strip()
            for kv in parts.split(","):
                kv = kv.strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        card["results"][k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        elif line.startswith("Tags:"):
            card["tags"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.startswith("Lesson:"):
            card["lesson"] = line.split(":", 1)[1].strip()
        elif line.startswith("Commit:"):
            card["parent_commit"] = line.split(":", 1)[1].strip()
        elif line.startswith("Time:"):
            card["timestamp"] = line.split(":", 1)[1].strip()
    return card


# ─── Metrics ──────────────────────────────────────────────

def compute_best_bpb(cards: list[dict]) -> float:
    """Best (minimum) val_bpb across all cards."""
    bpbs = [c["results"].get("val_bpb", float("inf")) for c in cards if c.get("status") != "crash"]
    valid = [b for b in bpbs if b > 0]
    return min(valid) if valid else float("inf")


def compute_trajectory(cards: list[dict]) -> list[tuple[int, float]]:
    """Running minimum val_bpb over experiment count."""
    trajectory = []
    best = float("inf")
    for i, card in enumerate(cards):
        bpb = card.get("results", {}).get("val_bpb", float("inf"))
        if bpb > 0 and card.get("status") != "crash":
            best = min(best, bpb)
        trajectory.append((i + 1, best))
    return trajectory


def compute_redundancy(cards: list[dict]) -> float:
    """Cross-agent redundancy rate: fraction of card pairs sharing same primary tag
    and >= 80% config_diff key overlap. Only counts cross-agent pairs."""
    if len(cards) < 2:
        return 0.0

    # Infer agent ID from card ID pattern or timestamp grouping
    agents = _infer_agent_ids(cards)

    redundant = 0
    total_cross = 0

    for i, c1 in enumerate(cards):
        for j, c2 in enumerate(cards):
            if j <= i:
                continue
            # Only cross-agent pairs
            if agents.get(c1.get("id")) == agents.get(c2.get("id")):
                continue

            total_cross += 1

            # Same primary tag?
            tags1 = c1.get("tags", [])
            tags2 = c2.get("tags", [])
            if not tags1 or not tags2:
                continue
            if tags1[0] != tags2[0]:
                continue

            # Config overlap >= 80%?
            keys1 = set(c1.get("config_diff", {}).keys())
            keys2 = set(c2.get("config_diff", {}).keys())
            if not keys1 and not keys2:
                redundant += 1
                continue
            union = keys1 | keys2
            if union:
                overlap = len(keys1 & keys2) / len(union)
                if overlap >= 0.8:
                    redundant += 1

    return redundant / total_cross if total_cross > 0 else 0.0


def _infer_agent_ids(cards: list[dict]) -> dict[str, str]:
    """Infer agent ID for each card. Uses card ID prefix or falls back to timestamp clustering."""
    agents = {}
    for card in cards:
        card_id = card.get("id", "")
        # Convention: card IDs often start with agent prefix (e.g., "agent1_001")
        parts = card_id.split("_")
        if len(parts) >= 2 and parts[0].startswith("agent"):
            agents[card_id] = parts[0]
        else:
            # Fall back to first character (e.g., "A_1_001" → agent "A_1")
            agents[card_id] = "_".join(parts[:2]) if len(parts) >= 2 else card_id
    return agents


def compute_coverage_entropy(cards: list[dict]) -> float:
    """Shannon entropy over tag distribution (higher = more diverse exploration)."""
    tag_counts: Counter = Counter()
    for card in cards:
        for tag in card.get("tags", []):
            tag_counts[tag] += 1

    total = sum(tag_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in tag_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def count_transfer_events(cards: list[dict]) -> int:
    """Count knowledge transfer events: cards citing another agent's card."""
    agents = _infer_agent_ids(cards)
    card_ids = {c.get("id"): c for c in cards}
    transfers = 0

    for card in cards:
        card_agent = agents.get(card.get("id"))
        for cited_id in card.get("prior_knowledge_used", []):
            cited_agent = agents.get(cited_id)
            if cited_agent and card_agent and cited_agent != card_agent:
                transfers += 1
    return transfers


def compute_category_distribution(cards: list[dict]) -> dict[str, int]:
    """Count cards per tag."""
    dist: Counter = Counter()
    for card in cards:
        for tag in card.get("tags", []):
            dist[tag] += 1
    return dict(dist)


# ─── Aggregation ──────────────────────────────────────────

def aggregate_over_seeds(
    metric_fn,
    results_dir: Path,
    condition: str,
    seeds: list[int] = [1, 2, 3],
) -> tuple[float, float]:
    """Compute mean and std of a metric over seeds."""
    values = []
    for seed in seeds:
        cards = load_cards(results_dir, condition, seed)
        if cards:
            val = metric_fn(cards)
            if val != float("inf"):
                values.append(val)

    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return (mean, std)


def compute_trajectory_aggregated(
    results_dir: Path,
    condition: str,
    seeds: list[int] = [1, 2, 3],
) -> list[dict]:
    """Compute mean trajectory over seeds, aligned by experiment index."""
    trajectories = []
    for seed in seeds:
        cards = load_cards(results_dir, condition, seed)
        if cards:
            trajectories.append(compute_trajectory(cards))

    if not trajectories:
        return []

    max_len = max(len(t) for t in trajectories)
    result = []
    for k in range(max_len):
        vals = []
        for t in trajectories:
            if k < len(t):
                vals.append(t[k][1])
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) if len(vals) > 1 else 0.0
        result.append({"k": k + 1, "mean": mean, "std": std, "n": len(vals)})
    return result


# ─── Main ─────────────────────────────────────────────────

CONDITIONS = ["condA", "condB", "condC", "condD", "condE", "condF", "condG"]
ABLATIONS = ["ablation_G-cov", "ablation_G-lin", "ablation_G-dir", "ablation_G-syn", "ablation_G-struct"]


def compute_all(results_dir: Path, seeds: list[int] = [1, 2, 3]) -> dict:
    """Compute all paper metrics."""
    results = {"conditions": {}, "ablations": {}, "trajectories": {}}

    # Per-condition metrics
    for cond in CONDITIONS:
        best_mean, best_std = aggregate_over_seeds(compute_best_bpb, results_dir, cond, seeds)
        red_mean, red_std = aggregate_over_seeds(compute_redundancy, results_dir, cond, seeds)
        ent_mean, ent_std = aggregate_over_seeds(compute_coverage_entropy, results_dir, cond, seeds)
        xfer_mean, xfer_std = aggregate_over_seeds(
            lambda c: float(count_transfer_events(c)), results_dir, cond, seeds
        )

        # Card count
        counts = []
        cat_dist_agg: Counter = Counter()
        for seed in seeds:
            cards = load_cards(results_dir, cond, seed)
            counts.append(len(cards))
            for tag, count in compute_category_distribution(cards).items():
                cat_dist_agg[tag] += count

        results["conditions"][cond] = {
            "best_bpb": {"mean": best_mean, "std": best_std},
            "redundancy": {"mean": red_mean, "std": red_std},
            "coverage_entropy": {"mean": ent_mean, "std": ent_std},
            "transfer_events": {"mean": xfer_mean, "std": xfer_std},
            "total_cards": sum(counts),
            "cards_per_seed": counts,
            "category_distribution": dict(cat_dist_agg),
        }

        # Trajectory
        results["trajectories"][cond] = compute_trajectory_aggregated(results_dir, cond, seeds)

    # Ablation metrics (single seed typically, but support multiple)
    g_best = results["conditions"]["condG"]["best_bpb"]["mean"]
    for abl in ABLATIONS:
        # Check if any seed directory exists for this ablation
        abl_exists = any(
            (results_dir / f"{abl}_seed{s}").exists() for s in seeds
        ) or (results_dir / abl).exists()
        if not abl_exists:
            continue

        best_mean, best_std = aggregate_over_seeds(compute_best_bpb, results_dir, abl, seeds)
        red_mean, red_std = aggregate_over_seeds(compute_redundancy, results_dir, abl, seeds)
        ent_mean, ent_std = aggregate_over_seeds(compute_coverage_entropy, results_dir, abl, seeds)

        results["ablations"][abl] = {
            "best_bpb": {"mean": best_mean, "std": best_std},
            "delta_vs_G": best_mean - g_best if best_mean > 0 else None,
            "redundancy": {"mean": red_mean, "std": red_std},
            "coverage_entropy": {"mean": ent_mean, "std": ent_std},
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute experiment results for paper")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing archived results")
    parser.add_argument("--output", type=Path, default=Path("analysis/results.json"), help="Output JSON file")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated seed list")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    results = compute_all(args.results_dir, seeds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("=== Results Summary ===")
    for cond in CONDITIONS:
        if cond in results["conditions"]:
            m = results["conditions"][cond]
            print(f"{cond}: best_bpb = {m['best_bpb']['mean']:.6f} +/- {m['best_bpb']['std']:.6f} "
                  f"| cards = {m['total_cards']} "
                  f"| redundancy = {m['redundancy']['mean']:.3f} "
                  f"| entropy = {m['coverage_entropy']['mean']:.3f}")

    if results["ablations"]:
        print("\n=== Ablation Deltas ===")
        for abl, m in results["ablations"].items():
            delta = m.get("delta_vs_G")
            delta_str = f"{delta:+.6f}" if delta is not None else "N/A"
            print(f"{abl}: best_bpb = {m['best_bpb']['mean']:.6f} | delta vs G = {delta_str}")

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
