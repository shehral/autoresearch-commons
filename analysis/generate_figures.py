#!/usr/bin/env python3
"""Generate all paper figures from computed results.

Reads analysis/results.json (from compute_results.py) and generates:
- Figure 3: Learning curves (val_bpb* vs experiment count, all conditions)
- Figure 4: Redundancy bar chart (conditions with 2+ agents)
- Figure 5: Coverage map evolution (3 snapshots from a representative G run)
- Figure 6: Knowledge graph (DAG from a representative G run)

Usage:
    python analysis/generate_figures.py --results analysis/results.json --output-dir paper/figures/
    python analysis/generate_figures.py --results analysis/results.json --output-dir analysis/figures/ --format png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# NeurIPS-compatible styling
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette — one per condition, grayscale-friendly
COLORS = {
    "condA": "#1f77b4",  # blue
    "condB": "#ff7f0e",  # orange
    "condC": "#2ca02c",  # green
    "condD": "#d62728",  # red
    "condE": "#9467bd",  # purple
    "condF": "#8c564b",  # brown
    "condG": "#e377c2",  # pink
}

LABELS = {
    "condA": "A: Memoryless",
    "condB": "B: Memory",
    "condC": "C: Isolated-2",
    "condD": "D: Shared-2",
    "condE": "E: Shared-2+Dir",
    "condF": "F: Isolated-4",
    "condG": "G: Shared-4+Dir",
}

CONDITIONS_ORDER = ["condA", "condB", "condC", "condD", "condE", "condF", "condG"]


def plot_learning_curves(results: dict, output_path: Path, fmt: str = "pdf"):
    """Figure 3: val_bpb* trajectory for all conditions."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for cond in CONDITIONS_ORDER:
        traj = results.get("trajectories", {}).get(cond, [])
        if not traj:
            continue

        ks = [t["k"] for t in traj]
        means = [t["mean"] for t in traj]
        stds = [t["std"] for t in traj]

        color = COLORS[cond]
        ax.plot(ks, means, label=LABELS[cond], color=color, linewidth=1.5)
        ax.fill_between(
            ks,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Experiment count $k$")
    ax.set_ylabel("$\\mathrm{val\\_bpb}^*_k$ (running min)")
    ax.set_title("Learning Curves by Condition")
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    path = output_path / f"learning_curves.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_redundancy_bar(results: dict, output_path: Path, fmt: str = "pdf"):
    """Figure 4: Redundancy rate for multi-agent conditions."""
    multi_agent = ["condC", "condD", "condE", "condF", "condG"]
    fig, ax = plt.subplots(figsize=(4, 3))

    x_positions = range(len(multi_agent))
    means = []
    stds = []
    colors = []

    for cond in multi_agent:
        m = results.get("conditions", {}).get(cond, {})
        means.append(m.get("redundancy", {}).get("mean", 0))
        stds.append(m.get("redundancy", {}).get("std", 0))
        colors.append(COLORS[cond])

    bars = ax.bar(x_positions, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([LABELS[c].split(":")[0] for c in multi_agent])
    ax.set_ylabel("Redundancy Rate $R$")
    ax.set_title("Cross-Agent Redundancy")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)

    path = output_path / f"redundancy_bar.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_coverage_evolution(results: dict, results_dir: Path, output_path: Path, fmt: str = "pdf"):
    """Figure 5: Coverage map evolution — tag counts at experiments 10, 50, 100.
    Uses raw card data from a representative condG seed 1 run."""
    # Try to load cards directly
    card_dir = results_dir / "condG_seed1" / "cards" if results_dir else None

    if card_dir and card_dir.exists():
        cards = _load_cards_from_dir(card_dir)
    else:
        # Fallback: use aggregated category distribution
        dist = results.get("conditions", {}).get("condG", {}).get("category_distribution", {})
        if not dist:
            print("  Skipping coverage evolution (no condG data)")
            return
        # Fake it with available data
        cards = None

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
    snapshots = [10, 50, 100]

    if cards:
        for ax, n in zip(axes, snapshots):
            subset = cards[:min(n, len(cards))]
            tag_counts = {}
            for card in subset:
                for tag in card.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            if tag_counts:
                sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
                tags = [t[0] for t in sorted_tags[:10]]
                counts = [t[1] for t in sorted_tags[:10]]
                ax.barh(range(len(tags)), counts, color=COLORS["condG"], edgecolor="black", linewidth=0.5)
                ax.set_yticks(range(len(tags)))
                ax.set_yticklabels(tags, fontsize=7)
                ax.invert_yaxis()
            ax.set_title(f"$k={n}$")
            ax.set_xlabel("Count")
    else:
        # Static fallback using aggregated data
        dist = results["conditions"]["condG"]["category_distribution"]
        sorted_tags = sorted(dist.items(), key=lambda x: -x[1])[:10]
        for ax, n in zip(axes, snapshots):
            tags = [t[0] for t in sorted_tags]
            counts = [int(t[1] * n / 100) for t in sorted_tags]
            ax.barh(range(len(tags)), counts, color=COLORS["condG"], edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(tags)))
            ax.set_yticklabels(tags, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"$k={n}$")
            ax.set_xlabel("Count")

    fig.suptitle("Coverage Map Evolution (Condition G)", fontsize=10)
    fig.tight_layout()

    path = output_path / f"coverage_evolution.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_knowledge_graph(results_dir: Path, output_path: Path, fmt: str = "pdf"):
    """Figure 6: Knowledge citation DAG from a representative G run."""
    try:
        import networkx as nx
    except ImportError:
        print("  Skipping knowledge graph (networkx not installed)")
        return

    card_dir = results_dir / "condG_seed1" / "cards" if results_dir else None
    if not card_dir or not card_dir.exists():
        print("  Skipping knowledge graph (no condG data)")
        return

    cards = _load_cards_from_dir(card_dir)

    G = nx.DiGraph()
    agent_colors = {}
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for card in cards:
        card_id = card.get("id", "")
        # Infer agent
        parts = card_id.split("_")
        agent = parts[0] if len(parts) >= 2 else "unknown"
        if agent not in agent_colors:
            agent_colors[agent] = color_list[len(agent_colors) % len(color_list)]

        G.add_node(card_id, agent=agent)
        for cited in card.get("prior_knowledge_used", []):
            if cited in {c.get("id") for c in cards}:
                G.add_edge(cited, card_id)

    if len(G.nodes) == 0:
        print("  Skipping knowledge graph (no nodes)")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    # Layout
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except Exception:
        pos = nx.shell_layout(G)

    node_colors = [agent_colors.get(G.nodes[n].get("agent", ""), "#999") for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", arrows=True, arrowsize=8, alpha=0.5)

    # Legend
    for agent, color in agent_colors.items():
        ax.scatter([], [], c=color, s=30, label=agent)
    ax.legend(loc="upper left", framealpha=0.9, title="Agent")

    ax.set_title("Knowledge Citation Graph (Condition G)")
    ax.axis("off")

    path = output_path / f"knowledge_graph.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_ablation_table(results: dict, output_path: Path, fmt: str = "pdf"):
    """Generate a visual table of ablation results."""
    ablations = results.get("ablations", {})
    if not ablations:
        print("  Skipping ablation table (no ablation data)")
        return

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.axis("off")

    headers = ["Ablation", "val_bpb*", "Delta vs G", "Redundancy", "Entropy"]
    rows = []
    for abl_name in ["ablation_G-cov", "ablation_G-lin", "ablation_G-dir", "ablation_G-syn", "ablation_G-struct"]:
        if abl_name not in ablations:
            continue
        m = ablations[abl_name]
        delta = m.get("delta_vs_G")
        rows.append([
            abl_name.replace("ablation_", ""),
            f"{m['best_bpb']['mean']:.4f}",
            f"{delta:+.4f}" if delta is not None else "N/A",
            f"{m['redundancy']['mean']:.3f}",
            f"{m['coverage_entropy']['mean']:.3f}",
        ])

    if rows:
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)

        path = output_path / f"ablation_table.{fmt}"
        fig.savefig(path)
        print(f"  Saved {path}")

    plt.close(fig)


def _load_cards_from_dir(card_dir: Path) -> list[dict]:
    """Load and sort cards from a directory."""
    cards = []
    for fpath in sorted(card_dir.iterdir()):
        if fpath.suffix == ".json":
            with open(fpath) as f:
                cards.append(json.load(f))
    cards.sort(key=lambda c: c.get("timestamp", ""))
    return cards


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=Path, required=True, help="Path to results.json")
    parser.add_argument("--results-dir", type=Path, default=None, help="Raw results dir (for card-level figures)")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"), help="Output directory")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"], help="Output format")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_learning_curves(results, args.output_dir, args.format)
    plot_redundancy_bar(results, args.output_dir, args.format)
    plot_coverage_evolution(results, args.results_dir, args.output_dir, args.format)
    plot_ablation_table(results, args.output_dir, args.format)

    if args.results_dir:
        plot_knowledge_graph(args.results_dir, args.output_dir, args.format)

    print("Done.")


if __name__ == "__main__":
    main()
