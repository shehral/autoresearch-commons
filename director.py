"""
director.py — Queue management and orchestration for autoresearch agents.

An optional layer that reads the knowledge base (via commons.py), identifies
under-explored areas, generates experiment ideas, and manages a queue of
pending experiments.  For v1 the orchestration loop prints instructions
rather than spawning subprocesses.

Usage (CLI):
    uv run director.py plan
    uv run director.py run --synthesis-interval 20 --max-experiments 5
    uv run director.py status
    uv run director.py add --hypothesis "Try SwiGLU" --category architecture --priority 1
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from commons import get_coverage_map, get_open_questions, load_cards

# ---------------------------------------------------------------------------
# Default knowledge directory: knowledge/ relative to this script
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------


def _queue_path(knowledge_dir: Path | str) -> Path:
    """Return the path to queue.json inside the knowledge directory."""
    return Path(knowledge_dir) / "queue.json"


def _empty_queue() -> dict:
    """Return a fresh, empty queue structure."""
    return {
        "version": 1,
        "last_updated": None,
        "experiments": [],
    }


def _next_exp_id(queue: dict) -> str:
    """Generate the next sequential experiment id (exp-0001, exp-0002, ...)."""
    existing_ids = [e["id"] for e in queue.get("experiments", [])]
    if not existing_ids:
        return "exp-0001"
    # Extract numeric parts and find the max
    max_num = 0
    for eid in existing_ids:
        try:
            num = int(eid.split("-")[1])
            if num > max_num:
                max_num = num
        except (IndexError, ValueError):
            continue
    return f"exp-{max_num + 1:04d}"


def load_queue(knowledge_dir: Path | str) -> dict:
    """Load queue.json from the knowledge directory.

    Returns an empty queue structure if the file does not exist.
    """
    qpath = _queue_path(knowledge_dir)
    if not qpath.exists():
        return _empty_queue()
    try:
        return json.loads(qpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_queue()


def save_queue(knowledge_dir: Path | str, queue: dict) -> None:
    """Save queue.json with an updated timestamp."""
    knowledge_dir = Path(knowledge_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    queue["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    qpath = _queue_path(knowledge_dir)
    qpath.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")


def add_to_queue(
    knowledge_dir: Path | str,
    hypothesis: str,
    category: str,
    priority: int = 5,
) -> dict:
    """Add an experiment idea to the queue.

    Returns the newly created experiment entry.
    """
    queue = load_queue(knowledge_dir)
    exp_id = _next_exp_id(queue)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    experiment = {
        "id": exp_id,
        "hypothesis": hypothesis,
        "category": category,
        "priority": priority,
        "status": "pending",
        "assigned_to": None,
        "created_at": now,
    }

    queue["experiments"].append(experiment)
    save_queue(knowledge_dir, queue)
    return experiment


def claim_next_experiment(
    knowledge_dir: Path | str,
    worker_id: str,
) -> dict | None:
    """Claim the highest-priority pending experiment for a worker.

    Priority 1 is highest.  Among equal priorities the oldest (earliest
    created_at) experiment is chosen.  Returns the claimed experiment dict,
    or None if no pending experiments are available.
    """
    queue = load_queue(knowledge_dir)
    pending = [e for e in queue["experiments"] if e["status"] == "pending"]

    if not pending:
        return None

    # Sort by priority (ascending = higher priority first), then by created_at
    pending.sort(key=lambda e: (e["priority"], e["created_at"]))
    chosen = pending[0]

    # Mutate the experiment in the queue list
    for exp in queue["experiments"]:
        if exp["id"] == chosen["id"]:
            exp["status"] = "in_progress"
            exp["assigned_to"] = worker_id
            break

    save_queue(knowledge_dir, queue)

    # Return the updated version
    chosen["status"] = "in_progress"
    chosen["assigned_to"] = worker_id
    return chosen


def complete_experiment(
    knowledge_dir: Path | str,
    exp_id: str,
    status: str = "completed",
) -> None:
    """Mark an experiment as completed (or another terminal status)."""
    queue = load_queue(knowledge_dir)
    for exp in queue["experiments"]:
        if exp["id"] == exp_id:
            exp["status"] = status
            break
    save_queue(knowledge_dir, queue)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_experiments(knowledge_dir: Path | str) -> list[dict]:
    """Generate experiment ideas from the knowledge base.

    Prioritisation:
        Priority 1 — Under-explored categories (< 3 experiments)
        Priority 2 — High-potential areas (best_delta < -0.005 and < 10 experiments)
        Priority 3 — Open questions from meta-synthesis
    """
    knowledge_dir = Path(knowledge_dir)
    coverage = get_coverage_map(knowledge_dir)
    open_questions = get_open_questions(knowledge_dir)

    ideas: list[dict] = []

    # --- Priority 1: Under-explored categories ---
    for tag, stats in coverage.items():
        if stats["count"] < 3:
            ideas.append({
                "hypothesis": f"Further explore {tag} (only {stats['count']} experiments so far)",
                "category": tag,
                "priority": 1,
            })

    # --- Priority 2: High-potential areas ---
    for tag, stats in coverage.items():
        if (
            stats["best_delta"] is not None
            and stats["best_delta"] < -0.005
            and stats["count"] < 10
        ):
            # Avoid duplicating priority-1 ideas
            already_planned = any(
                i["category"] == tag and i["priority"] == 1 for i in ideas
            )
            if not already_planned:
                ideas.append({
                    "hypothesis": (
                        f"Deepen exploration of {tag} "
                        f"(best_delta={stats['best_delta']:.6f}, "
                        f"{stats['count']} experiments)"
                    ),
                    "category": tag,
                    "priority": 2,
                })

    # --- Priority 3: Open questions ---
    for question in open_questions:
        ideas.append({
            "hypothesis": question,
            "category": "open_question",
            "priority": 3,
        })

    # Sort by priority
    ideas.sort(key=lambda i: i["priority"])
    return ideas


# ---------------------------------------------------------------------------
# Orchestration loop
# ---------------------------------------------------------------------------


def run_director_loop(
    knowledge_dir: Path | str,
    synthesis_interval: int = 20,
    max_experiments: int = 0,
) -> None:
    """Plan experiments, enqueue them, and print dispatch instructions.

    For v1 this does NOT spawn subprocesses — it prints what a worker
    should do and marks experiments through their lifecycle.

    Args:
        knowledge_dir: Path to the knowledge directory.
        synthesis_interval: After this many completed experiments, suggest
            running synthesis.  (Not enforced in v1.)
        max_experiments: Stop after enqueuing this many.  0 means plan all.
    """
    knowledge_dir = Path(knowledge_dir)

    # Step 1: Plan
    ideas = plan_experiments(knowledge_dir)
    if not ideas:
        print("No experiment ideas generated. Knowledge base may be empty.")
        return

    if max_experiments > 0:
        ideas = ideas[:max_experiments]

    # Step 2: Enqueue
    print(f"Enqueuing {len(ideas)} experiment(s)...\n")
    for idea in ideas:
        exp = add_to_queue(
            knowledge_dir,
            hypothesis=idea["hypothesis"],
            category=idea["category"],
            priority=idea["priority"],
        )
        print(f"  [{exp['id']}] (priority {exp['priority']}) {exp['hypothesis']}")

    # Step 3: Dispatch instructions
    queue = load_queue(knowledge_dir)
    pending = [e for e in queue["experiments"] if e["status"] == "pending"]
    print(f"\n--- Queue has {len(pending)} pending experiment(s) ---")
    print("\nTo dispatch, a worker should call:")
    print("  from director import claim_next_experiment, complete_experiment")
    print('  exp = claim_next_experiment(knowledge_dir, worker_id="worker-1")')
    print("  # ... run the experiment ...")
    print('  complete_experiment(knowledge_dir, exp["id"])')

    # Step 4: Synthesis reminder
    completed_count = sum(
        1 for e in queue["experiments"] if e["status"] == "completed"
    )
    if completed_count > 0 and completed_count % synthesis_interval == 0:
        print(
            f"\n[Synthesis reminder] {completed_count} experiments completed. "
            "Consider running: uv run commons.py read-brief"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Director — queue management and orchestration for autoresearch.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # plan
    subparsers.add_parser("plan", help="Show planned experiments from knowledge base")

    # run
    run_parser = subparsers.add_parser("run", help="Run the director loop")
    run_parser.add_argument(
        "--synthesis-interval",
        type=int,
        default=20,
        help="Suggest synthesis after this many completed experiments (default: 20)",
    )
    run_parser.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        help="Maximum experiments to enqueue (0 = all, default: 0)",
    )

    # status
    subparsers.add_parser("status", help="Show queue status")

    # add
    add_parser = subparsers.add_parser("add", help="Add an experiment to the queue")
    add_parser.add_argument(
        "--hypothesis", required=True, help="Experiment hypothesis"
    )
    add_parser.add_argument(
        "--category", required=True, help="Experiment category/tag"
    )
    add_parser.add_argument(
        "--priority",
        type=int,
        default=5,
        help="Priority (1=high, 2=medium, 3=low, 5=default)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    knowledge_dir = Path(os.environ.get("KNOWLEDGE_DIR", str(KNOWLEDGE_DIR)))

    if args.command == "plan":
        ideas = plan_experiments(knowledge_dir)
        if not ideas:
            print("No experiment ideas generated. Knowledge base may be empty.")
        else:
            print(f"Planned {len(ideas)} experiment(s):\n")
            for idea in ideas:
                print(
                    f"  [priority {idea['priority']}] ({idea['category']}) "
                    f"{idea['hypothesis']}"
                )

    elif args.command == "run":
        run_director_loop(
            knowledge_dir,
            synthesis_interval=args.synthesis_interval,
            max_experiments=args.max_experiments,
        )

    elif args.command == "status":
        queue = load_queue(knowledge_dir)
        experiments = queue.get("experiments", [])
        total = len(experiments)
        pending = sum(1 for e in experiments if e["status"] == "pending")
        in_progress = sum(1 for e in experiments if e["status"] == "in_progress")
        completed = sum(1 for e in experiments if e["status"] == "completed")

        print(f"Queue status (last updated: {queue.get('last_updated', 'never')})")
        print(f"  Total:       {total}")
        print(f"  Pending:     {pending}")
        print(f"  In progress: {in_progress}")
        print(f"  Completed:   {completed}")

        if experiments:
            print("\nExperiments:")
            for exp in experiments:
                assigned = exp.get("assigned_to") or "unassigned"
                print(
                    f"  {exp['id']}  priority={exp['priority']}  "
                    f"status={exp['status']}  assigned={assigned}  "
                    f"{exp['hypothesis'][:60]}"
                )

    elif args.command == "add":
        exp = add_to_queue(
            knowledge_dir,
            hypothesis=args.hypothesis,
            category=args.category,
            priority=args.priority,
        )
        print(f"Added: {exp['id']} (priority {exp['priority']}) {exp['hypothesis']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
