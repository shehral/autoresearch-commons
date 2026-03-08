"""
Tests for director.py — queue management and orchestration.

Uses tmp_path fixture for isolated knowledge directories so tests
never touch the real knowledge/ folder.
"""

import json
from pathlib import Path

import pytest

from director import (
    add_to_queue,
    claim_next_experiment,
    complete_experiment,
    load_queue,
    save_queue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a minimal knowledge directory structure."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir(parents=True)
    (kdir / "cards").mkdir()
    (kdir / "synthesis").mkdir()
    return kdir


# ---------------------------------------------------------------------------
# load_queue
# ---------------------------------------------------------------------------


class TestLoadQueue:
    def test_load_empty_queue_no_file(self, knowledge_dir: Path):
        """Loading from a dir without queue.json returns an empty queue."""
        queue = load_queue(knowledge_dir)
        assert queue["version"] == 1
        assert queue["last_updated"] is None
        assert queue["experiments"] == []

    def test_load_existing_queue(self, knowledge_dir: Path):
        """Loading reads back a previously saved queue."""
        queue = {
            "version": 1,
            "last_updated": "2026-01-01T00:00:00Z",
            "experiments": [
                {
                    "id": "exp-0001",
                    "hypothesis": "Test",
                    "category": "lr",
                    "priority": 1,
                    "status": "pending",
                    "assigned_to": None,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
        }
        (knowledge_dir / "queue.json").write_text(
            json.dumps(queue), encoding="utf-8"
        )
        loaded = load_queue(knowledge_dir)
        assert len(loaded["experiments"]) == 1
        assert loaded["experiments"][0]["id"] == "exp-0001"

    def test_load_corrupt_file_returns_empty(self, knowledge_dir: Path):
        """A corrupt queue.json should return an empty queue, not crash."""
        (knowledge_dir / "queue.json").write_text("NOT JSON", encoding="utf-8")
        queue = load_queue(knowledge_dir)
        assert queue["experiments"] == []


# ---------------------------------------------------------------------------
# add_to_queue
# ---------------------------------------------------------------------------


class TestAddToQueue:
    def test_add_single_experiment(self, knowledge_dir: Path):
        exp = add_to_queue(
            knowledge_dir,
            hypothesis="Try larger LR",
            category="learning_rate",
            priority=2,
        )
        assert exp["id"] == "exp-0001"
        assert exp["hypothesis"] == "Try larger LR"
        assert exp["category"] == "learning_rate"
        assert exp["priority"] == 2
        assert exp["status"] == "pending"
        assert exp["assigned_to"] is None
        assert exp["created_at"] is not None

    def test_add_default_priority(self, knowledge_dir: Path):
        exp = add_to_queue(
            knowledge_dir,
            hypothesis="Default priority test",
            category="misc",
        )
        assert exp["priority"] == 5

    def test_add_multiple_sequential_ids(self, knowledge_dir: Path):
        e1 = add_to_queue(knowledge_dir, hypothesis="First", category="a")
        e2 = add_to_queue(knowledge_dir, hypothesis="Second", category="b")
        e3 = add_to_queue(knowledge_dir, hypothesis="Third", category="c")
        assert e1["id"] == "exp-0001"
        assert e2["id"] == "exp-0002"
        assert e3["id"] == "exp-0003"

    def test_add_persists_to_disk(self, knowledge_dir: Path):
        add_to_queue(knowledge_dir, hypothesis="Persisted", category="test")
        queue = load_queue(knowledge_dir)
        assert len(queue["experiments"]) == 1
        assert queue["last_updated"] is not None


# ---------------------------------------------------------------------------
# claim_next_experiment
# ---------------------------------------------------------------------------


class TestClaimNextExperiment:
    def test_claim_returns_highest_priority(self, knowledge_dir: Path):
        """Priority 1 should be claimed before priority 3."""
        add_to_queue(knowledge_dir, hypothesis="Low priority", category="a", priority=3)
        add_to_queue(knowledge_dir, hypothesis="High priority", category="b", priority=1)
        add_to_queue(knowledge_dir, hypothesis="Medium priority", category="c", priority=2)

        exp = claim_next_experiment(knowledge_dir, worker_id="worker-1")
        assert exp is not None
        assert exp["hypothesis"] == "High priority"
        assert exp["priority"] == 1
        assert exp["status"] == "in_progress"
        assert exp["assigned_to"] == "worker-1"

    def test_claim_skips_assigned(self, knowledge_dir: Path):
        """Already claimed (in_progress) experiments should be skipped."""
        add_to_queue(knowledge_dir, hypothesis="First", category="a", priority=1)
        add_to_queue(knowledge_dir, hypothesis="Second", category="b", priority=1)

        # Claim the first
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        # Second claim should get the other experiment
        exp = claim_next_experiment(knowledge_dir, worker_id="worker-2")
        assert exp is not None
        assert exp["hypothesis"] == "Second"
        assert exp["assigned_to"] == "worker-2"

    def test_claim_empty_queue_returns_none(self, knowledge_dir: Path):
        result = claim_next_experiment(knowledge_dir, worker_id="worker-1")
        assert result is None

    def test_claim_all_assigned_returns_none(self, knowledge_dir: Path):
        add_to_queue(knowledge_dir, hypothesis="Only one", category="a", priority=1)
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        result = claim_next_experiment(knowledge_dir, worker_id="worker-2")
        assert result is None

    def test_claim_persists_status(self, knowledge_dir: Path):
        """After claiming, the queue on disk should reflect in_progress status."""
        add_to_queue(knowledge_dir, hypothesis="Track this", category="a")
        claim_next_experiment(knowledge_dir, worker_id="worker-1")

        queue = load_queue(knowledge_dir)
        exp = queue["experiments"][0]
        assert exp["status"] == "in_progress"
        assert exp["assigned_to"] == "worker-1"

    def test_claim_fifo_within_same_priority(self, knowledge_dir: Path):
        """Among same-priority experiments, oldest (first added) wins."""
        add_to_queue(knowledge_dir, hypothesis="Oldest", category="a", priority=2)
        add_to_queue(knowledge_dir, hypothesis="Newest", category="b", priority=2)

        exp = claim_next_experiment(knowledge_dir, worker_id="worker-1")
        assert exp is not None
        assert exp["hypothesis"] == "Oldest"


# ---------------------------------------------------------------------------
# complete_experiment
# ---------------------------------------------------------------------------


class TestCompleteExperiment:
    def test_complete_marks_status(self, knowledge_dir: Path):
        exp = add_to_queue(knowledge_dir, hypothesis="To complete", category="a")
        claim_next_experiment(knowledge_dir, worker_id="worker-1")
        complete_experiment(knowledge_dir, exp["id"])

        queue = load_queue(knowledge_dir)
        assert queue["experiments"][0]["status"] == "completed"

    def test_complete_custom_status(self, knowledge_dir: Path):
        exp = add_to_queue(knowledge_dir, hypothesis="To fail", category="a")
        complete_experiment(knowledge_dir, exp["id"], status="failed")

        queue = load_queue(knowledge_dir)
        assert queue["experiments"][0]["status"] == "failed"

    def test_complete_nonexistent_id_is_noop(self, knowledge_dir: Path):
        """Completing a non-existent id should not raise."""
        add_to_queue(knowledge_dir, hypothesis="Exists", category="a")
        complete_experiment(knowledge_dir, "exp-9999")

        queue = load_queue(knowledge_dir)
        assert queue["experiments"][0]["status"] == "pending"
