"""Task DAG status helpers."""

from __future__ import annotations


PENDING = "pending"
READY = "ready"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
BLOCKED = "blocked"

VALID_STATUSES = {
    PENDING,
    READY,
    RUNNING,
    COMPLETED,
    FAILED,
    BLOCKED,
}


def normalize_status(status: str | None) -> str:
    value = str(status or PENDING)
    if value not in VALID_STATUSES:
        raise ValueError(f"Unknown task status: {value}")
    return value


def ensure_transition_allowed(current: str, target: str) -> None:
    current = normalize_status(current)
    target = normalize_status(target)
    if current == COMPLETED and target != COMPLETED:
        raise ValueError("Completed tasks cannot transition to another status")
    if current == FAILED and target not in {FAILED, BLOCKED}:
        raise ValueError("Failed tasks cannot transition back to active status")
