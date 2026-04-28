"""Structured event helpers for centralized coordination rules."""

from __future__ import annotations

from typing import Any


def decision_result(
    allowed: bool,
    reason: str,
    conflict_type: str | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "reason": reason,
        "conflict_type": conflict_type,
        "owner": owner,
    }


def rule_event(
    timestep: int,
    robot_id: str,
    event_type: str,
    allowed: bool,
    reason: str,
    conflict_type: str | None = None,
    owner: str | None = None,
    from_cell: tuple[int, int] | list[int] | None = None,
    to_cell: tuple[int, int] | list[int] | None = None,
    resource_id: str | None = None,
    resource_type: str | None = None,
    safety_layer: str = "none",
    decision: str | None = None,
    converted_to_wait: bool = False,
    action_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestep": timestep,
        "robot_id": robot_id,
        "event_type": event_type,
        "action_id": action_id or event_type,
        "allowed": allowed,
        "reason": reason,
        "conflict_type": conflict_type,
        "owner": owner,
        "from_cell": list(from_cell) if from_cell is not None else None,
        "to_cell": list(to_cell) if to_cell is not None else None,
        "resource_id": resource_id,
        "resource_type": resource_type,
        "safety_layer": safety_layer,
        "decision": decision or ("allowed" if allowed else "denied"),
        "converted_to_wait": converted_to_wait,
        "details": dict(details or {}),
    }
