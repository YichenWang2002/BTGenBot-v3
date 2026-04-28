"""Deterministic recovery obstacle injection helpers."""

from __future__ import annotations

import random
from typing import Any


def generate_recovery_config(
    robot_starts: dict[str, list[int]],
    seed: int,
    block_prob: float,
    recovery_zone_radius: int = 10,
) -> dict[str, Any]:
    rng = random.Random(seed)
    blocked_cells: list[list[int]] = []
    temporary_obstacles: list[dict[str, Any]] = []

    for index, start in enumerate(robot_starts.values()):
        if rng.random() <= block_prob or index == 0:
            blocked_cells.append([start[0] + 1, start[1]])
        else:
            temporary_obstacles.append(
                {"cell": [start[0] + 2, start[1]], "appear_at": 1, "disappear_at": 8}
            )

    if len(blocked_cells) < min(2, len(robot_starts)):
        starts = list(robot_starts.values())
        for start in starts[: min(2, len(starts))]:
            cell = [start[0] + 1, start[1]]
            if cell not in blocked_cells:
                blocked_cells.append(cell)

    return {
        "enabled": True,
        "trigger_mode": "blocked_path",
        "recovery_zone_radius": recovery_zone_radius,
        "recovery_lock_ttl": 5,
        "actions": ["ClearEntireCostmap", "Spin", "Wait", "BackUp"],
        "blocked_cells": blocked_cells,
        "temporary_obstacles": temporary_obstacles,
        "stuck_threshold": 3,
    }


def active_temporary_obstacles(
    temporary_obstacles: list[dict[str, Any]], timestep: int
) -> set[tuple[int, int]]:
    active: set[tuple[int, int]] = set()
    for item in temporary_obstacles:
        if int(item.get("appear_at", 0)) <= timestep < int(item.get("disappear_at", 10**9)):
            cell = item["cell"]
            active.add((int(cell[0]), int(cell[1])))
    return active


def recovery_zone_id(cell: list[int] | tuple[int, int], radius: int) -> str:
    return f"recovery_zone_{int(cell[0]) // radius}_{int(cell[1]) // radius}"

