"""Waypoint assignment strategies for navigation benchmarks."""

from __future__ import annotations

import random
from typing import Any


def assign_waypoints(
    strategy: str,
    robot_ids: list[str],
    waypoints: list[dict[str, Any]],
    robot_starts: dict[str, list[int]] | None = None,
    seed: int = 0,
) -> dict[str, list[str]]:
    if not robot_ids:
        raise ValueError("At least one robot is required")

    normalized = strategy.lower()
    if normalized == "sequential_single_robot":
        return sequential_single_robot(robot_ids, waypoints)
    if normalized == "round_robin":
        return round_robin(robot_ids, waypoints)
    if normalized == "nearest_robot":
        if robot_starts is None:
            raise ValueError("nearest_robot requires robot_starts")
        return nearest_robot(robot_ids, waypoints, robot_starts)
    if normalized == "random_balanced":
        return random_balanced(robot_ids, waypoints, seed)
    raise ValueError(f"Unknown assignment strategy: {strategy}")


def sequential_single_robot(
    robot_ids: list[str], waypoints: list[dict[str, Any]]
) -> dict[str, list[str]]:
    assignments = {robot_id: [] for robot_id in robot_ids}
    assignments[robot_ids[0]] = [str(waypoint["id"]) for waypoint in waypoints]
    return assignments


def round_robin(
    robot_ids: list[str], waypoints: list[dict[str, Any]]
) -> dict[str, list[str]]:
    assignments = {robot_id: [] for robot_id in robot_ids}
    for index, waypoint in enumerate(waypoints):
        robot_id = robot_ids[index % len(robot_ids)]
        assignments[robot_id].append(str(waypoint["id"]))
    return assignments


def nearest_robot(
    robot_ids: list[str],
    waypoints: list[dict[str, Any]],
    robot_starts: dict[str, list[int]],
) -> dict[str, list[str]]:
    assignments = {robot_id: [] for robot_id in robot_ids}

    for waypoint in waypoints:
        waypoint_position = waypoint["position"]
        robot_id = min(
            robot_ids,
            key=lambda candidate: (
                manhattan(robot_starts[candidate], waypoint_position),
                len(assignments[candidate]),
                candidate,
            ),
        )
        assignments[robot_id].append(str(waypoint["id"]))
    return assignments


def random_balanced(
    robot_ids: list[str], waypoints: list[dict[str, Any]], seed: int
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = [str(waypoint["id"]) for waypoint in waypoints]
    rng.shuffle(shuffled)

    assignments = {robot_id: [] for robot_id in robot_ids}
    for waypoint_id in shuffled:
        min_load = min(len(items) for items in assignments.values())
        candidates = [
            robot_id for robot_id in robot_ids if len(assignments[robot_id]) == min_load
        ]
        assignments[rng.choice(candidates)].append(waypoint_id)
    return assignments


def manhattan(left: list[int], right: list[int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
