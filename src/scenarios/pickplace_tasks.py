"""Pick/place task construction and assignment helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.scenarios.waypoint_assignment import manhattan


def build_pickplace_tasks(objects: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for object_id, obj in sorted(objects.items()):
        tasks.append(
            {
                "task_id": f"task_{object_id}",
                "object_id": object_id,
                "pickup_position": list(obj["position"]),
                "drop_position": list(obj["target_position"]),
                "assigned_robot": None,
                "status": "pending",
                "attempts": 0,
            }
        )
    return tasks


def assign_tasks_nearest_robot(
    tasks: list[dict[str, Any]], robots: dict[str, Any]
) -> list[dict[str, Any]]:
    assigned = deepcopy(tasks)
    robot_positions = {
        robot_id: list(robot["start"]) for robot_id, robot in sorted(robots.items())
    }
    for task in assigned:
        robot_id = min(
            robot_positions,
            key=lambda candidate: (
                manhattan(robot_positions[candidate], task["pickup_position"]),
                candidate,
            ),
        )
        task["assigned_robot"] = robot_id
        task["status"] = "assigned"
    return assigned


def assign_tasks_round_robin(
    tasks: list[dict[str, Any]], robots: dict[str, Any]
) -> list[dict[str, Any]]:
    robot_ids = sorted(robots)
    assigned = deepcopy(tasks)
    for index, task in enumerate(assigned):
        task["assigned_robot"] = robot_ids[index % len(robot_ids)]
        task["status"] = "assigned"
    return assigned


def reassign_failed_task(
    task: dict[str, Any],
    robots: dict[str, Any],
    failed_robot_id: str,
    strategy: str = "nearest_robot",
) -> dict[str, Any]:
    candidates = {
        robot_id: robot for robot_id, robot in robots.items() if robot_id != failed_robot_id
    }
    if not candidates:
        reassigned = deepcopy(task)
        reassigned["assigned_robot"] = failed_robot_id
        reassigned["status"] = "assigned"
        return reassigned
    reassigned = deepcopy(task)
    if strategy == "round_robin":
        reassigned["assigned_robot"] = sorted(candidates)[0]
    else:
        reassigned["assigned_robot"] = min(
            candidates,
            key=lambda robot_id: manhattan(candidates[robot_id]["start"], task["pickup_position"]),
        )
    reassigned["status"] = "reassigned"
    return reassigned


def all_tasks_completed(tasks: list[dict[str, Any]]) -> bool:
    return all(task.get("status") == "placed" for task in tasks)


def assignments_by_robot(tasks: list[dict[str, Any]], robot_ids: list[str]) -> dict[str, list[str]]:
    assignments = {robot_id: [] for robot_id in robot_ids}
    for task in tasks:
        robot_id = task.get("assigned_robot")
        if robot_id in assignments:
            assignments[robot_id].append(task["task_id"])
    return assignments
