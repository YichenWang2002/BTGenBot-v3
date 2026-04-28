"""Validation utilities for task dependency DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.dag.task_dag import TaskDAG


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def validate_task_dag(
    dag: TaskDAG,
    scenario: Any | None = None,
    robot_ids: Iterable[str] | None = None,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    validate_dependency_endpoints(dag, errors)
    validate_acyclic(dag, errors)
    validate_pick_place_dependencies(dag, errors)
    validate_terminal_task_dependencies(dag, errors)
    validate_assigned_robots(dag, scenario, robot_ids, errors)

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_dependency_endpoints(dag: TaskDAG, errors: list[str]) -> None:
    for edge in dag.dependencies:
        if edge.source_task_id not in dag.tasks:
            errors.append(
                f"Dependency references missing source_task_id: {edge.source_task_id}"
            )
        if edge.target_task_id not in dag.tasks:
            errors.append(
                f"Dependency references missing target_task_id: {edge.target_task_id}"
            )


def validate_acyclic(dag: TaskDAG, errors: list[str]) -> None:
    try:
        dag.topological_sort()
    except ValueError as exc:
        errors.append(str(exc))


def validate_pick_place_dependencies(dag: TaskDAG, errors: list[str]) -> None:
    pick_tasks_by_object: dict[str, list[str]] = {}
    for task in dag.tasks.values():
        if task.task_type == "pick" and task.object_id:
            pick_tasks_by_object.setdefault(task.object_id, []).append(task.task_id)

    for task in dag.tasks.values():
        if task.task_type != "place" or not task.object_id:
            continue
        candidate_picks = pick_tasks_by_object.get(task.object_id, [])
        if not candidate_picks:
            errors.append(
                f"Place task {task.task_id} has no pick task for object_id {task.object_id}"
            )
            continue
        if not any(safe_has_path(dag, pick_task_id, task.task_id) for pick_task_id in candidate_picks):
            errors.append(
                f"Place task {task.task_id} must depend on a pick task for object_id {task.object_id}"
            )


def validate_terminal_task_dependencies(dag: TaskDAG, errors: list[str]) -> None:
    for task in dag.tasks.values():
        if task.task_type not in {"assembly", "final_check"}:
            continue
        if not dag.get_predecessors(task.task_id):
            errors.append(
                f"{task.task_type} task {task.task_id} must depend on at least one predecessor"
            )


def validate_assigned_robots(
    dag: TaskDAG,
    scenario: Any | None,
    robot_ids: Iterable[str] | None,
    errors: list[str],
) -> None:
    known_robot_ids = set(str(robot_id) for robot_id in (robot_ids or []))
    known_robot_ids.update(extract_robot_ids_from_scenario(scenario))
    if not known_robot_ids:
        return
    for task in dag.tasks.values():
        if task.assigned_robot and task.assigned_robot not in known_robot_ids:
            errors.append(
                f"Task {task.task_id} assigned_robot {task.assigned_robot} is not in scenario robots"
            )


def extract_robot_ids_from_scenario(scenario: Any | None) -> set[str]:
    if scenario is None:
        return set()
    if isinstance(scenario, dict):
        robots = scenario.get("robots") or {}
        if isinstance(robots, dict):
            return set(str(robot_id) for robot_id in robots)
        return set()
    state = getattr(scenario, "state", None)
    robots = getattr(state, "robots", None)
    if isinstance(robots, dict):
        return set(str(robot_id) for robot_id in robots)
    return set()


def safe_has_path(dag: TaskDAG, source_task_id: str, target_task_id: str) -> bool:
    try:
        return dag.has_dependency_path(source_task_id, target_task_id)
    except KeyError:
        return False
