"""Deterministic mock backend for failure-driven repair tests."""

from __future__ import annotations

from typing import Any
import json

from src.llm.backends.base import BaseLLMBackend


class MockLLMBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        context = extract_context(prompt)
        repaired = "failure_report" in prompt or self.calls > 1
        payload = build_payload(context, repaired=repaired)
        return json.dumps(payload, ensure_ascii=False)


def extract_context(prompt: str) -> dict[str, Any]:
    marker = "Scenario context JSON:\n"
    if marker not in prompt:
        return {"active_robots": ["robot_0"], "task_ids": [], "scenario": {}}
    tail = prompt.split(marker, 1)[1]
    end_markers = [
        "\n\nDAG repair instructions:",
        "\n\nStrict XML schema",
        "\n\nRepair iteration:",
        "\n\nPrevious robot_trees:",
    ]
    for end_marker in end_markers:
        if end_marker in tail:
            tail = tail.split(end_marker, 1)[0]
    try:
        return json.loads(tail)
    except json.JSONDecodeError:
        return {"active_robots": ["robot_0"], "task_ids": [], "scenario": {}}


def build_payload(context: dict[str, Any], repaired: bool) -> dict[str, Any]:
    robots = list(context.get("active_robots") or ["robot_0"])
    task_ids = list(context.get("task_ids") or [])
    scenario = context.get("scenario") or {}
    method = str(context.get("method") or "")
    task_type = str(context.get("task_type") or scenario.get("task", {}).get("type") or "")
    assignment = build_assignment(robots, task_ids, task_type, repaired, scenario, method)
    robot_trees = {
        robot_id: build_tree(
            task_ids_for_robot(assignment, robot_id),
            task_type,
            scenario,
            method,
            repaired,
            robot_id,
            robots,
        )
        for robot_id in robots
    }
    return {
        "robot_trees": robot_trees,
        "assignment": assignment,
        "coordination_notes": [
            "Use simulator movement primitives and centralized rule layer.",
            "Request resources before manipulation or recovery actions.",
        ],
        "assumptions": [
            "Grid coordinates are discrete.",
            "The centralized rule layer handles movement reservations and locks.",
        ],
    }


def build_assignment(
    robots: list[str],
    task_ids: list[str],
    task_type: str,
    repaired: bool,
    scenario: dict[str, Any],
    method: str,
) -> list[dict[str, Any]]:
    if method.startswith("llm_dag_") or task_type == "dag_pickplace" or scenario.get("task_dag"):
        scenario_assignments = (scenario.get("task") or {}).get("assignments") or {}
        if scenario_assignments:
            return [
                {
                    "robot_id": robot_id,
                    "task_ids": list(scenario_assignments.get(robot_id, [])),
                    "reason": "dag mock uses scenario assignment",
                }
                for robot_id in robots
            ]
        rows: list[dict[str, Any]] = []
        for index, robot_id in enumerate(robots):
            rows.append(
                {
                    "robot_id": robot_id,
                    "task_ids": [
                        task_id
                        for offset, task_id in enumerate(task_ids)
                        if offset % max(len(robots), 1) == index
                    ],
                    "reason": "dag mock balanced assignment",
                }
            )
        return rows
    rows: list[dict[str, Any]] = []
    if not repaired and len(robots) > 1 and task_ids:
        rows.append(
            {
                "robot_id": robots[0],
                "task_ids": list(task_ids),
                "reason": "mock initial plan assigns all tasks to the first robot",
            }
        )
        rows.append(
            {
                "robot_id": robots[1],
                "task_ids": [task_ids[0]],
                "reason": "mock initial plan intentionally duplicates one task",
            }
        )
        for robot_id in robots[2:]:
            rows.append({"robot_id": robot_id, "task_ids": [], "reason": "idle"})
        return rows
    scenario_assignments = (scenario.get("task") or {}).get("assignments") or {}
    if scenario_assignments:
        return [
            {
                "robot_id": robot_id,
                "task_ids": list(scenario_assignments.get(robot_id, [])),
                "reason": "repaired assignment follows scenario baseline",
            }
            for robot_id in robots
        ]
    for index, robot_id in enumerate(robots):
        rows.append(
            {
                "robot_id": robot_id,
                "task_ids": [
                    task_id for offset, task_id in enumerate(task_ids) if offset % len(robots) == index
                ],
                "reason": "balanced repaired assignment",
            }
        )
    return rows


def task_ids_for_robot(assignment: list[dict[str, Any]], robot_id: str) -> list[str]:
    for item in assignment:
        if item.get("robot_id") == robot_id:
            return list(item.get("task_ids") or [])
    return []


def build_tree(
    task_ids: list[str],
    task_type: str,
    scenario: dict[str, Any],
    method: str,
    repaired: bool,
    robot_id: str,
    robots: list[str],
) -> str:
    actions: list[str] = []
    if scenario.get("task_dag") or task_type == "dag_pickplace":
        return build_dag_tree(task_ids, scenario, method, repaired, robot_id, robots)
    if task_type == "navigation_pickplace":
        task_by_id = {
            task["task_id"]: task
            for task in scenario.get("pickplace", {}).get("tasks", [])
            if "task_id" in task
        }
        for task_id in task_ids:
            object_id = task_by_id.get(task_id, {}).get("object_id", task_id.replace("task_", ""))
            actions.extend(
                [
                    f'<Action ID="RequestResource" resource_id="{object_id}" resource_type="object"/>',
                    f'<Action ID="NavigateToPickup" object_id="{object_id}"/>',
                    f'<Action ID="PickObject" object_id="{object_id}"/>',
                    f'<Action ID="NavigateToDrop" object_id="{object_id}"/>',
                    f'<Action ID="PlaceObject" object_id="{object_id}"/>',
                    f'<Action ID="ReleaseResource" resource_id="{object_id}"/>',
                ]
            )
    else:
        if task_type == "navigation_recovery":
            actions.append(
                '<Action ID="RequestResource" resource_id="recovery_zone_auto" resource_type="recovery_zone"/>'
            )
            actions.append('<Action ID="ClearCostmap"/>')
        for task_id in task_ids:
            actions.append(f'<Action ID="NavigateToWaypoint" waypoint_id="{task_id}"/>')
    actions.append('<Action ID="ReportStatus" status="done"/>')
    body = "".join(actions)
    return (
        '<root main_tree_to_execute="MainTree">'
        '<BehaviorTree ID="MainTree">'
        f"<Sequence>{body}</Sequence>"
        "</BehaviorTree>"
        "</root>"
    )


def build_dag_tree(
    task_ids: list[str],
    scenario: dict[str, Any],
    method: str,
    repaired: bool,
    robot_id: str,
    robots: list[str],
) -> str:
    actions: list[str] = []
    dependency_guards_enabled = "without_dependency" not in method or repaired
    dag_tasks = {
        task["task_id"]: task
        for task in scenario.get("task_dag", {}).get("tasks", [])
        if "task_id" in task
    }
    pick_tasks = {
        task.get("task_id"): task
        for task in scenario.get("pickplace", {}).get("tasks", [])
        if task.get("task_id")
    }
    final_check_robot = dag_tasks.get("final_check", {}).get("assigned_robot")
    for task_id in task_ids:
        task = pick_tasks.get(task_id, {})
        object_id = str(task.get("object_id") or task_id.replace("task_", ""))
        pick_task_id = str(task.get("pick_task_id") or f"pick_{object_id}")
        place_task_id = str(task.get("place_task_id") or f"place_{object_id}")
        if dependency_guards_enabled:
            actions.append(f'<Action ID="WaitForDependency" task_id="{pick_task_id}"/>')
        actions.append(
            f'<Action ID="RequestResource" resource_id="{object_id}" resource_type="object"/>'
        )
        actions.append(
            f'<Action ID="PickObject" object_id="{object_id}" task_id="{pick_task_id}"/>'
        )
        if dependency_guards_enabled:
            actions.append(f'<Action ID="WaitForDependency" task_id="{place_task_id}"/>')
        actions.append(
            f'<Action ID="PlaceObject" object_id="{object_id}" task_id="{place_task_id}"/>'
        )
        actions.append(f'<Action ID="ReleaseResource" resource_id="{object_id}"/>')
    if final_check_robot in {None, ""}:
        final_check_robot = robots_first_robot(scenario)
    if final_check_robot == robot_id:
        if dependency_guards_enabled:
            actions.append('<Action ID="WaitForDependency" task_id="final_check"/>')
        actions.append('<Action ID="FinalCheck" task_id="final_check"/>')
    actions.append('<Action ID="ReportStatus" status="done"/>')
    body = "".join(actions)
    return (
        '<root main_tree_to_execute="MainTree">'
        '<BehaviorTree ID="MainTree">'
        f"<Sequence>{body}</Sequence>"
        "</BehaviorTree>"
        "</root>"
    )


def robots_first_robot(scenario: dict[str, Any]) -> str | None:
    robots = sorted((scenario.get("robots") or {}).keys())
    return robots[0] if robots else None
