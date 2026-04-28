"""Build structured failure reports from validation and simulation outputs."""

from __future__ import annotations

from typing import Any

from src.llm.schema import scenario_task_ids


def build_failure_report(
    scenario: Any,
    assignment: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    trace: list[dict[str, Any]] | None = None,
    static_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assignment = assignment or []
    metrics = metrics or {}
    trace = trace or []
    static_validation = static_validation or {}
    failure_types: list[str] = []
    details: list[dict[str, Any]] = []

    if static_validation.get("errors"):
        for error in static_validation["errors"]:
            detail_type = classify_static_error(str(error))
            add_failure(failure_types, detail_type)
            details.append({"type": detail_type, "message": str(error)})

    assigned = [str(task_id) for row in assignment for task_id in row.get("task_ids", [])]
    duplicates = sorted({task_id for task_id in assigned if assigned.count(task_id) > 1})
    if duplicates:
        add_failure(failure_types, "duplicate_task")
        details.append(
            {
                "type": "duplicate_task",
                "task_ids": duplicates,
                "message": "one or more tasks were assigned to multiple robots",
            }
        )
    missing = sorted(scenario_task_ids(scenario) - set(assigned))
    if missing:
        add_failure(failure_types, "unassigned_task")
        details.append(
            {
                "type": "unassigned_task",
                "task_ids": missing,
                "message": "one or more scenario tasks were not assigned",
            }
        )

    metric_checks = [
        ("collision_count", "collision"),
        ("edge_conflict_count", "edge_conflict"),
        ("resource_conflict_count", "resource_conflict"),
        ("recovery_conflict_count", "recovery_conflict"),
        ("dag_violation_count", "blocked_task_executed"),
        ("deadlock_count", "deadlock"),
        ("pick_failure_count", "pick_failure"),
        ("place_failure_count", "place_failure"),
    ]
    for metric_key, failure_type in metric_checks:
        if int(metrics.get(metric_key, 0) or 0) > 0:
            add_failure(failure_types, failure_type)
            details.append(
                {
                    "type": failure_type,
                    "metric": metric_key,
                    "value": metrics.get(metric_key),
                    "message": f"{metric_key} was non-zero",
                }
            )
    if int(metrics.get("object_conflict_count", 0) or 0) > 0 or int(
        metrics.get("duplicate_pick_attempt_count", 0) or 0
    ) > 0:
        add_failure(failure_types, "resource_conflict")
        add_failure(failure_types, "duplicate_task")
        details.extend(extract_pickplace_conflict_details(trace))
    if int(metrics.get("dependency_wait_count", 0) or 0) > 0:
        add_failure(failure_types, "missing_dependency_wait")
        details.append(
            {
                "type": "missing_dependency_wait",
                "metric": "dependency_wait_count",
                "value": metrics.get("dependency_wait_count"),
                "message": "one or more tasks waited on DAG dependencies",
            }
        )
    if int(metrics.get("dag_violation_count", 0) or 0) > 0:
        add_failure(failure_types, "dependency_not_satisfied")
        details.append(
            {
                "type": "dependency_not_satisfied",
                "metric": "dag_violation_count",
                "value": metrics.get("dag_violation_count"),
                "message": "one or more tasks executed before their dependencies were satisfied",
            }
        )
    if int(metrics.get("dag_violation_count", 0) or 0) > 0:
        add_failure(failure_types, "blocked_task_executed")
    if metrics.get("timeout"):
        add_failure(failure_types, "timeout")
        details.append({"type": "timeout", "message": "simulation timed out"})

    total_tasks = len(scenario_task_ids(scenario))
    if (
        scenario.pickplace.get("enabled")
        and total_tasks
        and int(metrics.get("tasks_completed", 0) or 0) < total_tasks
    ):
        add_failure(failure_types, "incomplete_task")
        details.append(
            {
                "type": "incomplete_task",
                "tasks_completed": metrics.get("tasks_completed", 0),
                "total_tasks": total_tasks,
                "message": "not all tasks completed",
            }
        )

    status = "failed" if failure_types else "success"
    return {
        "status": status,
        "failure_types": failure_types,
        "details": details,
        "suggested_repairs": suggested_repairs(failure_types),
    }


def add_failure(failure_types: list[str], failure_type: str) -> None:
    if failure_type not in failure_types:
        failure_types.append(failure_type)


def classify_static_error(error: str) -> str:
    lowered = error.lower()
    if "duplicate" in lowered:
        return "duplicate_task"
    if "unassigned" in lowered:
        return "unassigned_task"
    if "invalid dag task_id" in lowered:
        return "invalid_task_id"
    if "assigned to" in lowered and "executed in this robot tree" in lowered:
        return "wrong_assigned_robot_for_task"
    if "blocked task" in lowered and "dependency check" in lowered:
        return "missing_dependency_wait"
    if "place task" in lowered and "missing a pick dependency" in lowered:
        return "dependency_not_satisfied"
    if "final_check" in lowered and "not ready" in lowered:
        return "final_check_not_ready"
    if "resource" in lowered:
        return "resource_conflict"
    return "static_validation"


def extract_pickplace_conflict_details(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for step in trace:
        for event in step.get("pickplace_events", []):
            if event.get("event_type") in {
                "object_conflict",
                "duplicate_pick_attempt",
                "pickup_zone_conflict",
                "drop_zone_conflict",
            }:
                details.append(
                    {
                        "type": "resource_conflict",
                        "timestep": step.get("timestep"),
                        "robots": event.get("robot_ids") or [event.get("robot_id")],
                        "resource_id": event.get("object_id")
                        or event.get("pickup_zone_id")
                        or event.get("drop_zone_id"),
                        "message": "multiple robots attempted to use the same pick/place resource",
                    }
                )
    if not details:
        details.append(
            {
                "type": "resource_conflict",
                "message": "object or duplicate pick conflict metrics were non-zero",
            }
        )
    return details


def suggested_repairs(failure_types: list[str]) -> list[str]:
    suggestions: list[str] = []
    if "duplicate_task" in failure_types:
        suggestions.append("assign each task_id to exactly one robot")
    if "unassigned_task" in failure_types or "incomplete_task" in failure_types:
        suggestions.append("ensure all waypoints or pick/place tasks appear in assignment")
    if "resource_conflict" in failure_types:
        suggestions.append("use RequestResource before PickObject, PlaceObject, or recovery actions")
    if "missing_dependency_wait" in failure_types:
        suggestions.append("insert WaitForDependency before blocked DAG task actions")
    if "invalid_task_id" in failure_types:
        suggestions.append("use only task_id values defined in task_dag")
    if "wrong_assigned_robot_for_task" in failure_types:
        suggestions.append("respect assigned_robot for each DAG task")
    if "dependency_not_satisfied" in failure_types or "blocked_task_executed" in failure_types:
        suggestions.append("do not run a DAG task until all predecessors are complete")
    if "final_check_not_ready" in failure_types:
        suggestions.append("run final_check only after all required place tasks complete")
    if "collision" in failure_types or "edge_conflict" in failure_types:
        suggestions.append("enable centralized_rule or stagger robot motion with Wait actions")
    if "recovery_conflict" in failure_types:
        suggestions.append("use recovery_zone locks before ClearCostmap or Spin")
    if "timeout" in failure_types:
        suggestions.append("rebalance task assignment to reduce makespan")
    return suggestions or ["no repair needed"]
