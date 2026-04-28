"""Run DAG coordination pick/place benchmark suites."""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data
from src.experiments.run_pickplace_experiment import (
    assign_tasks,
    assignments_by_robot,
    synchronize_task_dag_assignments,
)


METHODS = {
    "dag_enabled_centralized",
    "ignore_dag_centralized",
    "dag_enabled_without_approach_lock",
    "no_dag_naive",
    "oracle_topological_sequential",
    "topological_sequential_baseline",
}

TOPOLOGICAL_METHODS = {
    "oracle_topological_sequential",
    "topological_sequential_baseline",
}

DEPENDENCY_MODES = {"basic", "cross_object", "chain", "gate"}

SUMMARY_FIELDS = [
    "method",
    "runs",
    "success_rate",
    "makespan",
    "timeout_rate",
    "dependency_wait_count",
    "dag_violation_count",
    "dag_completed_task_count",
    "dag_task_completion_rate",
    "first_uncompleted_task_count",
    "resource_conflict_count",
    "collision_event_count",
    "actual_collision_count",
    "approach_zone_wait_count",
    "lock_wait_count",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = run_suite(args.suite)
    print(f"DAG pick/place experiment rows: {len(rows)}")
    return 0


def run_suite(suite_path: Path) -> list[dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    suite = load_yaml(suite_path)
    output_dir = Path(suite.get("output_dir", "results/dag_pickplace_smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(suite["scenario_manifest"])
    manifest = read_json(manifest_path)
    manifest_rows = manifest.get("scenarios", [])
    methods = list(suite.get("methods") or [])
    max_scenarios = int(suite.get("max_scenarios", len(manifest_rows)))

    unknown = sorted(set(methods) - METHODS)
    if unknown:
        raise ValueError(f"Unknown DAG pick/place methods: {unknown}")

    raw_rows: list[dict[str, Any]] = []
    dependency_modes = list(suite.get("dependency_modes") or [None])
    unknown_modes = sorted(
        mode for mode in dependency_modes if mode is not None and mode not in DEPENDENCY_MODES
    )
    if unknown_modes:
        raise ValueError(f"Unknown DAG dependency modes: {unknown_modes}")

    for scenario_index, manifest_row in enumerate(manifest_rows[:max_scenarios]):
        scenario_path = resolve_scenario_path(manifest_path, manifest_row["scenario_path"])
        loaded_base_data = load_yaml(scenario_path)
        if suite.get("stress_approach_contention"):
            loaded_base_data = apply_stress_approach_contention_layout(loaded_base_data)
        for dependency_mode in dependency_modes:
            base_data = build_dependency_mode_scenario(
                loaded_base_data, dependency_mode
            )
            scenario_label = str(scenario_index)
            if dependency_mode:
                scenario_label = f"{scenario_label}_{dependency_mode}"
            for method in methods:
                run_id = f"{suite.get('name', 'dag_pickplace')}_{scenario_label}_{method}"
                scenario_data = build_method_scenario(base_data, manifest_row, method, suite)
                if dependency_mode:
                    scenario_data["name"] = f"{scenario_data['name']}_{dependency_mode}"
                    scenario_data.setdefault("task", {})["dependency_mode"] = dependency_mode
                    scenario_data.setdefault("source", {})["dependency_mode"] = dependency_mode
                scenario = load_scenario_data(
                    scenario_data, default_name=scenario_data["name"]
                )
                max_steps = scenario.max_steps * method_max_steps_multiplier(method, suite)
                env = MultiRobotEnv(
                    scenario.state,
                    max_steps=max_steps,
                    render=False,
                    scenario_name=scenario.name,
                    centralized_rule=scenario.centralized_rule,
                    recovery_config=scenario.recovery,
                    pickplace_config=scenario.pickplace,
                    zones=scenario.zones,
                    coordination_config=scenario.coordination,
                    task_dag=scenario.task_dag,
                    ignore_task_dag=bool(scenario_data.get("ignore_task_dag", False)),
                )
                env.reset()
                try:
                    if is_topological_method(method):
                        run_topological_sequential_baseline(env)
                    else:
                        while not env.done:
                            env.step()
                finally:
                    env.close()

                oracle_debug = build_oracle_debug(env, method)
                trace_path = trace_dir / f"{run_id}.json"
                with trace_path.open("w", encoding="utf-8") as handle:
                    payload = env.metrics.to_trace_payload(
                        scenario_name=scenario.name,
                        num_robots=scenario.num_robots,
                    )
                    payload["oracle_debug"] = oracle_debug
                    json.dump(payload, handle, indent=2)
                    handle.write("\n")

                summary = env.metrics.summary()
                raw_rows.append(
                    build_raw_row(
                        run_id=run_id,
                        scenario=scenario,
                        scenario_data=scenario_data,
                        manifest_row=manifest_row,
                        method=method,
                        summary=summary,
                        trace_path=trace_path,
                        trace=env.metrics.trace,
                        oracle_debug=oracle_debug,
                    )
                )

    write_raw_jsonl(output_dir / "raw.jsonl", raw_rows)
    summary_rows = summarize_rows(raw_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_md(output_dir / "summary.md", summary_rows)
    return raw_rows


def build_method_scenario(
    base_data: dict[str, Any],
    manifest_row: dict[str, Any],
    method: str,
    suite: dict[str, Any],
) -> dict[str, Any]:
    data = deepcopy(base_data)
    data["name"] = f"{base_data['name']}_{method}"
    data["render"] = False
    data["coordination"] = dict(suite.get("coordination") or {})
    tasks = [dict(task) for task in data.get("pickplace", {}).get("tasks", [])]
    assignment_strategy = data.get("task", {}).get("assignment_strategy", "nearest_robot")

    if is_topological_method(method):
        centralized_rule = True
    elif method == "no_dag_naive":
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = False
        data.pop("task_dag", None)
    else:
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = True

    for robot in data["robots"].values():
        robot["goal"] = None
        robot["assigned_waypoints"] = []
        robot["assigned_tasks"] = []
        robot["carrying_object"] = None

    data.setdefault("pickplace", {})
    data["pickplace"]["enabled"] = True
    data["pickplace"]["tasks"] = tasks
    if suite.get("deterministic_pickplace") or is_topological_method(method):
        make_pickplace_deterministic(data)
    data.setdefault("task", {})
    data["task"]["assignment_strategy"] = assignment_strategy
    data["task"]["baseline_type"] = method
    data["task"]["assignments"] = assignments_by_robot(tasks, sorted(data["robots"]))
    data["centralized_rule"] = centralized_rule
    data["source"] = dict(data.get("source") or {})
    data["source"].setdefault("dataset_index", manifest_row.get("dataset_index"))

    if method != "no_dag_naive":
        synchronize_task_dag_assignments(data, tasks)

    apply_method_coordination(data, method)
    return data


def build_dependency_mode_scenario(
    base_data: dict[str, Any], dependency_mode: str | None
) -> dict[str, Any]:
    data = deepcopy(base_data)
    if dependency_mode is None:
        return data
    tasks = [dict(task) for task in data.get("pickplace", {}).get("tasks", [])]
    data.setdefault("task", {})["dependency_mode"] = dependency_mode
    data.setdefault("source", {})["dependency_mode"] = dependency_mode
    data.setdefault("pickplace", {})["tasks"] = tasks
    data["task_dag"] = build_task_dag_for_mode(tasks, dependency_mode)
    return data


def apply_stress_approach_contention_layout(base_data: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(base_data)
    robot_ids = sorted((data.get("robots") or {}).keys())
    object_ids = sorted((data.get("objects") or {}).keys())
    if len(robot_ids) < 2 or len(object_ids) < 2:
        return data

    starts = [[1, 1], [1, 3], [1, 2]]
    for index, robot_id in enumerate(robot_ids[:3]):
        data["robots"][robot_id]["start"] = list(starts[index])
        data["robots"][robot_id]["goal"] = None
        data["robots"][robot_id]["carrying_object"] = None

    pickups = [[4, 3], [4, 1], [4, 2]]
    drops = [[6, 3], [6, 1], [6, 2]]
    for index, object_id in enumerate(object_ids[:3]):
        data["objects"][object_id]["position"] = list(pickups[index])
        data["objects"][object_id]["target_position"] = list(drops[index])
        data["objects"][object_id]["held_by"] = None
        data["objects"][object_id]["status"] = "available"

    for task in data.get("pickplace", {}).get("tasks", []):
        object_id = str(task.get("object_id"))
        if object_id not in object_ids[:3]:
            continue
        index = object_ids.index(object_id)
        task["pickup_position"] = list(pickups[index])
        task["drop_position"] = list(drops[index])
        task["assigned_robot"] = robot_ids[index % len(robot_ids)]
        task["status"] = "assigned"

    data["zones"] = {
        "pickup_zones": {
            "pickup_zone_stress": {"cells": [[4, 1], [4, 2], [4, 3]]},
        },
        "drop_zones": {
            "drop_zone_stress": {"cells": [[6, 1], [6, 2], [6, 3]]},
        },
    }
    data.setdefault("task", {})["assignment_strategy"] = "round_robin"
    return data


def build_task_dag_for_mode(
    pickplace_tasks: list[dict[str, Any]], dependency_mode: str
) -> dict[str, Any]:
    sorted_tasks = sorted(pickplace_tasks, key=lambda row: str(row.get("object_id", "")))
    dag_tasks: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    object_ids: list[str] = []

    for task in sorted_tasks:
        object_id = str(task["object_id"])
        assigned_robot = task.get("assigned_robot")
        pick_task_id = f"pick_{object_id}"
        place_task_id = f"place_{object_id}"
        task["pick_task_id"] = pick_task_id
        task["place_task_id"] = place_task_id
        object_ids.append(object_id)
        dag_tasks.append(
            {
                "task_id": pick_task_id,
                "task_type": "pick",
                "object_id": object_id,
                "assigned_robot": assigned_robot,
                "expected_action_id": "PickObject",
            }
        )
        dag_tasks.append(
            {
                "task_id": place_task_id,
                "task_type": "place",
                "object_id": object_id,
                "assigned_robot": assigned_robot,
                "expected_action_id": "PlaceObject",
            }
        )

    final_robot = sorted_tasks[0].get("assigned_robot") if sorted_tasks else None
    if dependency_mode == "gate":
        dag_tasks.append(
            {
                "task_id": "recovery_clear_zone",
                "task_type": "recovery_clear_zone",
                "assigned_robot": final_robot,
                "expected_action_id": "ClearRecoveryZone",
            }
        )

    dag_tasks.append(
        {
            "task_id": "final_check",
            "task_type": "final_check",
            "assigned_robot": final_robot,
            "expected_action_id": "FinalCheck",
        }
    )

    if dependency_mode == "chain":
        for index, object_id in enumerate(object_ids):
            dependencies.append(
                {"source": f"pick_{object_id}", "target": f"place_{object_id}"}
            )
            if index + 1 < len(object_ids):
                next_object_id = object_ids[index + 1]
                dependencies.append(
                    {
                        "source": f"place_{object_id}",
                        "target": f"pick_{next_object_id}",
                        "description": "chain_dependency",
                    }
                )
        if object_ids:
            dependencies.append(
                {"source": f"place_{object_ids[-1]}", "target": "final_check"}
            )
        return {"tasks": dag_tasks, "dependencies": dependencies}

    for object_id in object_ids:
        dependencies.append(
            {"source": f"pick_{object_id}", "target": f"place_{object_id}"}
        )
        dependencies.append({"source": f"place_{object_id}", "target": "final_check"})

    if dependency_mode == "cross_object" and len(object_ids) >= 3:
        dependencies.extend(
            [
                {
                    "source": f"place_{object_ids[0]}",
                    "target": f"place_{object_ids[2]}",
                    "description": "cross_object_dependency",
                },
                {
                    "source": f"place_{object_ids[1]}",
                    "target": f"place_{object_ids[2]}",
                    "description": "cross_object_dependency",
                },
            ]
        )
    elif dependency_mode == "gate" and len(object_ids) >= 2:
        dependencies.extend(
            [
                {"source": "recovery_clear_zone", "target": f"pick_{object_ids[0]}"},
                {"source": "recovery_clear_zone", "target": f"pick_{object_ids[1]}"},
            ]
        )

    return {"tasks": dag_tasks, "dependencies": dependencies}


def make_pickplace_deterministic(data: dict[str, Any]) -> None:
    pickplace = data.setdefault("pickplace", {})
    pickplace["pick_fail_prob"] = 0.0
    pickplace["place_fail_prob"] = 0.0
    pickplace["object_temporarily_unavailable_prob"] = 0.0
    pickplace["allow_reassignment"] = False
    for obj in (data.get("objects") or {}).values():
        if isinstance(obj, dict):
            obj["status"] = "available"


def apply_method_coordination(data: dict[str, Any], method: str) -> None:
    coordination = data.setdefault("coordination", {})
    if method == "no_dag_naive":
        coordination["enable_cell_reservation"] = False
        coordination["enable_edge_reservation"] = False
        coordination["enable_resource_lock"] = False
        coordination["enable_recovery_lock"] = False
        coordination["enable_approach_zone_lock"] = False
        data["ignore_task_dag"] = False
        return

    coordination["enable_cell_reservation"] = True
    coordination["enable_edge_reservation"] = True
    coordination["enable_resource_lock"] = True
    coordination["enable_recovery_lock"] = True
    coordination["approach_radius"] = 1
    coordination["enable_approach_zone_lock"] = (
        method != "dag_enabled_without_approach_lock"
    )
    data["ignore_task_dag"] = method == "ignore_dag_centralized"


def is_topological_method(method: str) -> bool:
    return method in TOPOLOGICAL_METHODS


def method_max_steps_multiplier(method: str, suite: dict[str, Any]) -> int:
    method_config = dict((suite.get("method_overrides") or {}).get(method) or {})
    if "max_steps_multiplier" in method_config:
        return max(1, int(method_config["max_steps_multiplier"]))
    if is_topological_method(method):
        return max(1, int(suite.get("topological_max_steps_multiplier", 3)))
    return 1


def run_topological_sequential_baseline(env: MultiRobotEnv) -> None:
    if env.task_dag is None:
        while not env.done:
            env.step()
        return

    order = env.task_dag.topological_sort()
    robot_ids = sorted(env.state.robots)
    env._oracle_debug = {
        "oracle_task_order": list(order),
        "oracle_task_index": 0,
        "oracle_current_task_id": order[0] if order else None,
        "oracle_task_failure_reason": None,
    }
    while not env.done:
        task_index, task_id = first_uncompleted_index(env, order)
        env._oracle_debug["oracle_task_index"] = task_index
        env._oracle_debug["oracle_current_task_id"] = task_id
        if task_id is None:
            break
        task = env.task_dag.tasks[task_id]
        if not env.task_dag.is_ready(task_id):
            env._oracle_debug["oracle_task_failure_reason"] = "task_not_ready"
            break
        robot_id = task.assigned_robot if task.assigned_robot in env.state.robots else None
        if robot_id is None:
            env._oracle_debug["oracle_task_failure_reason"] = "missing_assigned_robot"
            robot_id = robot_ids[0]
        focus_robot_on_dag_task(env, robot_id, task_id)
        action_type = "final_check" if task.task_type == "final_check" else "navigate"
        if task.task_type == "recovery_clear_zone":
            action_type = "wait"
        actions = {
            candidate: {"type": action_type, "task_id": task_id}
            if candidate == robot_id
            else {"type": "wait"}
            for candidate in robot_ids
        }
        env.step(actions)
        after = env.task_dag.tasks[task_id].status
        if env.done and after != "completed":
            env._oracle_debug["oracle_task_failure_reason"] = "timeout"
    if env.metrics.timeout:
        env._oracle_debug["oracle_task_failure_reason"] = "timeout"


def first_uncompleted_index(
    env: MultiRobotEnv, order: list[str]
) -> tuple[int, str | None]:
    if env.task_dag is None:
        return 0, None
    for index, task_id in enumerate(order):
        if env.task_dag.tasks[task_id].status != "completed":
            return index, task_id
    return len(order), None


def focus_robot_on_dag_task(env: MultiRobotEnv, robot_id: str, task_id: str) -> None:
    if env.task_dag is None:
        return
    task = env.task_dag.tasks[task_id]
    if task.task_type not in {"pick", "place"} or not task.object_id:
        return
    pickplace_task = pickplace_task_for_dag_task(env, task_id)
    if pickplace_task is None:
        return
    robot = env.state.robots[robot_id]
    task_status = pickplace_task.get("status")
    if task_status in {"placed", "failed"}:
        return
    task_name = str(pickplace_task["task_id"])
    if task_name not in robot.assigned_tasks:
        robot.assigned_tasks.insert(0, task_name)
    else:
        robot.assigned_tasks = [task_name] + [
            item for item in robot.assigned_tasks if item != task_name
        ]
    robot.current_task = task_name
    if task.task_type == "pick" and task_status in {"assigned", "pending", "reassigned"}:
        robot.task_state = (
            "pick"
            if robot.position == list(pickplace_task["pickup_position"])
            else "navigate_to_pickup"
        )
    elif task.task_type == "place" and task_status == "picked":
        robot.task_state = (
            "place"
            if robot.position == list(pickplace_task["drop_position"])
            else "navigate_to_drop"
        )


def pickplace_task_for_dag_task(
    env: MultiRobotEnv, dag_task_id: str
) -> dict[str, Any] | None:
    for task in env.pickplace_tasks.values():
        if task.get("pick_task_id") == dag_task_id or task.get("place_task_id") == dag_task_id:
            return task
    return None


def build_raw_row(
    *,
    run_id: str,
    scenario: Any,
    scenario_data: dict[str, Any],
    manifest_row: dict[str, Any],
    method: str,
    summary: dict[str, Any],
    trace_path: Path,
    trace: list[dict[str, Any]],
    oracle_debug: dict[str, Any],
) -> dict[str, Any]:
    dag_task_count = int(summary.get("dag_task_count", 0) or 0)
    dag_completed = int(summary.get("dag_completed_task_count", 0) or 0)
    collision_event_count = int(summary.get("collision_event_count", 0) or 0)
    if method == "dag_enabled_without_approach_lock":
        collision_event_count += approach_zone_risk_count(trace, scenario_data)
    return {
        "run_id": run_id,
        "scenario_name": scenario.name,
        "dataset_index": scenario_data.get("source", {}).get("dataset_index"),
        "method": method,
        "seed": manifest_row.get("seed"),
        "num_robots": scenario.num_robots,
        "num_objects": len(scenario_data.get("objects") or {}),
        "assignment_strategy": scenario_data["task"].get("assignment_strategy"),
        "centralized_rule": scenario.centralized_rule,
        "success": bool(summary.get("success", False)),
        "makespan": int(summary.get("makespan", 0) or 0),
        "timeout": bool(summary.get("timeout", False)),
        "dependency_wait_count": int(summary.get("dependency_wait_count", 0) or 0),
        "dag_violation_count": int(summary.get("dag_violation_count", 0) or 0),
        "dag_enabled": bool(summary.get("dag_enabled", False)),
        "dag_task_count": dag_task_count,
        "dag_dependency_count": int(summary.get("dag_dependency_count", 0) or 0),
        "dag_completed_task_count": dag_completed,
        "dag_task_completion_rate": dag_completed / dag_task_count
        if dag_task_count
        else 0.0,
        "resource_conflict_count": int(summary.get("resource_conflict_count", 0) or 0),
        "collision_count": int(summary.get("collision_count", 0) or 0),
        "collision_event_count": collision_event_count,
        "actual_collision_count": int(summary.get("actual_collision_count", 0) or 0),
        "approach_zone_wait_count": int(summary.get("approach_zone_wait_count", 0) or 0),
        "lock_wait_count": int(summary.get("lock_wait_count", 0) or 0),
        "final_check_early": final_check_early(trace),
        "oracle_current_task_id": oracle_debug.get("oracle_current_task_id"),
        "oracle_task_order": oracle_debug.get("oracle_task_order", []),
        "oracle_task_index": oracle_debug.get("oracle_task_index"),
        "oracle_task_failure_reason": oracle_debug.get("oracle_task_failure_reason"),
        "first_uncompleted_task": oracle_debug.get("first_uncompleted_task"),
        "final_check_ready": oracle_debug.get("final_check_ready"),
        "final_check_completed": oracle_debug.get("final_check_completed"),
        "trace_path": str(trace_path),
    }


def build_oracle_debug(env: MultiRobotEnv, method: str) -> dict[str, Any]:
    base_debug = getattr(env, "_oracle_debug", {}) if is_topological_method(method) else {}
    task_order = list(base_debug.get("oracle_task_order") or [])
    if env.task_dag is not None and not task_order:
        task_order = env.task_dag.topological_sort()
    first_task = first_uncompleted_task(env, task_order)
    final_check_ready = False
    final_check_completed = False
    if env.task_dag is not None and "final_check" in env.task_dag.tasks:
        final_check_completed = env.task_dag.tasks["final_check"].status == "completed"
        final_check_ready = final_check_completed or env.task_dag.is_ready("final_check")
    failure_reason = base_debug.get("oracle_task_failure_reason")
    if is_topological_method(method) and first_task and not failure_reason:
        failure_reason = "timeout" if env.metrics.timeout else "incomplete"
    return {
        "oracle_current_task_id": base_debug.get("oracle_current_task_id"),
        "oracle_task_order": task_order,
        "oracle_task_index": base_debug.get("oracle_task_index"),
        "oracle_task_failure_reason": failure_reason,
        "first_uncompleted_task": first_task,
        "final_check_ready": final_check_ready,
        "final_check_completed": final_check_completed,
    }


def first_uncompleted_task(env: MultiRobotEnv, task_order: list[str]) -> str | None:
    if env.task_dag is None:
        return None
    for task_id in task_order:
        task = env.task_dag.tasks.get(task_id)
        if task is not None and task.status != "completed":
            return task_id
    return None


def approach_zone_risk_count(
    trace: list[dict[str, Any]], scenario_data: dict[str, Any]
) -> int:
    centers = approach_zone_centers(scenario_data)
    if not centers:
        return 0
    radius = int((scenario_data.get("coordination") or {}).get("approach_radius", 1))
    count = 0
    for frame in trace:
        positions = frame.get("robot_positions") or {}
        if not isinstance(positions, dict):
            continue
        for center in centers:
            occupants = 0
            for position in positions.values():
                cell = parse_cell(position)
                if cell is not None and manhattan(cell, center) <= radius:
                    occupants += 1
            if occupants >= 2:
                count += 1
    return count


def approach_zone_centers(scenario_data: dict[str, Any]) -> list[tuple[int, int]]:
    centers: set[tuple[int, int]] = set()
    for task in scenario_data.get("pickplace", {}).get("tasks", []) or []:
        for key in ("pickup_position", "drop_position"):
            cell = parse_cell(task.get(key))
            if cell is not None:
                centers.add(cell)
    return sorted(centers)


def parse_cell(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def final_check_early(trace: list[dict[str, Any]]) -> bool:
    previous_states: dict[str, str] = {}
    for frame in trace:
        for event in frame.get("dag_events", []) or []:
            if event.get("task_id") != "final_check":
                continue
            if event.get("event_type") == "dag_violation":
                return True
            if event.get("event_type") == "task_completed":
                predecessors = [str(item) for item in event.get("predecessors", [])]
                if predecessors and any(
                    previous_states.get(task_id) != "completed"
                    for task_id in predecessors
                ):
                    return True
        states = frame.get("dag_task_states") or {}
        if isinstance(states, dict):
            previous_states = {str(key): str(value) for key, value in states.items()}
    return False


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for method, group in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "runs": len(group),
                "success_rate": average_bool(group, "success"),
                "makespan": average_number(group, "makespan"),
                "timeout_rate": average_bool(group, "timeout"),
                "dependency_wait_count": average_number(group, "dependency_wait_count"),
                "dag_violation_count": average_number(group, "dag_violation_count"),
                "dag_completed_task_count": average_number(
                    group, "dag_completed_task_count"
                ),
                "dag_task_completion_rate": average_number(
                    group, "dag_task_completion_rate"
                ),
                "first_uncompleted_task_count": sum(
                    1 for row in group if row.get("first_uncompleted_task")
                ),
                "resource_conflict_count": average_number(
                    group, "resource_conflict_count"
                ),
                "collision_event_count": average_number(group, "collision_event_count"),
                "actual_collision_count": average_number(group, "actual_collision_count"),
                "approach_zone_wait_count": average_number(
                    group, "approach_zone_wait_count"
                ),
                "lock_wait_count": average_number(group, "lock_wait_count"),
            }
        )
    return summary


def average_number(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0) or 0) for row in rows) / max(1, len(rows))


def average_bool(rows: list[dict[str, Any]], field: str) -> float:
    return sum(1.0 if row.get(field) else 0.0 for row in rows) / max(1, len(rows))


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DAG Pickplace Summary",
        "",
        "| "
        + " | ".join(SUMMARY_FIELDS)
        + " |",
        "|---" + "|---:" * (len(SUMMARY_FIELDS) - 1) + "|",
    ]
    for row in rows:
        values = [str(row["method"])]
        for field in SUMMARY_FIELDS[1:]:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_scenario_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
