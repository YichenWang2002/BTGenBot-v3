"""Run navigation + pick/place resource arbitration benchmark suites."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.analysis.audit_trace_collisions import audit_trace_payload
from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data
from src.experiments.summarize import summarize_rows, write_summary_csv, write_summary_md
from src.scenarios.pickplace_tasks import (
    assign_tasks_nearest_robot,
    assign_tasks_round_robin,
    assignments_by_robot,
)


RAW_FIELDS = [
    "run_id",
    "scenario_name",
    "dataset_index",
    "method",
    "seed",
    "num_robots",
    "num_objects",
    "assignment_strategy",
    "centralized_rule",
    "success",
    "makespan",
    "total_robot_steps",
    "collision_count",
    "collision_event_count_legacy",
    "collision_event_count",
    "actual_collision_count",
    "vertex_conflict_event_count",
    "edge_conflict_event_count",
    "audited_vertex_conflict_count",
    "audited_edge_swap_conflict_count",
    "vertex_conflict_count",
    "edge_conflict_count",
    "motion_wait_count",
    "rule_prevented_motion_conflict_count",
    "rule_rejection_count",
    "wait_count",
    "deadlock_count",
    "resource_conflict_count",
    "resource_request_denied_count",
    "rule_prevented_resource_conflict_count",
    "object_lock_denied_count",
    "pickup_zone_denied_count",
    "drop_zone_denied_count",
    "recovery_zone_denied_count",
    "pick_attempts",
    "pick_success_count",
    "pick_failure_count",
    "place_attempts",
    "place_success_count",
    "place_failure_count",
    "object_conflict_count",
    "pickup_zone_conflict_count",
    "drop_zone_conflict_count",
    "duplicate_pick_attempt_count",
    "approach_zone_denied_count",
    "approach_zone_wait_count",
    "corridor_wait_count",
    "rule_prevented_approach_conflict_count",
    "approach_lock_hold_time",
    "approach_lock_starvation_count",
    "approach_lock_reassignment_count",
    "lock_wait_count",
    "lock_wait_time",
    "reassignment_count",
    "tasks_completed",
    "tasks_failed",
    "all_objects_placed",
    "dag_enabled",
    "dag_task_count",
    "dag_dependency_count",
    "dag_completed_task_count",
    "dag_ready_task_count",
    "dag_blocked_task_count",
    "dependency_wait_count",
    "dag_violation_count",
    "critical_path_length",
    "timeout",
    "audit_warning",
    "trace_path",
]

COORDINATION_SUMMARY_FIELDS = [
    "method",
    "runs",
    "success_rate",
    "makespan",
    "collision_event_count",
    "actual_collision_count",
    "collision_count",
    "audited_vertex_conflict_count",
    "audited_edge_swap_conflict_count",
    "motion_wait_count",
    "rule_prevented_motion_conflict_count",
    "resource_conflict_count",
    "approach_zone_wait_count",
    "corridor_wait_count",
    "rule_prevented_approach_conflict_count",
    "approach_lock_hold_time",
    "approach_lock_starvation_count",
    "approach_lock_reassignment_count",
    "timeout_rate",
]

DAG_PICKPLACE_SUMMARY_FIELDS = [
    "method",
    "runs",
    "success_rate",
    "makespan",
    "dependency_wait_count",
    "dag_violation_count",
    "dag_completed_task_count",
    "collision_event_count",
    "collision_count",
    "actual_collision_count",
    "resource_conflict_count",
    "approach_zone_wait_count",
    "timeout_rate",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = run_suite(args.suite)
    print(f"Pick/place experiment rows: {len(rows)}")
    return 0


def run_suite(suite_path: Path) -> list[dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    suite = load_yaml(suite_path)
    output_dir = Path(suite.get("output_dir", "results/pickplace_smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(Path(suite["scenario_manifest"]))
    manifest_rows = manifest.get("scenarios", [])
    methods = list(suite.get("methods") or [])
    max_scenarios = int(suite.get("max_scenarios", len(manifest_rows)))

    raw_rows: list[dict[str, Any]] = []
    for scenario_index, manifest_row in enumerate(manifest_rows[:max_scenarios]):
        base_data = load_yaml(Path(manifest_row["scenario_path"]))
        for method in methods:
            run_id = f"{suite.get('name', 'pickplace')}_{scenario_index}_{method}"
            scenario_data = build_method_scenario(base_data, manifest_row, method, suite)
            scenario = load_scenario_data(scenario_data, default_name=scenario_data["name"])
            env = MultiRobotEnv(
                scenario.state,
                max_steps=scenario.max_steps,
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
                while not env.done:
                    env.step()
            finally:
                env.close()

            trace_path = trace_dir / f"{run_id}.json"
            trace_payload = env.metrics.to_trace_payload(
                scenario_name=scenario.name,
                num_robots=scenario.num_robots,
            )
            with trace_path.open("w", encoding="utf-8") as handle:
                json.dump(trace_payload, handle, indent=2)
                handle.write("\n")

            summary = env.metrics.summary()
            audit = audit_collision_payload(trace_payload)
            raw_rows.append(
                {
                    "run_id": run_id,
                    "scenario_name": scenario.name,
                    "dataset_index": scenario_data["source"]["dataset_index"],
                    "method": method,
                    "seed": manifest_row["seed"],
                    "num_robots": scenario.num_robots,
                    "num_objects": len(scenario_data.get("objects") or {}),
                    "assignment_strategy": scenario_data["task"]["assignment_strategy"],
                    "centralized_rule": scenario.centralized_rule,
                    "success": summary["success"],
                    "makespan": summary["makespan"],
                    "total_robot_steps": summary["total_robot_steps"],
                    "collision_count": summary["collision_count"],
                    "collision_event_count_legacy": summary[
                        "collision_event_count_legacy"
                    ],
                    "collision_event_count": summary["collision_event_count"],
                    "actual_collision_count": audit["actual_collision_count"],
                    "vertex_conflict_event_count": summary[
                        "vertex_conflict_event_count"
                    ],
                    "edge_conflict_event_count": summary["edge_conflict_event_count"],
                    "audited_vertex_conflict_count": audit[
                        "audited_vertex_conflict_count"
                    ],
                    "audited_edge_swap_conflict_count": audit[
                        "audited_edge_swap_conflict_count"
                    ],
                    "vertex_conflict_count": summary["vertex_conflict_count"],
                    "edge_conflict_count": summary["edge_conflict_count"],
                    "motion_wait_count": summary["motion_wait_count"],
                    "rule_prevented_motion_conflict_count": summary[
                        "rule_prevented_motion_conflict_count"
                    ],
                    "rule_rejection_count": summary["rule_rejection_count"],
                    "wait_count": summary["wait_count"],
                    "deadlock_count": summary["deadlock_count"],
                    "resource_conflict_count": summary["resource_conflict_count"],
                    "resource_request_denied_count": summary[
                        "resource_request_denied_count"
                    ],
                    "rule_prevented_resource_conflict_count": summary[
                        "rule_prevented_resource_conflict_count"
                    ],
                    "object_lock_denied_count": summary["object_lock_denied_count"],
                    "pickup_zone_denied_count": summary["pickup_zone_denied_count"],
                    "drop_zone_denied_count": summary["drop_zone_denied_count"],
                    "recovery_zone_denied_count": summary["recovery_zone_denied_count"],
                    "pick_attempts": summary["pick_attempts"],
                    "pick_success_count": summary["pick_success_count"],
                    "pick_failure_count": summary["pick_failure_count"],
                    "place_attempts": summary["place_attempts"],
                    "place_success_count": summary["place_success_count"],
                    "place_failure_count": summary["place_failure_count"],
                    "object_conflict_count": summary["object_conflict_count"],
                    "pickup_zone_conflict_count": summary["pickup_zone_conflict_count"],
                    "drop_zone_conflict_count": summary["drop_zone_conflict_count"],
                    "duplicate_pick_attempt_count": summary["duplicate_pick_attempt_count"],
                    "approach_zone_denied_count": summary[
                        "approach_zone_denied_count"
                    ],
                    "approach_zone_wait_count": summary["approach_zone_wait_count"],
                    "corridor_wait_count": summary["corridor_wait_count"],
                    "rule_prevented_approach_conflict_count": summary[
                        "rule_prevented_approach_conflict_count"
                    ],
                    "approach_lock_hold_time": summary["approach_lock_hold_time"],
                    "approach_lock_starvation_count": summary[
                        "approach_lock_starvation_count"
                    ],
                    "approach_lock_reassignment_count": summary[
                        "approach_lock_reassignment_count"
                    ],
                    "lock_wait_count": summary["lock_wait_count"],
                    "lock_wait_time": summary["lock_wait_time"],
                    "reassignment_count": summary["reassignment_count"],
                    "tasks_completed": summary["tasks_completed"],
                    "tasks_failed": summary["tasks_failed"],
                    "all_objects_placed": summary["all_objects_placed"],
                    "dag_enabled": summary["dag_enabled"],
                    "dag_task_count": summary["dag_task_count"],
                    "dag_dependency_count": summary["dag_dependency_count"],
                    "dag_completed_task_count": summary["dag_completed_task_count"],
                    "dag_ready_task_count": summary["dag_ready_task_count"],
                    "dag_blocked_task_count": summary["dag_blocked_task_count"],
                    "dependency_wait_count": summary["dependency_wait_count"],
                    "dag_violation_count": summary["dag_violation_count"],
                    "critical_path_length": summary["critical_path_length"],
                    "timeout": summary["timeout"],
                    "audit_warning": audit["audit_warning"],
                    "trace_path": str(trace_path),
                }
            )

    write_raw_jsonl(output_dir / "raw.jsonl", raw_rows)
    if suite.get("summary_style") == "coordination_ablation":
        summary_rows = summarize_coordination_rows(raw_rows)
        write_coordination_summary_csv(output_dir / "summary.csv", summary_rows)
        write_coordination_summary_md(output_dir / "summary.md", summary_rows)
    elif suite.get("summary_style") == "dag_pickplace":
        summary_rows = summarize_dag_pickplace_rows(raw_rows)
        write_dag_pickplace_summary_csv(output_dir / "summary.csv", summary_rows)
        write_dag_pickplace_summary_md(output_dir / "summary.md", summary_rows)
    else:
        summary_rows = summarize_rows(raw_rows)
        write_summary_csv(output_dir / "summary.csv", summary_rows)
        write_summary_md(output_dir / "summary.md", summary_rows)
    return raw_rows


def build_method_scenario(
    base_data: dict[str, Any],
    manifest_row: dict[str, Any],
    method: str,
    suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = deepcopy(base_data)
    data["name"] = f"{base_data['name']}_{method}"
    data["render"] = False
    data["coordination"] = dict((suite or {}).get("coordination") or {})
    tasks = [dict(task) for task in data["pickplace"].get("tasks", [])]

    if method == "single_robot_pickplace":
        first_robot = deepcopy(data["robots"][sorted(data["robots"])[0]])
        data["robots"] = {"robot_0": first_robot}
        for task in tasks:
            task["assigned_robot"] = "robot_0"
            task["status"] = "assigned"
        assignment_strategy = "sequential_single_robot"
        centralized_rule = False
    elif method == "naive_multi_robot_pickplace":
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = False
    elif method == "centralized_rule_multi_robot_pickplace":
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = True
    elif method in {
        "dag_enabled_centralized",
        "ignore_dag_centralized",
        "dag_enabled_without_approach_lock",
    }:
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = True
        data["coordination"]["enable_cell_reservation"] = True
        data["coordination"]["enable_edge_reservation"] = True
        data["coordination"]["enable_resource_lock"] = True
        data["coordination"]["enable_recovery_lock"] = True
        data["coordination"].setdefault("approach_radius", 1)
        if method == "dag_enabled_without_approach_lock":
            data["coordination"]["enable_approach_zone_lock"] = False
        else:
            data["coordination"]["enable_approach_zone_lock"] = True
        data["ignore_task_dag"] = method == "ignore_dag_centralized"
    elif method == "centralized_rule_with_approach_lock_pickplace":
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = True
        data["coordination"]["enable_approach_zone_lock"] = True
        data["coordination"].setdefault("approach_radius", 1)
    elif method in {
        "approach_radius_0",
        "approach_radius_1",
        "approach_radius_2",
        "centralized_rule_with_approach_radius_0_pickplace",
        "centralized_rule_with_approach_radius_1_pickplace",
        "centralized_rule_with_approach_radius_2_pickplace",
    }:
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = True
        radius = int(method.split("_")[-2] if method.endswith("_pickplace") else method.split("_")[-1])
        data["coordination"]["enable_approach_zone_lock"] = True
        data["coordination"]["approach_radius"] = radius
    elif method == "centralized_rule_with_approach_and_corridor_lock_pickplace":
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        tasks = assign_tasks(tasks, data["robots"], assignment_strategy)
        centralized_rule = True
        data["coordination"]["enable_approach_zone_lock"] = True
        data["coordination"].setdefault("approach_radius", 1)
        data["coordination"]["enable_corridor_lock"] = True
        if not data["coordination"].get("corridor_cells"):
            data["coordination"]["corridor_cells"] = build_corridor_cells(tasks)
    else:
        raise ValueError(f"Unknown method: {method}")

    for robot in data["robots"].values():
        robot["goal"] = None
        robot["assigned_waypoints"] = []
        robot["assigned_tasks"] = []
        robot["carrying_object"] = None
    data["pickplace"]["enabled"] = True
    data["pickplace"]["tasks"] = tasks
    synchronize_task_dag_assignments(data, tasks)
    data["task"]["assignment_strategy"] = assignment_strategy
    data["task"]["baseline_type"] = method
    data["task"]["assignments"] = assignments_by_robot(tasks, sorted(data["robots"]))
    data["centralized_rule"] = centralized_rule
    return data


def build_corridor_cells(tasks: list[dict[str, Any]]) -> list[list[list[int]]]:
    corridors: list[list[list[int]]] = []
    for task in tasks:
        pickup = list(task["pickup_position"])
        drop = list(task["drop_position"])
        cells: list[list[int]] = []
        x, y = pickup
        cells.append([x, y])
        while x != drop[0]:
            x += 1 if x < drop[0] else -1
            cells.append([x, y])
        while y != drop[1]:
            y += 1 if y < drop[1] else -1
            cells.append([x, y])
        corridors.append(cells)
    return corridors


def assign_tasks(
    tasks: list[dict[str, Any]], robots: dict[str, Any], strategy: str
) -> list[dict[str, Any]]:
    if strategy == "round_robin":
        return assign_tasks_round_robin(tasks, robots)
    return assign_tasks_nearest_robot(tasks, robots)


def synchronize_task_dag_assignments(
    scenario_data: dict[str, Any], tasks: list[dict[str, Any]]
) -> None:
    task_dag = scenario_data.get("task_dag")
    if not isinstance(task_dag, dict):
        return
    dag_tasks = task_dag.get("tasks")
    if not isinstance(dag_tasks, list):
        return
    assignments: dict[str, str] = {}
    for task in tasks:
        robot_id = task.get("assigned_robot")
        if not robot_id:
            continue
        if task.get("pick_task_id"):
            assignments[str(task["pick_task_id"])] = str(robot_id)
        if task.get("place_task_id"):
            assignments[str(task["place_task_id"])] = str(robot_id)
    for dag_task in dag_tasks:
        if not isinstance(dag_task, dict):
            continue
        task_id = str(dag_task.get("task_id"))
        if task_id in assignments:
            dag_task["assigned_robot"] = assignments[task_id]


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_coordination_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "collision_event_count": average_number(group, "collision_event_count"),
                "actual_collision_count": average_number(group, "actual_collision_count"),
                "collision_count": average_number(group, "collision_count"),
                "audited_vertex_conflict_count": average_number(
                    group, "audited_vertex_conflict_count"
                ),
                "audited_edge_swap_conflict_count": average_number(
                    group, "audited_edge_swap_conflict_count"
                ),
                "motion_wait_count": average_number(group, "motion_wait_count"),
                "rule_prevented_motion_conflict_count": average_number(
                    group, "rule_prevented_motion_conflict_count"
                ),
                "resource_conflict_count": average_number(group, "resource_conflict_count"),
                "approach_zone_wait_count": average_number(group, "approach_zone_wait_count"),
                "corridor_wait_count": average_number(group, "corridor_wait_count"),
                "rule_prevented_approach_conflict_count": average_number(
                    group, "rule_prevented_approach_conflict_count"
                ),
                "approach_lock_hold_time": average_number(group, "approach_lock_hold_time"),
                "approach_lock_starvation_count": average_number(
                    group, "approach_lock_starvation_count"
                ),
                "approach_lock_reassignment_count": average_number(
                    group, "approach_lock_reassignment_count"
                ),
                "timeout_rate": average_bool(group, "timeout"),
            }
        )
    return summary


def summarize_dag_pickplace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "dependency_wait_count": average_number(group, "dependency_wait_count"),
                "dag_violation_count": average_number(group, "dag_violation_count"),
                "dag_completed_task_count": average_number(
                    group, "dag_completed_task_count"
                ),
                "collision_event_count": average_number(group, "collision_event_count"),
                "collision_count": average_number(group, "collision_count"),
                "actual_collision_count": average_number(group, "actual_collision_count"),
                "resource_conflict_count": average_number(group, "resource_conflict_count"),
                "approach_zone_wait_count": average_number(group, "approach_zone_wait_count"),
                "timeout_rate": average_bool(group, "timeout"),
            }
        )
    return summary


def average_number(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field, 0) or 0) for row in rows) / max(1, len(rows))


def average_bool(rows: list[dict[str, Any]], field: str) -> float:
    return sum(1.0 if row.get(field) else 0.0 for row in rows) / max(1, len(rows))


def audit_collision_payload(trace_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        audit = audit_trace_payload(trace_payload, [])
        return {
            "actual_collision_count": int(audit["audited_collision_count"]),
            "audited_vertex_conflict_count": int(
                audit["audited_vertex_conflict_count"]
            ),
            "audited_edge_swap_conflict_count": int(
                audit["audited_edge_swap_conflict_count"]
            ),
            "audit_warning": "; ".join(str(item) for item in audit.get("warnings", [])),
        }
    except Exception as exc:
        return {
            "actual_collision_count": -1,
            "audited_vertex_conflict_count": -1,
            "audited_edge_swap_conflict_count": -1,
            "audit_warning": str(exc),
        }


def write_coordination_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COORDINATION_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_coordination_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pickplace Coordination Ablation Summary",
        "",
        "| method | runs | success_rate | makespan | collision_event_count | actual_collision_count | collision_count | audited_vertex_conflict_count | audited_edge_swap_conflict_count | motion_wait_count | rule_prevented_motion_conflict_count | resource_conflict_count | approach_zone_wait_count | corridor_wait_count | rule_prevented_approach_conflict_count | approach_lock_hold_time | approach_lock_starvation_count | approach_lock_reassignment_count | timeout_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {runs} | {success_rate:.3f} | {makespan:.3f} | "
            "{collision_event_count:.3f} | {actual_collision_count:.3f} | "
            "{collision_count:.3f} | {audited_vertex_conflict_count:.3f} | "
            "{audited_edge_swap_conflict_count:.3f} | {motion_wait_count:.3f} | "
            "{rule_prevented_motion_conflict_count:.3f} | "
            "{resource_conflict_count:.3f} | "
            "{approach_zone_wait_count:.3f} | {corridor_wait_count:.3f} | "
            "{rule_prevented_approach_conflict_count:.3f} | "
            "{approach_lock_hold_time:.3f} | {approach_lock_starvation_count:.3f} | "
            "{approach_lock_reassignment_count:.3f} | {timeout_rate:.3f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dag_pickplace_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAG_PICKPLACE_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dag_pickplace_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DAG Pickplace Summary",
        "",
        "| method | runs | success_rate | makespan | dependency_wait_count | dag_violation_count | dag_completed_task_count | collision_event_count | collision_count | actual_collision_count | resource_conflict_count | approach_zone_wait_count | timeout_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {runs} | {success_rate:.3f} | {makespan:.3f} | "
            "{dependency_wait_count:.3f} | {dag_violation_count:.3f} | "
            "{dag_completed_task_count:.3f} | {collision_event_count:.3f} | "
            "{collision_count:.3f} | {actual_collision_count:.3f} | "
            "{resource_conflict_count:.3f} | {approach_zone_wait_count:.3f} | "
            "{timeout_rate:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
