"""Run unified baseline and ablation suites for paper comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data
from src.llm.backends.mock import MockLLMBackend
from src.llm.generate_multi_bt import run_generated_plan
from src.llm.output_parser import parse_and_validate
from src.llm.prompt_builder import build_prompt
from src.repair.failure_report import build_failure_report
from src.scenarios.pickplace_tasks import (
    assign_tasks_nearest_robot,
    assign_tasks_round_robin,
    assignments_by_robot,
)
from src.scenarios.waypoint_assignment import assign_waypoints


BASELINE_METHODS = {
    "single_robot_sequential",
    "naive_multi_robot",
    "centralized_without_resource_lock",
    "centralized_without_recovery_lock",
    "prioritized_planning_multi_robot",
}
LLM_METHODS = {
    "llm_only_multi_robot",
    "centralized_rule_multi_bt",
    "centralized_without_repair",
}
SUMMARY_FIELDS = [
    "summary_scope",
    "scenario_family",
    "method",
    "runs",
    "success_rate",
    "makespan",
    "total_robot_steps",
    "collision_event_count",
    "actual_collision_count",
    "collision_count",
    "audited_vertex_conflict_count",
    "audited_edge_swap_conflict_count",
    "motion_wait_count",
    "rule_prevented_motion_conflict_count",
    "vertex_conflict_count",
    "edge_conflict_count",
    "resource_conflict_count",
    "recovery_conflict_count",
    "lock_wait_count",
    "rule_prevented_resource_conflict_count",
    "timeout_rate",
    "tasks_completed",
    "object_success_rate",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = run_suite(args.suite)
    print(f"Baseline experiment rows: {len(rows)}")
    return 0


def run_suite(suite_path: Path) -> list[dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    suite = load_yaml(suite_path)
    output_dir = Path(suite.get("output_dir", "results/baseline_small"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    max_iters = int(suite.get("max_iters", 3))
    run_specs = expand_run_specs(suite)
    raw_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(run_specs):
        scenario_path = Path(spec["scenario_path"])
        family = str(spec["scenario_family"])
        method = str(spec["method"])
        base_data = load_yaml(scenario_path)
        run_id = f"{suite.get('name', 'baseline')}_{index}_{family}_{method}"
        trace_path = trace_dir / f"{safe_name(run_id)}.json"
        if method in LLM_METHODS:
            row = run_mock_llm_method(
                run_id=run_id,
                scenario_path=scenario_path,
                scenario_family=family,
                method=method,
                trace_path=trace_path,
                max_iters=max_iters,
            )
        else:
            scenario_data = build_method_scenario(base_data, spec, method, family)
            row = run_simulator_method(
                run_id=run_id,
                scenario_data=scenario_data,
                scenario_family=family,
                method=method,
                trace_path=trace_path,
                prioritized_planning=method == "prioritized_planning_multi_robot",
            )
        raw_rows.append(row)

    write_raw_jsonl(output_dir / "raw.jsonl", raw_rows)
    summary_rows = summarize_baseline_rows(raw_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_md(output_dir / "summary.md", summary_rows)
    return raw_rows


def expand_run_specs(suite: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for family_entry in suite.get("families", []):
        family = str(family_entry["name"])
        manifest = read_json(Path(family_entry["scenario_manifest"]))
        manifest_rows = manifest.get("scenarios", [])
        max_scenarios = int(family_entry.get("max_scenarios", len(manifest_rows)))
        methods = list(family_entry.get("methods") or [])
        for manifest_row in manifest_rows[:max_scenarios]:
            for method in methods:
                specs.append(
                    {
                        **manifest_row,
                        "scenario_family": family,
                        "method": method,
                    }
                )
    return specs


def run_simulator_method(
    run_id: str,
    scenario_data: dict[str, Any],
    scenario_family: str,
    method: str,
    trace_path: Path,
    prioritized_planning: bool = False,
) -> dict[str, Any]:
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
        prioritized_planning=prioritized_planning,
        task_dag=scenario.task_dag,
    )
    env.reset()
    try:
        while not env.done:
            env.step()
    finally:
        env.close()
    trace_payload = env.metrics.to_trace_payload(
        scenario_name=scenario.name,
        num_robots=scenario.num_robots,
    )
    write_json(trace_path, trace_payload)
    row = build_raw_row(
        run_id=run_id,
        scenario_name=scenario.name,
        scenario_family=scenario_family,
        method=method,
        summary=env.metrics.summary(),
        trace_path=trace_path,
        scenario_data=scenario_data,
        num_robots=scenario.num_robots,
        num_iters=0,
    )
    return row


def run_mock_llm_method(
    run_id: str,
    scenario_path: Path,
    scenario_family: str,
    method: str,
    trace_path: Path,
    max_iters: int,
) -> dict[str, Any]:
    backend = MockLLMBackend()
    scenario_data = load_yaml(scenario_path)
    scenario = load_scenario_data(scenario_data, default_name=scenario_data.get("name", "scenario"))
    previous_report: dict[str, Any] | None = None
    final_summary: dict[str, Any] = {}
    final_trace: dict[str, Any] = {"trace": []}
    iterations = 0
    limit = 1 if method in {"llm_only_multi_robot", "centralized_without_repair"} else max_iters

    for iteration in range(limit):
        iterations = iteration + 1
        prompt_method = (
            "centralized_rule_multi_bt"
            if method in {"centralized_rule_multi_bt", "centralized_without_repair"}
            else method
        )
        prompt = build_prompt(
            scenario_path,
            prompt_method,
            previous_failure_report=previous_report,
        )
        raw_response = backend.generate(prompt)
        parsed, schema_result = parse_and_validate(raw_response, scenario)
        if parsed is None or not schema_result.valid:
            final_summary = {"success": False, "timeout": False}
            final_trace = {"trace": []}
        else:
            run_method = (
                "centralized_rule_multi_bt"
                if method in {"centralized_rule_multi_bt", "centralized_without_repair"}
                else "llm_only_multi_robot"
            )
            final_summary, final_trace = run_generated_plan(
                scenario_path=scenario_path,
                payload=parsed,
                method=run_method,
                trace_path=trace_path,
            )
        if final_summary.get("success"):
            break
        previous_report = build_failure_report(
            scenario=scenario,
            assignment=(parsed or {}).get("assignment") if parsed else [],
            metrics=final_summary,
            trace=final_trace.get("trace", []),
            static_validation={"valid": bool(parsed and schema_result.valid)},
        )

    if not trace_path.exists():
        write_json(trace_path, final_trace)
    row = build_raw_row(
        run_id=run_id,
        scenario_name=scenario.name,
        scenario_family=scenario_family,
        method=method,
        summary=final_summary,
        trace_path=trace_path,
        scenario_data=scenario_data,
        num_robots=len(scenario_data.get("robots", {})),
        num_iters=iterations,
    )
    row["centralized_rule"] = method in {
        "centralized_rule_multi_bt",
        "centralized_without_repair",
    }
    row["coordination"] = coordination_for_method(method)
    return row


def build_method_scenario(
    base_data: dict[str, Any],
    manifest_row: dict[str, Any],
    method: str,
    family: str,
) -> dict[str, Any]:
    data = deepcopy(base_data)
    data["name"] = f"{base_data['name']}_{method}"
    data["render"] = False
    data["coordination"] = coordination_for_method(method)
    data["centralized_rule"] = method in {
        "centralized_without_resource_lock",
        "centralized_without_recovery_lock",
    }
    if method == "naive_multi_robot":
        data["centralized_rule"] = False
    elif method == "prioritized_planning_multi_robot":
        data["centralized_rule"] = False

    if family in {"nav", "recovery"}:
        build_waypoint_method_scenario(data, manifest_row, method)
    elif family == "pickplace":
        build_pickplace_method_scenario(data, method)
    else:
        raise ValueError(f"Unknown scenario family: {family}")
    data["task"]["baseline_type"] = method
    return data


def build_waypoint_method_scenario(
    data: dict[str, Any], manifest_row: dict[str, Any], method: str
) -> None:
    waypoints = data.get("waypoints", [])
    seed = int(manifest_row.get("seed", 0))
    if method == "single_robot_sequential":
        first_robot_id = sorted(data["robots"])[0]
        first_robot = deepcopy(data["robots"][first_robot_id])
        data["robots"] = {"robot_0": first_robot}
        robot_ids = ["robot_0"]
        strategy = "sequential_single_robot"
    else:
        robot_ids = sorted(data["robots"])
        strategy = data["task"].get("assignment_strategy", "nearest_robot")
    robot_starts = {robot_id: data["robots"][robot_id]["start"] for robot_id in robot_ids}
    assignments = assign_waypoints(
        strategy,
        robot_ids,
        waypoints,
        robot_starts=robot_starts,
        seed=seed,
    )
    for robot in data["robots"].values():
        robot["goal"] = None
        robot["assigned_waypoints"] = []
    data["task"]["assignment_strategy"] = strategy
    data["task"]["assignments"] = assignments
    if "recovery" in data:
        data["recovery"]["enabled"] = bool(data["recovery"].get("enabled", False))


def build_pickplace_method_scenario(data: dict[str, Any], method: str) -> None:
    tasks = [dict(task) for task in data["pickplace"].get("tasks", [])]
    if method == "single_robot_sequential":
        first_robot = deepcopy(data["robots"][sorted(data["robots"])[0]])
        data["robots"] = {"robot_0": first_robot}
        for task in tasks:
            task["assigned_robot"] = "robot_0"
            task["status"] = "assigned"
            task["attempts"] = 0
        strategy = "sequential_single_robot"
    else:
        strategy = data["task"].get("assignment_strategy", "nearest_robot")
        tasks = assign_tasks(tasks, data["robots"], strategy)
    for robot in data["robots"].values():
        robot["goal"] = None
        robot["assigned_waypoints"] = []
        robot["assigned_tasks"] = []
        robot["carrying_object"] = None
    data["pickplace"]["enabled"] = True
    data["pickplace"]["tasks"] = tasks
    data["task"]["assignment_strategy"] = strategy
    data["task"]["assignments"] = assignments_by_robot(tasks, sorted(data["robots"]))


def coordination_for_method(method: str) -> dict[str, bool]:
    config = {
        "enable_cell_reservation": True,
        "enable_edge_reservation": True,
        "enable_resource_lock": True,
        "enable_recovery_lock": True,
    }
    if method in {"single_robot_sequential", "naive_multi_robot", "prioritized_planning_multi_robot"}:
        return config
    if method == "centralized_without_resource_lock":
        config["enable_resource_lock"] = False
    elif method == "centralized_without_recovery_lock":
        config["enable_recovery_lock"] = False
    return config


def assign_tasks(
    tasks: list[dict[str, Any]], robots: dict[str, Any], strategy: str
) -> list[dict[str, Any]]:
    if strategy == "round_robin":
        return assign_tasks_round_robin(tasks, robots)
    return assign_tasks_nearest_robot(tasks, robots)


def build_raw_row(
    run_id: str,
    scenario_name: str,
    scenario_family: str,
    method: str,
    summary: dict[str, Any],
    trace_path: Path,
    scenario_data: dict[str, Any],
    num_robots: int,
    num_iters: int,
) -> dict[str, Any]:
    source = scenario_data.get("source") or {}
    task = scenario_data.get("task") or {}
    coordination = scenario_data.get("coordination") or {}
    return {
        "run_id": run_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "dataset_index": source.get("dataset_index"),
        "method": method,
        "num_iters": num_iters,
        "num_robots": num_robots,
        "assignment_strategy": task.get("assignment_strategy"),
        "centralized_rule": bool(scenario_data.get("centralized_rule", False)),
        "coordination": coordination,
        "success": bool(summary.get("success", False)),
        "makespan": int(summary.get("makespan", 0) or 0),
        "total_robot_steps": int(summary.get("total_robot_steps", 0) or 0),
        "collision_event_count": int(
            summary.get("collision_event_count", summary.get("collision_count", 0)) or 0
        ),
        "actual_collision_count": int(summary.get("actual_collision_count", 0) or 0),
        "collision_count": int(summary.get("collision_count", 0) or 0),
        "audited_vertex_conflict_count": int(
            summary.get("audited_vertex_conflict_count", 0) or 0
        ),
        "audited_edge_swap_conflict_count": int(
            summary.get("audited_edge_swap_conflict_count", 0) or 0
        ),
        "motion_wait_count": int(summary.get("motion_wait_count", 0) or 0),
        "rule_prevented_motion_conflict_count": int(
            summary.get("rule_prevented_motion_conflict_count", 0) or 0
        ),
        "vertex_conflict_count": int(summary.get("vertex_conflict_count", 0) or 0),
        "edge_conflict_count": int(summary.get("edge_conflict_count", 0) or 0),
        "resource_conflict_count": int(summary.get("resource_conflict_count", 0) or 0),
        "recovery_conflict_count": int(summary.get("recovery_conflict_count", 0) or 0),
        "lock_wait_count": int(summary.get("lock_wait_count", 0) or 0),
        "rule_prevented_resource_conflict_count": int(
            summary.get("rule_prevented_resource_conflict_count", 0) or 0
        ),
        "timeout": bool(summary.get("timeout", False)),
        "tasks_completed": int(summary.get("tasks_completed", 0) or 0),
        "object_success_rate": float(summary.get("object_success_rate", 0.0) or 0.0),
        "trace_path": str(trace_path),
    }


def summarize_baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[(row["scenario_family"], row["method"])].append(row)
        overall[row["method"]].append(row)
    for (family, method), group in sorted(by_family.items()):
        summary.append(build_summary_row("by_family", family, method, group))
    for method, group in sorted(overall.items()):
        summary.append(build_summary_row("overall", "all", method, group))
    return summary


def build_summary_row(
    summary_scope: str, scenario_family: str, method: str, group: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "summary_scope": summary_scope,
        "scenario_family": scenario_family,
        "method": method,
        "runs": len(group),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in group),
        "makespan": mean(float(row["makespan"]) for row in group),
        "total_robot_steps": mean(float(row["total_robot_steps"]) for row in group),
        "collision_event_count": mean(float(row["collision_event_count"]) for row in group),
        "actual_collision_count": mean(float(row["actual_collision_count"]) for row in group),
        "collision_count": mean(float(row["collision_count"]) for row in group),
        "audited_vertex_conflict_count": mean(
            float(row["audited_vertex_conflict_count"]) for row in group
        ),
        "audited_edge_swap_conflict_count": mean(
            float(row["audited_edge_swap_conflict_count"]) for row in group
        ),
        "motion_wait_count": mean(float(row["motion_wait_count"]) for row in group),
        "rule_prevented_motion_conflict_count": mean(
            float(row["rule_prevented_motion_conflict_count"]) for row in group
        ),
        "vertex_conflict_count": mean(float(row["vertex_conflict_count"]) for row in group),
        "edge_conflict_count": mean(float(row["edge_conflict_count"]) for row in group),
        "resource_conflict_count": mean(float(row["resource_conflict_count"]) for row in group),
        "recovery_conflict_count": mean(float(row["recovery_conflict_count"]) for row in group),
        "lock_wait_count": mean(float(row["lock_wait_count"]) for row in group),
        "rule_prevented_resource_conflict_count": mean(
            float(row["rule_prevented_resource_conflict_count"]) for row in group
        ),
        "timeout_rate": mean(1.0 if row["timeout"] else 0.0 for row in group),
        "tasks_completed": mean(float(row["tasks_completed"]) for row in group),
        "object_success_rate": mean(float(row["object_success_rate"]) for row in group),
    }


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
        "# Baseline Experiment Summary",
        "",
        "| scope | family | method | runs | success_rate | makespan | total_robot_steps | collision_event_count | actual_collision_count | collision_count | resource_conflict_count | recovery_conflict_count | lock_wait_count | timeout_rate | tasks_completed | object_success_rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {summary_scope} | {scenario_family} | {method} | {runs} | "
            "{success_rate:.3f} | {makespan:.3f} | {total_robot_steps:.3f} | "
            "{collision_event_count:.3f} | {actual_collision_count:.3f} | "
            "{collision_count:.3f} | {resource_conflict_count:.3f} | "
            "{recovery_conflict_count:.3f} | {lock_wait_count:.3f} | "
            "{timeout_rate:.3f} | {tasks_completed:.3f} | "
            "{object_success_rate:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
