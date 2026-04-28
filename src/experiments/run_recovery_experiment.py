"""Run navigation + recovery benchmark suites."""

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
from src.experiments.summarize import summarize_rows, write_summary_csv, write_summary_md
from src.scenarios.waypoint_assignment import assign_waypoints


RAW_FIELDS = [
    "run_id",
    "scenario_name",
    "dataset_index",
    "method",
    "seed",
    "num_robots",
    "num_waypoints",
    "assignment_strategy",
    "centralized_rule",
    "success",
    "makespan",
    "total_robot_steps",
    "collision_count",
    "vertex_conflict_count",
    "edge_conflict_count",
    "rule_rejection_count",
    "wait_count",
    "deadlock_count",
    "resource_conflict_count",
    "resource_request_denied_count",
    "lock_wait_count",
    "lock_wait_time",
    "rule_prevented_resource_conflict_count",
    "recovery_attempts",
    "recovery_conflict_count",
    "recovery_lock_wait_count",
    "recovery_wait_time",
    "successful_recoveries",
    "failed_recoveries",
    "stuck_count",
    "timeout",
    "trace_path",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = run_suite(args.suite)
    print(f"Recovery experiment rows: {len(rows)}")
    return 0


def run_suite(suite_path: Path) -> list[dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    suite = load_yaml(suite_path)
    output_dir = Path(suite.get("output_dir", "results/recovery_smoke"))
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
            run_id = f"{suite.get('name', 'recovery')}_{scenario_index}_{method}"
            scenario_data = build_method_scenario(base_data, manifest_row, method)
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
                task_dag=scenario.task_dag,
            )
            env.reset()
            try:
                while not env.done:
                    env.step()
            finally:
                env.close()

            trace_path = trace_dir / f"{run_id}.json"
            with trace_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    env.metrics.to_trace_payload(
                        scenario_name=scenario.name,
                        num_robots=scenario.num_robots,
                    ),
                    handle,
                    indent=2,
                )
                handle.write("\n")

            task = scenario_data["task"]
            summary = env.metrics.summary()
            raw_rows.append(
                {
                    "run_id": run_id,
                    "scenario_name": scenario.name,
                    "dataset_index": scenario_data["source"]["dataset_index"],
                    "method": method,
                    "seed": manifest_row["seed"],
                    "num_robots": scenario.num_robots,
                    "num_waypoints": task["num_waypoints"],
                    "assignment_strategy": task["assignment_strategy"],
                    "centralized_rule": scenario.centralized_rule,
                    "success": summary["success"],
                    "makespan": summary["makespan"],
                    "total_robot_steps": summary["total_robot_steps"],
                    "collision_count": summary["collision_count"],
                    "vertex_conflict_count": summary["vertex_conflict_count"],
                    "edge_conflict_count": summary["edge_conflict_count"],
                    "rule_rejection_count": summary["rule_rejection_count"],
                    "wait_count": summary["wait_count"],
                    "deadlock_count": summary["deadlock_count"],
                    "resource_conflict_count": summary["resource_conflict_count"],
                    "resource_request_denied_count": summary[
                        "resource_request_denied_count"
                    ],
                    "lock_wait_count": summary["lock_wait_count"],
                    "lock_wait_time": summary["lock_wait_time"],
                    "rule_prevented_resource_conflict_count": summary[
                        "rule_prevented_resource_conflict_count"
                    ],
                    "recovery_attempts": summary["recovery_attempts"],
                    "recovery_conflict_count": summary["recovery_conflict_count"],
                    "recovery_lock_wait_count": summary["recovery_lock_wait_count"],
                    "recovery_wait_time": summary["recovery_wait_time"],
                    "successful_recoveries": summary["successful_recoveries"],
                    "failed_recoveries": summary["failed_recoveries"],
                    "stuck_count": summary["stuck_count"],
                    "timeout": summary["timeout"],
                    "trace_path": str(trace_path),
                }
            )

    write_raw_jsonl(output_dir / "raw.jsonl", raw_rows)
    summary_rows = summarize_rows(raw_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_md(output_dir / "summary.md", summary_rows)
    return raw_rows


def build_method_scenario(
    base_data: dict[str, Any], manifest_row: dict[str, Any], method: str
) -> dict[str, Any]:
    data = deepcopy(base_data)
    data["name"] = f"{base_data['name']}_{method}"
    waypoints = data["waypoints"]
    seed = int(manifest_row["seed"])

    if method == "single_robot_recovery":
        first_robot_id = sorted(data["robots"])[0]
        first_robot = deepcopy(data["robots"][first_robot_id])
        data["robots"] = {"robot_0": first_robot}
        robot_ids = ["robot_0"]
        robot_starts = {"robot_0": data["robots"]["robot_0"]["start"]}
        assignment_strategy = "sequential_single_robot"
        centralized_rule = False
    elif method == "naive_multi_robot_recovery":
        robot_ids = sorted(data["robots"])
        robot_starts = {
            robot_id: data["robots"][robot_id]["start"] for robot_id in robot_ids
        }
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        centralized_rule = False
    elif method == "centralized_rule_multi_robot_recovery":
        robot_ids = sorted(data["robots"])
        robot_starts = {
            robot_id: data["robots"][robot_id]["start"] for robot_id in robot_ids
        }
        assignment_strategy = data["task"].get("assignment_strategy", "nearest_robot")
        centralized_rule = True
    else:
        raise ValueError(f"Unknown method: {method}")

    assignments = assign_waypoints(
        assignment_strategy,
        robot_ids,
        waypoints,
        robot_starts=robot_starts,
        seed=seed,
    )
    for robot_id in robot_ids:
        data["robots"][robot_id]["goal"] = None
    data["task"]["assignment_strategy"] = assignment_strategy
    data["task"]["baseline_type"] = method
    data["task"]["assignments"] = assignments
    data["centralized_rule"] = centralized_rule
    data["recovery"]["enabled"] = True
    data["render"] = False
    return data


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
