"""Failure-driven LLM multi-robot BT repair loop."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from src.env.scenario_loader import load_scenario
from src.llm.generate_multi_bt import build_backend, run_generated_plan
from src.llm.output_parser import parse_and_validate
from src.llm.prompt_builder import build_prompt
from src.repair.critic import static_validate
from src.repair.failure_report import build_failure_report


OPENAI_ENV_VARS = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")

RAW_FIELDS = [
    "run_id",
    "scenario_name",
    "scenario_family",
    "method",
    "backend",
    "success",
    "num_iters",
    "final_failure_types",
    "provider_error",
    "error_message",
    "xml_valid",
    "assignment_valid",
    "collision_count",
    "collision_event_count",
    "actual_collision_count",
    "resource_conflict_count",
    "resource_request_denied_count",
    "lock_wait_count",
    "rule_prevented_resource_conflict_count",
    "recovery_conflict_count",
    "duplicate_pick_attempt_count",
    "approach_zone_wait_count",
    "approach_lock_hold_time",
    "approach_lock_starvation_count",
    "approach_lock_reassignment_count",
    "tasks_completed",
    "timeout",
    "final_trace_path",
]

SUMMARY_FIELDS = [
    "summary_scope",
    "method",
    "backend",
    "scenario_family",
    "runs",
    "success_rate",
    "avg_num_iters",
    "xml_valid_rate",
    "assignment_valid_rate",
    "provider_error_count",
    "provider_error_rate",
    "avg_collision_count",
    "avg_collision_event_count",
    "avg_actual_collision_count",
    "avg_resource_conflict_count",
    "avg_resource_request_denied_count",
    "avg_lock_wait_count",
    "avg_rule_prevented_resource_conflict_count",
    "avg_recovery_conflict_count",
    "avg_duplicate_pick_attempt_count",
    "timeout_rate",
]

LLM_PICKPLACE_APPROACH_V2_SUMMARY_STYLE = "llm_pickplace_approach_v2"
DAG_LLM_MOCK_SUMMARY_STYLE = "dag_llm_mock"

LLM_PICKPLACE_APPROACH_V2_SUMMARY_FIELDS = [
    "summary_scope",
    "method",
    "backend",
    "scenario_family",
    "runs",
    "success_rate",
    "avg_num_iters",
    "collision_count",
    "collision_event_count",
    "actual_collision_count",
    "resource_conflict_count",
    "approach_zone_wait_count",
    "approach_lock_hold_time",
    "approach_lock_starvation_count",
    "approach_lock_reassignment_count",
    "timeout_rate",
    "provider_error_rate",
]

DAG_LLM_MOCK_SUMMARY_FIELDS = [
    "summary_scope",
    "method",
    "backend",
    "scenario_family",
    "runs",
    "success_rate",
    "avg_num_iters",
    "xml_valid_rate",
    "assignment_valid_rate",
    "dag_static_error_count",
    "missing_dependency_wait_count",
    "invalid_task_id_count",
    "wrong_assigned_robot_for_task_count",
    "dag_violation_count",
    "dependency_wait_count",
    "dag_task_completion_rate",
    "resource_conflict_count",
    "collision_event_count",
    "actual_collision_count",
    "timeout_rate",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    rows = run_suite(
        args.suite,
        backend_name=args.backend,
        max_iters_override=args.max_iters,
        output_dir_override=args.output_dir,
        max_runs=args.max_runs,
        fail_fast=args.fail_fast,
    )
    print(f"LLM repair rows: {len(rows)}")
    return 0


def run_suite(
    suite_path: Path,
    backend_name: str = "mock",
    max_iters_override: int | None = None,
    output_dir_override: Path | None = None,
    max_runs: int | None = None,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    suite = load_yaml(suite_path)
    output_dir = Path(output_dir_override or suite.get("output_dir", "results/llm_repair_smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend_requires_openai_env(backend_name) and not openai_env_configured():
        print_openai_env_help()
        return []
    methods = list(suite.get("methods") or [])
    method_configs = dict(suite.get("method_configs") or {})
    scenarios = load_suite_scenarios(suite)
    max_iters = int(max_iters_override or suite.get("max_iters", 3))
    profile = load_profile(Path(suite.get("profile", "data/derived/bt_dataset_profile.json")))
    run_specs: list[tuple[Path, str, str, dict[str, Any]]] = [
        (
            scenario_path,
            scenario_family,
            method,
            dict(method_configs.get(method) or {}),
        )
        for scenario_path, scenario_family in scenarios
        for method in methods
    ]
    if max_runs is not None:
        run_specs = run_specs[: max(0, max_runs)]
    total_runs = len(run_specs)

    rows: list[dict[str, Any]] = []
    for run_index, (
        scenario_path,
        scenario_family,
        method,
        method_config,
    ) in enumerate(run_specs, start=1):
        scenario = load_scenario(scenario_path)
        source_sample = sample_for_scenario(scenario.raw, profile)
        backend = build_backend(backend_name)
        run_id = f"{suite.get('name', 'llm_repair')}_{scenario.name}_{method}"
        print(
            f"[{run_index}/{total_runs}] scenario={scenario.name} method={method} "
            f"backend={backend_name} model={backend_model(backend)}"
        )
        row = run_one(
            run_id=run_id,
            scenario_path=scenario_path,
            scenario_family=scenario_family,
            method=method,
            backend_name=backend_name,
            backend=backend,
            source_sample=source_sample,
            output_dir=output_dir,
            max_iters=max_iters,
            method_config=method_config,
        )
        rows.append(row)
        if row.get("provider_error"):
            print(
                "[provider_error] saved artifacts to "
                f"{output_dir / 'runs' / safe_name(run_id)}"
            )
            if fail_fast:
                break

    write_raw_jsonl(output_dir / "raw.jsonl", rows)
    if suite.get("summary_style") == LLM_PICKPLACE_APPROACH_V2_SUMMARY_STYLE:
        summary = summarize_llm_pickplace_approach_rows(rows)
        write_llm_pickplace_approach_summary_csv(output_dir / "summary.csv", summary)
        write_llm_pickplace_approach_summary_md(output_dir / "summary.md", summary)
    elif suite.get("summary_style") == DAG_LLM_MOCK_SUMMARY_STYLE:
        summary = summarize_dag_llm_mock_rows(rows)
        write_dag_llm_mock_summary_csv(output_dir / "summary.csv", summary)
        write_dag_llm_mock_summary_md(output_dir / "summary.md", summary)
    else:
        summary = summarize_rows(rows)
        write_summary_csv(output_dir / "summary.csv", summary)
        write_summary_md(output_dir / "summary.md", summary)
    return rows


def run_one(
    run_id: str,
    scenario_path: Path,
    scenario_family: str,
    method: str,
    backend_name: str,
    backend: Any,
    source_sample: dict[str, Any] | None,
    output_dir: Path,
    max_iters: int,
    method_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    run_dir = output_dir / "runs" / safe_name(run_id)
    previous_report: dict[str, Any] | None = None
    previous_robot_trees: dict[str, str] | None = None
    final_metrics: dict[str, Any] = {}
    final_report: dict[str, Any] = {"status": "failed", "failure_types": ["not_run"]}
    final_static = {"xml_valid": False, "assignment_valid": False}
    final_trace_path = ""
    parsed: dict[str, Any] | None = None
    success = False
    iterations_used = 0

    for iteration in range(max_iters):
        iterations_used = iteration + 1
        iter_dir = run_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(
            scenario_path,
            method,
            source_sample=source_sample,
            previous_failure_report=previous_report,
            previous_robot_trees=previous_robot_trees,
        )
        write_text(iter_dir / "prompt.txt", prompt)
        try:
            raw_response = backend.generate(prompt)
        except Exception as exc:
            message = sanitize_error_message(str(exc), backend)
            return write_provider_error_artifacts(
                iter_dir=iter_dir,
                run_id=run_id,
                scenario=scenario,
                scenario_family=scenario_family,
                method=method,
                backend_name=backend_name,
                backend=backend,
                message=message,
                iteration=iteration,
            )
        write_text(iter_dir / "raw_response.txt", raw_response)

        parsed, schema_result = parse_and_validate(raw_response, scenario)
        write_json(iter_dir / "parsed_response.json", parsed or {})
        if parsed is not None:
            previous_robot_trees = dict(parsed.get("robot_trees") or {})
        static_result = static_validate(scenario, parsed)
        if not schema_result.valid:
            static_result["valid"] = False
            static_result["errors"] = list(static_result.get("errors", [])) + schema_result.errors
            static_result["xml_valid"] = False
        write_json(iter_dir / "static_validation.json", static_result)

        trace_path = iter_dir / "trace.json"
        metrics: dict[str, Any] = {}
        trace_payload: dict[str, Any] = {"trace": []}
        if parsed is not None and schema_result.valid:
            metrics, trace_payload = run_generated_plan(
                scenario_path=scenario_path,
                payload=parsed,
                method=method,
                trace_path=trace_path,
                coordination_override=(method_config or {}).get("coordination"),
            )
        else:
            write_json(trace_path, trace_payload)
        final_trace_path = str(trace_path)
        final_metrics = metrics
        write_json(iter_dir / "metrics.json", metrics)

        if schema_result.valid and static_result.get("valid") and metrics.get("success"):
            final_report = {
                "status": "success",
                "failure_types": [],
                "details": [],
                "suggested_repairs": ["no repair needed"],
            }
            write_json(iter_dir / "failure_report.json", final_report)
            success = True
            final_static = static_result
            break

        final_report = build_failure_report(
            scenario=scenario,
            assignment=(parsed or {}).get("assignment") if parsed else [],
            metrics=metrics,
            trace=trace_payload.get("trace", []),
            static_validation=static_result,
        )
        write_json(iter_dir / "failure_report.json", final_report)
        previous_report = final_report
        final_static = static_result

    return {
        "run_id": run_id,
        "scenario_name": scenario.name,
        "scenario_family": scenario_family,
        "method": method,
        "backend": backend_name,
        "success": success,
        "num_iters": iterations_used,
        "final_failure_types": final_report.get("failure_types", []),
        "provider_error": False,
        "error_message": "",
        "xml_valid": bool(final_static.get("xml_valid")),
        "assignment_valid": bool(final_static.get("assignment_valid")),
        "dag_static_error_count": int(
            final_static.get("dag_static_error_count", 0) or 0
        ),
        "missing_dependency_wait_count": int(
            final_static.get("missing_dependency_wait_count", 0) or 0
        ),
        "invalid_task_id_count": int(final_static.get("invalid_task_id_count", 0) or 0),
        "wrong_assigned_robot_for_task_count": int(
            final_static.get("wrong_assigned_robot_for_task_count", 0) or 0
        ),
        "dag_violation_count": int(final_metrics.get("dag_violation_count", 0) or 0),
        "dependency_wait_count": int(final_metrics.get("dependency_wait_count", 0) or 0),
        "dag_task_count": int(final_metrics.get("dag_task_count", 0) or 0),
        "dag_completed_task_count": int(
            final_metrics.get("dag_completed_task_count", 0) or 0
        ),
        "dag_task_completion_rate": (
            int(final_metrics.get("dag_completed_task_count", 0) or 0)
            / int(final_metrics.get("dag_task_count", 0) or 0)
            if int(final_metrics.get("dag_task_count", 0) or 0) > 0
            else 0.0
        ),
        "collision_count": int(final_metrics.get("collision_count", 0) or 0),
        "collision_event_count": int(
            final_metrics.get(
                "collision_event_count", final_metrics.get("collision_count", 0)
            )
            or 0
        ),
        "actual_collision_count": int(
            final_metrics.get("actual_collision_count", 0) or 0
        ),
        "resource_conflict_count": int(final_metrics.get("resource_conflict_count", 0) or 0),
        "resource_request_denied_count": int(
            final_metrics.get("resource_request_denied_count", 0) or 0
        ),
        "lock_wait_count": int(final_metrics.get("lock_wait_count", 0) or 0),
        "rule_prevented_resource_conflict_count": int(
            final_metrics.get("rule_prevented_resource_conflict_count", 0) or 0
        ),
        "recovery_conflict_count": int(final_metrics.get("recovery_conflict_count", 0) or 0),
        "duplicate_pick_attempt_count": int(
            final_metrics.get("duplicate_pick_attempt_count", 0) or 0
        ),
        "approach_zone_wait_count": int(
            final_metrics.get("approach_zone_wait_count", 0) or 0
        ),
        "approach_lock_hold_time": int(
            final_metrics.get("approach_lock_hold_time", 0) or 0
        ),
        "approach_lock_starvation_count": int(
            final_metrics.get("approach_lock_starvation_count", 0) or 0
        ),
        "approach_lock_reassignment_count": int(
            final_metrics.get("approach_lock_reassignment_count", 0) or 0
        ),
        "tasks_completed": int(final_metrics.get("tasks_completed", 0) or 0),
        "timeout": bool(final_metrics.get("timeout", False)),
        "final_trace_path": final_trace_path,
    }


def write_provider_error_artifacts(
    iter_dir: Path,
    run_id: str,
    scenario: Any,
    scenario_family: str,
    method: str,
    backend_name: str,
    backend: Any,
    message: str,
    iteration: int,
) -> dict[str, Any]:
    failure_report = provider_error_failure_report(backend_name, backend, message)
    metrics = {"success": False, "timeout": False, "provider_error": True}
    static_validation = {
        "valid": False,
        "xml_valid": False,
        "assignment_valid": False,
        "errors": [message],
    }
    write_text(iter_dir / "raw_response.txt", f"PROVIDER_ERROR: {message}\n")
    write_json(iter_dir / "parsed_response.json", {})
    write_json(iter_dir / "static_validation.json", static_validation)
    write_json(iter_dir / "metrics.json", metrics)
    write_json(iter_dir / "trace.json", {"trace": []})
    write_json(iter_dir / "failure_report.json", failure_report)
    return {
        "run_id": run_id,
        "scenario_name": scenario.name,
        "scenario_family": scenario_family,
        "method": method,
        "backend": backend_name,
        "success": False,
        "num_iters": iteration + 1,
        "final_failure_types": ["provider_error"],
        "provider_error": True,
        "error_message": message,
        "xml_valid": False,
        "assignment_valid": False,
        "dag_static_error_count": 0,
        "missing_dependency_wait_count": 0,
        "invalid_task_id_count": 0,
        "wrong_assigned_robot_for_task_count": 0,
        "dag_violation_count": 0,
        "dependency_wait_count": 0,
        "dag_task_count": 0,
        "dag_completed_task_count": 0,
        "dag_task_completion_rate": 0.0,
        "collision_count": 0,
        "collision_event_count": 0,
        "actual_collision_count": 0,
        "resource_conflict_count": 0,
        "resource_request_denied_count": 0,
        "lock_wait_count": 0,
        "rule_prevented_resource_conflict_count": 0,
        "recovery_conflict_count": 0,
        "duplicate_pick_attempt_count": 0,
        "approach_zone_wait_count": 0,
        "approach_lock_hold_time": 0,
        "approach_lock_starvation_count": 0,
        "approach_lock_reassignment_count": 0,
        "tasks_completed": 0,
        "timeout": False,
        "final_trace_path": str(iter_dir / "trace.json"),
    }


def provider_error_failure_report(
    backend_name: str, backend: Any, message: str
) -> dict[str, Any]:
    return {
        "status": "failed",
        "failure_types": ["provider_error"],
        "details": [
            {
                "type": "provider_error",
                "message": message,
                "backend": backend_name,
                "model": backend_model(backend),
                "provider_profile": backend_provider_profile(backend),
            }
        ],
        "suggested_repairs": [
            "Retry with OPENAI_PROVIDER_PROFILE=opencode-zen",
            "Set OPENAI_DISABLE_RESPONSE_FORMAT=1",
            "Use --max-runs for debugging",
            "Try an alternative OPENAI_MODEL",
        ],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["method"], row["backend"], str(row.get("scenario_family", "unknown"))),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (method, backend, scenario_family), group in sorted(groups.items()):
        summary.append(build_summary_row("by_family", method, backend, scenario_family, group))

    overall_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        overall_groups.setdefault((row["method"], row["backend"]), []).append(row)
    for (method, backend), group in sorted(overall_groups.items()):
        summary.append(build_summary_row("overall", method, backend, "all", group))
    return summary


def build_summary_row(
    summary_scope: str,
    method: str,
    backend: str,
    scenario_family: str,
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary_scope": summary_scope,
        "method": method,
        "backend": backend,
        "scenario_family": scenario_family,
        "runs": len(group),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in group),
        "avg_num_iters": mean(float(row["num_iters"]) for row in group),
        "xml_valid_rate": mean(1.0 if row["xml_valid"] else 0.0 for row in group),
        "assignment_valid_rate": mean(
            1.0 if row["assignment_valid"] else 0.0 for row in group
        ),
        "provider_error_count": sum(1 for row in group if row.get("provider_error")),
        "provider_error_rate": mean(
            1.0 if row.get("provider_error") else 0.0 for row in group
        ),
        "avg_collision_count": mean(float(row["collision_count"]) for row in group),
        "avg_collision_event_count": average_row_number(
            group, "collision_event_count"
        ),
        "avg_actual_collision_count": average_row_number(
            group, "actual_collision_count"
        ),
        "avg_resource_conflict_count": mean(
            float(row["resource_conflict_count"]) for row in group
        ),
        "avg_resource_request_denied_count": mean(
            float(row["resource_request_denied_count"]) for row in group
        ),
        "avg_lock_wait_count": mean(float(row["lock_wait_count"]) for row in group),
        "avg_rule_prevented_resource_conflict_count": mean(
            float(row["rule_prevented_resource_conflict_count"]) for row in group
        ),
        "avg_recovery_conflict_count": mean(
            float(row["recovery_conflict_count"]) for row in group
        ),
        "avg_duplicate_pick_attempt_count": mean(
            float(row["duplicate_pick_attempt_count"]) for row in group
        ),
        "timeout_rate": mean(1.0 if row["timeout"] else 0.0 for row in group),
    }


def summarize_llm_pickplace_approach_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["method"], row["backend"], str(row.get("scenario_family", "unknown"))),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (method, backend, scenario_family), group in sorted(groups.items()):
        summary.append(
            build_llm_pickplace_approach_summary_row(
                "by_family", method, backend, scenario_family, group
            )
        )

    overall_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        overall_groups.setdefault((row["method"], row["backend"]), []).append(row)
    for (method, backend), group in sorted(overall_groups.items()):
        summary.append(
            build_llm_pickplace_approach_summary_row(
                "overall", method, backend, "all", group
            )
        )
    return summary


def build_llm_pickplace_approach_summary_row(
    summary_scope: str,
    method: str,
    backend: str,
    scenario_family: str,
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary_scope": summary_scope,
        "method": method,
        "backend": backend,
        "scenario_family": scenario_family,
        "runs": len(group),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in group),
        "avg_num_iters": mean(float(row["num_iters"]) for row in group),
        "collision_count": average_row_number(group, "collision_count"),
        "collision_event_count": average_row_number(group, "collision_event_count"),
        "actual_collision_count": average_row_number(group, "actual_collision_count"),
        "resource_conflict_count": average_row_number(group, "resource_conflict_count"),
        "approach_zone_wait_count": average_row_number(
            group, "approach_zone_wait_count"
        ),
        "approach_lock_hold_time": average_row_number(
            group, "approach_lock_hold_time"
        ),
        "approach_lock_starvation_count": average_row_number(
            group, "approach_lock_starvation_count"
        ),
        "approach_lock_reassignment_count": average_row_number(
            group, "approach_lock_reassignment_count"
        ),
        "timeout_rate": mean(1.0 if row["timeout"] else 0.0 for row in group),
        "provider_error_rate": mean(
            1.0 if row.get("provider_error") else 0.0 for row in group
        ),
    }


def average_row_number(group: list[dict[str, Any]], field: str) -> float:
    return mean(float(row.get(field, 0) or 0) for row in group)


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
        "# LLM Repair Summary",
        "",
    ]
    append_summary_table(lines, "Overall By Method/Backend", rows, "overall")
    append_summary_table(lines, "By Method/Backend/Scenario Family", rows, "by_family")
    write_text(path, "\n".join(lines) + "\n")


def write_llm_pickplace_approach_summary_csv(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=LLM_PICKPLACE_APPROACH_V2_SUMMARY_FIELDS
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_llm_pickplace_approach_summary_md(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    lines = [
        "# LLM Pickplace Approach V2 Summary",
        "",
    ]
    append_llm_pickplace_approach_summary_table(
        lines, "Overall By Method/Backend", rows, "overall"
    )
    append_llm_pickplace_approach_summary_table(
        lines, "By Method/Backend/Scenario Family", rows, "by_family"
    )
    write_text(path, "\n".join(lines) + "\n")


def append_llm_pickplace_approach_summary_table(
    lines: list[str], title: str, rows: list[dict[str, Any]], summary_scope: str
) -> None:
    scoped_rows = [row for row in rows if row.get("summary_scope") == summary_scope]
    if not scoped_rows:
        return
    lines.extend(
        [
            f"## {title}",
            "",
            "| method | backend | scenario_family | runs | success_rate | avg_num_iters | collision_count | collision_event_count | actual_collision_count | resource_conflict_count | approach_zone_wait_count | approach_lock_hold_time | approach_lock_starvation_count | approach_lock_reassignment_count | timeout_rate | provider_error_rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scoped_rows:
        lines.append(
            "| {method} | {backend} | {scenario_family} | {runs} | "
            "{success_rate:.3f} | {avg_num_iters:.3f} | {collision_count:.3f} | "
            "{collision_event_count:.3f} | {actual_collision_count:.3f} | "
            "{resource_conflict_count:.3f} | {approach_zone_wait_count:.3f} | "
            "{approach_lock_hold_time:.3f} | {approach_lock_starvation_count:.3f} | "
            "{approach_lock_reassignment_count:.3f} | {timeout_rate:.3f} | "
            "{provider_error_rate:.3f} |".format(**row)
    )
    lines.append("")


def append_summary_table(
    lines: list[str], title: str, rows: list[dict[str, Any]], summary_scope: str
) -> None:
    scoped_rows = [row for row in rows if row.get("summary_scope") == summary_scope]
    if not scoped_rows:
        return
    lines.extend(
        [
            f"## {title}",
            "",
            "| method | backend | scenario_family | runs | success_rate | avg_num_iters | xml_valid_rate | assignment_valid_rate | provider_error_count | provider_error_rate | avg_collision_count | avg_collision_event_count | avg_actual_collision_count | avg_resource_conflict_count | avg_resource_request_denied_count | avg_lock_wait_count | avg_rule_prevented_resource_conflict_count | avg_recovery_conflict_count | avg_duplicate_pick_attempt_count | timeout_rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scoped_rows:
        lines.append(
            "| {method} | {backend} | {scenario_family} | {runs} | {success_rate:.3f} | "
            "{avg_num_iters:.3f} | {xml_valid_rate:.3f} | {assignment_valid_rate:.3f} | "
            "{provider_error_count} | {provider_error_rate:.3f} | "
            "{avg_collision_count:.3f} | {avg_collision_event_count:.3f} | "
            "{avg_actual_collision_count:.3f} | {avg_resource_conflict_count:.3f} | "
            "{avg_resource_request_denied_count:.3f} | {avg_lock_wait_count:.3f} | "
            "{avg_rule_prevented_resource_conflict_count:.3f} | "
            "{avg_recovery_conflict_count:.3f} | {avg_duplicate_pick_attempt_count:.3f} | "
            "{timeout_rate:.3f} |".format(**row)
    )
    lines.append("")


def summarize_dag_llm_mock_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["method"], row["backend"], str(row.get("scenario_family", "unknown"))),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (method, backend, scenario_family), group in sorted(groups.items()):
        summary.append(
            build_dag_llm_mock_summary_row(
                "by_family", method, backend, scenario_family, group
            )
        )

    overall_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        overall_groups.setdefault((row["method"], row["backend"]), []).append(row)
    for (method, backend), group in sorted(overall_groups.items()):
        summary.append(
            build_dag_llm_mock_summary_row("overall", method, backend, "all", group)
        )
    return summary


def build_dag_llm_mock_summary_row(
    summary_scope: str,
    method: str,
    backend: str,
    scenario_family: str,
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary_scope": summary_scope,
        "method": method,
        "backend": backend,
        "scenario_family": scenario_family,
        "runs": len(group),
        "success_rate": mean(1.0 if row.get("success") else 0.0 for row in group),
        "avg_num_iters": mean(float(row.get("num_iters", 0) or 0) for row in group),
        "xml_valid_rate": mean(1.0 if row.get("xml_valid") else 0.0 for row in group),
        "assignment_valid_rate": mean(
            1.0 if row.get("assignment_valid") else 0.0 for row in group
        ),
        "dag_static_error_count": mean(
            float(row.get("dag_static_error_count", 0) or 0) for row in group
        ),
        "missing_dependency_wait_count": mean(
            float(row.get("missing_dependency_wait_count", 0) or 0) for row in group
        ),
        "invalid_task_id_count": mean(
            float(row.get("invalid_task_id_count", 0) or 0) for row in group
        ),
        "wrong_assigned_robot_for_task_count": mean(
            float(row.get("wrong_assigned_robot_for_task_count", 0) or 0)
            for row in group
        ),
        "dag_violation_count": mean(
            float(row.get("dag_violation_count", 0) or 0) for row in group
        ),
        "dependency_wait_count": mean(
            float(row.get("dependency_wait_count", 0) or 0) for row in group
        ),
        "dag_task_completion_rate": mean(
            float(row.get("dag_task_completion_rate", 0.0) or 0.0) for row in group
        ),
        "resource_conflict_count": mean(
            float(row.get("resource_conflict_count", 0) or 0) for row in group
        ),
        "collision_event_count": average_row_number(group, "collision_event_count"),
        "actual_collision_count": average_row_number(group, "actual_collision_count"),
        "timeout_rate": mean(1.0 if row.get("timeout") else 0.0 for row in group),
    }


def write_dag_llm_mock_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAG_LLM_MOCK_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dag_llm_mock_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DAG LLM Mock Summary",
        "",
    ]
    append_dag_llm_mock_summary_table(lines, "Overall By Method/Backend", rows, "overall")
    append_dag_llm_mock_summary_table(
        lines, "By Method/Backend/Scenario Family", rows, "by_family"
    )
    write_text(path, "\n".join(lines) + "\n")


def append_dag_llm_mock_summary_table(
    lines: list[str], title: str, rows: list[dict[str, Any]], summary_scope: str
) -> None:
    scoped_rows = [row for row in rows if row.get("summary_scope") == summary_scope]
    if not scoped_rows:
        return
    lines.extend(
        [
            f"## {title}",
            "",
            "| method | backend | scenario_family | runs | success_rate | avg_num_iters | xml_valid_rate | assignment_valid_rate | dag_static_error_count | missing_dependency_wait_count | invalid_task_id_count | wrong_assigned_robot_for_task_count | dag_violation_count | dependency_wait_count | dag_task_completion_rate | resource_conflict_count | collision_event_count | actual_collision_count | timeout_rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scoped_rows:
        lines.append(
            "| {method} | {backend} | {scenario_family} | {runs} | {success_rate:.3f} | "
            "{avg_num_iters:.3f} | {xml_valid_rate:.3f} | {assignment_valid_rate:.3f} | "
            "{dag_static_error_count:.3f} | {missing_dependency_wait_count:.3f} | "
            "{invalid_task_id_count:.3f} | {wrong_assigned_robot_for_task_count:.3f} | "
            "{dag_violation_count:.3f} | {dependency_wait_count:.3f} | "
            "{dag_task_completion_rate:.3f} | {resource_conflict_count:.3f} | "
            "{collision_event_count:.3f} | {actual_collision_count:.3f} | "
            "{timeout_rate:.3f} |".format(**row)
        )
    lines.append("")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_scenario_entry(entry: Any) -> tuple[Path, str]:
    if isinstance(entry, dict):
        scenario_path = Path(entry["path"])
        scenario_family = entry.get("scenario_family") or infer_scenario_family(scenario_path)
        return scenario_path, str(scenario_family)
    scenario_path = Path(entry)
    return scenario_path, infer_scenario_family(scenario_path)


def load_suite_scenarios(suite: dict[str, Any]) -> list[tuple[Path, str]]:
    if suite.get("scenarios"):
        return [normalize_scenario_entry(entry) for entry in suite.get("scenarios", [])]
    manifest_path = suite.get("scenario_manifest")
    if not manifest_path:
        return []
    manifest = read_json(Path(manifest_path))
    manifest_rows = manifest.get("scenarios", [])
    max_scenarios = int(suite.get("max_scenarios", len(manifest_rows)))
    scenarios: list[tuple[Path, str]] = []
    for manifest_row in manifest_rows[:max_scenarios]:
        scenario_path = Path(
            manifest_row.get("scenario_path")
            or manifest_row.get("path")
            or manifest_row.get("scenario")
        )
        scenario_family = str(
            manifest_row.get("scenario_family")
            or manifest_row.get("task_family")
            or infer_scenario_family(scenario_path)
        )
        scenarios.append((scenario_path, scenario_family))
    return scenarios


def infer_scenario_family(path: Path) -> str:
    value = str(path)
    if "nav_easy" in value:
        return "nav"
    if "recovery_medium" in value:
        return "recovery"
    if "pickplace_hard" in value:
        return "pickplace"
    return "unknown"


def backend_requires_openai_env(backend_name: str) -> bool:
    return backend_name in {"openai", "openai-compatible", "openai_compatible"}


def openai_env_configured() -> bool:
    return all(os.environ.get(name) for name in OPENAI_ENV_VARS)


def print_openai_env_help() -> None:
    missing = [name for name in OPENAI_ENV_VARS if not os.environ.get(name)]
    print("Missing OpenAI-compatible environment variables: " + ", ".join(missing))
    print("To run openai-compatible DeepSeek benchmark, set:")
    print('export OPENAI_BASE_URL="https://api.deepseek.com"')
    print('export OPENAI_API_KEY="..."')
    print('export OPENAI_MODEL="deepseek-v4-pro"')


def backend_model(backend: Any) -> str:
    return str(getattr(backend, "model", os.environ.get("OPENAI_MODEL", "unknown")))


def backend_provider_profile(backend: Any) -> str:
    return str(
        getattr(
            backend,
            "provider_profile",
            os.environ.get("OPENAI_PROVIDER_PROFILE", "default"),
        )
    )


def sanitize_error_message(message: str, backend: Any) -> str:
    secrets = [
        os.environ.get("OPENAI_API_KEY"),
        getattr(backend, "api_key", None),
    ]
    sanitized = message
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(str(secret), "***")
    return sanitized


def load_profile(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    return {int(sample["index"]): sample for sample in profile.get("samples", [])}


def sample_for_scenario(raw_scenario: dict[str, Any], profile: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    index = (raw_scenario.get("source") or {}).get("dataset_index")
    if index is None:
        return None
    return profile.get(int(index))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
