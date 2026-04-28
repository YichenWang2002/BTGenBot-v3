"""Run SafetyGuarantor ablations with mock or OpenAI-compatible LLM backends."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from src.env.scenario_loader import load_scenario
from src.llm.backends.base import BaseLLMBackend
from src.llm.backends.mock import build_payload, extract_context
from src.llm.generate_multi_bt import build_backend, run_generated_plan
from src.llm.output_parser import parse_and_validate
from src.llm.prompt_builder import build_prompt
from src.repair.critic import static_validate
from src.repair.failure_report import build_failure_report


METHOD_ORDER = [
    "llm_only_multi_bt",
    "llm_static_critic_only",
    "llm_rule_no_repair",
    "llm_rule_with_repair_no_approach",
    "llm_rule_with_repair_full",
]

SUMMARY_FIELDS = [
    "summary_scope",
    "scenario_family",
    "method",
    "backend",
    "model",
    "runs",
    "success_rate",
    "avg_num_iters",
    "xml_valid_rate",
    "assignment_valid_rate",
    "resource_request_valid_rate",
    "provider_error_rate",
    "timeout_rate",
    "makespan",
    "collision_event_count",
    "actual_collision_count",
    "resource_conflict_count",
    "recovery_conflict_count",
    "approach_zone_wait_count",
    "lock_wait_count",
    "rule_prevented_motion_conflict_count",
    "rule_prevented_resource_conflict_count",
    "rule_prevented_approach_conflict_count",
]

RAW_FIELDS = [
    "run_id",
    "scenario_name",
    "scenario_family",
    "method",
    "backend",
    "model",
    "success",
    "num_iters",
    "xml_valid",
    "assignment_valid",
    "resource_request_valid",
    "static_valid",
    "provider_error",
    "error_message",
    "final_failure_types",
    "makespan",
    "timeout",
    "collision_event_count",
    "actual_collision_count",
    "resource_conflict_count",
    "recovery_conflict_count",
    "approach_zone_wait_count",
    "lock_wait_count",
    "rule_prevented_motion_conflict_count",
    "rule_prevented_resource_conflict_count",
    "rule_prevented_approach_conflict_count",
    "trace_path",
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    prompt_method: str
    execution_method: str
    static_hard_gate: bool
    repair_enabled: bool
    max_iters: int
    initial_repaired: bool
    coordination: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-scenarios-per-family", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--balanced-check", type=int, default=None)
    parser.add_argument("--families", default=None)
    args = parser.parse_args(argv)
    rows = run_suite(
        args.suite,
        backend_name=args.backend,
        output_dir_override=args.output_dir,
        max_scenarios_per_family=args.max_scenarios_per_family,
        max_runs=args.max_runs,
        balanced_check=args.balanced_check,
        families=parse_family_filter(args.families),
    )
    print(f"Safety ablation rows: {len(rows)}")
    return 0


def run_suite(
    suite_path: Path,
    backend_name: str = "mock",
    output_dir_override: Path | None = None,
    max_scenarios_per_family: int | None = None,
    max_runs: int | None = None,
    balanced_check: int | None = None,
    families: set[str] | None = None,
) -> list[dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    backend = build_backend(backend_name)
    model_name = backend_model_name(backend_name, backend)
    suite = load_yaml(suite_path)
    output_dir = Path(output_dir_override or suite.get("output_dir", "results/safety_guarantor_ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    methods = list(suite.get("methods") or METHOD_ORDER)
    specs = method_specs()
    scenarios = load_suite_scenarios(
        suite,
        max_scenarios_per_family=max_scenarios_per_family,
        balanced_check=balanced_check,
        families=families,
    )
    run_specs = [
        (scenario_path, scenario_family, specs[method])
        for scenario_path, scenario_family in scenarios
        for method in methods
    ]
    if max_runs is not None:
        run_specs = run_specs[: max(0, max_runs)]

    rows: list[dict[str, Any]] = []
    total_runs = len(run_specs)
    for index, (scenario_path, scenario_family, spec) in enumerate(run_specs, start=1):
        scenario = load_scenario(scenario_path)
        run_id = f"{suite.get('name', 'safety_ablation')}_{scenario.name}_{spec.name}"
        print(f"[{index}/{total_runs}] scenario={scenario.name} method={spec.name}")
        rows.append(
            run_one(
                run_id=run_id,
                scenario_path=scenario_path,
                scenario_family=scenario_family,
                spec=spec,
                trace_dir=trace_dir,
                backend=backend,
                backend_name=backend_name,
                model_name=model_name,
            )
        )

    write_raw_jsonl(output_dir / "raw.jsonl", rows)
    summary_rows = summarize_rows(rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_summary_md(output_dir / "summary.md", summary_rows)
    return rows


def run_one(
    run_id: str,
    scenario_path: Path,
    scenario_family: str,
    spec: MethodSpec,
    trace_dir: Path,
    backend: BaseLLMBackend,
    backend_name: str,
    model_name: str,
) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    previous_report: dict[str, Any] | None = None
    previous_robot_trees: dict[str, str] | None = None
    final_static: dict[str, Any] = {
        "xml_valid": False,
        "assignment_valid": False,
        "resource_valid": False,
        "valid": False,
    }
    final_metrics: dict[str, Any] = {}
    final_trace: dict[str, Any] = {"trace": []}
    final_report: dict[str, Any] = {"status": "failed", "failure_types": ["not_run"]}
    iterations_used = 0
    success = False
    trace_path = trace_dir / f"{safe_name(run_id)}_iter0.json"

    for iteration in range(spec.max_iters):
        iterations_used = iteration + 1
        force_repaired = spec.initial_repaired or previous_report is not None
        prompt = build_prompt(
            scenario_path,
            spec.prompt_method,
            previous_failure_report=previous_report,
            previous_robot_trees=previous_robot_trees,
        )
        trace_path = trace_dir / f"{safe_name(run_id)}_iter{iteration}.json"
        try:
            raw_response = generate_response(
                backend=backend,
                backend_name=backend_name,
                prompt=prompt,
                repaired=force_repaired,
            )
        except Exception as exc:
            error_message = sanitize_error_message(str(exc))
            write_json(trace_path, empty_trace_payload(scenario.name))
            final_report = {
                "status": "failed",
                "failure_types": ["provider_error"],
                "details": [{"type": "provider_error", "message": error_message}],
                "suggested_repairs": [],
            }
            return build_raw_row(
                run_id=run_id,
                scenario_name=scenario.name,
                scenario_family=scenario_family,
                method=spec.name,
                backend_name=backend_name,
                model_name=model_name,
                success=False,
                num_iters=iterations_used,
                static_result=final_static,
                metrics={},
                failure_report=final_report,
                trace_path=trace_path,
                provider_error=True,
                error_message=error_message,
            )
        parsed, schema_result = parse_and_validate(raw_response, scenario)
        if parsed is not None:
            previous_robot_trees = dict(parsed.get("robot_trees") or {})
        static_result = static_validate(scenario, parsed)
        if not schema_result.valid:
            static_result["valid"] = False
            static_result["errors"] = list(static_result.get("errors", [])) + schema_result.errors
            static_result["xml_valid"] = False
        final_static = static_result

        should_execute = parsed is not None and schema_result.valid
        if spec.static_hard_gate and not static_result.get("valid"):
            should_execute = False

        if should_execute:
            final_metrics, final_trace = run_generated_plan(
                scenario_path=scenario_path,
                payload=parsed or {},
                method=spec.execution_method,
                trace_path=trace_path,
                coordination_override=spec.coordination,
            )
        else:
            write_json(trace_path, empty_trace_payload(scenario.name))
            final_metrics = {}
            final_trace = {"trace": []}

        if (
            parsed is not None
            and schema_result.valid
            and (not spec.static_hard_gate or static_result.get("valid"))
            and final_metrics.get("success")
        ):
            success = True
            final_report = {
                "status": "success",
                "failure_types": [],
                "details": [],
                "suggested_repairs": ["no repair needed"],
            }
            break

        final_report = build_failure_report(
            scenario=scenario,
            assignment=(parsed or {}).get("assignment") if parsed else [],
            metrics=final_metrics,
            trace=final_trace.get("trace", []),
            static_validation=static_result,
        )
        if not spec.repair_enabled:
            break
        previous_report = final_report

    return build_raw_row(
        run_id=run_id,
        scenario_name=scenario.name,
        scenario_family=scenario_family,
        method=spec.name,
        backend_name=backend_name,
        model_name=model_name,
        success=success,
        num_iters=iterations_used,
        static_result=final_static,
        metrics=final_metrics,
        failure_report=final_report,
        trace_path=trace_path,
        provider_error=False,
        error_message="",
    )


def generate_response(
    backend: BaseLLMBackend,
    backend_name: str,
    prompt: str,
    repaired: bool,
) -> str:
    if backend_name == "mock":
        return generate_mock_response(prompt, repaired=repaired)
    return backend.generate(prompt)


def generate_mock_response(prompt: str, repaired: bool) -> str:
    context = extract_context(prompt)
    payload = build_payload(context, repaired=repaired)
    return json.dumps(payload, ensure_ascii=False)


def build_raw_row(
    run_id: str,
    scenario_name: str,
    scenario_family: str,
    method: str,
    backend_name: str,
    model_name: str,
    success: bool,
    num_iters: int,
    static_result: dict[str, Any],
    metrics: dict[str, Any],
    failure_report: dict[str, Any],
    trace_path: Path,
    provider_error: bool,
    error_message: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "method": method,
        "backend": backend_name,
        "model": model_name,
        "success": bool(success),
        "num_iters": int(num_iters),
        "xml_valid": bool(static_result.get("xml_valid")),
        "assignment_valid": bool(static_result.get("assignment_valid")),
        "resource_request_valid": bool(static_result.get("resource_valid")),
        "static_valid": bool(static_result.get("valid")),
        "provider_error": bool(provider_error),
        "error_message": sanitize_error_message(error_message),
        "final_failure_types": list(failure_report.get("failure_types", [])),
        "makespan": int(metrics.get("makespan", 0) or 0),
        "timeout": bool(metrics.get("timeout", False)),
        "collision_event_count": int(
            metrics.get("collision_event_count", metrics.get("collision_count", 0)) or 0
        ),
        "actual_collision_count": int(metrics.get("actual_collision_count", 0) or 0),
        "resource_conflict_count": int(metrics.get("resource_conflict_count", 0) or 0),
        "recovery_conflict_count": int(metrics.get("recovery_conflict_count", 0) or 0),
        "approach_zone_wait_count": int(metrics.get("approach_zone_wait_count", 0) or 0),
        "lock_wait_count": int(metrics.get("lock_wait_count", 0) or 0),
        "rule_prevented_motion_conflict_count": int(
            metrics.get("rule_prevented_motion_conflict_count", 0) or 0
        ),
        "rule_prevented_resource_conflict_count": int(
            metrics.get("rule_prevented_resource_conflict_count", 0) or 0
        ),
        "rule_prevented_approach_conflict_count": int(
            metrics.get("rule_prevented_approach_conflict_count", 0) or 0
        ),
        "trace_path": str(trace_path),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    by_method: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    by_family_method: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row["method"])
        backend = str(row.get("backend", "unknown"))
        model = str(row.get("model", "unknown"))
        by_method.setdefault((method, backend, model), []).append(row)
        by_family_method.setdefault(
            (str(row["scenario_family"]), method, backend, model), []
        ).append(row)
    for method in METHOD_ORDER:
        for (group_method, backend, model), group in sorted(by_method.items()):
            if group_method == method:
                summary.append(
                    build_summary_row("overall", "all", method, backend, model, group)
                )
    for (family, method, backend, model), group in sorted(by_family_method.items()):
        summary.append(build_summary_row("by_family", family, method, backend, model, group))
    return summary


def build_summary_row(
    summary_scope: str,
    scenario_family: str,
    method: str,
    backend: str,
    model: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary_scope": summary_scope,
        "scenario_family": scenario_family,
        "method": method,
        "backend": backend,
        "model": model,
        "runs": len(rows),
        "success_rate": average_bool(rows, "success"),
        "avg_num_iters": average_number(rows, "num_iters"),
        "xml_valid_rate": average_bool(rows, "xml_valid"),
        "assignment_valid_rate": average_bool(rows, "assignment_valid"),
        "resource_request_valid_rate": average_bool(rows, "resource_request_valid"),
        "provider_error_rate": average_bool(rows, "provider_error"),
        "timeout_rate": average_bool(rows, "timeout"),
        "makespan": average_number(rows, "makespan"),
        "collision_event_count": average_number(rows, "collision_event_count"),
        "actual_collision_count": average_number(rows, "actual_collision_count"),
        "resource_conflict_count": average_number(rows, "resource_conflict_count"),
        "recovery_conflict_count": average_number(rows, "recovery_conflict_count"),
        "approach_zone_wait_count": average_number(rows, "approach_zone_wait_count"),
        "lock_wait_count": average_number(rows, "lock_wait_count"),
        "rule_prevented_motion_conflict_count": average_number(
            rows, "rule_prevented_motion_conflict_count"
        ),
        "rule_prevented_resource_conflict_count": average_number(
            rows, "rule_prevented_resource_conflict_count"
        ),
        "rule_prevented_approach_conflict_count": average_number(
            rows, "rule_prevented_approach_conflict_count"
        ),
    }


def method_specs() -> dict[str, MethodSpec]:
    no_approach = {
        "enable_cell_reservation": True,
        "enable_edge_reservation": True,
        "enable_resource_lock": True,
        "enable_recovery_lock": True,
        "enable_approach_zone_lock": False,
    }
    full = {
        **no_approach,
        "enable_approach_zone_lock": True,
        "approach_radius": 1,
        "max_wait_ticks": 3,
        "reassignment_wait_ticks": 8,
    }
    return {
        "llm_only_multi_bt": MethodSpec(
            name="llm_only_multi_bt",
            prompt_method="llm_only_multi_robot",
            execution_method="llm_only_multi_robot",
            static_hard_gate=False,
            repair_enabled=False,
            max_iters=1,
            initial_repaired=False,
            coordination={"enable_approach_zone_lock": False},
        ),
        "llm_static_critic_only": MethodSpec(
            name="llm_static_critic_only",
            prompt_method="llm_only_multi_robot",
            execution_method="llm_only_multi_robot",
            static_hard_gate=True,
            repair_enabled=False,
            max_iters=1,
            initial_repaired=True,
            coordination={"enable_approach_zone_lock": False},
        ),
        "llm_rule_no_repair": MethodSpec(
            name="llm_rule_no_repair",
            prompt_method="centralized_rule_multi_bt",
            execution_method="centralized_rule_multi_bt",
            static_hard_gate=True,
            repair_enabled=False,
            max_iters=1,
            initial_repaired=True,
            coordination=no_approach,
        ),
        "llm_rule_with_repair_no_approach": MethodSpec(
            name="llm_rule_with_repair_no_approach",
            prompt_method="centralized_rule_multi_bt",
            execution_method="centralized_rule_multi_bt",
            static_hard_gate=True,
            repair_enabled=True,
            max_iters=3,
            initial_repaired=False,
            coordination=no_approach,
        ),
        "llm_rule_with_repair_full": MethodSpec(
            name="llm_rule_with_repair_full",
            prompt_method="centralized_rule_with_approach_radius_1",
            execution_method="centralized_rule_with_approach_radius_1",
            static_hard_gate=True,
            repair_enabled=True,
            max_iters=3,
            initial_repaired=False,
            coordination=full,
        ),
    }


def load_suite_scenarios(
    suite: dict[str, Any],
    max_scenarios_per_family: int | None = None,
    balanced_check: int | None = None,
    families: set[str] | None = None,
) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for family in suite.get("families", []):
        family_name = str(family["name"])
        if families is not None and family_name not in families:
            continue
        manifest = load_json(Path(family["scenario_manifest"]))
        limit = int(family.get("max_scenarios", len(manifest.get("scenarios", []))))
        if max_scenarios_per_family is not None:
            limit = min(limit, int(max_scenarios_per_family))
        if balanced_check is not None:
            limit = min(limit, max(0, int(balanced_check)))
        for item in manifest.get("scenarios", [])[:limit]:
            rows.append((Path(item["scenario_path"]), family_name))
    return rows


def parse_family_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    families = {item.strip() for item in value.split(",") if item.strip()}
    return families or None


def backend_model_name(backend_name: str, backend: BaseLLMBackend) -> str:
    if backend_name == "mock":
        return "mock"
    model = getattr(backend, "model", None) or os.environ.get("OPENAI_MODEL")
    return str(model or "unknown")


def sanitize_error_message(message: str) -> str:
    sanitized = str(message)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        sanitized = sanitized.replace(api_key, "***")
    return sanitized


def average_number(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key, 0) or 0) for row in rows) if rows else 0.0


def average_bool(rows: list[dict[str, Any]], key: str) -> float:
    return mean(1.0 if row.get(key) else 0.0 for row in rows) if rows else 0.0


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SafetyGuarantor Ablation Summary",
        "",
        "| " + " | ".join(SUMMARY_FIELDS) + " |",
        "|" + "|".join(["---"] * 5 + ["---:"] * (len(SUMMARY_FIELDS) - 5)) + "|",
    ]
    for row in rows:
        values = []
        for field in SUMMARY_FIELDS:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def empty_trace_payload(scenario_name: str) -> dict[str, Any]:
    return {
        "scenario_name": scenario_name,
        "num_robots": 0,
        "metrics": {"success": False, "timeout": False},
        "trace": [],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
