"""Audit robot collisions directly from saved trace.json files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


METHOD_NAMES = (
    "centralized_rule_with_approach_radius_1",
    "centralized_rule_multi_bt",
    "llm_only_multi_robot",
    "centralized_without_repair",
)

BATCH_FIELDS = [
    "run_dir",
    "trace_file",
    "iteration",
    "is_final_trace",
    "metrics_file",
    "reported_collision_source",
    "method",
    "scenario_name",
    "success",
    "reported_collision_count",
    "audited_vertex_conflict_count",
    "audited_edge_swap_conflict_count",
    "audited_collision_count",
    "match",
    "first_conflict_timestep",
]


Cell = tuple[Any, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trace", type=Path, help="Path to one trace.json file.")
    group.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing run subdirectories with trace.json files.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only audit final traces, using raw.jsonl final_trace_path when available.",
    )
    parser.add_argument("--out", type=Path, help="CSV output path for --runs-dir mode.")
    args = parser.parse_args(argv)

    if args.trace is not None:
        result = audit_trace_file(args.trace)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.out is None:
        parser.error("--out is required with --runs-dir")
    rows = audit_runs_dir(args.runs_dir, final_only=args.final_only)
    write_batch_csv(args.out, rows)
    summary = summarize_batch_rows(rows)
    write_summary_json(summary_path_for_csv(args.out), summary)
    for row in rows:
        warnings = row.pop("_warnings", [])
        for warning in warnings:
            print(f"warning: {row['run_dir']}: {warning}", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote collision audit rows: {len(rows)} -> {args.out}")
    return 0


def audit_trace_file(path: Path) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        payload = load_json(path)
    except Exception as exc:
        return audit_error_result(f"failed to read trace JSON: {exc}")
    return audit_trace_payload(payload, warnings)


def audit_trace_payload(payload: Any, warnings: list[str] | None = None) -> dict[str, Any]:
    warnings = warnings if warnings is not None else []
    timesteps = extract_timesteps(payload, warnings)
    frames = [extract_frame(item, index, warnings) for index, item in enumerate(timesteps)]

    vertex_conflicts = find_vertex_conflicts(frames)
    edge_swap_conflicts = find_edge_swap_conflicts(frames)
    conflicts = vertex_conflicts + edge_swap_conflicts
    first_timestep = first_conflict_timestep(conflicts)
    audited_collision_count = len(vertex_conflicts) + len(edge_swap_conflicts)
    reported_collision_count = extract_reported_collision_count(payload)

    return {
        "audited_vertex_conflict_count": len(vertex_conflicts),
        "audited_edge_swap_conflict_count": len(edge_swap_conflicts),
        "audited_collision_count": audited_collision_count,
        "first_conflict_timestep": first_timestep,
        "conflict_examples": conflicts[:10],
        "reported_collision_count": reported_collision_count,
        "match": (
            None
            if reported_collision_count is None
            else int(reported_collision_count) == audited_collision_count
        ),
        "warnings": warnings,
    }


def audit_runs_dir(runs_dir: Path, final_only: bool = False) -> list[dict[str, Any]]:
    final_trace_paths, final_warnings = discover_final_trace_paths(runs_dir)
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(runs_dir.rglob("trace.json")):
        warnings: list[str] = []
        try:
            payload = load_json(trace_path)
            audit = audit_trace_payload(payload, warnings)
        except Exception as exc:
            payload = {}
            audit = audit_error_result(f"failed to audit trace JSON: {exc}")
            warnings = list(audit["warnings"])
        run_dir = infer_run_dir(trace_path)
        trace_metadata = build_trace_metadata(trace_path, run_dir, final_trace_paths)
        if trace_metadata["final_source"] == "unknown":
            warnings.append("could not determine whether trace is final")
        warnings.extend(final_warnings)
        if final_only and not trace_metadata["is_final_trace"]:
            continue
        metrics_path, metrics_payload, metrics_warnings = load_same_round_metrics(trace_path)
        warnings.extend(metrics_warnings)
        reported_collision_count, reported_collision_source = reported_collision_from_metrics(
            metrics_path, metrics_payload
        )
        success = extract_success(metrics_payload)
        if success == "":
            success = extract_success(payload)
        row = {
            "run_dir": str(run_dir),
            "trace_file": str(trace_path),
            "iteration": trace_metadata["iteration"],
            "is_final_trace": trace_metadata["is_final_trace"],
            "metrics_file": str(metrics_path) if metrics_path is not None else "",
            "reported_collision_source": reported_collision_source,
            "method": infer_method(run_dir.name),
            "scenario_name": extract_scenario_name(payload) or run_dir.name,
            "success": success,
            "reported_collision_count": reported_collision_count,
            "audited_vertex_conflict_count": audit["audited_vertex_conflict_count"],
            "audited_edge_swap_conflict_count": audit["audited_edge_swap_conflict_count"],
            "audited_collision_count": audit["audited_collision_count"],
            "match": (
                None
                if reported_collision_count is None
                else int(reported_collision_count) == int(audit["audited_collision_count"])
            ),
            "first_conflict_timestep": audit["first_conflict_timestep"],
            "_warnings": warnings,
        }
        rows.append(row)
    return rows


def discover_final_trace_paths(runs_dir: Path) -> tuple[set[Path], list[str]]:
    warnings: list[str] = []
    final_trace_paths: set[Path] = set()
    raw_jsonl = runs_dir.parent / "raw.jsonl"
    if raw_jsonl.exists():
        try:
            with raw_jsonl.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    final_trace_path = row.get("final_trace_path")
                    if final_trace_path:
                        final_trace_paths.add(normalize_path(Path(final_trace_path)))
        except Exception as exc:
            warnings.append(f"failed to read final_trace_path values from raw.jsonl: {exc}")
    else:
        warnings.append("raw.jsonl not found; using highest iter_N trace per run as final")
        final_trace_paths.update(discover_highest_iteration_traces(runs_dir))

    for trace_path in runs_dir.glob("*/trace.json"):
        final_trace_paths.add(normalize_path(trace_path))
    return final_trace_paths, warnings


def discover_highest_iteration_traces(runs_dir: Path) -> set[Path]:
    latest_by_run: dict[Path, tuple[int, Path]] = {}
    for trace_path in runs_dir.rglob("trace.json"):
        run_dir = infer_run_dir(trace_path)
        iteration = infer_iteration(trace_path)
        if iteration is None:
            continue
        existing = latest_by_run.get(run_dir)
        if existing is None or iteration > existing[0]:
            latest_by_run[run_dir] = (iteration, trace_path)
    return {normalize_path(trace_path) for _, trace_path in latest_by_run.values()}


def build_trace_metadata(
    trace_path: Path, run_dir: Path, final_trace_paths: set[Path]
) -> dict[str, Any]:
    normalized_trace_path = normalize_path(trace_path)
    iteration = infer_iteration(trace_path)
    is_run_root_trace = trace_path.parent == run_dir
    final_source = "raw_jsonl_or_run_root" if final_trace_paths else "unknown"
    return {
        "iteration": "" if iteration is None else iteration,
        "is_final_trace": normalized_trace_path in final_trace_paths or is_run_root_trace,
        "final_source": final_source,
    }


def infer_iteration(trace_path: Path) -> int | None:
    match = re.fullmatch(r"iter_(\d+)", trace_path.parent.name)
    if match is None:
        return None
    return int(match.group(1))


def load_same_round_metrics(
    trace_path: Path,
) -> tuple[Path | None, Any, list[str]]:
    metrics_path = trace_path.with_name("metrics.json")
    if not metrics_path.exists():
        return None, None, [f"same-round metrics.json not found for {trace_path}"]
    try:
        return metrics_path, load_json(metrics_path), []
    except Exception as exc:
        return metrics_path, None, [f"failed to read same-round metrics.json: {exc}"]


def reported_collision_from_metrics(
    metrics_path: Path | None, metrics_payload: Any
) -> tuple[int | None, str]:
    if metrics_path is None or not isinstance(metrics_payload, dict):
        return None, "unknown"
    for key in ("collision_event_count", "collision_count"):
        if key not in metrics_payload:
            continue
        try:
            return int(metrics_payload[key]), f"{metrics_path}:{key}"
        except (TypeError, ValueError):
            return None, "unknown"
    return None, "unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_path(path: Path) -> Path:
    return path.resolve(strict=False)


def extract_timesteps(payload: Any, warnings: list[str]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("trace", "steps", "timesteps"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        warnings.append("trace payload has no list field among trace/steps/timesteps")
        return []
    warnings.append(f"trace payload must be a list or object, got {type(payload).__name__}")
    return []


def extract_frame(item: Any, index: int, warnings: list[str]) -> dict[str, Any]:
    if not isinstance(item, dict):
        warnings.append(f"timestep index {index} is not an object")
        return {"timestep": index, "positions": {}}
    timestep = extract_timestep(item, index)
    positions = extract_positions(item, timestep, warnings)
    return {"timestep": timestep, "positions": positions}


def extract_timestep(item: dict[str, Any], fallback: int) -> Any:
    for key in ("timestep", "time", "step", "t"):
        if key in item:
            return item[key]
    return fallback


def extract_positions(
    item: dict[str, Any], timestep: Any, warnings: list[str]
) -> dict[str, Cell]:
    positions_obj = None
    for key in ("robot_positions", "positions", "robots"):
        if key in item:
            positions_obj = item[key]
            break
    if positions_obj is None:
        warnings.append(f"timestep {timestep}: missing robot_positions")
        return {}

    positions: dict[str, Cell] = {}
    if isinstance(positions_obj, dict):
        for robot_id, raw_position in positions_obj.items():
            cell = parse_cell(raw_position)
            if cell is None:
                warnings.append(
                    f"timestep {timestep}: malformed position for robot {robot_id}"
                )
                continue
            positions[str(robot_id)] = cell
        return positions

    if isinstance(positions_obj, list):
        for index, row in enumerate(positions_obj):
            robot_id, cell = parse_robot_position_row(row)
            if robot_id is None or cell is None:
                warnings.append(
                    f"timestep {timestep}: malformed robot_positions[{index}]"
                )
                continue
            positions[str(robot_id)] = cell
        return positions

    warnings.append(
        f"timestep {timestep}: robot_positions must be object or list, "
        f"got {type(positions_obj).__name__}"
    )
    return {}


def parse_robot_position_row(row: Any) -> tuple[str | None, Cell | None]:
    if isinstance(row, dict):
        robot_id = row.get("robot_id") or row.get("id") or row.get("name")
        return str(robot_id) if robot_id is not None else None, parse_cell(row)
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return str(row[0]), parse_cell(row[1])
    return None, None


def parse_cell(value: Any) -> Cell | None:
    if isinstance(value, dict):
        for key in ("position", "pos", "cell", "location"):
            if key in value:
                return parse_cell(value[key])
        if "x" in value and "y" in value:
            return normalize_cell(value["x"], value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return normalize_cell(value[0], value[1])
    return None


def normalize_cell(x_value: Any, y_value: Any) -> Cell | None:
    try:
        x = int(x_value)
        y = int(y_value)
    except (TypeError, ValueError):
        return None
    return (x, y)


def find_vertex_conflicts(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for frame in frames:
        robots_by_cell: dict[Cell, list[str]] = {}
        for robot_id, cell in frame["positions"].items():
            robots_by_cell.setdefault(cell, []).append(robot_id)
        for cell, robots in sorted(robots_by_cell.items(), key=lambda item: item[0]):
            if len(robots) >= 2:
                conflicts.append(
                    {
                        "type": "vertex",
                        "timestep": frame["timestep"],
                        "cell": list(cell),
                        "robots": sorted(robots),
                    }
                )
    return conflicts


def find_edge_swap_conflicts(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for previous, current in zip(frames, frames[1:]):
        previous_positions = previous["positions"]
        current_positions = current["positions"]
        shared_robots = sorted(set(previous_positions) & set(current_positions))
        for left_index, robot_i in enumerate(shared_robots):
            for robot_j in shared_robots[left_index + 1 :]:
                i_from = previous_positions[robot_i]
                i_to = current_positions[robot_i]
                j_from = previous_positions[robot_j]
                j_to = current_positions[robot_j]
                if i_from == j_to and j_from == i_to and i_from != i_to:
                    conflicts.append(
                        {
                            "type": "edge_swap",
                            "timestep": current["timestep"],
                            "robot_i": robot_i,
                            "robot_j": robot_j,
                            "robot_i_from": list(i_from),
                            "robot_i_to": list(i_to),
                            "robot_j_from": list(j_from),
                            "robot_j_to": list(j_to),
                        }
                    )
    return conflicts


def first_conflict_timestep(conflicts: list[dict[str, Any]]) -> Any:
    if not conflicts:
        return None
    return min(conflict["timestep"] for conflict in conflicts)


def extract_reported_collision_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in ("collision_event_count", "collision_count"):
            if key in metrics:
                return int(metrics[key])
    for key in ("collision_event_count", "collision_count"):
        if key in payload:
            return int(payload[key])
    return None


def extract_success(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return ""
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and "success" in metrics:
        return metrics["success"]
    return payload.get("success", "")


def extract_scenario_name(payload: Any) -> str | None:
    if isinstance(payload, dict) and payload.get("scenario_name") is not None:
        return str(payload["scenario_name"])
    return None


def infer_run_dir(trace_path: Path) -> Path:
    if trace_path.parent.name.startswith("iter_"):
        return trace_path.parent.parent
    return trace_path.parent


def infer_method(run_dir_name: str) -> str:
    for method in METHOD_NAMES:
        suffix = f"_{method}"
        if run_dir_name.endswith(suffix):
            return method
    return ""


def audit_error_result(message: str) -> dict[str, Any]:
    return {
        "audited_vertex_conflict_count": 0,
        "audited_edge_swap_conflict_count": 0,
        "audited_collision_count": 0,
        "first_conflict_timestep": None,
        "conflict_examples": [],
        "reported_collision_count": None,
        "match": None,
        "warnings": [message],
    }


def write_batch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BATCH_FIELDS})


def summarize_batch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_dirs = {row.get("run_dir") for row in rows if row.get("run_dir")}
    mismatches = [row for row in rows if str(row.get("match")) != "True"]
    return {
        "total_traces_audited": len(rows),
        "total_runs": len(run_dirs),
        "approach_method_matches": count_matches(
            rows, "centralized_rule_with_approach_radius_1"
        ),
        "baseline_method_matches": count_matches(rows, "centralized_rule_multi_bt"),
        "mismatch_count": len(mismatches),
        "mismatch_examples": [
            {
                "run_dir": row.get("run_dir", ""),
                "trace_file": row.get("trace_file", ""),
                "method": row.get("method", ""),
                "reported_collision_count": row.get("reported_collision_count"),
                "audited_collision_count": row.get("audited_collision_count"),
                "first_conflict_timestep": row.get("first_conflict_timestep"),
            }
            for row in mismatches[:10]
        ],
    }


def count_matches(rows: list[dict[str, Any]], method: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("method") == method and str(row.get("match")) == "True"
    )


def summary_path_for_csv(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".summary.json")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
