"""Aggregate experiment raw JSONL rows into a compact CSV/Markdown summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


FIELDS = [
    "summary_scope",
    "method",
    "backend",
    "scenario_family",
    "runs",
    "success_rate",
    "avg_collision_event_count",
    "avg_actual_collision_count",
    "avg_collision_count",
    "avg_resource_conflict_count",
    "avg_resource_request_denied_count",
    "avg_lock_wait_count",
    "avg_rule_prevented_resource_conflict_count",
    "avg_recovery_conflict_count",
    "avg_duplicate_pick_attempt_count",
    "timeout_rate",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args(argv)
    rows = read_jsonl(args.raw)
    summary = aggregate_rows(rows)
    write_csv(args.out_csv, summary)
    if args.out_md:
        write_md(args.out_md, summary)
    print(f"Aggregated rows: {len(rows)}")
    return 0


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row.get("method", "unknown")),
                str(row.get("backend", "unknown")),
                str(row.get("scenario_family", "unknown")),
            ),
            [],
        ).append(row)
    result = []
    for (method, backend, scenario_family), group_rows in sorted(grouped.items()):
        result.append(build_row("by_family", method, backend, scenario_family, group_rows))

    overall: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        overall.setdefault(
            (str(row.get("method", "unknown")), str(row.get("backend", "unknown"))),
            [],
        ).append(row)
    for (method, backend), group_rows in sorted(overall.items()):
        result.append(build_row("overall", method, backend, "all", group_rows))
    return result


def build_row(
    summary_scope: str,
    method: str,
    backend: str,
    scenario_family: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary_scope": summary_scope,
        "method": method,
        "backend": backend,
        "scenario_family": scenario_family,
        "runs": len(rows),
        "success_rate": mean(1.0 if row.get("success") else 0.0 for row in rows),
        "avg_collision_event_count": avg(
            rows, "collision_event_count", fallback="collision_count"
        ),
        "avg_actual_collision_count": avg(rows, "actual_collision_count"),
        "avg_collision_count": avg(rows, "collision_count"),
        "avg_resource_conflict_count": avg(rows, "resource_conflict_count"),
        "avg_resource_request_denied_count": avg(rows, "resource_request_denied_count"),
        "avg_lock_wait_count": avg(rows, "lock_wait_count"),
        "avg_rule_prevented_resource_conflict_count": avg(
            rows, "rule_prevented_resource_conflict_count"
        ),
        "avg_recovery_conflict_count": avg(rows, "recovery_conflict_count"),
        "avg_duplicate_pick_attempt_count": avg(rows, "duplicate_pick_attempt_count"),
        "timeout_rate": mean(1.0 if row.get("timeout") else 0.0 for row in rows),
    }


def avg(rows: list[dict[str, Any]], key: str, fallback: str | None = None) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""} and fallback is not None:
            value = row.get(fallback)
        values.append(float(value or 0))
    return mean(values)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Aggregated Results",
        "",
        "| " + " | ".join(FIELDS) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(FIELDS) - 1)) + "|",
    ]
    for row in rows:
        values = [
            str(row.get("summary_scope", "")),
            str(row["method"]),
            str(row.get("backend", "")),
            str(row.get("scenario_family", "")),
        ] + [
            f"{float(row[field]):.3f}" if field != "runs" else str(row[field])
            for field in FIELDS[4:]
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
