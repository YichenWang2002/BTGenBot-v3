"""Generate compact Markdown tables for paper-facing experiment summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


PAPER_COLUMNS = [
    "summary_scope",
    "method",
    "backend",
    "scenario_family",
    "runs",
    "success_rate",
    "collision_event_count",
    "actual_collision_count",
    "approach_zone_wait_count",
    "avg_resource_conflict_count",
    "avg_resource_request_denied_count",
    "avg_lock_wait_count",
    "avg_rule_prevented_resource_conflict_count",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--collision-audit-csv", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = read_csv(args.summary_csv)
    audit_actual_by_method = (
        read_audit_actual_by_method(args.collision_audit_csv)
        if args.collision_audit_csv is not None
        else {}
    )
    paper_rows = build_paper_rows(rows, audit_actual_by_method)
    write_table(args.out, paper_rows)
    print(f"Paper table rows: {len(rows)}")
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_audit_actual_by_method(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    values_by_method: dict[str, list[float]] = {}
    for row in rows:
        method = row.get("method", "")
        if not method:
            continue
        values_by_method.setdefault(method, []).append(
            float(row.get("audited_collision_count", 0) or 0)
        )
    return {
        method: mean(values)
        for method, values in values_by_method.items()
        if values
    }


def build_paper_rows(
    rows: list[dict[str, str]], audit_actual_by_method: dict[str, float] | None = None
) -> list[dict[str, str]]:
    audit_actual_by_method = audit_actual_by_method or {}
    paper_rows: list[dict[str, str]] = []
    for row in rows:
        method = row.get("method", "")
        paper_row = dict(row)
        paper_row["collision_event_count"] = first_present(
            row,
            [
                "collision_event_count",
                "avg_collision_event_count",
                "collision_count",
                "avg_collision_count",
            ],
        )
        if method in audit_actual_by_method:
            paper_row["actual_collision_count"] = format_number(
                audit_actual_by_method[method]
            )
        else:
            paper_row["actual_collision_count"] = first_present(
                row,
                [
                    "actual_collision_count",
                    "avg_actual_collision_count",
                ],
            )
        paper_row["approach_zone_wait_count"] = first_present(
            row,
            [
                "approach_zone_wait_count",
                "avg_approach_zone_wait_count",
            ],
        )
        paper_rows.append(paper_row)
    return paper_rows


def first_present(row: dict[str, str], columns: list[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value not in {None, ""}:
            return str(value)
    return "0"


def format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Table",
        "",
        "| " + " | ".join(PAPER_COLUMNS) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(PAPER_COLUMNS) - 1)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(row.get(column, "0") for column in PAPER_COLUMNS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
