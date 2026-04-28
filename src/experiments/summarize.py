"""Summarize navigation experiment raw rows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SUMMARY_FIELDS = [
    "method",
    "runs",
    "success_rate",
    "avg_makespan",
    "avg_total_robot_steps",
    "avg_collision_event_count",
    "avg_actual_collision_count",
    "avg_collision_count",
    "avg_vertex_conflict_event_count",
    "avg_edge_conflict_event_count",
    "avg_audited_vertex_conflict_count",
    "avg_audited_edge_swap_conflict_count",
    "avg_motion_wait_count",
    "avg_rule_prevented_motion_conflict_count",
    "avg_rule_rejection_count",
    "avg_wait_count",
    "avg_deadlock_count",
    "avg_recovery_attempts",
    "avg_recovery_conflict_count",
    "avg_recovery_lock_wait_count",
    "avg_recovery_wait_time",
    "avg_successful_recoveries",
    "avg_failed_recoveries",
    "avg_resource_conflict_count",
    "avg_resource_request_denied_count",
    "avg_rule_prevented_resource_conflict_count",
    "avg_object_lock_denied_count",
    "avg_pickup_zone_denied_count",
    "avg_drop_zone_denied_count",
    "avg_recovery_zone_denied_count",
    "avg_approach_zone_wait_count",
    "avg_object_conflict_count",
    "avg_pickup_zone_conflict_count",
    "avg_drop_zone_conflict_count",
    "avg_duplicate_pick_attempt_count",
    "avg_pick_success_count",
    "avg_place_success_count",
    "avg_lock_wait_count",
    "avg_reassignment_count",
    "avg_tasks_completed",
    "all_objects_placed_rate",
]


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary: list[dict[str, Any]] = []
    for method, method_rows in sorted(grouped.items()):
        runs = len(method_rows)
        summary.append(
            {
                "method": method,
                "runs": runs,
                "success_rate": mean(1.0 if row["success"] else 0.0 for row in method_rows),
                "avg_makespan": mean(float(row["makespan"]) for row in method_rows),
                "avg_total_robot_steps": mean(
                    float(row["total_robot_steps"]) for row in method_rows
                ),
                "avg_collision_event_count": average_number(
                    method_rows, "collision_event_count", fallback="collision_count"
                ),
                "avg_actual_collision_count": average_number(
                    method_rows, "actual_collision_count"
                ),
                "avg_collision_count": mean(
                    float(row["collision_count"]) for row in method_rows
                ),
                "avg_vertex_conflict_event_count": average_number(
                    method_rows,
                    "vertex_conflict_event_count",
                    fallback="vertex_conflict_count",
                ),
                "avg_edge_conflict_event_count": average_number(
                    method_rows,
                    "edge_conflict_event_count",
                    fallback="edge_conflict_count",
                ),
                "avg_audited_vertex_conflict_count": average_number(
                    method_rows, "audited_vertex_conflict_count"
                ),
                "avg_audited_edge_swap_conflict_count": average_number(
                    method_rows, "audited_edge_swap_conflict_count"
                ),
                "avg_motion_wait_count": average_number(method_rows, "motion_wait_count"),
                "avg_rule_prevented_motion_conflict_count": average_number(
                    method_rows, "rule_prevented_motion_conflict_count"
                ),
                "avg_rule_rejection_count": mean(
                    float(row["rule_rejection_count"]) for row in method_rows
                ),
                "avg_wait_count": mean(float(row["wait_count"]) for row in method_rows),
                "avg_deadlock_count": mean(
                    float(row["deadlock_count"]) for row in method_rows
                ),
                "avg_recovery_attempts": mean(
                    float(row.get("recovery_attempts", 0)) for row in method_rows
                ),
                "avg_recovery_conflict_count": mean(
                    float(row.get("recovery_conflict_count", 0)) for row in method_rows
                ),
                "avg_recovery_lock_wait_count": mean(
                    float(row.get("recovery_lock_wait_count", 0)) for row in method_rows
                ),
                "avg_recovery_wait_time": mean(
                    float(row.get("recovery_wait_time", 0)) for row in method_rows
                ),
                "avg_successful_recoveries": mean(
                    float(row.get("successful_recoveries", 0)) for row in method_rows
                ),
                "avg_failed_recoveries": mean(
                    float(row.get("failed_recoveries", 0)) for row in method_rows
                ),
                "avg_resource_conflict_count": mean(
                    float(row.get("resource_conflict_count", 0)) for row in method_rows
                ),
                "avg_resource_request_denied_count": mean(
                    float(row.get("resource_request_denied_count", 0))
                    for row in method_rows
                ),
                "avg_rule_prevented_resource_conflict_count": mean(
                    float(row.get("rule_prevented_resource_conflict_count", 0))
                    for row in method_rows
                ),
                "avg_object_lock_denied_count": average_number(
                    method_rows, "object_lock_denied_count"
                ),
                "avg_pickup_zone_denied_count": average_number(
                    method_rows, "pickup_zone_denied_count"
                ),
                "avg_drop_zone_denied_count": average_number(
                    method_rows, "drop_zone_denied_count"
                ),
                "avg_recovery_zone_denied_count": average_number(
                    method_rows, "recovery_zone_denied_count"
                ),
                "avg_approach_zone_wait_count": average_number(
                    method_rows, "approach_zone_wait_count"
                ),
                "avg_object_conflict_count": mean(
                    float(row.get("object_conflict_count", 0)) for row in method_rows
                ),
                "avg_pickup_zone_conflict_count": mean(
                    float(row.get("pickup_zone_conflict_count", 0)) for row in method_rows
                ),
                "avg_drop_zone_conflict_count": mean(
                    float(row.get("drop_zone_conflict_count", 0)) for row in method_rows
                ),
                "avg_duplicate_pick_attempt_count": mean(
                    float(row.get("duplicate_pick_attempt_count", 0)) for row in method_rows
                ),
                "avg_pick_success_count": mean(
                    float(row.get("pick_success_count", 0)) for row in method_rows
                ),
                "avg_place_success_count": mean(
                    float(row.get("place_success_count", 0)) for row in method_rows
                ),
                "avg_lock_wait_count": mean(
                    float(row.get("lock_wait_count", 0)) for row in method_rows
                ),
                "avg_reassignment_count": mean(
                    float(row.get("reassignment_count", 0)) for row in method_rows
                ),
                "avg_tasks_completed": mean(
                    float(row.get("tasks_completed", 0)) for row in method_rows
                ),
                "all_objects_placed_rate": mean(
                    1.0 if row.get("all_objects_placed", False) else 0.0
                    for row in method_rows
                ),
            }
        )
    return summary


def average_number(
    rows: list[dict[str, Any]], key: str, fallback: str | None = None
) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""} and fallback is not None:
            value = row.get(fallback)
        values.append(float(value or 0))
    return mean(values)


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_summary_md(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Summary",
        "",
        "| method | runs | success_rate | avg_makespan | avg_total_robot_steps | avg_collision_event_count | avg_actual_collision_count | avg_collision_count | avg_motion_wait_count | avg_rule_prevented_motion_conflict_count | avg_resource_conflict_count | avg_resource_request_denied_count | avg_lock_wait_count | avg_rule_prevented_resource_conflict_count | avg_object_lock_denied_count | avg_pickup_zone_denied_count | avg_drop_zone_denied_count | avg_recovery_zone_denied_count | avg_approach_zone_wait_count | all_objects_placed_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {runs} | {success_rate:.3f} | {avg_makespan:.3f} | "
            "{avg_total_robot_steps:.3f} | {avg_collision_event_count:.3f} | "
            "{avg_actual_collision_count:.3f} | {avg_collision_count:.3f} | "
            "{avg_motion_wait_count:.3f} | "
            "{avg_rule_prevented_motion_conflict_count:.3f} | "
            "{avg_resource_conflict_count:.3f} | "
            "{avg_resource_request_denied_count:.3f} | {avg_lock_wait_count:.3f} | "
            "{avg_rule_prevented_resource_conflict_count:.3f} | "
            "{avg_object_lock_denied_count:.3f} | "
            "{avg_pickup_zone_denied_count:.3f} | "
            "{avg_drop_zone_denied_count:.3f} | "
            "{avg_recovery_zone_denied_count:.3f} | "
            "{avg_approach_zone_wait_count:.3f} | "
            "{all_objects_placed_rate:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
