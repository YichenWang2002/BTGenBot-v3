"""Prompt construction for multi-robot BT generation and repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import yaml

from src.env.scenario_loader import load_scenario
from src.llm.methods import CENTRALIZED_RULE_LLM_METHODS
from src.llm.schema import ALLOWED_ACTION_IDS, OUTPUT_SCHEMA_EXAMPLE


FORBIDDEN_XML_TAGS = [
    "RecoveryNode",
    "ReactiveFallback",
    "RetryUntilSuccessful",
    "RetryUntilSuccesful",
    "Decorator",
    "SubTree",
    "Parallel",
    "RateController",
    "SequenceStar",
    "PipelineSequence",
]


def build_prompt(
    scenario_path: str | Path,
    method: str,
    source_sample: dict[str, Any] | None = None,
    previous_failure_report: dict[str, Any] | None = None,
    previous_robot_trees: dict[str, str] | None = None,
) -> str:
    scenario = load_scenario(scenario_path)
    raw_scenario = load_yaml(Path(scenario_path))
    context = {
        "scenario_name": scenario.name,
        "method": method,
        "active_robots": sorted(scenario.state.robots),
        "task_type": (raw_scenario.get("task") or {}).get("type"),
        "task_ids": task_ids_from_scenario(raw_scenario),
        "source_dataset_index": (raw_scenario.get("source") or {}).get("dataset_index"),
        "scenario": raw_scenario,
        "source_sample": source_sample or {},
    }
    lines = [
        "You generate minimal executable multi-robot Behavior Tree XML for this project.",
        "Return JSON only. Do not use markdown, code fences, comments, or prose outside JSON.",
        "Do not modify the original dataset or scenario YAML. Do not bypass the centralized rule layer.",
        "",
        "Task:",
        f"- Method: {method}",
        "- Produce one robot-level BT XML string for every active robot.",
        "- Produce assignment, coordination_notes, and assumptions.",
        "",
        "Available action primitives:",
        ", ".join(sorted(ALLOWED_ACTION_IDS)),
        "",
        "Centralized rule layer:",
        "- Movement must go through the simulator and centralized rule manager when enabled.",
        "- Object, pickup_zone, drop_zone, and recovery_zone use resource locks.",
        "- Use RequestResource before PickObject, PlaceObject, ClearCostmap, or Spin when relevant.",
    ]
    if scenario.task_dag is not None:
        lines.extend(dag_task_table_lines(scenario))
    lines.extend(
        [
            "",
            "Output JSON schema example:",
            json.dumps(OUTPUT_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2),
            "",
            "Scenario context JSON:",
            json.dumps(context, ensure_ascii=False, indent=2),
        ]
    )
    if method in CENTRALIZED_RULE_LLM_METHODS:
        lines.extend(centralized_xml_schema_lines(raw_scenario))
    if scenario.task_dag is not None and previous_failure_report is not None:
        lines.extend(
            [
                "",
                "DAG repair instructions:",
                "Add WaitForDependency before any blocked task action.",
                "Use only valid task_id values from the DAG.",
                "Respect assigned_robot for every task action.",
                "Preserve pick -> place ordering for each object.",
                "Do not run final_check before all required place tasks are completed.",
            ]
        )
    if previous_failure_report is not None:
        lines.extend(
            [
                "",
                "Repair iteration:",
                "Use this failure_report to repair the previous result.",
                "Only modify assignment and robot_trees unless the failure report explicitly requires coordination_notes or assumptions.",
                "failure_report:",
                json.dumps(previous_failure_report, ensure_ascii=False, indent=2),
            ]
        )
        if requires_recovery_resource_fix(previous_failure_report):
            lines.extend(
                [
                    "",
                    "Mandatory recovery resource repair:",
                    'Replace any resource="recovery_zone" with resource_id="recovery_zone_0" resource_type="recovery_zone".',
                    'Use exactly: <Action ID="RequestResource" resource_id="recovery_zone_0" resource_type="recovery_zone"/>',
                    'Release with exactly: <Action ID="ReleaseResource" resource_id="recovery_zone_0"/>',
                ]
            )
    if previous_robot_trees is not None:
        lines.extend(
            [
                "",
                "Previous robot_trees:",
                json.dumps(previous_robot_trees, ensure_ascii=False, indent=2),
            ]
        )
    return "\n".join(lines) + "\n"


def dag_task_table_lines(scenario: Any) -> list[str]:
    dag = getattr(scenario, "task_dag", None)
    if dag is None:
        return []
    lines = [
        "",
        "DAG Task Table:",
        "| task_id | task_type | object_id | assigned_robot | predecessors | expected_action_id |",
        "|---|---|---|---|---|---|",
    ]
    try:
        ordered_task_ids = dag.topological_sort()
    except Exception:
        ordered_task_ids = sorted(dag.tasks)
    for task_id in ordered_task_ids:
        task = dag.tasks[task_id]
        predecessors = ", ".join(
            predecessor.task_id for predecessor in dag.get_predecessors(task_id)
        ) or "-"
        lines.append(
            "| {task_id} | {task_type} | {object_id} | {assigned_robot} | {predecessors} | {expected_action_id} |".format(
                task_id=task.task_id,
                task_type=task.task_type,
                object_id=task.object_id or "-",
                assigned_robot=task.assigned_robot or "-",
                predecessors=predecessors,
                expected_action_id=task.expected_action_id or "-",
            )
        )
    lines.extend(
        [
            "",
            "DAG task rules:",
            "1. Do not execute a task before all predecessor tasks are completed.",
            "2. If a task has predecessors, insert a dependency check before the task action.",
            "3. Use only valid task_id values from the DAG.",
            "4. For pick/place tasks, preserve pick -> place dependencies.",
            "5. final_check must only run after all required place tasks are completed.",
            "6. If a task is not ready, use WaitForDependency.",
            "",
            "DAG XML examples:",
            '<Action ID="WaitForDependency" task_id="place_obj_0"/>',
            '<Condition ID="IsTaskReady" task_id="place_obj_0"/>',
            '<Action ID="PickObject" object_id="obj_0" task_id="pick_obj_0"/>',
            '<Action ID="PlaceObject" object_id="obj_0" task_id="place_obj_0"/>',
            '<Action ID="FinalCheck" task_id="final_check"/>',
        ]
    )
    return lines


def centralized_xml_schema_lines(raw_scenario: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "Strict XML schema for centralized_rule_multi_bt:",
        "- Allowed XML tags only: root, BehaviorTree, Sequence, Fallback, ReactiveSequence, Action, Condition.",
        "- Forbidden XML tags: " + ", ".join(FORBIDDEN_XML_TAGS) + ".",
        "- Do not use a generic resource attribute on RequestResource.",
        "Valid RequestResource examples:",
        '<Action ID="RequestResource" resource_id="recovery_zone_0" resource_type="recovery_zone"/>',
        '<Action ID="RequestResource" resource_id="obj_0" resource_type="object"/>',
        '<Action ID="RequestResource" resource_id="drop_zone_0" resource_type="drop_zone"/>',
        '<Action ID="RequestResource" resource_id="pickup_zone_0" resource_type="pickup_zone"/>',
        "Valid ReleaseResource examples:",
        '<Action ID="ReleaseResource" resource_id="recovery_zone_0"/>',
        '<Action ID="ReleaseResource" resource_id="obj_0"/>',
        '<Action ID="ReleaseResource" resource_id="drop_zone_0"/>',
    ]
    task_type = (raw_scenario.get("task") or {}).get("type")
    if task_type == "navigation_recovery" or raw_scenario.get("recovery", {}).get("enabled"):
        lines.extend(
            [
                "",
                "Recovery BT XML example:",
                '<root main_tree_to_execute="MainTree">',
                '  <BehaviorTree ID="MainTree">',
                "    <Sequence>",
                '      <Action ID="RequestResource" resource_id="recovery_zone_0" resource_type="recovery_zone"/>',
                '      <Action ID="ClearCostmap" scope="local"/>',
                '      <Action ID="Spin"/>',
                '      <Action ID="Wait" duration="1"/>',
                '      <Action ID="ReleaseResource" resource_id="recovery_zone_0"/>',
                '      <Action ID="ReportStatus" status="done"/>',
                "    </Sequence>",
                "  </BehaviorTree>",
                "</root>",
            ]
        )
    return lines


def requires_recovery_resource_fix(failure_report: dict[str, Any]) -> bool:
    text = json.dumps(failure_report, ensure_ascii=False)
    return (
        "recovery action lacks recovery_zone RequestResource" in text
        or 'resource="recovery_zone"' in text
        or "resource='recovery_zone'" in text
        or "Invalid RequestResource format" in text
    )


def task_ids_from_scenario(raw_scenario: dict[str, Any]) -> list[str]:
    task_type = (raw_scenario.get("task") or {}).get("type", "")
    if task_type == "navigation_pickplace" or raw_scenario.get("pickplace", {}).get("enabled"):
        return [
            str(task["task_id"])
            for task in raw_scenario.get("pickplace", {}).get("tasks", [])
            if "task_id" in task
        ]
    return [str(item["id"]) for item in raw_scenario.get("waypoints", []) if "id" in item]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
