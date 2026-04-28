"""Schema and XML validation for LLM multi-robot BT output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import xml.etree.ElementTree as ET


ALLOWED_XML_TAGS = {
    "root",
    "BehaviorTree",
    "Sequence",
    "Fallback",
    "ReactiveSequence",
    "Action",
    "Condition",
}

ALLOWED_ACTION_IDS = {
    "RequestMove",
    "NavigateToWaypoint",
    "NavigateToPickup",
    "PickObject",
    "NavigateToDrop",
    "PlaceObject",
    "RequestResource",
    "ReleaseResource",
    "ClearCostmap",
    "Spin",
    "Wait",
    "WaitForDependency",
    "Backup",
    "FinalCheck",
    "Assembly",
    "ReportStatus",
}

OUTPUT_SCHEMA_EXAMPLE = {
    "robot_trees": {
        "robot_0": '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence><Action ID="NavigateToWaypoint" waypoint_id="wp_0"/></Sequence></BehaviorTree></root>'
    },
    "assignment": [
        {
            "robot_id": "robot_0",
            "task_ids": ["wp_0"],
            "reason": "nearest robot",
        }
    ],
    "coordination_notes": ["Use centralized rule layer for collisions and locks."],
    "assumptions": ["Grid coordinates are discrete."],
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def scenario_robot_ids(scenario: Any) -> list[str]:
    return sorted(scenario.state.robots)


def scenario_task_ids(scenario: Any) -> set[str]:
    waypoint_ids = set(scenario.state.waypoints)
    pickplace_ids = {
        str(task["task_id"])
        for task in scenario.pickplace.get("tasks", [])
        if "task_id" in task
    }
    return waypoint_ids | pickplace_ids


def scenario_dag_task_ids(scenario: Any) -> set[str]:
    task_dag = getattr(scenario, "task_dag", None)
    if task_dag is None:
        return set()
    return set(task_dag.tasks)


def validate_llm_output(payload: Any, scenario: Any) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ValidationResult(False, ["LLM output must be a JSON object"])

    robot_ids = scenario_robot_ids(scenario)
    task_ids = scenario_task_ids(scenario)
    robot_trees = payload.get("robot_trees")
    if not isinstance(robot_trees, dict):
        errors.append("robot_trees must be an object")
    else:
        missing = [robot_id for robot_id in robot_ids if robot_id not in robot_trees]
        extra = [robot_id for robot_id in robot_trees if robot_id not in robot_ids]
        if missing:
            errors.append(f"robot_trees missing active robots: {missing}")
        if extra:
            errors.append(f"robot_trees contains unknown robots: {extra}")
        for robot_id, xml_text in robot_trees.items():
            if not isinstance(xml_text, str):
                errors.append(f"robot tree for {robot_id} must be an XML string")
                continue
            xml_result = validate_bt_xml(xml_text)
            errors.extend(f"{robot_id}: {error}" for error in xml_result.errors)
            warnings.extend(f"{robot_id}: {warning}" for warning in xml_result.warnings)

    assignment = payload.get("assignment")
    if not isinstance(assignment, list):
        errors.append("assignment must be a list")
    else:
        for index, item in enumerate(assignment):
            if not isinstance(item, dict):
                errors.append(f"assignment[{index}] must be an object")
                continue
            robot_id = item.get("robot_id")
            if robot_id not in robot_ids:
                errors.append(f"assignment[{index}] has unknown robot_id: {robot_id}")
            task_list = item.get("task_ids")
            if not isinstance(task_list, list):
                errors.append(f"assignment[{index}].task_ids must be a list")
                continue
            for task_id in task_list:
                if task_id not in task_ids:
                    errors.append(f"assignment[{index}] has unknown task_id: {task_id}")

    for key in ("coordination_notes", "assumptions"):
        if not isinstance(payload.get(key), list):
            errors.append(f"{key} must be a list")
        elif not all(isinstance(value, str) for value in payload[key]):
            errors.append(f"{key} must contain only strings")

    return ValidationResult(not errors, errors, warnings)


def validate_bt_xml(xml_text: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return ValidationResult(False, [f"invalid XML: {exc}"])
    if root.tag != "root":
        errors.append("BT XML root tag must be <root>")
    for element in root.iter():
        if element.tag not in ALLOWED_XML_TAGS:
            errors.append(f"disallowed XML tag: {element.tag}")
        if element.tag == "Action":
            action_id = element.attrib.get("ID")
            if action_id not in ALLOWED_ACTION_IDS:
                errors.append(f"disallowed Action ID: {action_id}")
        if element.tag == "Condition":
            warnings.append("Condition nodes are accepted but not executed in v1")
    return ValidationResult(not errors, errors, warnings)
