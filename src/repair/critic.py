"""Static validation for LLM-generated multi-robot BT plans."""

from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET

from src.llm.schema import (
    ValidationResult,
    scenario_dag_task_ids,
    scenario_robot_ids,
    scenario_task_ids,
    validate_bt_xml,
)


DAG_DEPENDENCY_GUARD_ACTIONS = {"WaitForDependency"}
DAG_DEPENDENCY_GUARD_CONDITIONS = {"IsTaskReady"}
DAG_TASK_ACTION_IDS = {"PickObject", "PlaceObject", "FinalCheck", "Assembly"}


def validate_assignment(scenario: Any, assignment: list[dict[str, Any]]) -> ValidationResult:
    errors: list[str] = []
    robots = set(scenario_robot_ids(scenario))
    tasks = scenario_task_ids(scenario)
    assigned: list[str] = []
    for index, item in enumerate(assignment):
        robot_id = item.get("robot_id")
        if robot_id not in robots:
            errors.append(f"unknown robot in assignment[{index}]: {robot_id}")
        task_ids = item.get("task_ids")
        if not isinstance(task_ids, list):
            errors.append(f"assignment[{index}].task_ids must be a list")
            continue
        for task_id in task_ids:
            if task_id not in tasks:
                errors.append(f"unknown task id in assignment[{index}]: {task_id}")
            assigned.append(str(task_id))
    duplicates = sorted({task_id for task_id in assigned if assigned.count(task_id) > 1})
    if duplicates:
        errors.append(f"duplicate task assignment: {duplicates}")
    missing = sorted(tasks - set(assigned))
    if missing:
        errors.append(f"unassigned tasks: {missing}")
    return ValidationResult(not errors, errors)


def validate_robot_trees(robot_trees: dict[str, str]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    for robot_id, xml_text in robot_trees.items():
        result = validate_bt_xml(xml_text)
        errors.extend(f"{robot_id}: {error}" for error in result.errors)
        warnings.extend(f"{robot_id}: {warning}" for warning in result.warnings)
    return ValidationResult(not errors, errors, warnings)


def validate_required_resources(robot_trees: dict[str, str], scenario: Any) -> ValidationResult:
    errors: list[str] = []
    invalid_requests = collect_invalid_resource_requests(robot_trees)
    errors.extend(item["message"] for item in invalid_requests)
    if scenario.pickplace.get("enabled"):
        for robot_id, xml_text in robot_trees.items():
            actions = extract_actions(xml_text)
            for action in actions:
                if action.get("ID") in {"PickObject", "PlaceObject"}:
                    object_id = action.get("object_id")
                    if object_id and not has_prior_resource_action(actions, object_id, action):
                        errors.append(
                            f"{robot_id}: {action.get('ID')} for {object_id} lacks RequestResource"
                        )
    if scenario.recovery.get("enabled"):
        for robot_id, xml_text in robot_trees.items():
            actions = extract_actions(xml_text)
            recovery_actions = [a for a in actions if a.get("ID") in {"ClearCostmap", "Spin"}]
            has_request = any(
                a.get("ID") == "RequestResource" and a.get("resource_type") == "recovery_zone"
                for a in actions
            )
            if recovery_actions and not has_request:
                errors.append(f"{robot_id}: recovery action lacks recovery_zone RequestResource")
    return ValidationResult(not errors, errors)


def validate_dag_robot_trees(robot_trees: dict[str, str], scenario: Any) -> dict[str, Any]:
    dag = getattr(scenario, "task_dag", None)
    if dag is None:
        return {
            "valid": True,
            "errors": [],
            "dag_static_error_count": 0,
            "dag_static_error_examples": [],
            "missing_dependency_wait_count": 0,
            "invalid_task_id_count": 0,
            "wrong_assigned_robot_for_task_count": 0,
            "dependency_not_satisfied_count": 0,
            "blocked_task_executed_count": 0,
            "final_check_not_ready_count": 0,
        }

    dag_task_ids = scenario_dag_task_ids(scenario)
    errors: list[str] = []
    examples: list[str] = []
    missing_dependency_wait_count = 0
    invalid_task_id_count = 0
    wrong_assigned_robot_for_task_count = 0
    dependency_not_satisfied_count = 0
    blocked_task_executed_count = 0
    final_check_not_ready_count = 0

    for robot_id, xml_text in robot_trees.items():
        dependency_guards: set[str] = set()
        for node in extract_behavior_nodes(xml_text):
            tag = node["tag"]
            attrib = node["attrib"]
            if tag == "Condition" and attrib.get("ID") in DAG_DEPENDENCY_GUARD_CONDITIONS:
                guard_task_id = str(attrib.get("task_id") or "")
                if not guard_task_id or guard_task_id not in dag_task_ids:
                    invalid_task_id_count += 1
                    message = f"{robot_id}: invalid DAG task_id in condition: {guard_task_id or '<missing>'}"
                    errors.append(message)
                    examples.append(message)
                    continue
                dependency_guards.add(guard_task_id)
                continue
            if tag != "Action":
                continue
            action_id = str(attrib.get("ID") or "")
            task_id = str(attrib.get("task_id") or "")
            if action_id in DAG_DEPENDENCY_GUARD_ACTIONS:
                if not task_id or task_id not in dag_task_ids:
                    invalid_task_id_count += 1
                    message = (
                        f"{robot_id}: invalid DAG task_id in WaitForDependency: "
                        f"{task_id or '<missing>'}"
                    )
                    errors.append(message)
                    examples.append(message)
                    continue
                dependency_guards.add(task_id)
                continue

            if action_id not in DAG_TASK_ACTION_IDS:
                continue
            if not task_id or task_id not in dag.tasks:
                invalid_task_id_count += 1
                message = f"{robot_id}: invalid DAG task_id for {action_id}: {task_id or '<missing>'}"
                errors.append(message)
                examples.append(message)
                continue

            task = dag.tasks[task_id]
            if task.assigned_robot and task.assigned_robot != robot_id:
                wrong_assigned_robot_for_task_count += 1
                message = (
                    f"{robot_id}: task {task_id} assigned to {task.assigned_robot} "
                    f"but executed in this robot tree"
                )
                errors.append(message)
                examples.append(message)

            if task.task_type == "place" and not place_depends_on_pick(dag, task_id):
                dependency_not_satisfied_count += 1
                message = f"{robot_id}: place task {task_id} is missing a pick dependency"
                errors.append(message)
                examples.append(message)

            predecessors = dag.get_predecessors(task_id)
            if predecessors and task_id not in dependency_guards:
                missing_dependency_wait_count += 1
                blocked_task_executed_count += 1
                dependency_not_satisfied_count += 1
                message = (
                    f"{robot_id}: blocked task {task_id} executed without dependency check"
                )
                errors.append(message)
                examples.append(message)

            if task.task_type in {"final_check", "assembly"} and predecessors and task_id not in dependency_guards:
                final_check_not_ready_count += 1
                message = f"{robot_id}: {task.task_type} task {task_id} is not ready"
                errors.append(message)
                examples.append(message)

            if action_id in {"PickObject", "PlaceObject"}:
                object_id = str(attrib.get("object_id") or "")
                if task.object_id and object_id != task.object_id:
                    invalid_task_id_count += 1
                    message = (
                        f"{robot_id}: task {task_id} object_id {object_id or '<missing>'} "
                        f"does not match DAG object_id {task.object_id}"
                    )
                    errors.append(message)
                    examples.append(message)

    return {
        "valid": not errors,
        "errors": errors,
        "dag_static_error_count": len(errors),
        "dag_static_error_examples": examples[:5],
        "missing_dependency_wait_count": missing_dependency_wait_count,
        "invalid_task_id_count": invalid_task_id_count,
        "wrong_assigned_robot_for_task_count": wrong_assigned_robot_for_task_count,
        "dependency_not_satisfied_count": dependency_not_satisfied_count,
        "blocked_task_executed_count": blocked_task_executed_count,
        "final_check_not_ready_count": final_check_not_ready_count,
    }


def detect_static_conflicts(assignment: list[dict[str, Any]]) -> ValidationResult:
    task_to_robots: dict[str, list[str]] = {}
    for item in assignment:
        robot_id = str(item.get("robot_id"))
        for task_id in item.get("task_ids") or []:
            task_to_robots.setdefault(str(task_id), []).append(robot_id)
    duplicates = {
        task_id: sorted(set(robots))
        for task_id, robots in task_to_robots.items()
        if len(set(robots)) > 1 or len(robots) > 1
    }
    if not duplicates:
        return ValidationResult(True)
    return ValidationResult(False, [f"duplicate_task: {duplicates}"])


def static_validate(scenario: Any, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "valid": False,
            "xml_valid": False,
            "assignment_valid": False,
            "resource_valid": False,
            "invalid_resource_request_count": 0,
            "invalid_resource_request_examples": [],
            "dag_static_error_count": 0,
            "dag_static_error_examples": [],
            "missing_dependency_wait_count": 0,
            "invalid_task_id_count": 0,
            "wrong_assigned_robot_for_task_count": 0,
            "dependency_not_satisfied_count": 0,
            "blocked_task_executed_count": 0,
            "final_check_not_ready_count": 0,
            "suggested_xml_fix": "",
            "errors": ["no parsed payload"],
        }
    assignment_result = validate_assignment(scenario, payload.get("assignment") or [])
    tree_result = validate_robot_trees(payload.get("robot_trees") or {})
    resource_result = validate_required_resources(payload.get("robot_trees") or {}, scenario)
    dag_result = validate_dag_robot_trees(payload.get("robot_trees") or {}, scenario)
    conflict_result = detect_static_conflicts(payload.get("assignment") or [])
    invalid_requests = collect_invalid_resource_requests(payload.get("robot_trees") or {})
    errors = (
        assignment_result.errors
        + tree_result.errors
        + resource_result.errors
        + dag_result.get("errors", [])
        + conflict_result.errors
    )
    return {
        "valid": not errors,
        "xml_valid": tree_result.valid,
        "assignment_valid": assignment_result.valid and conflict_result.valid,
        "resource_valid": resource_result.valid,
        "invalid_resource_request_count": len(invalid_requests),
        "invalid_resource_request_examples": invalid_requests,
        "dag_static_error_count": int(dag_result.get("dag_static_error_count", 0) or 0),
        "dag_static_error_examples": list(dag_result.get("dag_static_error_examples") or []),
        "missing_dependency_wait_count": int(
            dag_result.get("missing_dependency_wait_count", 0) or 0
        ),
        "invalid_task_id_count": int(dag_result.get("invalid_task_id_count", 0) or 0),
        "wrong_assigned_robot_for_task_count": int(
            dag_result.get("wrong_assigned_robot_for_task_count", 0) or 0
        ),
        "dependency_not_satisfied_count": int(
            dag_result.get("dependency_not_satisfied_count", 0) or 0
        ),
        "blocked_task_executed_count": int(
            dag_result.get("blocked_task_executed_count", 0) or 0
        ),
        "final_check_not_ready_count": int(
            dag_result.get("final_check_not_ready_count", 0) or 0
        ),
        "suggested_xml_fix": suggested_xml_fix(invalid_requests),
        "errors": errors,
        "warnings": tree_result.warnings,
    }


def extract_actions(xml_text: str) -> list[dict[str, str]]:
    return [
        node["attrib"]
        for node in extract_behavior_nodes(xml_text)
        if node["tag"] == "Action"
    ]


def extract_behavior_nodes(xml_text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    return [{"tag": element.tag, "attrib": dict(element.attrib)} for element in root.iter()]


def has_prior_resource_action(
    actions: list[dict[str, str]], object_id: str, target_action: dict[str, str]
) -> bool:
    for action in actions:
        if action is target_action:
            break
        if action.get("ID") == "RequestResource" and action.get("resource_id") == object_id:
            return True
    return False


def collect_invalid_resource_requests(
    robot_trees: dict[str, str]
) -> list[dict[str, str]]:
    invalid: list[dict[str, str]] = []
    for robot_id, xml_text in robot_trees.items():
        for action in extract_actions(xml_text):
            if action.get("ID") != "RequestResource":
                continue
            if "resource" in action:
                resource = action.get("resource", "")
                resource_type = infer_resource_type(resource)
                invalid.append(
                    {
                        "robot_id": robot_id,
                        "resource": resource,
                        "message": invalid_resource_message(resource, resource_type),
                        "suggested_xml": resource_request_example(resource_type),
                    }
                )
            elif not action.get("resource_id") or not action.get("resource_type"):
                resource_id = action.get("resource_id", "")
                resource_type = infer_resource_type(resource_id or action.get("resource_type", ""))
                invalid.append(
                    {
                        "robot_id": robot_id,
                        "resource": resource_id,
                        "message": (
                            "Invalid RequestResource format: include both resource_id "
                            "and resource_type attributes."
                        ),
                        "suggested_xml": resource_request_example(resource_type),
                    }
                )
    return invalid


def infer_resource_type(resource: str) -> str:
    if resource == "recovery_zone" or resource.startswith("recovery_zone"):
        return "recovery_zone"
    if resource.startswith("drop_zone"):
        return "drop_zone"
    if resource.startswith("pickup_zone"):
        return "pickup_zone"
    if resource.startswith("obj_") or resource.startswith("object"):
        return "object"
    return "object"


def invalid_resource_message(resource: str, resource_type: str) -> str:
    if resource_type == "recovery_zone":
        return (
            "Invalid RequestResource format: use resource_id='recovery_zone_0' "
            "and resource_type='recovery_zone', not resource='recovery_zone'."
        )
    if resource_type == "drop_zone":
        return (
            "Invalid RequestResource format: use resource_id='drop_zone_0' "
            "and resource_type='drop_zone', not resource='drop_zone'."
        )
    if resource_type == "pickup_zone":
        return (
            "Invalid RequestResource format: use resource_id='pickup_zone_0' "
            "and resource_type='pickup_zone', not resource='pickup_zone'."
        )
    example_id = resource if resource.startswith("obj_") else "obj_0"
    return (
        f"Invalid RequestResource format: use resource_id='{example_id}' "
        "and resource_type='object', not a generic resource attribute."
    )


def resource_request_example(resource_type: str) -> str:
    examples = {
        "recovery_zone": '<Action ID="RequestResource" resource_id="recovery_zone_0" resource_type="recovery_zone"/>',
        "object": '<Action ID="RequestResource" resource_id="obj_0" resource_type="object"/>',
        "pickup_zone": '<Action ID="RequestResource" resource_id="pickup_zone_0" resource_type="pickup_zone"/>',
        "drop_zone": '<Action ID="RequestResource" resource_id="drop_zone_0" resource_type="drop_zone"/>',
    }
    return examples.get(resource_type, examples["object"])


def suggested_xml_fix(invalid_requests: list[dict[str, str]]) -> str:
    if not invalid_requests:
        return ""
    unique = []
    for item in invalid_requests:
        suggested = item["suggested_xml"]
        if suggested not in unique:
            unique.append(suggested)
    return "Replace invalid RequestResource actions with: " + " ".join(unique)


def place_depends_on_pick(dag: Any, place_task_id: str) -> bool:
    task = dag.tasks.get(place_task_id)
    if task is None or task.task_type != "place" or not task.object_id:
        return True
    return any(
        predecessor.task_type == "pick" and predecessor.object_id == task.object_id
        for predecessor in dag.get_predecessors(place_task_id)
    )
