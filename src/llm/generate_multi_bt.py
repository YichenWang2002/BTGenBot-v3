"""Utilities for turning LLM BT output into simulator runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json
import os
import xml.etree.ElementTree as ET

import yaml

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data
from src.llm.backends.base import BaseLLMBackend
from src.llm.backends.mock import MockLLMBackend
from src.llm.backends.openai_compatible import OpenAICompatibleBackend
from src.llm.methods import (
    APPROACH_RADIUS_1_COORDINATION,
    APPROACH_RADIUS_1_METHOD,
    CENTRALIZED_RULE_LLM_METHODS,
)


def build_backend(name: str) -> BaseLLMBackend:
    if name == "mock":
        return MockLLMBackend()
    if name in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleBackend()
    raise ValueError(f"Unknown LLM backend: {name}")


def extract_bt_actions(robot_trees: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    actions_by_robot: dict[str, list[dict[str, str]]] = {}
    for robot_id, xml_text in robot_trees.items():
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            actions_by_robot[robot_id] = []
            continue
        actions_by_robot[robot_id] = [
            dict(element.attrib) for element in root.iter() if element.tag == "Action"
        ]
    return actions_by_robot


def run_generated_plan(
    scenario_path: str | Path,
    payload: dict[str, Any],
    method: str,
    trace_path: str | Path | None = None,
    coordination_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    scenario_data = load_yaml(Path(scenario_path))
    injected = inject_llm_assignment(
        scenario_data, payload, method, coordination_override=coordination_override
    )
    scenario = load_scenario_data(injected, default_name=injected.get("name", "llm_scenario"))
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=scenario.centralized_rule,
        recovery_config=scenario.recovery,
        pickplace_config=scenario.pickplace,
        zones=scenario.zones,
        coordination_config=scenario.coordination,
        task_dag=scenario.task_dag,
    )
    env.reset()
    try:
        while not env.done:
            env.step()
    finally:
        env.close()
    trace_payload = env.metrics.to_trace_payload(
        scenario_name=scenario.name,
        num_robots=scenario.num_robots,
    )
    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(trace_payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return env.metrics.summary(), trace_payload


def inject_llm_assignment(
    scenario_data: dict[str, Any],
    payload: dict[str, Any],
    method: str,
    coordination_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = deepcopy(scenario_data)
    assignment_rows = payload.get("assignment") or []
    assignments = {
        str(row.get("robot_id")): list(row.get("task_ids") or [])
        for row in assignment_rows
        if row.get("robot_id") in data.get("robots", {})
    }
    for robot in data.get("robots", {}).values():
        robot["goal"] = None
        robot["assigned_waypoints"] = []
        robot["assigned_tasks"] = []
        robot["carrying_object"] = None

    task_type = (data.get("task") or {}).get("type")
    if task_type == "navigation_pickplace" or data.get("pickplace", {}).get("enabled"):
        first_owner_by_task: dict[str, str] = {}
        for robot_id, task_ids in assignments.items():
            for task_id in task_ids:
                first_owner_by_task.setdefault(str(task_id), robot_id)
        for task in data.get("pickplace", {}).get("tasks", []):
            task_id = str(task.get("task_id"))
            if task_id in first_owner_by_task:
                task["assigned_robot"] = first_owner_by_task[task_id]
                task["status"] = "assigned"
                task["attempts"] = 0
        data.setdefault("task", {})["assignments"] = assignments
    else:
        data.setdefault("task", {})["assignments"] = assignments

    data["centralized_rule"] = method in CENTRALIZED_RULE_LLM_METHODS
    if method == APPROACH_RADIUS_1_METHOD:
        coordination = dict(data.get("coordination") or {})
        coordination.update(APPROACH_RADIUS_1_COORDINATION)
        if coordination_override is not None:
            coordination.update(coordination_override)
        data["coordination"] = coordination
    elif coordination_override is not None:
        coordination = dict(data.get("coordination") or {})
        coordination.update(coordination_override)
        data["coordination"] = coordination
    data["render"] = False
    data["name"] = f"{data.get('name', 'scenario')}_{method}"
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
