"""Load YAML scenarios for the multi-robot grid environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.dag.dag_validator import validate_task_dag
from src.dag.task_dag import TaskDAG
from src.env.multi_robot_state import (
    MultiRobotState,
    ObjectState,
    Position,
    RobotState,
    normalize_position,
    position_key,
)


@dataclass
class Scenario:
    name: str
    state: MultiRobotState
    max_steps: int
    render: bool = False
    centralized_rule: bool = False
    recovery: dict[str, Any] = field(default_factory=dict)
    pickplace: dict[str, Any] = field(default_factory=dict)
    zones: dict[str, Any] = field(default_factory=dict)
    coordination: dict[str, Any] = field(default_factory=dict)
    task_dag: TaskDAG | None = None
    cell_size: int = 48
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def num_robots(self) -> int:
        return len(self.state.robots)


def load_scenario(path: str | Path) -> Scenario:
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return load_scenario_data(data, default_name=scenario_path.stem)


def load_scenario_data(data: dict[str, Any], default_name: str = "scenario") -> Scenario:
    if not isinstance(data, dict):
        raise ValueError("Scenario must be a mapping")

    map_data = data.get("map") or {}
    width = int(map_data["width"])
    height = int(map_data["height"])
    obstacles = {
        position_key(normalize_position(position))
        for position in map_data.get("obstacles", [])
    }

    waypoints = load_waypoints(data.get("waypoints") or [])
    task_data = data.get("task") or {}
    assignments = task_data.get("assignments") or {}
    pickplace_data = dict(data.get("pickplace") or {})
    waypoint_assignments = {} if pickplace_data.get("enabled") else assignments
    robots = load_robots(data.get("robots") or {}, waypoint_assignments)
    if pickplace_data.get("enabled"):
        for robot_id, robot in robots.items():
            robot.assigned_tasks = list(assignments.get(robot_id, []))
    objects = load_objects(data.get("objects") or {})
    task_dag = load_task_dag(data)
    state = MultiRobotState(
        timestep=0,
        grid_width=width,
        grid_height=height,
        obstacles=obstacles,
        robots=robots,
        objects=objects,
        waypoints=waypoints,
    )

    return Scenario(
        name=str(data.get("name") or default_name),
        state=state,
        max_steps=int(data.get("max_steps", 100)),
        render=bool(data.get("render", False)),
        centralized_rule=bool(data.get("centralized_rule", False)),
        recovery=dict(data.get("recovery") or {}),
        pickplace=pickplace_data,
        zones=dict(data.get("zones") or {}),
        coordination=dict(data.get("coordination") or {}),
        task_dag=task_dag,
        cell_size=int(data.get("cell_size", 48)),
        raw=data,
    )


def load_task_dag(data: dict[str, Any]) -> TaskDAG | None:
    if not data.get("task_dag"):
        return None
    task_dag = TaskDAG.from_dict(data)
    result = validate_task_dag(task_dag, scenario=data)
    result.raise_for_errors()
    return task_dag


def load_robots(
    raw_robots: dict[str, Any], assignments: dict[str, list[str]] | None = None
) -> dict[str, RobotState]:
    if not raw_robots:
        raise ValueError("Scenario requires at least one robot")
    robots: dict[str, RobotState] = {}
    assignments = assignments or {}
    for robot_id, robot_data in raw_robots.items():
        start: Position = normalize_position(robot_data["start"])
        raw_goal = robot_data.get("goal")
        goal: Position | None = normalize_position(raw_goal) if raw_goal is not None else None
        robots[str(robot_id)] = RobotState(
            robot_id=str(robot_id),
            position=start,
            goal=goal,
            status=str(robot_data.get("status", "running")),
            path=[list(start)],
            carrying_object=robot_data.get("carrying_object"),
            start=start,
            assigned_waypoints=list(
                robot_data.get("assigned_waypoints")
                or assignments.get(str(robot_id), [])
            ),
            assigned_tasks=list(robot_data.get("assigned_tasks") or []),
        )
    return robots


def load_waypoints(raw_waypoints: list[dict[str, Any]]) -> dict[str, Position]:
    waypoints: dict[str, Position] = {}
    for item in raw_waypoints:
        waypoint_id = str(item["id"])
        waypoints[waypoint_id] = normalize_position(item["position"])
    return waypoints


def load_objects(raw_objects: dict[str, Any]) -> dict[str, ObjectState]:
    objects: dict[str, ObjectState] = {}
    for object_id, object_data in raw_objects.items():
        objects[str(object_id)] = ObjectState(
            object_id=str(object_id),
            position=object_data.get("position"),
            held_by=object_data.get("held_by"),
            target_position=object_data.get("target_position"),
            status=str(object_data.get("status", "available")),
        )
    return objects
