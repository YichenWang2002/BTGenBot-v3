"""Action helpers for the multi-robot grid environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.env.multi_robot_state import Position, RobotState


DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "wait": (0, 0),
}


@dataclass(frozen=True)
class Action:
    robot_id: str
    action_type: str
    direction: str | None = None
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"robot_id": self.robot_id, "action_type": self.action_type}
        if self.direction is not None:
            payload["direction"] = self.direction
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        return payload


def move_one_step(position: Position, direction: str) -> Position:
    normalized = direction.lower()
    if normalized not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    dx, dy = DIRECTIONS[normalized]
    return [position[0] + dx, position[1] + dy]


def navigate_one_step(robot: RobotState) -> tuple[Position, str]:
    if robot.goal is None or robot.position == robot.goal:
        return list(robot.position), "wait"

    x, y = robot.position
    goal_x, goal_y = robot.goal
    if x < goal_x:
        return [x + 1, y], "east"
    if x > goal_x:
        return [x - 1, y], "west"
    if y < goal_y:
        return [x, y + 1], "south"
    return [x, y - 1], "north"


def wait(position: Position) -> Position:
    return list(position)


def normalize_step_actions(
    actions: Any, robot_ids: list[str], default_robot_id: str = "robot_0"
) -> dict[str, dict[str, Any]]:
    if actions is None:
        return {
            robot_id: {"type": "navigate"}
            for robot_id in robot_ids
        }

    if isinstance(actions, str):
        return {default_robot_id: {"type": actions}}

    if isinstance(actions, list):
        if len(actions) != 2:
            raise ValueError("List action shorthand must be a [x, y] position or use dict actions")
        raise ValueError("Absolute-position list actions are not supported; use a direction")

    if not isinstance(actions, dict):
        raise TypeError("actions must be None, str, list, or dict")

    if "type" in actions or "action" in actions or "direction" in actions:
        return {default_robot_id: _normalize_action_value(actions)}

    normalized: dict[str, dict[str, Any]] = {}
    for robot_id, action_value in actions.items():
        normalized[str(robot_id)] = _normalize_action_value(action_value)
    return normalized


def _normalize_action_value(action_value: Any) -> dict[str, Any]:
    if action_value is None:
        return {"type": "wait"}
    if isinstance(action_value, str):
        if action_value.lower() in DIRECTIONS and action_value.lower() != "wait":
            return {"type": "move", "direction": action_value}
        return {"type": action_value}
    if isinstance(action_value, dict):
        action_type = action_value.get("type") or action_value.get("action")
        direction = action_value.get("direction")
        if action_type is None and direction is not None:
            action_type = "move"
        if action_type is None:
            raise ValueError(f"Action missing type/action: {action_value}")
        payload = {"type": str(action_type)}
        if direction is not None:
            payload["direction"] = str(direction)
        if action_value.get("task_id") is not None:
            payload["task_id"] = str(action_value["task_id"])
        return payload
    raise TypeError(f"Unsupported action value: {action_value!r}")
