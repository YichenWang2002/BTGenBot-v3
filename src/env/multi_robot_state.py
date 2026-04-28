"""State containers for the multi-robot grid environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Position = list[int]
PositionKey = tuple[int, int]

ROBOT_STATUSES = {"idle", "running", "waiting", "success", "failure"}


@dataclass
class RobotState:
    robot_id: str
    position: Position
    goal: Position | None
    status: str = "idle"
    path: list[Position] = field(default_factory=list)
    carrying_object: str | None = None
    start: Position | None = None
    assigned_waypoints: list[str] = field(default_factory=list)
    current_waypoint_index: int = 0
    completed_waypoints: list[str] = field(default_factory=list)
    recovery_state: str = "none"
    recovery_attempts: int = 0
    stuck_ticks: int = 0
    last_positions: list[Position] = field(default_factory=list)
    previous_position: Position | None = None
    assigned_tasks: list[str] = field(default_factory=list)
    current_task: str | None = None
    task_state: str = "idle"

    def __post_init__(self) -> None:
        self.position = normalize_position(self.position)
        if self.goal is not None:
            self.goal = normalize_position(self.goal)
        self.path = [normalize_position(position) for position in self.path]
        if self.status not in ROBOT_STATUSES:
            raise ValueError(f"Invalid robot status for {self.robot_id}: {self.status}")
        if self.start is None:
            self.start = list(self.position)
        else:
            self.start = normalize_position(self.start)

    @property
    def at_goal(self) -> bool:
        return self.goal is not None and self.position == self.goal

    @property
    def has_waypoint_task(self) -> bool:
        return bool(self.assigned_waypoints)

    def reset(self) -> None:
        self.position = list(self.start or self.position)
        self.path = [list(self.position)]
        self.current_waypoint_index = 0
        self.completed_waypoints = []
        self.recovery_state = "none"
        self.recovery_attempts = 0
        self.stuck_ticks = 0
        self.last_positions = [list(self.position)]
        self.previous_position = None
        self.current_task = self.assigned_tasks[0] if self.assigned_tasks else None
        self.task_state = "navigate_to_pickup" if self.current_task else "idle"
        self.status = "running"
        self.carrying_object = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "position": list(self.position),
            "goal": list(self.goal) if self.goal is not None else None,
            "status": self.status,
            "path": [list(position) for position in self.path],
            "carrying_object": self.carrying_object,
            "assigned_waypoints": list(self.assigned_waypoints),
            "current_waypoint_index": self.current_waypoint_index,
            "completed_waypoints": list(self.completed_waypoints),
            "recovery_state": self.recovery_state,
            "recovery_attempts": self.recovery_attempts,
            "stuck_ticks": self.stuck_ticks,
            "last_positions": [list(position) for position in self.last_positions],
            "assigned_tasks": list(self.assigned_tasks),
            "current_task": self.current_task,
            "task_state": self.task_state,
        }


@dataclass
class ObjectState:
    object_id: str
    position: Position | None = None
    held_by: str | None = None
    target_position: Position | None = None
    status: str = "available"

    def __post_init__(self) -> None:
        if self.position is not None:
            self.position = normalize_position(self.position)
        if self.target_position is not None:
            self.target_position = normalize_position(self.target_position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "position": list(self.position) if self.position is not None else None,
            "held_by": self.held_by,
            "target_position": (
                list(self.target_position) if self.target_position is not None else None
            ),
            "status": self.status,
        }


@dataclass
class MultiRobotState:
    timestep: int
    grid_width: int
    grid_height: int
    obstacles: set[PositionKey] = field(default_factory=set)
    robots: dict[str, RobotState] = field(default_factory=dict)
    objects: dict[str, ObjectState] = field(default_factory=dict)
    waypoints: dict[str, Position] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("Grid width and height must be positive")
        self.obstacles = {position_key(position) for position in self.obstacles}
        if not self.robots:
            raise ValueError("MultiRobotState requires at least one robot")
        for robot in self.robots.values():
            self.validate_position(robot.position, f"{robot.robot_id} position")
            if robot.goal is not None:
                self.validate_position(robot.goal, f"{robot.robot_id} goal")
            for waypoint_id in robot.assigned_waypoints:
                if waypoint_id not in self.waypoints:
                    raise ValueError(
                        f"{robot.robot_id} assigned unknown waypoint: {waypoint_id}"
                    )
        self.waypoints = {
            waypoint_id: normalize_position(position)
            for waypoint_id, position in self.waypoints.items()
        }
        for waypoint_id, position in self.waypoints.items():
            self.validate_position(position, f"{waypoint_id} waypoint")
        for obj in self.objects.values():
            if obj.position is not None:
                self.validate_position(obj.position, f"{obj.object_id} position")
            if obj.target_position is not None:
                self.validate_position(obj.target_position, f"{obj.object_id} target")

    @classmethod
    def from_single_robot(
        cls,
        grid_width: int,
        grid_height: int,
        start: Position,
        goal: Position,
        obstacles: list[Position] | set[PositionKey] | None = None,
    ) -> "MultiRobotState":
        robot = RobotState(
            robot_id="robot_0",
            position=start,
            goal=goal,
            status="running",
            path=[normalize_position(start)],
            start=start,
        )
        return cls(
            timestep=0,
            grid_width=grid_width,
            grid_height=grid_height,
            obstacles={position_key(position) for position in obstacles or []},
            robots={"robot_0": robot},
        )

    def default_robot_id(self, robot_id: str | None = None) -> str:
        return robot_id or "robot_0"

    def get_robot(self, robot_id: str | None = None) -> RobotState:
        resolved = self.default_robot_id(robot_id)
        try:
            return self.robots[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown robot_id: {resolved}") from exc

    def is_in_bounds(self, position: Position | PositionKey) -> bool:
        x, y = position_key(position)
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def is_obstacle(self, position: Position | PositionKey) -> bool:
        return position_key(position) in self.obstacles

    def is_blocked(self, position: Position | PositionKey) -> bool:
        return not self.is_in_bounds(position) or self.is_obstacle(position)

    def validate_position(self, position: Position | PositionKey, label: str) -> None:
        if not self.is_in_bounds(position):
            raise ValueError(f"{label} is outside the grid: {position}")

    def reset(self) -> None:
        self.timestep = 0
        for robot in self.robots.values():
            robot.reset()
            self.activate_robot_waypoint(robot.robot_id)

    def activate_robot_waypoint(self, robot_id: str) -> None:
        robot = self.robots[robot_id]
        if not robot.assigned_waypoints:
            if robot.goal is None:
                robot.status = "success"
                return
            robot.status = "success" if robot.at_goal else "running"
            return
        if robot.current_waypoint_index >= len(robot.assigned_waypoints):
            robot.goal = None
            robot.status = "success"
            return
        waypoint_id = robot.assigned_waypoints[robot.current_waypoint_index]
        robot.goal = list(self.waypoints[waypoint_id])
        robot.status = "success" if robot.position == robot.goal else "running"

    def advance_robot_waypoint_if_reached(self, robot_id: str) -> None:
        robot = self.robots[robot_id]
        if not robot.assigned_waypoints:
            if robot.at_goal:
                robot.status = "success"
            return
        if robot.goal is None or robot.position != robot.goal:
            return
        waypoint_id = robot.assigned_waypoints[robot.current_waypoint_index]
        if waypoint_id not in robot.completed_waypoints:
            robot.completed_waypoints.append(waypoint_id)
        robot.current_waypoint_index += 1
        self.activate_robot_waypoint(robot_id)

    def robot_positions(self) -> dict[str, Position]:
        return {
            robot_id: list(robot.position)
            for robot_id, robot in sorted(self.robots.items())
        }

    def robot_statuses(self) -> dict[str, str]:
        return {
            robot_id: robot.status
            for robot_id, robot in sorted(self.robots.items())
        }

    def all_robots_success(self) -> bool:
        return all(robot.status == "success" for robot in self.robots.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "obstacles": [list(position) for position in sorted(self.obstacles)],
            "robots": {
                robot_id: robot.to_dict()
                for robot_id, robot in sorted(self.robots.items())
            },
            "objects": {
                object_id: obj.to_dict()
                for object_id, obj in sorted(self.objects.items())
            },
            "waypoints": {
                waypoint_id: list(position)
                for waypoint_id, position in sorted(self.waypoints.items())
            },
        }


def normalize_position(position: Position | tuple[int, int]) -> Position:
    if len(position) != 2:
        raise ValueError(f"Position must contain exactly two values: {position}")
    return [int(position[0]), int(position[1])]


def position_key(position: Position | PositionKey) -> PositionKey:
    if len(position) != 2:
        raise ValueError(f"Position must contain exactly two values: {position}")
    return int(position[0]), int(position[1])
