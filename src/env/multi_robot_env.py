"""Minimal multi-robot grid environment with pygame rendering."""

from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from typing import Any

from src.coordination.rules import CentralizedRuleManager
from src.dag.task_dag import TaskDAG
from src.env.actions import (
    Action,
    move_one_step as move_position_one_step,
    navigate_one_step as navigate_robot_one_step,
    normalize_step_actions,
    wait as wait_position,
)
from src.env.metrics import MetricsRecorder
from src.env.multi_robot_state import MultiRobotState, Position, position_key
from src.scenarios.pickplace_injection import pick_fails, place_fails
from src.scenarios.pickplace_tasks import reassign_failed_task


class MultiRobotEnv:
    def __init__(
        self,
        state: MultiRobotState,
        max_steps: int = 100,
        render: bool = False,
        cell_size: int = 48,
        render_fps: int = 8,
        scenario_name: str = "scenario",
        centralized_rule: bool = False,
        recovery_config: dict[str, Any] | None = None,
        pickplace_config: dict[str, Any] | None = None,
        zones: dict[str, Any] | None = None,
        coordination_config: dict[str, Any] | None = None,
        prioritized_planning: bool = False,
        task_dag: TaskDAG | None = None,
        ignore_task_dag: bool = False,
    ) -> None:
        self.initial_state = state
        self.state = state
        self.max_steps = max_steps
        self.render_enabled = render
        self.cell_size = cell_size
        self.render_fps = max(1, int(render_fps))
        self.scenario_name = scenario_name
        self.centralized_rule_enabled = centralized_rule
        self.recovery_config = recovery_config or {}
        self.recovery_enabled = bool(self.recovery_config.get("enabled", False))
        self.recovery_blocked_cells: set[tuple[int, int]] = set()
        self.recovery_cleared_cells: set[tuple[int, int]] = set()
        self.recovery_locks_by_robot: dict[str, str] = {}
        self.pickplace_config = pickplace_config or {}
        self.pickplace_enabled = bool(self.pickplace_config.get("enabled", False))
        self.zones = zones or {}
        self.coordination_config = self._normalize_coordination_config(coordination_config)
        self.prioritized_planning_enabled = prioritized_planning
        self.initial_task_dag = task_dag
        self.task_dag: TaskDAG | None = None
        self.ignore_task_dag = ignore_task_dag
        self.pickplace_tasks: dict[str, dict[str, Any]] = {}
        self.pickplace_locks_by_robot: dict[str, set[str]] = {}
        self.pickplace_path_locks_by_robot: dict[str, set[str]] = {}
        self.pickplace_path_lock_acquired_at: dict[str, dict[str, int]] = {}
        self.pickplace_path_wait_ticks: dict[tuple[str, str], int] = {}
        self.pickplace_path_reassignments: set[tuple[str, str]] = set()
        self._placed_once = False
        self.rule_manager: CentralizedRuleManager | None = None
        self.metrics = MetricsRecorder()
        self.metrics.centralized_rule_enabled = centralized_rule
        self._pygame: Any = None
        self._screen: Any = None
        self._clock: Any = None
        self._font: Any = None
        self._small_font: Any = None
        self._last_collisions: list[dict[str, Any]] = []
        self._pending_dag_events: list[dict[str, Any]] = []

    @classmethod
    def from_single_robot(
        cls,
        grid_width: int,
        grid_height: int,
        start: Position,
        goal: Position,
        obstacles: list[Position] | None = None,
        max_steps: int = 100,
        render: bool = False,
        centralized_rule: bool = False,
        recovery_config: dict[str, Any] | None = None,
        pickplace_config: dict[str, Any] | None = None,
        zones: dict[str, Any] | None = None,
    ) -> "MultiRobotEnv":
        return cls(
            MultiRobotState.from_single_robot(
                grid_width=grid_width,
                grid_height=grid_height,
                start=start,
                goal=goal,
                obstacles=obstacles,
            ),
            max_steps=max_steps,
            render=render,
            scenario_name="single_robot",
            centralized_rule=centralized_rule,
            recovery_config=recovery_config,
            pickplace_config=pickplace_config,
            zones=zones,
        )

    @staticmethod
    def _normalize_coordination_config(
        coordination_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        config = dict(coordination_config or {})
        return {
            "enable_cell_reservation": bool(config.get("enable_cell_reservation", True)),
            "enable_edge_reservation": bool(config.get("enable_edge_reservation", True)),
            "enable_resource_lock": bool(config.get("enable_resource_lock", True)),
            "enable_recovery_lock": bool(config.get("enable_recovery_lock", True)),
            "enable_approach_zone_lock": bool(
                config.get("enable_approach_zone_lock", False)
            ),
            "approach_radius": int(config.get("approach_radius", 1)),
            "max_wait_ticks": int(config.get("max_wait_ticks", 3)),
            "reassignment_wait_ticks": int(
                config.get("reassignment_wait_ticks", 8)
            ),
            "enable_corridor_lock": bool(config.get("enable_corridor_lock", False)),
            "corridor_cells": list(config.get("corridor_cells") or []),
        }

    def reset(self) -> MultiRobotState:
        self.state.reset()
        self.recovery_enabled = bool(self.recovery_config.get("enabled", False))
        self.recovery_blocked_cells = {
            position_key(cell) for cell in self.recovery_config.get("blocked_cells", [])
        }
        self.recovery_cleared_cells = set()
        self.recovery_locks_by_robot = {}
        self.pickplace_enabled = bool(self.pickplace_config.get("enabled", False))
        self.pickplace_tasks = {
            task["task_id"]: deepcopy(task)
            for task in self.pickplace_config.get("tasks", [])
        }
        self.pickplace_locks_by_robot = {}
        self.pickplace_path_locks_by_robot = {}
        self.pickplace_path_lock_acquired_at = {}
        self.pickplace_path_wait_ticks = {}
        self.pickplace_path_reassignments = set()
        self._placed_once = False
        self._last_collisions = []
        self.task_dag = self.initial_task_dag.clone() if self.initial_task_dag else None
        if self.task_dag is not None:
            self.task_dag.reset_statuses()
        self._pending_dag_events = []
        if self.pickplace_enabled:
            for robot_id, robot in self.state.robots.items():
                robot.status = "running"
                robot.assigned_tasks = [
                    task_id
                    for task_id, task in self.pickplace_tasks.items()
                    if task.get("assigned_robot") == robot_id
                ] or list(robot.assigned_tasks)
                robot.current_task = robot.assigned_tasks[0] if robot.assigned_tasks else None
                robot.task_state = "navigate_to_pickup" if robot.current_task else "idle"
        self.metrics.reset()
        self.metrics.centralized_rule_enabled = self.centralized_rule_enabled
        self._update_metrics_dag_snapshot()
        if self.centralized_rule_enabled:
            self.rule_manager = CentralizedRuleManager(
                grid_width=self.state.grid_width,
                grid_height=self.state.grid_height,
                obstacles=self.state.obstacles,
                enable_cell_reservation=self.coordination_config[
                    "enable_cell_reservation"
                ],
                enable_edge_reservation=self.coordination_config[
                    "enable_edge_reservation"
                ],
            )
        else:
            self.rule_manager = None
        if self.render_enabled:
            self.render()
        return self.state

    def move_one_step(
        self, robot_id: str | None = None, direction: str = "wait"
    ) -> dict[str, Any]:
        resolved = self.state.default_robot_id(robot_id)
        return self.step({resolved: {"type": "move", "direction": direction}})

    def navigate_one_step(self, robot_id: str | None = None) -> dict[str, Any]:
        resolved = self.state.default_robot_id(robot_id)
        return self.step({resolved: {"type": "navigate"}})

    def wait(self, robot_id: str | None = None) -> dict[str, Any]:
        resolved = self.state.default_robot_id(robot_id)
        return self.step({resolved: {"type": "wait"}})

    def step(self, actions: Any = None) -> dict[str, Any]:
        if self.done:
            return self.observation([])

        robot_ids = sorted(self.state.robots)
        normalized_actions = normalize_step_actions(actions, robot_ids)
        proposals: dict[str, Position] = {}
        action_records: dict[str, dict[str, Any]] = {}
        self._pending_recovery_events: list[dict[str, Any]] = []
        self._pending_pickplace_events: list[dict[str, Any]] = []
        self._pending_dag_events: list[dict[str, Any]] = []
        self._step_recovery_blockers = self.current_recovery_blockers()

        for robot_id in robot_ids:
            robot = self.state.robots[robot_id]
            action_value = normalized_actions.get(robot_id, {"type": "wait"})
            target, action_record = self._propose_action(robot_id, action_value)
            proposals[robot_id] = target
            action_records[robot_id] = action_record.to_dict()
            if robot.status == "success":
                proposals[robot_id] = list(robot.position)
        self._step_recovery_blockers = None

        rule_events: list[dict[str, Any]] = []
        reservations: dict[str, Any] = {"cells": [], "edges": []}
        resource_locks: dict[str, Any] = {"locks": {}, "wait_queues": {}}
        recovery_events: list[dict[str, Any]] = list(self._pending_recovery_events)
        recovery_events.extend(self._detect_naive_recovery_conflicts(recovery_events))
        pickplace_events: list[dict[str, Any]] = list(getattr(self, "_pending_pickplace_events", []))
        pickplace_events.extend(self._detect_naive_pickplace_conflicts(pickplace_events))
        self._complete_ready_system_tasks()
        dag_events: list[dict[str, Any]] = list(getattr(self, "_pending_dag_events", []))
        deadlock_detected = False

        if self.centralized_rule_enabled:
            proposals, rule_events, reservations, resource_locks, deadlock_detected = (
                self._apply_centralized_rules(proposals, action_records)
            )
            if self.rule_manager is not None:
                resource_locks = self.rule_manager.snapshot()["resource_locks"]
        elif self.prioritized_planning_enabled:
            proposals = self._apply_prioritized_planning(proposals, action_records)

        if self._pickplace_path_locking_enabled():
            path_rule_event_start = (
                len(self.rule_manager.rule_events) if self.rule_manager is not None else 0
            )
            proposals = self._apply_pickplace_path_locks(
                proposals, action_records, pickplace_events
            )
            if self.rule_manager is not None:
                rule_events.extend(self.rule_manager.events_since(path_rule_event_start))
                resource_locks = self.rule_manager.snapshot()["resource_locks"]

        collisions = self._detect_collisions(proposals)
        self._last_collisions = collisions
        collided_robot_ids = {
            robot_id
            for collision in collisions
            for robot_id in collision["robot_ids"]
        }

        moved_robot_ids: list[str] = []
        previous_positions = {
            robot_id: list(robot.position)
            for robot_id, robot in self.state.robots.items()
        }
        for robot_id, target in proposals.items():
            robot = self.state.robots[robot_id]
            if robot_id in collided_robot_ids:
                target = list(robot.position)

            if target != robot.position:
                moved_robot_ids.append(robot_id)
                robot.previous_position = list(robot.position)
            robot.position = list(target)
            robot.path.append(list(robot.position))
            self.state.advance_robot_waypoint_if_reached(robot_id)
            if self.pickplace_enabled:
                if self._robot_pickplace_done(robot_id):
                    robot.status = "success"
                elif action_records[robot_id].get("rule_rejected"):
                    robot.status = "waiting"
                elif robot.status != "failure":
                    robot.status = "running"
            elif robot.status == "success":
                pass
            elif action_records[robot_id].get("rule_rejected"):
                robot.status = "waiting"
            elif robot.position == robot.goal:
                robot.status = "success"
            elif robot.status != "failure":
                robot.status = "running"
            is_recovery_action = str(action_records[robot_id].get("action_type", "")).startswith(
                "recovery_"
            )
            if (
                robot.position == previous_positions[robot_id]
                and robot.status == "running"
                and not is_recovery_action
            ):
                robot.stuck_ticks += 1
            elif not is_recovery_action:
                robot.stuck_ticks = 0
            robot.last_positions.append(list(robot.position))
            robot.last_positions = robot.last_positions[-5:]
            if (
                self.recovery_enabled
                and robot.stuck_ticks >= int(self.recovery_config.get("stuck_threshold", 3))
                and robot.recovery_state == "none"
                and robot.status == "running"
            ):
                robot.recovery_state = "clear_costmap"
                robot.recovery_attempts += 1
                recovery_events.append(
                    self._recovery_event(robot_id, "stuck", "stuck_threshold")
                )

        if self._pickplace_path_locking_enabled():
            self._release_inactive_pickplace_path_locks(pickplace_events)
            if self.rule_manager is not None:
                resource_locks = self.rule_manager.snapshot()["resource_locks"]

        self.state.timestep += 1
        timeout = self.state.timestep >= self.max_steps and not self._success()
        if timeout:
            for robot in self.state.robots.values():
                if robot.status != "success":
                    robot.status = "failure"

        self.metrics.record_step(
            timestep=self.state.timestep,
            robot_positions=self.state.robot_positions(),
            actions=action_records,
            collisions=collisions,
            status=self.state.robot_statuses(),
            moved_robot_ids=moved_robot_ids,
            rule_events=rule_events,
            reservations=reservations,
            resource_locks=resource_locks,
            deadlock_detected=deadlock_detected,
            recovery_events=recovery_events,
            recovery_states=self._recovery_states(),
            recovery_queues=self._recovery_queues(),
            temporary_obstacles=[
                list(cell) for cell in sorted(self.active_temporary_obstacles())
            ],
            pickplace_events=pickplace_events,
            object_states=self._object_states(),
            task_states=self._task_states(),
            pickup_zone_queues=self._zone_queues("pickup_zone"),
            drop_zone_queues=self._zone_queues("drop_zone"),
            dag_task_states=self._dag_task_states(),
            dag_ready_tasks=self._dag_ready_tasks(),
            dag_blocked_tasks=self._dag_blocked_tasks(),
            dag_events=dag_events,
            dag_summary=self._dag_summary(),
        )
        self.metrics.finish(success=self._success(), timeout=timeout)

        if self.render_enabled:
            self.render()

        return self.observation(collisions)

    @property
    def done(self) -> bool:
        return self._success() or self.metrics.timeout

    def observation(self, collisions: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "metrics": self.metrics.summary(),
            "collisions": collisions,
            "done": self.done,
        }

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None

    def pause_render_window(self) -> None:
        self._ensure_pygame()
        assert self._pygame is not None
        assert self._clock is not None

        pygame = self._pygame
        waiting = True
        while waiting and self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.render_enabled = False
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
            self._clock.tick(30)

    def render(self) -> None:
        self._ensure_pygame()
        assert self._pygame is not None
        assert self._screen is not None
        assert self._clock is not None

        pygame = self._pygame
        cell = self.cell_size
        width = self.state.grid_width * cell
        grid_height = self.state.grid_height * cell

        self._screen.fill((245, 245, 245))
        for x in range(self.state.grid_width):
            for y in range(self.state.grid_height):
                rect = pygame.Rect(x * cell, y * cell, cell, cell)
                pygame.draw.rect(self._screen, (220, 220, 220), rect, 1)

        for obstacle in self.state.obstacles:
            x, y = obstacle
            rect = pygame.Rect(x * cell, y * cell, cell, cell)
            pygame.draw.rect(self._screen, (40, 40, 40), rect)

        self._draw_collisions(pygame, cell)
        self._draw_pickplace_markers(pygame, cell)

        colors = [
            (31, 119, 180),
            (214, 39, 40),
            (44, 160, 44),
            (255, 127, 14),
            (148, 103, 189),
            (23, 190, 207),
        ]

        for index, (robot_id, robot) in enumerate(sorted(self.state.robots.items())):
            color = colors[index % len(colors)]
            if robot.goal is not None:
                goal_x, goal_y = robot.goal
                goal_rect = pygame.Rect(
                    goal_x * cell + cell // 4,
                    goal_y * cell + cell // 4,
                    cell // 2,
                    cell // 2,
                )
                pygame.draw.rect(self._screen, color, goal_rect, 2)

            x, y = robot.position
            center = (x * cell + cell // 2, y * cell + cell // 2)
            pygame.draw.circle(self._screen, color, center, max(8, cell // 3))
            if self._font is not None:
                label = self._font.render(robot_id.split("_")[-1], True, (255, 255, 255))
                self._screen.blit(label, label.get_rect(center=center))

        self._draw_objects(pygame, cell)
        self._draw_info_panel(pygame, width)

        pygame.display.set_caption(f"{self.scenario_name} t={self.state.timestep}")
        pygame.display.flip()
        self._clock.tick(self.render_fps)

        # Keep the window responsive for interactive demos.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.render_enabled = False

        if width <= 0 or grid_height <= 0:
            raise RuntimeError("Invalid render surface size")

    def _ensure_pygame(self) -> None:
        if self._pygame is not None:
            return

        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

        import pygame

        pygame.init()
        self._pygame = pygame
        width = self.state.grid_width * self.cell_size
        height = self.state.grid_height * self.cell_size
        panel_width = self._panel_width()
        self._screen = pygame.display.set_mode((width + panel_width, max(height, 360)))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, max(18, self.cell_size // 2))
        self._small_font = pygame.font.Font(None, 18)

    def _panel_width(self) -> int:
        return 390

    def _draw_collisions(self, pygame: Any, cell: int) -> None:
        for collision in self._last_collisions:
            if "position" in collision:
                x, y = collision["position"]
                rect = pygame.Rect(x * cell, y * cell, cell, cell)
                pygame.draw.rect(self._screen, (255, 205, 205), rect)
                pygame.draw.rect(self._screen, (210, 40, 40), rect, max(2, cell // 12))
            elif "from_cell" in collision and "to_cell" in collision:
                from_x, from_y = collision["from_cell"]
                to_x, to_y = collision["to_cell"]
                start = (from_x * cell + cell // 2, from_y * cell + cell // 2)
                end = (to_x * cell + cell // 2, to_y * cell + cell // 2)
                pygame.draw.line(self._screen, (210, 40, 40), start, end, max(3, cell // 10))

    def _draw_pickplace_markers(self, pygame: Any, cell: int) -> None:
        for task in self.pickplace_tasks.values():
            pickup = task.get("pickup_position")
            drop = task.get("drop_position")
            if pickup is not None:
                x, y = pickup
                rect = pygame.Rect(
                    x * cell + cell // 5,
                    y * cell + cell // 5,
                    max(6, cell * 3 // 5),
                    max(6, cell * 3 // 5),
                )
                pygame.draw.rect(self._screen, (65, 105, 225), rect, 2)
            if drop is not None:
                x, y = drop
                center = (x * cell + cell // 2, y * cell + cell // 2)
                pygame.draw.circle(self._screen, (20, 120, 80), center, max(6, cell // 3), 2)

    def _draw_objects(self, pygame: Any, cell: int) -> None:
        for obj in self.state.objects.values():
            color = self._object_color(obj.status)
            position = obj.position
            if position is None and obj.held_by in self.state.robots:
                robot = self.state.robots[obj.held_by]
                center = (
                    robot.position[0] * cell + cell * 3 // 4,
                    robot.position[1] * cell + cell // 4,
                )
            elif position is not None:
                center = (position[0] * cell + cell // 2, position[1] * cell + cell // 2)
            else:
                continue

            radius = max(5, cell // 6)
            pygame.draw.circle(self._screen, color, center, radius)
            pygame.draw.circle(self._screen, (35, 35, 35), center, radius, 1)
            if self._small_font is not None:
                label = self._small_font.render(obj.object_id.split("_")[-1], True, (20, 20, 20))
                self._screen.blit(label, label.get_rect(center=(center[0], center[1] + radius + 7)))

    @staticmethod
    def _object_color(status: str) -> tuple[int, int, int]:
        return {
            "available": (166, 110, 54),
            "held": (252, 186, 3),
            "placed": (70, 155, 105),
            "unavailable": (150, 150, 150),
        }.get(status, (166, 110, 54))

    def _draw_info_panel(self, pygame: Any, panel_x: int) -> None:
        assert self._screen is not None
        panel_rect = pygame.Rect(panel_x, 0, self._panel_width(), self._screen.get_height())
        pygame.draw.rect(self._screen, (250, 250, 250), panel_rect)
        pygame.draw.line(self._screen, (190, 190, 190), (panel_x, 0), (panel_x, self._screen.get_height()), 1)

        lines = self._render_panel_lines()
        font = self._small_font or self._font
        if font is None:
            return
        y = 12
        for line, color in lines:
            if y > self._screen.get_height() - 18:
                break
            text = font.render(line[:58], True, color)
            self._screen.blit(text, (panel_x + 12, y))
            y += 18

    def _render_panel_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
        lines: list[tuple[str, tuple[int, int, int]]] = []
        normal = (35, 35, 35)
        muted = (95, 95, 95)
        alert = (190, 40, 40)

        lines.append((f"scenario: {self.scenario_name}", normal))
        lines.append((f"timestep: {self.state.timestep}/{self.max_steps}", normal))
        lines.append((f"status: {self._status_label()}", normal))
        lines.append((f"centralized_rule_enabled: {self.centralized_rule_enabled}", normal))
        lines.append(("", normal))
        lines.append(("robots", muted))
        for robot_id, robot in sorted(self.state.robots.items()):
            task = f" task={robot.current_task}:{robot.task_state}" if robot.current_task else ""
            carrying = f" carrying={robot.carrying_object}" if robot.carrying_object else ""
            lines.append(
                (
                    f"{robot_id} pos={robot.position} status={robot.status}{task}{carrying}",
                    normal,
                )
            )

        if self.state.objects:
            task_by_object = {
                str(task.get("object_id")): task
                for task in self.pickplace_tasks.values()
                if task.get("object_id") is not None
            }
            lines.append(("", normal))
            lines.append(("objects", muted))
            for object_id, obj in sorted(self.state.objects.items()):
                task = task_by_object.get(object_id, {})
                held = f" held_by={obj.held_by}" if obj.held_by else ""
                lines.append((f"{object_id} status={obj.status}{held}", normal))
                lines.append(
                    (
                        f"  pickup={task.get('pickup_position')} drop={task.get('drop_position')}",
                        muted,
                    )
                )

        lines.append(("", normal))
        lines.append(("resource lock wait queues", muted))
        wait_lines = self._format_wait_queue_lines()
        if wait_lines:
            lines.extend((line, alert) for line in wait_lines)
        else:
            lines.append(("none", muted))

        lines.append(("", normal))
        lines.append(("collisions", muted))
        if self._last_collisions:
            lines.extend((self._format_collision(collision), alert) for collision in self._last_collisions[:6])
        else:
            lines.append(("none", muted))

        return lines

    def _status_label(self) -> str:
        if self.metrics.timeout:
            return "timeout"
        if self.metrics.success or self._success():
            return "success"
        return "running"

    def _format_wait_queue_lines(self) -> list[str]:
        locks = self._resource_locks_for_render()
        wait_queues = locks.get("wait_queues", {})
        lines: list[str] = []
        for resource_id, queue in sorted(wait_queues.items()):
            queued = [
                str(item.get("robot_id", item)) if isinstance(item, dict) else str(item)
                for item in queue
            ]
            if queued:
                lines.append(f"{resource_id}: {', '.join(queued)}")
        return lines

    def _resource_locks_for_render(self) -> dict[str, Any]:
        if self.rule_manager is not None:
            return self.rule_manager.snapshot().get("resource_locks", {})
        if self.metrics.trace:
            return self.metrics.trace[-1].get("resource_locks", {})
        return {"locks": {}, "wait_queues": {}}

    @staticmethod
    def _format_collision(collision: dict[str, Any]) -> str:
        if "position" in collision:
            return (
                f"{collision.get('type')} cell={collision.get('position')} "
                f"robots={collision.get('robot_ids')}"
            )
        return (
            f"{collision.get('type')} edge={collision.get('from_cell')}->{collision.get('to_cell')} "
            f"robots={collision.get('robot_ids')}"
        )

    def _propose_action(
        self, robot_id: str, action_value: dict[str, Any]
    ) -> tuple[Position, Action]:
        robot = self.state.robots[robot_id]
        action_type = str(action_value.get("type", "wait")).lower()

        if action_type == "navigate":
            if self.pickplace_enabled:
                return self._propose_pickplace_action(robot_id)
            target, direction = navigate_robot_one_step(robot)
            if self.recovery_enabled:
                if robot.recovery_state != "none":
                    return self._propose_recovery_action(robot_id)
                if self._is_recovery_blocked(target):
                    robot.recovery_state = "clear_costmap"
                    robot.recovery_attempts += 1
                    return self._propose_recovery_action(
                        robot_id, blocked_target=target
                    )
            return target, Action(robot_id=robot_id, action_type="navigate", direction=direction)
        if action_type == "move":
            direction = str(action_value.get("direction", "wait"))
            target = move_position_one_step(robot.position, direction)
            return target, Action(robot_id=robot_id, action_type="move", direction=direction)
        if action_type == "wait":
            return wait_position(robot.position), Action(robot_id=robot_id, action_type="wait")
        if action_type == "final_check":
            task_id = action_value.get("task_id") or self._find_dag_task_id(
                "final_check", robot_id=robot_id
            )
            if task_id and self._dag_should_wait(robot_id, str(task_id), "final_check"):
                return wait_position(robot.position), Action(
                    robot_id=robot_id,
                    action_type="dag_wait",
                    task_id=str(task_id),
                )
            if task_id:
                self._mark_dag_completed(str(task_id), robot_id, "final_check")
            return wait_position(robot.position), Action(
                robot_id=robot_id,
                action_type="final_check",
                task_id=str(task_id) if task_id else None,
            )

        raise ValueError(f"Unknown action type for {robot_id}: {action_type}")

    def _propose_recovery_action(
        self, robot_id: str, blocked_target: Position | None = None
    ) -> tuple[Position, Action]:
        robot = self.state.robots[robot_id]
        if not hasattr(self, "_pending_recovery_events"):
            self._pending_recovery_events = []
        events = self._pending_recovery_events
        if blocked_target is not None:
            events.append(self._recovery_event(robot_id, "recovery_start", "blocked_path"))
            if position_key(blocked_target) in self.active_temporary_obstacles():
                events.append(
                    self._recovery_event(
                        robot_id,
                        "temporary_obstacle_hit",
                        "temporary_obstacle",
                        cell=blocked_target,
                    )
                )

        state = robot.recovery_state
        recovery_task_id = self._find_dag_task_id("recovery", robot_id=robot_id)
        if recovery_task_id and self._dag_should_wait(
            robot_id, recovery_task_id, f"recovery_{state}"
        ):
            return list(robot.position), Action(
                robot_id, "dag_wait", task_id=recovery_task_id
            )
        if state == "clear_costmap":
            if not self._ensure_recovery_lock(robot_id, "clear_costmap", events):
                return list(robot.position), Action(robot_id, "recovery_clear_costmap")
            cleared = self._clear_recovery_zone(robot_id)
            events.append(
                self._recovery_event(
                    robot_id,
                    "clear_costmap",
                    "cleared_recovery_zone",
                    cells=[list(cell) for cell in sorted(cleared)],
                )
            )
            robot.recovery_state = "spin"
            return list(robot.position), Action(robot_id, "recovery_clear_costmap")

        if state == "spin":
            if not self._ensure_recovery_lock(robot_id, "spin", events):
                return list(robot.position), Action(robot_id, "recovery_spin")
            events.append(self._recovery_event(robot_id, "spin", "spin"))
            robot.recovery_state = "wait"
            return list(robot.position), Action(robot_id, "recovery_spin")

        if state == "wait":
            events.append(
                self._recovery_event(robot_id, "recovery_wait", "wait", duration=1)
            )
            robot.recovery_state = "backup"
            return list(robot.position), Action(robot_id, "recovery_wait")

        if state == "backup":
            target = self._backup_target(robot_id)
            events.append(
                self._recovery_event(robot_id, "backup", "backup", to_cell=target)
            )
            events.append(
                self._recovery_event(robot_id, "recovery_success", "recovery_done")
            )
            if recovery_task_id:
                self._mark_dag_completed(recovery_task_id, robot_id, "recovery_success")
            robot.recovery_state = "none"
            self._release_recovery_lock(robot_id)
            return target, Action(robot_id, "recovery_backup", task_id=recovery_task_id)

        robot.recovery_state = "none"
        return list(robot.position), Action(robot_id, "wait")

    def _propose_pickplace_action(self, robot_id: str) -> tuple[Position, Action]:
        if not hasattr(self, "_pending_pickplace_events"):
            self._pending_pickplace_events = []
        events = self._pending_pickplace_events
        robot = self.state.robots[robot_id]
        task = self._current_task_for_robot(robot_id)
        if task is None:
            robot.task_state = "done"
            return list(robot.position), Action(robot_id, "wait")

        obj = self.state.objects[task["object_id"]]
        if robot.task_state in {"idle", "done"}:
            robot.task_state = "navigate_to_pickup"

        if robot.task_state == "navigate_to_pickup":
            target = list(obj.position or task["pickup_position"])
            robot.goal = target
            if robot.position == target:
                robot.task_state = "pick"
                return list(robot.position), Action(robot_id, "pick")
            return self._step_toward(robot, target), Action(robot_id, "navigate_to_pickup")

        if robot.task_state == "pick":
            return self._pick(robot_id, task, obj, events)

        if robot.task_state == "navigate_to_drop":
            target = list(obj.target_position or task["drop_position"])
            robot.goal = target
            if robot.position == target:
                robot.task_state = "place"
                return list(robot.position), Action(robot_id, "place")
            return self._step_toward(robot, target), Action(robot_id, "navigate_to_drop")

        if robot.task_state == "place":
            return self._place(robot_id, task, obj, events)

        return list(robot.position), Action(robot_id, "wait")

    def _pick(
        self,
        robot_id: str,
        task: dict[str, Any],
        obj: Any,
        events: list[dict[str, Any]],
    ) -> tuple[Position, Action]:
        robot = self.state.robots[robot_id]
        dag_task_id = self._find_dag_task_id(
            "pick",
            robot_id=robot_id,
            object_id=str(task.get("object_id")),
            explicit_task_id=task.get("pick_task_id"),
        )
        if dag_task_id and self._dag_should_wait(robot_id, dag_task_id, "pick"):
            return list(robot.position), Action(robot_id, "dag_wait", task_id=dag_task_id)
        events.append(self._pickplace_event(robot_id, "pick_attempt", task))
        if self.centralized_rule_enabled:
            if not self._acquire_pickplace_lock(robot_id, obj.object_id, "object", "pick", events):
                return list(robot.position), Action(robot_id, "pick")
            zone_id = self._zone_for_cell("pickup_zones", obj.position or task["pickup_position"])
            if not self._acquire_pickplace_lock(robot_id, zone_id, "pickup_zone", "pick", events):
                return list(robot.position), Action(robot_id, "pick")
        elif obj.held_by and obj.held_by != robot_id:
            events.append(self._pickplace_event(robot_id, "object_conflict", task, object_id=obj.object_id))

        if obj.status == "unavailable":
            obj.status = "available"
            self._release_matching_approach_lock(robot_id, "pickup", task, events)
            self._fail_or_reassign(robot_id, task, events, "object_unavailable")
            return list(robot.position), Action(robot_id, "pick")

        task["attempts"] = int(task.get("attempts", 0)) + 1
        seed = int(self.pickplace_config.get("seed", 0))
        if pick_fails(seed, task["task_id"], task["attempts"], float(self.pickplace_config.get("pick_fail_prob", 0))):
            self._release_matching_approach_lock(robot_id, "pickup", task, events)
            self._fail_or_reassign(robot_id, task, events, "pick_failure")
            return list(robot.position), Action(robot_id, "pick")

        obj.held_by = robot_id
        obj.status = "held"
        obj.position = None
        robot.carrying_object = obj.object_id
        task["status"] = "picked"
        robot.task_state = "navigate_to_drop"
        self._release_matching_approach_lock(robot_id, "pickup", task, events)
        if dag_task_id:
            self._mark_dag_completed(dag_task_id, robot_id, "pick_success")
        events.append(self._pickplace_event(robot_id, "pick_success", task, object_id=obj.object_id))
        return list(robot.position), Action(robot_id, "pick", task_id=dag_task_id)

    def _place(
        self,
        robot_id: str,
        task: dict[str, Any],
        obj: Any,
        events: list[dict[str, Any]],
    ) -> tuple[Position, Action]:
        robot = self.state.robots[robot_id]
        dag_task_id = self._find_dag_task_id(
            "place",
            robot_id=robot_id,
            object_id=str(task.get("object_id")),
            explicit_task_id=task.get("place_task_id"),
        )
        if dag_task_id and self._dag_should_wait(robot_id, dag_task_id, "place"):
            return list(robot.position), Action(robot_id, "dag_wait", task_id=dag_task_id)
        events.append(self._pickplace_event(robot_id, "place_attempt", task))
        if self.centralized_rule_enabled:
            zone_id = self._zone_for_cell("drop_zones", obj.target_position or task["drop_position"])
            if not self._acquire_pickplace_lock(robot_id, zone_id, "drop_zone", "place", events):
                return list(robot.position), Action(robot_id, "place")

        task["attempts"] = int(task.get("attempts", 0)) + 1
        seed = int(self.pickplace_config.get("seed", 0))
        if place_fails(seed, task["task_id"], task["attempts"], float(self.pickplace_config.get("place_fail_prob", 0))):
            self._release_matching_approach_lock(robot_id, "drop", task, events)
            events.append(self._pickplace_event(robot_id, "place_failure", task, object_id=obj.object_id))
            return list(robot.position), Action(robot_id, "place")

        obj.position = list(obj.target_position or task["drop_position"])
        obj.held_by = None
        obj.status = "placed"
        robot.carrying_object = None
        task["status"] = "placed"
        robot.task_state = "done"
        self._release_matching_approach_lock(robot_id, "drop", task, events)
        self._release_pickplace_locks(robot_id)
        self._release_pickplace_path_locks(robot_id, events)
        self._advance_robot_task(robot_id)
        if dag_task_id:
            self._mark_dag_completed(dag_task_id, robot_id, "place_success")
        events.append(self._pickplace_event(robot_id, "place_success", task, object_id=obj.object_id))
        return list(robot.position), Action(robot_id, "place", task_id=dag_task_id)

    def _current_task_for_robot(self, robot_id: str) -> dict[str, Any] | None:
        robot = self.state.robots[robot_id]
        if robot.current_task and self.pickplace_tasks.get(robot.current_task, {}).get("status") != "placed":
            return self.pickplace_tasks[robot.current_task]
        self._advance_robot_task(robot_id)
        if robot.current_task:
            return self.pickplace_tasks[robot.current_task]
        return None

    def _advance_robot_task(self, robot_id: str) -> None:
        robot = self.state.robots[robot_id]
        for task_id in robot.assigned_tasks:
            task = self.pickplace_tasks.get(task_id)
            if task and task.get("status") not in {"placed", "failed"}:
                robot.current_task = task_id
                robot.task_state = "navigate_to_pickup"
                return
        robot.current_task = None
        robot.task_state = "done"
        robot.status = "success"

    def _fail_or_reassign(
        self, robot_id: str, task: dict[str, Any], events: list[dict[str, Any]], reason: str
    ) -> None:
        events.append(self._pickplace_event(robot_id, "pick_failure", task, reason=reason))
        task["status"] = "failed"
        old_robot = self.state.robots[robot_id]
        if self.pickplace_config.get("allow_reassignment", False):
            robots = {rid: {"start": r.position} for rid, r in self.state.robots.items()}
            reassigned = reassign_failed_task(task, robots, robot_id)
            task.update(reassigned)
            new_robot = task.get("assigned_robot")
            if new_robot:
                self._sync_dag_task_assignment(task, str(new_robot))
            if task["task_id"] in old_robot.assigned_tasks:
                old_robot.assigned_tasks.remove(task["task_id"])
            if new_robot and task["task_id"] not in self.state.robots[new_robot].assigned_tasks:
                self.state.robots[new_robot].assigned_tasks.append(task["task_id"])
            if new_robot and self.state.robots[new_robot].current_task is None:
                self.state.robots[new_robot].current_task = task["task_id"]
                self.state.robots[new_robot].task_state = "navigate_to_pickup"
                self.state.robots[new_robot].status = "running"
            events.append(self._pickplace_event(robot_id, "reassignment", task, reassigned_to=new_robot))
        self._release_pickplace_locks(robot_id)
        self._release_pickplace_path_locks(robot_id, events)
        self._advance_robot_task(robot_id)

    def _acquire_pickplace_lock(
        self,
        robot_id: str,
        resource_id: str,
        resource_type: str,
        reason: str,
        events: list[dict[str, Any]],
    ) -> bool:
        if not self.centralized_rule_enabled or self.rule_manager is None:
            return True
        if not self.coordination_config["enable_resource_lock"]:
            return True
        held = self.pickplace_locks_by_robot.setdefault(robot_id, set())
        if resource_id in held:
            return True
        result = self.rule_manager.request_resource(
            robot_id,
            resource_id,
            resource_type,
            self.state.timestep,
            int(self.pickplace_config.get("lock_ttl", 8)),
            reason,
        )
        if result["allowed"]:
            held.add(resource_id)
            return True
        events.append(
            self._pickplace_event(
                robot_id,
                "lock_wait",
                None,
                resource_id=resource_id,
                resource_type=resource_type,
                owner=result.get("owner"),
            )
        )
        return False

    def _release_pickplace_locks(self, robot_id: str) -> None:
        if self.rule_manager is None:
            self.pickplace_locks_by_robot.pop(robot_id, None)
            return
        for resource_id in list(self.pickplace_locks_by_robot.get(robot_id, set())):
            self.rule_manager.release_resource(robot_id, resource_id)
        self.pickplace_locks_by_robot.pop(robot_id, None)

    def _step_toward(self, robot: Any, target: Position) -> Position:
        if robot.position == target:
            return list(robot.position)
        x, y = robot.position
        if x < target[0]:
            return [x + 1, y]
        if x > target[0]:
            return [x - 1, y]
        if y < target[1]:
            return [x, y + 1]
        return [x, y - 1]

    def _detect_naive_pickplace_conflicts(
        self, pickplace_events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.centralized_rule_enabled and self.coordination_config["enable_resource_lock"]:
            return []

        conflicts: list[dict[str, Any]] = []
        pick_by_object: dict[str, list[str]] = defaultdict(list)
        pick_by_zone: dict[str, list[str]] = defaultdict(list)
        place_by_zone: dict[str, list[str]] = defaultdict(list)

        for event in pickplace_events:
            robot_id = str(event.get("robot_id", ""))
            event_type = event.get("event_type")
            if event_type == "pick_attempt":
                object_id = event.get("object_id")
                pickup_zone_id = event.get("pickup_zone_id")
                if object_id:
                    pick_by_object[str(object_id)].append(robot_id)
                if pickup_zone_id:
                    pick_by_zone[str(pickup_zone_id)].append(robot_id)
            elif event_type == "place_attempt":
                drop_zone_id = event.get("drop_zone_id")
                if drop_zone_id:
                    place_by_zone[str(drop_zone_id)].append(robot_id)

        for object_id, robot_ids in sorted(pick_by_object.items()):
            unique = sorted(set(robot_ids))
            if len(unique) > 1:
                conflicts.append(
                    self._pickplace_conflict_event(
                        "duplicate_pick_attempt", unique, object_id=object_id
                    )
                )
                conflicts.append(
                    self._pickplace_conflict_event(
                        "object_conflict", unique, object_id=object_id
                    )
                )

        for zone_id, robot_ids in sorted(pick_by_zone.items()):
            unique = sorted(set(robot_ids))
            if len(unique) > 1:
                conflicts.append(
                    self._pickplace_conflict_event(
                        "pickup_zone_conflict", unique, pickup_zone_id=zone_id
                    )
                )

        for zone_id, robot_ids in sorted(place_by_zone.items()):
            unique = sorted(set(robot_ids))
            if len(unique) > 1:
                conflicts.append(
                    self._pickplace_conflict_event(
                        "drop_zone_conflict", unique, drop_zone_id=zone_id
                    )
                )

        return conflicts

    def _pickplace_conflict_event(
        self, event_type: str, robot_ids: list[str], **extra: Any
    ) -> dict[str, Any]:
        event = {
            "timestep": self.state.timestep,
            "robot_id": ",".join(robot_ids),
            "robot_ids": list(robot_ids),
            "event_type": event_type,
            "reason": "simultaneous_pickplace_resource_use",
        }
        event.update(extra)
        return event

    def _robot_pickplace_done(self, robot_id: str) -> bool:
        if not self.pickplace_enabled:
            return False
        robot = self.state.robots[robot_id]
        return robot.current_task is None and robot.task_state == "done"

    def _object_states(self) -> dict[str, Any]:
        return {
            object_id: obj.to_dict()
            for object_id, obj in sorted(self.state.objects.items())
        }

    def _task_states(self) -> dict[str, Any]:
        return {
            task_id: dict(task)
            for task_id, task in sorted(self.pickplace_tasks.items())
        }

    def _zone_queues(self, resource_type: str) -> dict[str, Any]:
        if self.rule_manager is None:
            return {}
        queues = self.rule_manager.snapshot()["resource_locks"].get("wait_queues", {})
        return {
            resource_id: queue
            for resource_id, queue in queues.items()
            if resource_id.startswith(resource_type)
        }

    def _zone_for_cell(self, zone_group: str, cell: Position | None) -> str:
        if cell is None:
            return f"{zone_group.rstrip('s')}_unknown"
        key = position_key(cell)
        for zone_id, zone in sorted((self.zones.get(zone_group) or {}).items()):
            cells = {position_key(candidate) for candidate in zone.get("cells", [])}
            if key in cells:
                return str(zone_id)
        return f"{zone_group.rstrip('s')}_{key[0]}_{key[1]}"

    def _pickplace_event(
        self,
        robot_id: str,
        event_type: str,
        task: dict[str, Any] | None,
        **extra: Any,
    ) -> dict[str, Any]:
        robot = self.state.robots[robot_id]
        object_id = extra.get("object_id")
        pickup_zone_id = extra.get("pickup_zone_id")
        drop_zone_id = extra.get("drop_zone_id")
        if task is not None:
            object_id = object_id or task.get("object_id")
            obj = self.state.objects.get(str(object_id)) if object_id else None
            pickup_zone_id = pickup_zone_id or self._zone_for_cell(
                "pickup_zones",
                (obj.position if obj is not None else None) or task.get("pickup_position"),
            )
            drop_zone_id = drop_zone_id or self._zone_for_cell(
                "drop_zones",
                (obj.target_position if obj is not None else None) or task.get("drop_position"),
            )
        event = {
            "timestep": self.state.timestep,
            "robot_id": robot_id,
            "event_type": event_type,
            "reason": extra.get("reason", event_type),
            "task_id": task.get("task_id") if task else None,
            "object_id": object_id,
            "pickup_zone_id": pickup_zone_id,
            "drop_zone_id": drop_zone_id,
            "cell": list(robot.position),
            "task_state": robot.task_state,
        }
        event.update(extra)
        return event

    def _find_dag_task_id(
        self,
        task_type: str,
        robot_id: str | None = None,
        object_id: str | None = None,
        explicit_task_id: Any | None = None,
    ) -> str | None:
        if self.task_dag is None:
            return None
        if explicit_task_id is not None:
            task_id = str(explicit_task_id)
            return task_id if task_id in self.task_dag.tasks else None
        matches = self.task_dag.find_tasks(
            task_type=task_type,
            object_id=object_id,
            assigned_robot=robot_id,
        )
        if not matches:
            return None
        ready_matches = [task for task in matches if self.task_dag.is_ready(task.task_id)]
        if ready_matches:
            return sorted(task.task_id for task in ready_matches)[0]
        return sorted(task.task_id for task in matches)[0]

    def _sync_dag_task_assignment(self, task: dict[str, Any], robot_id: str) -> None:
        if self.task_dag is None:
            return
        for dag_task_key in ("pick_task_id", "place_task_id"):
            dag_task_id = task.get(dag_task_key)
            if dag_task_id is None:
                continue
            node = self.task_dag.tasks.get(str(dag_task_id))
            if node is not None:
                node.assigned_robot = robot_id

    def _complete_ready_system_tasks(self) -> None:
        if self.task_dag is None:
            return
        if not self.ignore_task_dag:
            for task in sorted(
                self.task_dag.find_tasks(task_type="recovery_clear_zone"),
                key=lambda node: node.task_id,
            ):
                if self.task_dag.is_ready(task.task_id):
                    self._mark_dag_completed(
                        task.task_id,
                        task.assigned_robot or "system",
                        "recovery_clear_zone",
                    )
        for task in sorted(
            self.task_dag.find_tasks(task_type="final_check"),
            key=lambda node: node.task_id,
        ):
            if self.task_dag.is_ready(task.task_id):
                self._mark_dag_completed(
                    task.task_id,
                    task.assigned_robot or "system",
                    "final_check",
                )

    def _dag_should_wait(self, robot_id: str, task_id: str, action_type: str) -> bool:
        if self.task_dag is None:
            return False
        if self.task_dag.is_ready(task_id):
            return False
        if self.ignore_task_dag:
            self._pending_dag_events.append(
                self._dag_event(robot_id, "dag_violation", task_id, action_type)
            )
            return False
        self._pending_dag_events.append(
            self._dag_event(robot_id, "dependency_wait", task_id, action_type)
        )
        return True

    def _mark_dag_completed(self, task_id: str, robot_id: str, action_type: str) -> None:
        if self.task_dag is None:
            return
        task = self.task_dag.tasks.get(task_id)
        if task is None or task.status == "completed":
            return
        self.task_dag.mark_completed(task_id)
        self._pending_dag_events.append(
            self._dag_event(robot_id, "task_completed", task_id, action_type)
        )

    def _dag_event(
        self, robot_id: str, event_type: str, task_id: str, action_type: str
    ) -> dict[str, Any]:
        return {
            "timestep": self.state.timestep,
            "robot_id": robot_id,
            "event_type": event_type,
            "task_id": task_id,
            "action_type": action_type,
            "ready": self.task_dag.is_ready(task_id) if self.task_dag is not None else True,
            "predecessors": [
                task.task_id
                for task in self.task_dag.get_predecessors(task_id)
            ]
            if self.task_dag is not None and task_id in self.task_dag.tasks
            else [],
        }

    def _dag_task_states(self) -> dict[str, str]:
        return self.task_dag.task_states() if self.task_dag is not None else {}

    def _dag_ready_tasks(self) -> list[str]:
        if self.task_dag is None:
            return []
        return sorted(task.task_id for task in self.task_dag.ready_tasks())

    def _dag_blocked_tasks(self) -> list[str]:
        if self.task_dag is None:
            return []
        return sorted(task.task_id for task in self.task_dag.blocked_tasks())

    def _dag_summary(self) -> dict[str, Any]:
        if self.task_dag is None:
            return {
                "dag_enabled": False,
                "dag_task_count": 0,
                "dag_dependency_count": 0,
                "dag_completed_task_count": 0,
                "dag_ready_task_count": 0,
                "dag_blocked_task_count": 0,
                "critical_path_length": 0,
            }
        return {
            "dag_enabled": True,
            "dag_task_count": len(self.task_dag.tasks),
            "dag_dependency_count": len(self.task_dag.dependencies),
            "dag_completed_task_count": len(self.task_dag.completed_tasks()),
            "dag_ready_task_count": len(self.task_dag.ready_tasks()),
            "dag_blocked_task_count": len(self.task_dag.blocked_tasks()),
            "critical_path_length": self.task_dag.critical_path_length(),
        }

    def _update_metrics_dag_snapshot(self) -> None:
        summary = self._dag_summary()
        self.metrics.dag_enabled = bool(summary["dag_enabled"])
        self.metrics.dag_task_count = int(summary["dag_task_count"])
        self.metrics.dag_dependency_count = int(summary["dag_dependency_count"])
        self.metrics.dag_completed_task_count = int(
            summary["dag_completed_task_count"]
        )
        self.metrics.dag_ready_task_count = int(summary["dag_ready_task_count"])
        self.metrics.dag_blocked_task_count = int(summary["dag_blocked_task_count"])
        self.metrics.critical_path_length = int(summary["critical_path_length"])

    def _success(self) -> bool:
        if self.pickplace_enabled:
            return bool(self.pickplace_tasks) and all(
                task.get("status") == "placed"
                for task in self.pickplace_tasks.values()
            )
        return self.state.all_robots_success()

    def _ensure_recovery_lock(
        self, robot_id: str, reason: str, events: list[dict[str, Any]]
    ) -> bool:
        if not self.centralized_rule_enabled:
            return True
        if not self.coordination_config["enable_recovery_lock"]:
            return True
        if self.rule_manager is None:
            return True

        resource_id = self._recovery_resource_id(self.state.robots[robot_id].position)
        if self.recovery_locks_by_robot.get(robot_id) == resource_id:
            return True
        result = self.rule_manager.request_resource(
            robot_id=robot_id,
            resource_id=resource_id,
            resource_type="recovery_zone",
            timestep=self.state.timestep,
            ttl=int(self.recovery_config.get("recovery_lock_ttl", 5)),
            reason=reason,
        )
        if result["allowed"]:
            self.recovery_locks_by_robot[robot_id] = resource_id
            events.append(
                self._recovery_event(
                    robot_id, "recovery_lock_acquired", reason, resource_id=resource_id
                )
            )
            return True
        events.append(
            self._recovery_event(
                robot_id,
                "recovery_lock_wait",
                reason,
                resource_id=resource_id,
                owner=result.get("owner"),
            )
        )
        return False

    def _release_recovery_lock(self, robot_id: str) -> None:
        resource_id = self.recovery_locks_by_robot.pop(robot_id, None)
        if resource_id and self.rule_manager is not None:
            self.rule_manager.release_resource(robot_id, resource_id)

    def _clear_recovery_zone(self, robot_id: str) -> set[tuple[int, int]]:
        robot = self.state.robots[robot_id]
        radius = int(self.recovery_config.get("recovery_zone_radius", 2))
        center = position_key(robot.position)
        cleared: set[tuple[int, int]] = set()
        candidates = set(self.recovery_blocked_cells) | self.active_temporary_obstacles()
        for cell in candidates:
            if self._manhattan(center, cell) <= radius:
                self.recovery_cleared_cells.add(cell)
                cleared.add(cell)
        return cleared

    def _backup_target(self, robot_id: str) -> Position:
        robot = self.state.robots[robot_id]
        if robot.previous_position is not None:
            target = list(robot.previous_position)
        elif len(robot.path) >= 2:
            target = list(robot.path[-2])
        else:
            target = list(robot.position)
        if self.state.is_blocked(target) or self._is_recovery_blocked(target):
            return list(robot.position)
        return target

    def _detect_naive_recovery_conflicts(
        self, recovery_events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.centralized_rule_enabled and self.coordination_config["enable_recovery_lock"]:
            return []
        by_zone: dict[str, list[str]] = defaultdict(list)
        for event in recovery_events:
            if event.get("event_type") not in {"clear_costmap", "spin"}:
                continue
            resource_id = event.get("resource_id") or self._recovery_resource_id(
                event.get("cell") or event.get("from_cell") or [0, 0]
            )
            by_zone[resource_id].append(event["robot_id"])

        conflicts: list[dict[str, Any]] = []
        for resource_id, robot_ids in sorted(by_zone.items()):
            unique = sorted(set(robot_ids))
            if len(unique) > 1:
                conflicts.append(
                    {
                        "timestep": self.state.timestep,
                        "robot_id": ",".join(unique),
                        "event_type": "recovery_conflict",
                        "reason": "simultaneous_recovery_zone_use",
                        "resource_id": resource_id,
                        "robot_ids": unique,
                    }
                )
        return conflicts

    def _is_recovery_blocked(self, position: Position) -> bool:
        cell = position_key(position)
        blockers = getattr(self, "_step_recovery_blockers", None)
        if blockers is not None:
            return cell in blockers
        return cell in self.current_recovery_blockers()

    def current_recovery_blockers(self) -> set[tuple[int, int]]:
        return {
            cell for cell in self.recovery_blocked_cells if cell not in self.recovery_cleared_cells
        } | self.active_temporary_obstacles()

    def active_temporary_obstacles(self, timestep: int | None = None) -> set[tuple[int, int]]:
        current = self.state.timestep if timestep is None else timestep
        active: set[tuple[int, int]] = set()
        for item in self.recovery_config.get("temporary_obstacles", []):
            cell = position_key(item["cell"])
            if cell in self.recovery_cleared_cells:
                continue
            if int(item.get("appear_at", 0)) <= current < int(item.get("disappear_at", 10**9)):
                active.add(cell)
        return active

    def _recovery_resource_id(self, cell: Position | tuple[int, int]) -> str:
        radius = max(1, int(self.recovery_config.get("recovery_zone_radius", 2)))
        x, y = position_key(cell)
        return f"recovery_zone_{x // radius}_{y // radius}"

    def _recovery_states(self) -> dict[str, str]:
        return {
            robot_id: robot.recovery_state
            for robot_id, robot in sorted(self.state.robots.items())
        }

    def _recovery_queues(self) -> dict[str, Any]:
        if self.rule_manager is None:
            return {}
        return self.rule_manager.snapshot()["resource_locks"].get("wait_queues", {})

    def _recovery_event(
        self,
        robot_id: str,
        event_type: str,
        reason: str,
        **extra: Any,
    ) -> dict[str, Any]:
        robot = self.state.robots[robot_id]
        event = {
            "timestep": self.state.timestep,
            "robot_id": robot_id,
            "event_type": event_type,
            "reason": reason,
            "recovery_state": robot.recovery_state,
            "cell": list(robot.position),
            "resource_id": self._recovery_resource_id(robot.position),
        }
        event.update(extra)
        return event

    @staticmethod
    def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _apply_centralized_rules(
        self,
        proposals: dict[str, Position],
        action_records: dict[str, dict[str, Any]],
    ) -> tuple[
        dict[str, Position],
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
        bool,
    ]:
        if self.rule_manager is None:
            return proposals, [], {"cells": [], "edges": []}, {"locks": {}, "wait_queues": {}}, False

        event_start = len(self.rule_manager.rule_events)
        self.rule_manager.tick(self.state.timestep)

        final_proposals: dict[str, Position] = {}
        for robot_id in sorted(proposals):
            robot = self.state.robots[robot_id]
            target = proposals[robot_id]
            result = self.rule_manager.request_move(
                robot_id=robot_id,
                from_cell=robot.position,
                to_cell=target,
                timestep=self.state.timestep,
            )
            action_records[robot_id]["rule_allowed"] = result["allowed"]
            action_records[robot_id]["rule_reason"] = result["reason"]
            if result["allowed"]:
                final_proposals[robot_id] = target
            else:
                action_records[robot_id]["rule_rejected"] = True
                final_proposals[robot_id] = list(robot.position)

        deadlock_detected = self.rule_manager.detect_deadlock(
            window_size=max(1, len(self.state.robots))
        )
        events = self.rule_manager.events_since(event_start)
        snapshot = self.rule_manager.snapshot()
        return (
            final_proposals,
            events,
            snapshot["reservations"],
            snapshot["resource_locks"],
            deadlock_detected,
        )

    def _apply_prioritized_planning(
        self,
        proposals: dict[str, Position],
        action_records: dict[str, dict[str, Any]],
    ) -> dict[str, Position]:
        final_proposals: dict[str, Position] = {}
        accepted_from: dict[str, tuple[int, int]] = {}
        accepted_to: dict[str, tuple[int, int]] = {}

        for robot_id in sorted(proposals):
            robot = self.state.robots[robot_id]
            from_cell = position_key(robot.position)
            target = position_key(proposals[robot_id])
            blocked = False
            for prior_id, prior_target in accepted_to.items():
                prior_from = accepted_from[prior_id]
                if target == prior_target:
                    blocked = True
                    break
                if target == prior_from and from_cell == prior_target:
                    blocked = True
                    break
            if blocked:
                final_proposals[robot_id] = list(robot.position)
                action_records[robot_id]["prioritized_wait"] = True
            else:
                final_proposals[robot_id] = list(proposals[robot_id])
            accepted_from[robot_id] = from_cell
            accepted_to[robot_id] = position_key(final_proposals[robot_id])

        return final_proposals

    def _pickplace_path_locking_enabled(self) -> bool:
        return (
            self.pickplace_enabled
            and self.centralized_rule_enabled
            and self.rule_manager is not None
            and (
                self.coordination_config["enable_approach_zone_lock"]
                or self.coordination_config["enable_corridor_lock"]
            )
        )

    def _apply_pickplace_path_locks(
        self,
        proposals: dict[str, Position],
        action_records: dict[str, dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> dict[str, Position]:
        final_proposals = {robot_id: list(target) for robot_id, target in proposals.items()}
        for robot_id in sorted(proposals):
            target = proposals[robot_id]
            required = self._path_resources_for_robot_target(
                robot_id, target, action_records.get(robot_id, {})
            )
            self._clear_unneeded_path_waits(robot_id, required)
            denied = self._occupied_approach_target_denial(
                robot_id, target, required, proposals
            )
            if denied is None:
                denied = self._first_denied_path_resource(robot_id, required)
            if denied is not None:
                denied = self._handle_denied_path_resource(robot_id, denied, events)
            if denied is not None:
                event_type = (
                    "approach_zone_wait"
                    if denied["resource_type"] == "approach_zone"
                    else "corridor_wait"
                )
                events.append(
                    self._pickplace_event(
                        robot_id,
                        event_type,
                        None,
                        resource_id=denied["resource_id"],
                        resource_type=denied["resource_type"],
                        owner=denied.get("owner"),
                        duration=1,
                    )
                )
                if self.rule_manager is not None:
                    self.rule_manager.make_wait_event(
                        robot_id=robot_id,
                        timestep=self.state.timestep,
                        event_type=event_type,
                        action_id=str(
                            action_records.get(robot_id, {}).get("action_type", event_type)
                        ),
                        safety_layer=(
                            "manipulation_area"
                            if denied["resource_type"] == "approach_zone"
                            else "semantic_resource"
                        ),
                        conflict_type=(
                            "approach_zone_denied"
                            if denied["resource_type"] == "approach_zone"
                            else f"{denied['resource_type']}_denied"
                        ),
                        reason=denied.get("reason", event_type),
                        resource_id=denied["resource_id"],
                        resource_type=denied["resource_type"],
                        owner=denied.get("owner"),
                        details={"source_event_type": event_type},
                    )
                action_records[robot_id]["rule_rejected"] = True
                action_records[robot_id]["path_lock_rejected"] = True
                final_proposals[robot_id] = list(self.state.robots[robot_id].position)
                continue
            for resource in required:
                if resource.get("passive"):
                    continue
                self._acquire_path_lock(robot_id, resource)
        return final_proposals

    def _path_resources_for_robot_target(
        self,
        robot_id: str,
        cell: Position,
        action_record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        if self.coordination_config["enable_approach_zone_lock"]:
            resources.extend(
                self._approach_resources_for_robot_target(robot_id, cell, action_record)
            )
            resources.extend(self._held_approach_blocks_for_cell(robot_id, cell, resources))
        if self.coordination_config["enable_corridor_lock"]:
            resources.extend(self._corridor_resources_for_cell(cell))
        return resources

    def _approach_resources_for_robot_target(
        self,
        robot_id: str,
        cell: Position,
        action_record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        robot = self.state.robots[robot_id]
        task_id = robot.current_task
        task = self.pickplace_tasks.get(task_id or "")
        if task is None:
            return []
        if task.get("assigned_robot") != robot_id:
            return []

        action_type = str(action_record.get("action_type", ""))
        target_key = position_key(cell)
        radius = max(0, int(self.coordination_config.get("approach_radius", 1)))
        resources: list[dict[str, Any]] = []

        if action_type in {"navigate_to_pickup", "pick"} and task.get("status") not in {
            "picked",
            "placed",
            "failed",
        }:
            pickup = position_key(task["pickup_position"])
            if self._manhattan(target_key, pickup) <= radius:
                resources.append(
                    {
                        "resource_id": self._approach_resource_id("pickup", pickup),
                        "resource_type": "approach_zone",
                        "reason": "pickup_approach_zone",
                    }
                )

        if action_type in {"navigate_to_drop", "place"} and task.get("status") == "picked":
            drop = position_key(task["drop_position"])
            if self._manhattan(target_key, drop) <= radius:
                resources.append(
                    {
                        "resource_id": self._approach_resource_id("drop", drop),
                        "resource_type": "approach_zone",
                        "reason": "drop_approach_zone",
                    }
                )

        return resources

    def _held_approach_blocks_for_cell(
        self,
        robot_id: str,
        cell: Position,
        active_resources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.rule_manager is None:
            return []
        active_ids = {resource["resource_id"] for resource in active_resources}
        key = position_key(cell)
        resources: list[dict[str, Any]] = []
        for resource_id, cells in self._approach_zone_cells_by_resource().items():
            if resource_id in active_ids or key not in cells:
                continue
            owner = self.rule_manager.resource_manager.get_owner(resource_id)
            if owner is None or owner == robot_id:
                continue
            resources.append(
                {
                    "resource_id": resource_id,
                    "resource_type": "approach_zone",
                    "reason": "approach_zone_occupied",
                    "passive": True,
                }
            )
        return resources

    def _corridor_resources_for_cell(self, cell: Position) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        key = position_key(cell)
        for resource_id, cells in self._corridor_cells_by_resource().items():
            if key in cells:
                resources.append(
                    {
                        "resource_id": resource_id,
                        "resource_type": "corridor",
                        "reason": "corridor",
                    }
                )
        return resources

    def _clear_unneeded_path_waits(
        self, robot_id: str, required: list[dict[str, Any]]
    ) -> None:
        required_ids = {resource["resource_id"] for resource in required}
        for key in list(self.pickplace_path_wait_ticks):
            waiting_robot, resource_id = key
            if waiting_robot == robot_id and resource_id not in required_ids:
                self.pickplace_path_wait_ticks.pop(key, None)

    def _occupied_approach_target_denial(
        self,
        robot_id: str,
        target: Position,
        required: list[dict[str, Any]],
        proposals: dict[str, Position],
    ) -> dict[str, Any] | None:
        active_approach = [
            resource
            for resource in required
            if resource["resource_type"] == "approach_zone" and not resource.get("passive")
        ]
        if not active_approach:
            return None
        target_key = position_key(target)
        for other_id, other in self.state.robots.items():
            if other_id == robot_id or position_key(other.position) != target_key:
                continue
            other_target = position_key(proposals.get(other_id, other.position))
            if other_target == target_key:
                return {
                    **active_approach[0],
                    "owner": other_id,
                    "occupancy": True,
                }
        return None

    def _first_denied_path_resource(
        self, robot_id: str, resources: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if self.rule_manager is None:
            return None
        for resource in resources:
            owner = self.rule_manager.resource_manager.get_owner(resource["resource_id"])
            if owner is not None and owner != robot_id:
                return {**resource, "owner": owner}
        return None

    def _handle_denied_path_resource(
        self,
        robot_id: str,
        resource: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if resource["resource_type"] != "approach_zone":
            return resource
        if resource.get("passive") or resource.get("occupancy"):
            return resource

        resource_id = resource["resource_id"]
        wait_key = (robot_id, resource_id)
        wait_ticks = self.pickplace_path_wait_ticks.get(wait_key, 0) + 1
        self.pickplace_path_wait_ticks[wait_key] = wait_ticks

        reassignment_wait_ticks = max(
            1, int(self.coordination_config.get("reassignment_wait_ticks", 8))
        )
        if wait_ticks >= reassignment_wait_ticks and self._maybe_reassign_approach_wait(
            robot_id, resource_id, events
        ):
            self.pickplace_path_wait_ticks.pop(wait_key, None)
            return resource

        max_wait_ticks = max(1, int(self.coordination_config.get("max_wait_ticks", 3)))
        if wait_ticks >= max_wait_ticks:
            owner = resource.get("owner")
            events.append(
                self._pickplace_event(
                    robot_id,
                    "approach_lock_starvation",
                    None,
                    resource_id=resource_id,
                    owner=owner,
                    wait_ticks=wait_ticks,
                )
            )
            if owner:
                self._release_path_lock_resource(owner, resource_id, events)
                return None
        return resource

    def _maybe_reassign_approach_wait(
        self, robot_id: str, resource_id: str, events: list[dict[str, Any]]
    ) -> bool:
        if not self.pickplace_config.get("allow_reassignment", False):
            return False
        robot = self.state.robots[robot_id]
        task_id = robot.current_task
        task = self.pickplace_tasks.get(task_id or "")
        if task is None:
            return False
        reassignment_key = (task["task_id"], resource_id)
        if reassignment_key in self.pickplace_path_reassignments:
            return False
        self.pickplace_path_reassignments.add(reassignment_key)

        if robot.carrying_object or task.get("status") == "picked":
            events.append(
                self._pickplace_event(
                    robot_id,
                    "approach_lock_reassignment",
                    task,
                    resource_id=resource_id,
                    mode="reroute",
                )
            )
            return True

        old_robot = self.state.robots[robot_id]
        robots = {rid: {"start": r.position} for rid, r in self.state.robots.items()}
        reassigned = reassign_failed_task(task, robots, robot_id)
        task.update(reassigned)
        new_robot = task.get("assigned_robot")
        if task["task_id"] in old_robot.assigned_tasks:
            old_robot.assigned_tasks.remove(task["task_id"])
        if new_robot and task["task_id"] not in self.state.robots[new_robot].assigned_tasks:
            self.state.robots[new_robot].assigned_tasks.append(task["task_id"])
        if old_robot.current_task == task["task_id"]:
            old_robot.current_task = None
            old_robot.task_state = "idle"
        if new_robot and self.state.robots[new_robot].current_task is None:
            self.state.robots[new_robot].current_task = task["task_id"]
            self.state.robots[new_robot].task_state = "navigate_to_pickup"
            self.state.robots[new_robot].status = "running"
        self._release_pickplace_path_locks(robot_id, events)
        self._advance_robot_task(robot_id)
        events.append(
            self._pickplace_event(
                robot_id,
                "approach_lock_reassignment",
                task,
                resource_id=resource_id,
                reassigned_to=new_robot,
                mode="task_reassignment",
            )
        )
        return True

    def _acquire_path_lock(self, robot_id: str, resource: dict[str, Any]) -> None:
        if self.rule_manager is None:
            return
        resource_id = resource["resource_id"]
        if resource_id in self.pickplace_path_locks_by_robot.get(robot_id, set()):
            return
        result = self.rule_manager.request_resource(
            robot_id=robot_id,
            resource_id=resource_id,
            resource_type=resource["resource_type"],
            timestep=self.state.timestep,
            ttl=int(self.pickplace_config.get("lock_ttl", 8)),
            reason=resource["reason"],
        )
        if result["allowed"]:
            self.pickplace_path_locks_by_robot.setdefault(robot_id, set()).add(resource_id)
            self.pickplace_path_lock_acquired_at.setdefault(robot_id, {})[
                resource_id
            ] = self.state.timestep
            self.pickplace_path_wait_ticks.pop((robot_id, resource_id), None)

    def _release_pickplace_path_locks(
        self,
        robot_id: str,
        events: list[dict[str, Any]] | None = None,
        resource_type: str | None = None,
    ) -> None:
        for resource_id in list(self.pickplace_path_locks_by_robot.get(robot_id, set())):
            if resource_type is not None and self._path_lock_type(resource_id) != resource_type:
                continue
            self._release_path_lock_resource(robot_id, resource_id, events)

    def _release_matching_approach_lock(
        self,
        robot_id: str,
        stage: str,
        task: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> None:
        position_field = "pickup_position" if stage == "pickup" else "drop_position"
        resource_id = self._approach_resource_id(stage, position_key(task[position_field]))
        self._release_path_lock_resource(robot_id, resource_id, events)

    def _release_path_lock_resource(
        self,
        robot_id: str,
        resource_id: str,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        resource_ids = self.pickplace_path_locks_by_robot.get(robot_id)
        if not resource_ids or resource_id not in resource_ids:
            return
        if self.rule_manager is not None:
            self.rule_manager.release_resource(robot_id, resource_id)
        resource_ids.remove(resource_id)
        acquired_at = self.pickplace_path_lock_acquired_at.get(robot_id, {}).pop(
            resource_id, self.state.timestep
        )
        if not self.pickplace_path_lock_acquired_at.get(robot_id):
            self.pickplace_path_lock_acquired_at.pop(robot_id, None)
        if not resource_ids:
            self.pickplace_path_locks_by_robot.pop(robot_id, None)
        if self._path_lock_type(resource_id) == "approach_zone" and events is not None:
            events.append(
                self._pickplace_event(
                    robot_id,
                    "approach_lock_release",
                    None,
                    resource_id=resource_id,
                    hold_time=max(0, self.state.timestep - int(acquired_at)),
                )
            )

    def _release_inactive_pickplace_path_locks(
        self, events: list[dict[str, Any]] | None = None
    ) -> None:
        if self.rule_manager is None:
            self.pickplace_path_locks_by_robot.clear()
            self.pickplace_path_lock_acquired_at.clear()
            return
        for robot_id, resource_ids in list(self.pickplace_path_locks_by_robot.items()):
            robot = self.state.robots[robot_id]
            for resource_id in list(resource_ids):
                cells = self._path_lock_cells(resource_id)
                if position_key(robot.position) not in cells:
                    self._release_path_lock_resource(robot_id, resource_id, events)

    def _path_resources_for_cell(self, cell: Position) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        key = position_key(cell)
        if self.coordination_config["enable_approach_zone_lock"]:
            for resource_id, cells in self._approach_zone_cells_by_resource().items():
                if key in cells:
                    resources.append(
                        {
                            "resource_id": resource_id,
                            "resource_type": "approach_zone",
                            "reason": "approach_zone",
                        }
                    )
        if self.coordination_config["enable_corridor_lock"]:
            for resource_id, cells in self._corridor_cells_by_resource().items():
                if key in cells:
                    resources.append(
                        {
                            "resource_id": resource_id,
                            "resource_type": "corridor",
                            "reason": "corridor",
                        }
                    )
        return resources

    def _path_lock_type(self, resource_id: str) -> str:
        if resource_id.startswith(("pickup_approach_zone_", "drop_approach_zone_")):
            return "approach_zone"
        if resource_id.startswith("corridor_"):
            return "corridor"
        return "unknown"

    def _path_lock_cells(self, resource_id: str) -> set[tuple[int, int]]:
        cells = self._approach_zone_cells_by_resource().get(resource_id)
        if cells is not None:
            return cells
        return self._corridor_cells_by_resource().get(resource_id, set())

    def _approach_zone_cells_by_resource(self) -> dict[str, set[tuple[int, int]]]:
        radius = max(0, int(self.coordination_config.get("approach_radius", 1)))
        zones: dict[str, set[tuple[int, int]]] = {}
        for task in self.pickplace_tasks.values():
            for stage, field in (
                ("pickup", "pickup_position"),
                ("drop", "drop_position"),
            ):
                position = task.get(field)
                if position is None:
                    continue
                center = position_key(position)
                zones[self._approach_resource_id(stage, center)] = self._manhattan_zone(
                    list(center), radius
                )
        return zones

    @staticmethod
    def _approach_resource_id(stage: str, center: tuple[int, int]) -> str:
        return f"{stage}_approach_zone_{center[0]}_{center[1]}"

    def _corridor_cells_by_resource(self) -> dict[str, set[tuple[int, int]]]:
        corridors: dict[str, set[tuple[int, int]]] = {}
        for index, raw_cells in enumerate(self.coordination_config.get("corridor_cells", [])):
            corridors[f"corridor_{index}"] = {
                position_key(cell) for cell in raw_cells
            }
        return corridors

    def _manhattan_zone(self, center: Position, radius: int) -> set[tuple[int, int]]:
        center_key = position_key(center)
        cells: set[tuple[int, int]] = set()
        for x in range(self.state.grid_width):
            for y in range(self.state.grid_height):
                if self._manhattan(center_key, (x, y)) <= radius:
                    cells.add((x, y))
        return cells

    def _detect_collisions(self, proposals: dict[str, Position]) -> list[dict[str, Any]]:
        collisions: list[dict[str, Any]] = []

        for robot_id, target in proposals.items():
            if self.state.is_blocked(target) or (
                self.recovery_enabled and self._is_recovery_blocked(target)
            ):
                collisions.append(
                    {
                        "type": "obstacle",
                        "robot_ids": [robot_id],
                        "position": list(target),
                    }
                )

        proposed_by_cell: dict[tuple[int, int], list[str]] = defaultdict(list)
        for robot_id, target in proposals.items():
            proposed_by_cell[position_key(target)].append(robot_id)

        for cell, robot_ids in sorted(proposed_by_cell.items()):
            if len(robot_ids) > 1:
                collisions.append(
                    {
                        "type": "vertex",
                        "robot_ids": sorted(robot_ids),
                        "position": list(cell),
                    }
                )

        robot_ids = sorted(proposals)
        for index, robot_id in enumerate(robot_ids):
            robot = self.state.robots[robot_id]
            from_cell = position_key(robot.position)
            to_cell = position_key(proposals[robot_id])
            if from_cell == to_cell:
                continue
            for other_id in robot_ids[index + 1 :]:
                other = self.state.robots[other_id]
                other_from = position_key(other.position)
                other_to = position_key(proposals[other_id])
                if from_cell == other_to and to_cell == other_from:
                    collisions.append(
                        {
                            "type": "edge",
                            "robot_ids": sorted([robot_id, other_id]),
                            "from_cell": list(from_cell),
                            "to_cell": list(to_cell),
                        }
                    )

        return collisions
