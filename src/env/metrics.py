"""Metrics and JSON trace recording for multi-robot simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsRecorder:
    success: bool = False
    makespan: int = 0
    total_robot_steps: int = 0
    collision_count: int = 0
    vertex_conflict_count: int = 0
    edge_conflict_count: int = 0
    motion_wait_count: int = 0
    rule_prevented_motion_conflict_count: int = 0
    rule_rejection_count: int = 0
    resource_conflict_count: int = 0
    resource_request_denied_count: int = 0
    rule_prevented_resource_conflict_count: int = 0
    object_lock_denied_count: int = 0
    pickup_zone_denied_count: int = 0
    drop_zone_denied_count: int = 0
    recovery_zone_denied_count: int = 0
    deadlock_count: int = 0
    wait_count: int = 0
    recovery_attempts: int = 0
    recovery_conflict_count: int = 0
    recovery_lock_wait_count: int = 0
    recovery_wait_time: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    stuck_count: int = 0
    temporary_obstacle_hits: int = 0
    pick_attempts: int = 0
    pick_success_count: int = 0
    pick_failure_count: int = 0
    place_attempts: int = 0
    place_success_count: int = 0
    place_failure_count: int = 0
    object_conflict_count: int = 0
    pickup_zone_conflict_count: int = 0
    drop_zone_conflict_count: int = 0
    duplicate_pick_attempt_count: int = 0
    approach_zone_denied_count: int = 0
    approach_zone_wait_count: int = 0
    rule_prevented_approach_conflict_count: int = 0
    approach_lock_hold_time: int = 0
    approach_lock_starvation_count: int = 0
    approach_lock_reassignment_count: int = 0
    corridor_wait_count: int = 0
    lock_wait_count: int = 0
    lock_wait_time: int = 0
    reassignment_count: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    all_objects_placed: bool = False
    centralized_rule_enabled: bool = False
    timeout: bool = False
    dag_enabled: bool = False
    dag_task_count: int = 0
    dag_dependency_count: int = 0
    dag_completed_task_count: int = 0
    dag_ready_task_count: int = 0
    dag_blocked_task_count: int = 0
    dependency_wait_count: int = 0
    dag_violation_count: int = 0
    critical_path_length: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.success = False
        self.makespan = 0
        self.total_robot_steps = 0
        self.collision_count = 0
        self.vertex_conflict_count = 0
        self.edge_conflict_count = 0
        self.motion_wait_count = 0
        self.rule_prevented_motion_conflict_count = 0
        self.rule_rejection_count = 0
        self.resource_conflict_count = 0
        self.resource_request_denied_count = 0
        self.rule_prevented_resource_conflict_count = 0
        self.object_lock_denied_count = 0
        self.pickup_zone_denied_count = 0
        self.drop_zone_denied_count = 0
        self.recovery_zone_denied_count = 0
        self.deadlock_count = 0
        self.wait_count = 0
        self.recovery_attempts = 0
        self.recovery_conflict_count = 0
        self.recovery_lock_wait_count = 0
        self.recovery_wait_time = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.stuck_count = 0
        self.temporary_obstacle_hits = 0
        self.pick_attempts = 0
        self.pick_success_count = 0
        self.pick_failure_count = 0
        self.place_attempts = 0
        self.place_success_count = 0
        self.place_failure_count = 0
        self.object_conflict_count = 0
        self.pickup_zone_conflict_count = 0
        self.drop_zone_conflict_count = 0
        self.duplicate_pick_attempt_count = 0
        self.approach_zone_denied_count = 0
        self.approach_zone_wait_count = 0
        self.rule_prevented_approach_conflict_count = 0
        self.approach_lock_hold_time = 0
        self.approach_lock_starvation_count = 0
        self.approach_lock_reassignment_count = 0
        self.corridor_wait_count = 0
        self.lock_wait_count = 0
        self.lock_wait_time = 0
        self.reassignment_count = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.all_objects_placed = False
        self.timeout = False
        self.dag_enabled = False
        self.dag_task_count = 0
        self.dag_dependency_count = 0
        self.dag_completed_task_count = 0
        self.dag_ready_task_count = 0
        self.dag_blocked_task_count = 0
        self.dependency_wait_count = 0
        self.dag_violation_count = 0
        self.critical_path_length = 0
        self.trace.clear()

    def record_step(
        self,
        timestep: int,
        robot_positions: dict[str, list[int]],
        actions: dict[str, dict[str, Any]],
        collisions: list[dict[str, Any]],
        status: dict[str, str],
        moved_robot_ids: list[str],
        rule_events: list[dict[str, Any]] | None = None,
        reservations: dict[str, Any] | None = None,
        resource_locks: dict[str, Any] | None = None,
        deadlock_detected: bool = False,
        recovery_events: list[dict[str, Any]] | None = None,
        recovery_states: dict[str, str] | None = None,
        recovery_queues: dict[str, Any] | None = None,
        temporary_obstacles: list[list[int]] | None = None,
        pickplace_events: list[dict[str, Any]] | None = None,
        object_states: dict[str, Any] | None = None,
        task_states: dict[str, Any] | None = None,
        pickup_zone_queues: dict[str, Any] | None = None,
        drop_zone_queues: dict[str, Any] | None = None,
        dag_task_states: dict[str, str] | None = None,
        dag_ready_tasks: list[str] | None = None,
        dag_blocked_tasks: list[str] | None = None,
        dag_events: list[dict[str, Any]] | None = None,
        dag_summary: dict[str, Any] | None = None,
    ) -> None:
        self.collision_count += len(collisions)
        self.vertex_conflict_count += sum(
            1 for collision in collisions if collision.get("type") in {"vertex", "obstacle"}
        )
        self.edge_conflict_count += sum(
            1 for collision in collisions if collision.get("type") == "edge"
        )
        for event in rule_events or []:
            if event.get("allowed") is False:
                self.rule_rejection_count += 1
                if event.get("safety_layer") == "motion":
                    self.motion_wait_count += 1
                    self.rule_prevented_motion_conflict_count += 1
                if (
                    event.get("conflict_type") == "resource"
                    or event.get("safety_layer") == "semantic_resource"
                ):
                    self.resource_request_denied_count += 1
                    self._record_resource_denial(event.get("conflict_type"))
                    if self.centralized_rule_enabled:
                        self.rule_prevented_resource_conflict_count += 1
                    else:
                        self.resource_conflict_count += 1
                if event.get("event_type") == "move_request":
                    self.wait_count += 1
        if deadlock_detected:
            self.deadlock_count += 1
        for event in recovery_events or []:
            event_type = event.get("event_type")
            if event_type == "recovery_start":
                self.recovery_attempts += 1
            elif event_type == "recovery_conflict":
                self.recovery_conflict_count += 1
            elif event_type == "recovery_lock_wait":
                self.recovery_lock_wait_count += 1
                self.recovery_zone_denied_count += 1
                self.lock_wait_count += 1
                self.lock_wait_time += int(event.get("duration", 1))
                self.resource_request_denied_count += 1
                if self.centralized_rule_enabled:
                    self.rule_prevented_resource_conflict_count += 1
                else:
                    self.resource_conflict_count += 1
                self.wait_count += 1
            elif event_type == "recovery_wait":
                self.recovery_wait_time += int(event.get("duration", 1))
            elif event_type == "recovery_success":
                self.successful_recoveries += 1
            elif event_type == "recovery_failure":
                self.failed_recoveries += 1
            elif event_type == "stuck":
                self.stuck_count += 1
            elif event_type == "temporary_obstacle_hit":
                self.temporary_obstacle_hits += 1
        for event in pickplace_events or []:
            event_type = event.get("event_type")
            if event_type == "pick_attempt":
                self.pick_attempts += 1
            elif event_type == "pick_success":
                self.pick_success_count += 1
            elif event_type == "pick_failure":
                self.pick_failure_count += 1
            elif event_type == "place_attempt":
                self.place_attempts += 1
            elif event_type == "place_success":
                self.place_success_count += 1
            elif event_type == "place_failure":
                self.place_failure_count += 1
            elif event_type == "object_conflict":
                self.object_conflict_count += 1
                self.resource_conflict_count += 1
            elif event_type == "pickup_zone_conflict":
                self.pickup_zone_conflict_count += 1
                self.resource_conflict_count += 1
            elif event_type == "drop_zone_conflict":
                self.drop_zone_conflict_count += 1
                self.resource_conflict_count += 1
            elif event_type == "duplicate_pick_attempt":
                self.duplicate_pick_attempt_count += 1
                self.resource_conflict_count += 1
            elif event_type == "approach_zone_wait":
                self.approach_zone_denied_count += 1
                self.approach_zone_wait_count += 1
                self.rule_prevented_approach_conflict_count += 1
                self.lock_wait_count += 1
                self.lock_wait_time += int(event.get("duration", 1))
                self.resource_request_denied_count += 1
                self.wait_count += 1
            elif event_type == "approach_lock_release":
                self.approach_lock_hold_time += int(event.get("hold_time", 0))
            elif event_type == "approach_lock_starvation":
                self.approach_lock_starvation_count += 1
            elif event_type == "approach_lock_reassignment":
                self.approach_lock_reassignment_count += 1
            elif event_type == "corridor_wait":
                self.corridor_wait_count += 1
                self.lock_wait_count += 1
                self.lock_wait_time += int(event.get("duration", 1))
                self.resource_request_denied_count += 1
                self.wait_count += 1
            elif event_type == "lock_wait":
                self.lock_wait_count += 1
                self.lock_wait_time += int(event.get("duration", 1))
                self.resource_request_denied_count += 1
                self._record_resource_type_denial(str(event.get("resource_type", "")))
                if self.centralized_rule_enabled:
                    self.rule_prevented_resource_conflict_count += 1
                else:
                    self.resource_conflict_count += 1
                self.wait_count += 1
            elif event_type == "reassignment":
                self.reassignment_count += 1
        for event in dag_events or []:
            event_type = event.get("event_type")
            if event_type == "dependency_wait":
                self.dependency_wait_count += 1
            elif event_type == "dag_violation":
                self.dag_violation_count += 1
        if dag_summary is not None:
            self.dag_enabled = bool(dag_summary.get("dag_enabled", False))
            self.dag_task_count = int(dag_summary.get("dag_task_count", 0) or 0)
            self.dag_dependency_count = int(
                dag_summary.get("dag_dependency_count", 0) or 0
            )
            self.dag_completed_task_count = int(
                dag_summary.get("dag_completed_task_count", 0) or 0
            )
            self.dag_ready_task_count = int(
                dag_summary.get("dag_ready_task_count", 0) or 0
            )
            self.dag_blocked_task_count = int(
                dag_summary.get("dag_blocked_task_count", 0) or 0
            )
            self.critical_path_length = int(
                dag_summary.get("critical_path_length", 0) or 0
            )
        if task_states is not None:
            self.tasks_completed = sum(
                1 for task in task_states.values() if task.get("status") == "placed"
            )
            self.tasks_failed = sum(
                1 for task in task_states.values() if task.get("status") == "failed"
            )
            self.all_objects_placed = bool(task_states) and all(
                task.get("status") == "placed" for task in task_states.values()
            )
        self.total_robot_steps += len(moved_robot_ids)
        self.makespan = timestep
        self.trace.append(
            {
                "timestep": timestep,
                "robot_positions": robot_positions,
                "actions": actions,
                "collisions": collisions,
                "rule_events": rule_events or [],
                "reservations": reservations or {"cells": [], "edges": []},
                "resource_locks": resource_locks or {"locks": {}, "wait_queues": {}},
                "recovery_events": recovery_events or [],
                "recovery_states": recovery_states or {},
                "recovery_queues": recovery_queues or {},
                "temporary_obstacles": temporary_obstacles or [],
                "object_states": object_states or {},
                "task_states": task_states or {},
                "pickup_zone_queues": pickup_zone_queues or {},
                "drop_zone_queues": drop_zone_queues or {},
                "pickplace_events": pickplace_events or [],
                "dag_task_states": dag_task_states or {},
                "dag_ready_tasks": dag_ready_tasks or [],
                "dag_blocked_tasks": dag_blocked_tasks or [],
                "dag_events": dag_events or [],
                "status": status,
            }
        )

    def finish(self, success: bool, timeout: bool) -> None:
        self.success = success
        self.timeout = timeout

    def _record_resource_denial(self, conflict_type: Any) -> None:
        mapping = {
            "object_lock_denied": "object",
            "pickup_zone_denied": "pickup_zone",
            "drop_zone_denied": "drop_zone",
            "recovery_zone_denied": "recovery_zone",
        }
        self._record_resource_type_denial(mapping.get(str(conflict_type), ""))

    def _record_resource_type_denial(self, resource_type: str) -> None:
        if resource_type == "object":
            self.object_lock_denied_count += 1
        elif resource_type == "pickup_zone":
            self.pickup_zone_denied_count += 1
        elif resource_type == "drop_zone":
            self.drop_zone_denied_count += 1
        elif resource_type == "recovery_zone":
            self.recovery_zone_denied_count += 1

    def actual_collision_count(self) -> int:
        return (
            self.actual_vertex_conflict_count()
            + self.actual_edge_swap_conflict_count()
        )

    def actual_vertex_conflict_count(self) -> int:
        count = 0
        for frame in self.trace:
            positions = frame.get("robot_positions") or {}
            robots_by_cell: dict[tuple[int, int], list[str]] = {}
            if not isinstance(positions, dict):
                continue
            for robot_id, value in positions.items():
                cell = parse_cell(value)
                if cell is None:
                    continue
                robots_by_cell.setdefault(cell, []).append(str(robot_id))
            count += sum(1 for robots in robots_by_cell.values() if len(robots) >= 2)
        return count

    def actual_edge_swap_conflict_count(self) -> int:
        count = 0
        for previous, current in zip(self.trace, self.trace[1:]):
            previous_positions = parse_positions(previous.get("robot_positions") or {})
            current_positions = parse_positions(current.get("robot_positions") or {})
            shared_robots = sorted(set(previous_positions) & set(current_positions))
            for left_index, robot_i in enumerate(shared_robots):
                for robot_j in shared_robots[left_index + 1 :]:
                    i_from = previous_positions[robot_i]
                    i_to = current_positions[robot_i]
                    j_from = previous_positions[robot_j]
                    j_to = current_positions[robot_j]
                    if i_from == j_to and j_from == i_to and i_from != i_to:
                        count += 1
        return count

    def summary(self) -> dict[str, Any]:
        audited_vertex_conflict_count = self.actual_vertex_conflict_count()
        audited_edge_swap_conflict_count = self.actual_edge_swap_conflict_count()
        actual_collision_count = (
            audited_vertex_conflict_count + audited_edge_swap_conflict_count
        )
        return {
            "success": self.success,
            "makespan": self.makespan,
            "timeout": self.timeout,
            "total_robot_steps": self.total_robot_steps,
            "collision_count": self.collision_count,
            "collision_event_count_legacy": self.collision_count,
            "collision_event_count": self.collision_count,
            "actual_collision_count": actual_collision_count,
            "vertex_conflict_count": self.vertex_conflict_count,
            "vertex_conflict_event_count": self.vertex_conflict_count,
            "edge_conflict_count": self.edge_conflict_count,
            "edge_conflict_event_count": self.edge_conflict_count,
            "audited_vertex_conflict_count": audited_vertex_conflict_count,
            "audited_edge_swap_conflict_count": audited_edge_swap_conflict_count,
            "motion_wait_count": self.motion_wait_count,
            "rule_prevented_motion_conflict_count": (
                self.rule_prevented_motion_conflict_count
            ),
            "rule_rejection_count": self.rule_rejection_count,
            "resource_conflict_count": self.resource_conflict_count,
            "resource_request_denied_count": self.resource_request_denied_count,
            "rule_prevented_resource_conflict_count": (
                self.rule_prevented_resource_conflict_count
            ),
            "object_lock_denied_count": self.object_lock_denied_count,
            "pickup_zone_denied_count": self.pickup_zone_denied_count,
            "drop_zone_denied_count": self.drop_zone_denied_count,
            "recovery_zone_denied_count": self.recovery_zone_denied_count,
            "deadlock_count": self.deadlock_count,
            "wait_count": self.wait_count,
            "recovery_attempts": self.recovery_attempts,
            "recovery_conflict_count": self.recovery_conflict_count,
            "recovery_lock_wait_count": self.recovery_lock_wait_count,
            "recovery_wait_time": self.recovery_wait_time,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "stuck_count": self.stuck_count,
            "temporary_obstacle_hits": self.temporary_obstacle_hits,
            "pick_attempts": self.pick_attempts,
            "pick_success_count": self.pick_success_count,
            "pick_failure_count": self.pick_failure_count,
            "place_attempts": self.place_attempts,
            "place_success_count": self.place_success_count,
            "place_failure_count": self.place_failure_count,
            "object_conflict_count": self.object_conflict_count,
            "pickup_zone_conflict_count": self.pickup_zone_conflict_count,
            "drop_zone_conflict_count": self.drop_zone_conflict_count,
            "duplicate_pick_attempt_count": self.duplicate_pick_attempt_count,
            "approach_zone_denied_count": self.approach_zone_denied_count,
            "approach_zone_wait_count": self.approach_zone_wait_count,
            "rule_prevented_approach_conflict_count": (
                self.rule_prevented_approach_conflict_count
            ),
            "approach_lock_hold_time": self.approach_lock_hold_time,
            "approach_lock_starvation_count": self.approach_lock_starvation_count,
            "approach_lock_reassignment_count": self.approach_lock_reassignment_count,
            "corridor_wait_count": self.corridor_wait_count,
            "lock_wait_count": self.lock_wait_count,
            "lock_wait_time": self.lock_wait_time,
            "reassignment_count": self.reassignment_count,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "object_success_rate": (
                self.tasks_completed / (self.tasks_completed + self.tasks_failed)
                if (self.tasks_completed + self.tasks_failed) > 0
                else 0.0
            ),
            "all_objects_placed": self.all_objects_placed,
            "centralized_rule_enabled": self.centralized_rule_enabled,
            "dag_enabled": self.dag_enabled,
            "dag_task_count": self.dag_task_count,
            "dag_dependency_count": self.dag_dependency_count,
            "dag_completed_task_count": self.dag_completed_task_count,
            "dag_ready_task_count": self.dag_ready_task_count,
            "dag_blocked_task_count": self.dag_blocked_task_count,
            "dependency_wait_count": self.dependency_wait_count,
            "dag_violation_count": self.dag_violation_count,
            "critical_path_length": self.critical_path_length,
        }

    def to_trace_payload(self, scenario_name: str, num_robots: int) -> dict[str, Any]:
        return {
            "scenario_name": scenario_name,
            "num_robots": num_robots,
            "metrics": self.summary(),
            "trace": self.trace,
        }


def parse_positions(value: Any) -> dict[str, tuple[int, int]]:
    if not isinstance(value, dict):
        return {}
    positions: dict[str, tuple[int, int]] = {}
    for robot_id, raw_cell in value.items():
        cell = parse_cell(raw_cell)
        if cell is not None:
            positions[str(robot_id)] = cell
    return positions


def parse_cell(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        if "position" in value:
            return parse_cell(value["position"])
        if "x" in value and "y" in value:
            return normalize_cell(value["x"], value["y"])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return normalize_cell(value[0], value[1])
    return None


def normalize_cell(x_value: Any, y_value: Any) -> tuple[int, int] | None:
    try:
        return int(x_value), int(y_value)
    except (TypeError, ValueError):
        return None
