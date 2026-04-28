"""Centralized rule manager for multi-robot movement and resources."""

from __future__ import annotations

from typing import Any

from src.coordination.events import rule_event
from src.coordination.reservation_table import ReservationTable, normalize_cell
from src.coordination.resource_manager import ResourceManager
from src.coordination.safety_guarantor import SafetyGuarantor


class CentralizedRuleManager:
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        obstacles: set[tuple[int, int]] | None = None,
        enable_cell_reservation: bool = True,
        enable_edge_reservation: bool = True,
    ) -> None:
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.obstacles = {normalize_cell(cell) for cell in obstacles or set()}
        self.enable_cell_reservation = enable_cell_reservation
        self.enable_edge_reservation = enable_edge_reservation
        self.reservation_table = ReservationTable()
        self.resource_manager = ResourceManager()
        self.safety_guarantor = SafetyGuarantor(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            obstacles=self.obstacles,
            reservation_table=self.reservation_table,
            resource_manager=self.resource_manager,
            enable_cell_reservation=self.enable_cell_reservation,
            enable_edge_reservation=self.enable_edge_reservation,
        )
        self.rule_events: list[dict[str, Any]] = []
        self._move_history: list[dict[str, Any]] = []

    def request_move(
        self,
        robot_id: str,
        from_cell: tuple[int, int] | list[int],
        to_cell: tuple[int, int] | list[int],
        timestep: int,
    ) -> dict[str, Any]:
        start = normalize_cell(from_cell)
        target = normalize_cell(to_cell)
        decision = self.safety_guarantor.check_action(
            robot_id,
            {
                "action_id": "move_request",
                "from_cell": start,
                "to_cell": target,
                "timestep": timestep,
            },
        )
        result = decision.to_result()

        event = rule_event(
            timestep=int(timestep),
            robot_id=robot_id,
            event_type="move_request",
            allowed=result["allowed"],
            reason=result["reason"],
            conflict_type=decision.conflict_type,
            owner=result["owner"],
            from_cell=start,
            to_cell=target,
            safety_layer=decision.safety_layer,
            decision=decision.decision,
            converted_to_wait=decision.converted_to_wait,
            action_id="move_request",
            details=decision.details,
        )
        self.rule_events.append(event)
        self._move_history.append(event)
        return result

    def request_resource(
        self,
        robot_id: str,
        resource_id: str,
        resource_type: str,
        timestep: int,
        ttl: int,
        reason: str,
    ) -> dict[str, Any]:
        decision = self.safety_guarantor.check_action(
            robot_id,
            {
                "action_id": (
                    "approach_zone_request"
                    if resource_type == "approach_zone"
                    else "resource_request"
                ),
                "resource_id": resource_id,
                "resource_type": resource_type,
                "timestep": timestep,
                "ttl": ttl,
                "reason": reason,
            },
        )
        result = decision.to_result()
        self.rule_events.append(
            rule_event(
                timestep=int(timestep),
                robot_id=robot_id,
                event_type="resource_request",
                allowed=result["allowed"],
                reason=result["reason"],
                conflict_type=decision.conflict_type,
                owner=result["owner"],
                resource_id=resource_id,
                resource_type=resource_type,
                safety_layer=decision.safety_layer,
                decision=decision.decision,
                converted_to_wait=decision.converted_to_wait,
                action_id="resource_request",
                details=decision.details,
            )
        )
        return result

    def release_resource(self, robot_id: str, resource_id: str) -> dict[str, Any]:
        result = self.resource_manager.release(resource_id, robot_id)
        self.rule_events.append(
            rule_event(
                timestep=-1,
                robot_id=robot_id,
                event_type="resource_release",
                allowed=result["allowed"],
                reason=result["reason"],
                conflict_type=result["conflict_type"],
                owner=result["owner"],
                resource_id=resource_id,
                resource_type=result.get("resource_type"),
                safety_layer="none",
                decision="allowed" if result["allowed"] else "denied",
                converted_to_wait=False,
                action_id="resource_release",
            )
        )
        return result

    def tick(self, timestep: int) -> list[dict[str, Any]]:
        releases = self.resource_manager.tick(timestep)
        for release in releases:
            self.rule_events.append(
                rule_event(
                    timestep=int(timestep),
                    robot_id=release["owner"],
                    event_type="resource_ttl_release",
                    allowed=True,
                    reason=release["reason"],
                    resource_id=release["resource_id"],
                    safety_layer="none",
                    action_id="resource_ttl_release",
                )
            )
        return releases

    def detect_deadlock(self, window_size: int) -> bool:
        if window_size <= 0:
            return False
        recent = self._move_history[-window_size:]
        if len(recent) < window_size:
            return False
        robot_ids = {event["robot_id"] for event in recent}
        if not robot_ids:
            return False
        rejected = [event for event in recent if not event["allowed"]]
        return len(rejected) == len(recent) and len(robot_ids) > 0

    def events_since(self, start_index: int) -> list[dict[str, Any]]:
        return [dict(event) for event in self.rule_events[start_index:]]

    def snapshot(self) -> dict[str, Any]:
        return {
            "reservations": self.reservation_table.snapshot(),
            "resource_locks": self.resource_manager.snapshot(),
        }

    def make_wait_event(
        self,
        robot_id: str,
        timestep: int,
        event_type: str,
        action_id: str,
        safety_layer: str,
        conflict_type: str,
        reason: str,
        resource_id: str | None = None,
        resource_type: str | None = None,
        owner: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = rule_event(
            timestep=int(timestep),
            robot_id=robot_id,
            event_type=event_type,
            allowed=False,
            reason=reason,
            conflict_type=conflict_type,
            owner=owner,
            resource_id=resource_id,
            resource_type=resource_type,
            safety_layer=safety_layer,
            decision="converted_to_wait",
            converted_to_wait=True,
            action_id=action_id,
            details=details,
        )
        self.rule_events.append(event)
        return event
