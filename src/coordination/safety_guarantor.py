"""Unified safety guard for centralized multi-robot coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.coordination.events import decision_result
from src.coordination.reservation_table import ReservationTable, normalize_cell
from src.coordination.resource_manager import ResourceManager


SAFETY_LAYERS = {"motion", "semantic_resource", "manipulation_area", "none"}


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str
    safety_layer: str = "none"
    conflict_type: str | None = None
    converted_to_wait: bool = False
    resource_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.safety_layer not in SAFETY_LAYERS:
            raise ValueError(f"Unsupported safety layer: {self.safety_layer}")

    @property
    def decision(self) -> str:
        if self.converted_to_wait:
            return "converted_to_wait"
        return "allowed" if self.allowed else "denied"

    def to_result(self) -> dict[str, Any]:
        legacy_conflict_type = self.details.get("legacy_conflict_type")
        result = {
            **decision_result(
                self.allowed,
                self.reason,
                conflict_type=legacy_conflict_type,
                owner=self.details.get("owner"),
            ),
            "safety_layer": self.safety_layer,
            "safety_conflict_type": self.conflict_type,
            "decision": self.decision,
            "converted_to_wait": self.converted_to_wait,
            "resource_id": self.resource_id,
            "details": dict(self.details),
        }
        if "next_owner" in self.details:
            result["next_owner"] = self.details["next_owner"]
        return result


class SafetyGuarantor:
    """Deterministic safety layer over motion reservations and resource locks."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        obstacles: set[tuple[int, int]] | None = None,
        reservation_table: ReservationTable | None = None,
        resource_manager: ResourceManager | None = None,
        enable_cell_reservation: bool = True,
        enable_edge_reservation: bool = True,
    ) -> None:
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.obstacles = {normalize_cell(cell) for cell in obstacles or set()}
        self.reservation_table = reservation_table or ReservationTable()
        self.resource_manager = resource_manager or ResourceManager()
        self.enable_cell_reservation = enable_cell_reservation
        self.enable_edge_reservation = enable_edge_reservation

    def check_action(
        self, robot_id: str, action: dict[str, Any], state: Any = None
    ) -> SafetyDecision:
        action_id = str(action.get("action_id") or action.get("type") or "")
        if action_id in {"move_request", "request_move", "move"}:
            return self._check_motion(robot_id, action)
        if action_id in {"resource_request", "request_resource"}:
            return self._check_resource(robot_id, action)
        if action_id in {"approach_zone_request", "request_approach_zone"}:
            return self._check_approach_zone(robot_id, action)
        return SafetyDecision(True, "no_safety_constraint", details={"action_id": action_id})

    def _check_motion(self, robot_id: str, action: dict[str, Any]) -> SafetyDecision:
        start = normalize_cell(action["from_cell"])
        target = normalize_cell(action["to_cell"])
        reservation_timestep = int(action["timestep"]) + 1

        target_result = self._validate_move_target(target, reservation_timestep)
        if not target_result.allowed:
            return target_result

        if self.enable_cell_reservation:
            owner = self.reservation_table.get_cell_owner(target, reservation_timestep)
            if owner is not None and owner != robot_id:
                return SafetyDecision(
                    False,
                    "cell_reserved",
                    safety_layer="motion",
                    conflict_type="cell_conflict",
                    converted_to_wait=True,
                    details={"owner": owner, "legacy_conflict_type": "vertex"},
                )

        if self.enable_edge_reservation:
            edge_result = self.reservation_table.is_edge_conflict(
                robot_id, start, target, reservation_timestep
            )
            if not edge_result["allowed"]:
                return SafetyDecision(
                    False,
                    edge_result["reason"],
                    safety_layer="motion",
                    conflict_type="edge_conflict",
                    converted_to_wait=True,
                    details={
                        "owner": edge_result.get("owner"),
                        "legacy_conflict_type": "edge",
                    },
                )

        if self.enable_cell_reservation:
            cell_result = self.reservation_table.reserve_cell(
                robot_id, target, reservation_timestep
            )
            if not cell_result["allowed"]:
                return SafetyDecision(
                    False,
                    cell_result["reason"],
                    safety_layer="motion",
                    conflict_type="cell_conflict",
                    converted_to_wait=True,
                    details={
                        "owner": cell_result.get("owner"),
                        "legacy_conflict_type": "vertex",
                    },
                )

        if self.enable_edge_reservation:
            edge_reserve = self.reservation_table.reserve_edge(
                robot_id, start, target, reservation_timestep
            )
            if not edge_reserve["allowed"]:
                return SafetyDecision(
                    False,
                    edge_reserve["reason"],
                    safety_layer="motion",
                    conflict_type="edge_conflict",
                    converted_to_wait=True,
                    details={
                        "owner": edge_reserve.get("owner"),
                        "legacy_conflict_type": "edge",
                    },
                )

        return SafetyDecision(
            True,
            f"move_allowed_at_t{reservation_timestep}",
            safety_layer="motion",
            details={"legacy_conflict_type": None},
        )

    def _check_resource(self, robot_id: str, action: dict[str, Any]) -> SafetyDecision:
        resource_id = str(action["resource_id"])
        resource_type = str(action["resource_type"])
        result = self.resource_manager.acquire(
            resource_id=resource_id,
            robot_id=robot_id,
            resource_type=resource_type,
            timestep=int(action["timestep"]),
            ttl=int(action["ttl"]),
            reason=str(action.get("reason", "")),
        )
        allowed = bool(result["allowed"])
        return SafetyDecision(
            allowed,
            str(result["reason"]),
            safety_layer=self._resource_safety_layer(resource_type),
            conflict_type=None if allowed else self._resource_conflict_type(resource_type),
            converted_to_wait=not allowed,
            resource_id=resource_id,
            details={
                "owner": result.get("owner"),
                "next_owner": result.get("next_owner"),
                "resource_type": resource_type,
                "legacy_conflict_type": result.get("conflict_type"),
            },
        )

    def _check_approach_zone(
        self, robot_id: str, action: dict[str, Any]
    ) -> SafetyDecision:
        action = {**action, "resource_type": "approach_zone"}
        return self._check_resource(robot_id, action)

    def _validate_move_target(
        self, target: tuple[int, int], timestep: int
    ) -> SafetyDecision:
        x, y = target
        if x < 0 or y < 0 or x >= self.grid_width or y >= self.grid_height:
            return SafetyDecision(
                False,
                "out_of_bounds",
                safety_layer="motion",
                conflict_type="cell_conflict",
                converted_to_wait=True,
                details={"legacy_conflict_type": "vertex"},
            )
        if target in self.obstacles:
            return SafetyDecision(
                False,
                "obstacle",
                safety_layer="motion",
                conflict_type="cell_conflict",
                converted_to_wait=True,
                details={"legacy_conflict_type": "vertex"},
            )
        return SafetyDecision(
            True,
            f"move_allowed_at_t{timestep}",
            safety_layer="motion",
            details={"legacy_conflict_type": None},
        )

    @staticmethod
    def _resource_safety_layer(resource_type: str) -> str:
        if resource_type == "approach_zone":
            return "manipulation_area"
        return "semantic_resource"

    @staticmethod
    def _resource_conflict_type(resource_type: str) -> str:
        mapping = {
            "object": "object_lock_denied",
            "pickup_zone": "pickup_zone_denied",
            "drop_zone": "drop_zone_denied",
            "recovery_zone": "recovery_zone_denied",
            "approach_zone": "approach_zone_denied",
        }
        return mapping.get(resource_type, f"{resource_type}_denied")
