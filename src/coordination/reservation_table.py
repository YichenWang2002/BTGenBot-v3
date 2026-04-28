"""Space-time reservation table for centralized multi-robot coordination."""

from __future__ import annotations

from typing import Any

from src.coordination.events import decision_result


Cell = tuple[int, int]
Edge = tuple[Cell, Cell, int]


class ReservationTable:
    def __init__(self) -> None:
        self._cell_reservations: dict[tuple[Cell, int], str] = {}
        self._edge_reservations: dict[Edge, str] = {}

    def reserve_cell(
        self, robot_id: str, cell: tuple[int, int] | list[int], timestep: int
    ) -> dict[str, Any]:
        normalized = normalize_cell(cell)
        key = (normalized, int(timestep))
        owner = self._cell_reservations.get(key)
        if owner is not None and owner != robot_id:
            return decision_result(
                allowed=False,
                reason="cell_reserved",
                conflict_type="vertex",
                owner=owner,
            )
        self._cell_reservations[key] = robot_id
        return decision_result(allowed=True, reason="cell_reserved")

    def release_cell(
        self, robot_id: str, cell: tuple[int, int] | list[int], timestep: int
    ) -> dict[str, Any]:
        normalized = normalize_cell(cell)
        key = (normalized, int(timestep))
        owner = self._cell_reservations.get(key)
        if owner is None:
            return decision_result(allowed=True, reason="cell_not_reserved")
        if owner != robot_id:
            return decision_result(
                allowed=False,
                reason="not_cell_owner",
                conflict_type="vertex",
                owner=owner,
            )
        del self._cell_reservations[key]
        return decision_result(allowed=True, reason="cell_released")

    def is_cell_reserved(self, cell: tuple[int, int] | list[int], timestep: int) -> bool:
        return (normalize_cell(cell), int(timestep)) in self._cell_reservations

    def get_cell_owner(
        self, cell: tuple[int, int] | list[int], timestep: int
    ) -> str | None:
        return self._cell_reservations.get((normalize_cell(cell), int(timestep)))

    def reserve_edge(
        self,
        robot_id: str,
        from_cell: tuple[int, int] | list[int],
        to_cell: tuple[int, int] | list[int],
        timestep: int,
    ) -> dict[str, Any]:
        conflict = self.is_edge_conflict(robot_id, from_cell, to_cell, timestep)
        if not conflict["allowed"]:
            return conflict

        edge = (normalize_cell(from_cell), normalize_cell(to_cell), int(timestep))
        owner = self._edge_reservations.get(edge)
        if owner is not None and owner != robot_id:
            return decision_result(
                allowed=False,
                reason="edge_reserved",
                conflict_type="edge",
                owner=owner,
            )
        self._edge_reservations[edge] = robot_id
        return decision_result(allowed=True, reason="edge_reserved")

    def is_edge_conflict(
        self,
        robot_id: str,
        from_cell: tuple[int, int] | list[int],
        to_cell: tuple[int, int] | list[int],
        timestep: int,
    ) -> dict[str, Any]:
        reverse_edge = (
            normalize_cell(to_cell),
            normalize_cell(from_cell),
            int(timestep),
        )
        owner = self._edge_reservations.get(reverse_edge)
        if owner is not None and owner != robot_id:
            return decision_result(
                allowed=False,
                reason="edge_swap_conflict",
                conflict_type="edge",
                owner=owner,
            )
        return decision_result(allowed=True, reason="no_edge_conflict")

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        cells = [
            {"cell": list(cell), "timestep": timestep, "owner": owner}
            for (cell, timestep), owner in sorted(self._cell_reservations.items())
        ]
        edges = [
            {
                "from_cell": list(from_cell),
                "to_cell": list(to_cell),
                "timestep": timestep,
                "owner": owner,
            }
            for (from_cell, to_cell, timestep), owner in sorted(
                self._edge_reservations.items()
            )
        ]
        return {"cells": cells, "edges": edges}


def normalize_cell(cell: tuple[int, int] | list[int]) -> Cell:
    if len(cell) != 2:
        raise ValueError(f"Cell must have two coordinates: {cell}")
    return int(cell[0]), int(cell[1])

