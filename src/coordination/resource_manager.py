"""FIFO resource locks for centralized robot coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.coordination.events import decision_result


RESOURCE_TYPES = {
    "object",
    "pickup_zone",
    "drop_zone",
    "recovery_zone",
    "charging_zone",
    "approach_zone",
    "corridor",
}


@dataclass
class ResourceLock:
    resource_id: str
    resource_type: str
    owner: str
    acquired_at: int
    ttl: int
    reason: str

    def expired(self, timestep: int) -> bool:
        return self.ttl >= 0 and int(timestep) >= self.acquired_at + self.ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "owner": self.owner,
            "acquired_at": self.acquired_at,
            "ttl": self.ttl,
            "reason": self.reason,
        }


class ResourceManager:
    def __init__(self) -> None:
        self._locks: dict[str, ResourceLock] = {}
        self._wait_queues: dict[str, list[dict[str, Any]]] = {}

    def acquire(
        self,
        resource_id: str,
        robot_id: str,
        resource_type: str,
        timestep: int,
        ttl: int,
        reason: str,
    ) -> dict[str, Any]:
        validate_resource_type(resource_type)
        existing = self._locks.get(resource_id)
        if existing is None:
            self._locks[resource_id] = ResourceLock(
                resource_id=resource_id,
                resource_type=resource_type,
                owner=robot_id,
                acquired_at=int(timestep),
                ttl=int(ttl),
                reason=reason,
            )
            return {
                **decision_result(True, "resource_acquired"),
                "resource_id": resource_id,
                "next_owner": None,
            }

        if existing.owner == robot_id:
            existing.acquired_at = int(timestep)
            existing.ttl = int(ttl)
            existing.reason = reason
            return {
                **decision_result(True, "resource_reentrant_acquire"),
                "resource_id": resource_id,
                "next_owner": None,
            }

        request = {
            "resource_id": resource_id,
            "robot_id": robot_id,
            "resource_type": resource_type,
            "timestep": int(timestep),
            "ttl": int(ttl),
            "reason": reason,
        }
        queue = self._wait_queues.setdefault(resource_id, [])
        if not any(item["robot_id"] == robot_id for item in queue):
            queue.append(request)

        return {
            **decision_result(
                False,
                "resource_locked",
                conflict_type="resource",
                owner=existing.owner,
            ),
            "resource_id": resource_id,
            "next_owner": queue[0]["robot_id"] if queue else None,
        }

    def release(self, resource_id: str, robot_id: str) -> dict[str, Any]:
        existing = self._locks.get(resource_id)
        if existing is None:
            return {
                **decision_result(True, "resource_not_locked"),
                "resource_id": resource_id,
                "next_owner": None,
            }
        if existing.owner != robot_id:
            return {
                **decision_result(
                    False,
                    "not_resource_owner",
                    conflict_type="resource",
                    owner=existing.owner,
                ),
                "resource_id": resource_id,
                "next_owner": None,
            }

        del self._locks[resource_id]
        next_owner = self._promote_next(resource_id, existing.acquired_at + existing.ttl)
        return {
            **decision_result(True, "resource_released"),
            "resource_id": resource_id,
            "next_owner": next_owner,
        }

    def is_locked(self, resource_id: str) -> bool:
        return resource_id in self._locks

    def get_owner(self, resource_id: str) -> str | None:
        lock = self._locks.get(resource_id)
        return lock.owner if lock is not None else None

    def tick(self, timestep: int) -> list[dict[str, Any]]:
        released: list[dict[str, Any]] = []
        for resource_id, lock in list(self._locks.items()):
            if lock.expired(timestep):
                owner = lock.owner
                del self._locks[resource_id]
                next_owner = self._promote_next(resource_id, int(timestep))
                released.append(
                    {
                        "resource_id": resource_id,
                        "owner": owner,
                        "reason": "ttl_expired",
                        "next_owner": next_owner,
                    }
                )
        return released

    def snapshot(self) -> dict[str, Any]:
        return {
            "locks": {
                resource_id: lock.to_dict()
                for resource_id, lock in sorted(self._locks.items())
            },
            "wait_queues": {
                resource_id: [dict(item) for item in queue]
                for resource_id, queue in sorted(self._wait_queues.items())
                if queue
            },
        }

    def _promote_next(self, resource_id: str, timestep: int) -> str | None:
        queue = self._wait_queues.get(resource_id) or []
        if not queue:
            self._wait_queues.pop(resource_id, None)
            return None

        request = queue.pop(0)
        self._locks[resource_id] = ResourceLock(
            resource_id=resource_id,
            resource_type=request["resource_type"],
            owner=request["robot_id"],
            acquired_at=int(timestep),
            ttl=request["ttl"],
            reason=request["reason"],
        )
        if not queue:
            self._wait_queues.pop(resource_id, None)
        return request["robot_id"]


def validate_resource_type(resource_type: str) -> None:
    if resource_type not in RESOURCE_TYPES:
        raise ValueError(f"Unsupported resource type: {resource_type}")
