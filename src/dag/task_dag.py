"""Task dependency DAG primitives for multi-robot task ordering."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.dag.dag_state import (
    BLOCKED,
    COMPLETED,
    FAILED,
    PENDING,
    READY,
    RUNNING,
    ensure_transition_allowed,
    normalize_status,
)


@dataclass
class TaskNode:
    task_id: str
    task_type: str
    assigned_robot: str | None = None
    object_id: str | None = None
    target: Any | None = None
    expected_action_id: str | None = None
    status: str = PENDING

    def __post_init__(self) -> None:
        self.task_id = str(self.task_id)
        self.task_type = str(self.task_type)
        self.status = normalize_status(self.status)
        if self.assigned_robot is not None:
            self.assigned_robot = str(self.assigned_robot)
        if self.object_id is not None:
            self.object_id = str(self.object_id)
        if self.expected_action_id is not None:
            self.expected_action_id = str(self.expected_action_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskNode":
        return cls(
            task_id=str(data["task_id"]),
            task_type=str(data["task_type"]),
            assigned_robot=data.get("assigned_robot"),
            object_id=data.get("object_id"),
            target=data.get("target"),
            expected_action_id=data.get("expected_action_id"),
            status=str(data.get("status", PENDING)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "assigned_robot": self.assigned_robot,
            "object_id": self.object_id,
            "target": self.target,
            "expected_action_id": self.expected_action_id,
            "status": self.status,
        }


@dataclass
class DependencyEdge:
    source_task_id: str
    target_task_id: str
    dependency_type: str = "finish_to_start"
    description: str | None = None

    def __post_init__(self) -> None:
        self.source_task_id = str(self.source_task_id)
        self.target_task_id = str(self.target_task_id)
        self.dependency_type = str(self.dependency_type or "finish_to_start")
        if self.description is not None:
            self.description = str(self.description)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DependencyEdge":
        return cls(
            source_task_id=str(data.get("source_task_id", data.get("source"))),
            target_task_id=str(data.get("target_task_id", data.get("target"))),
            dependency_type=str(
                data.get("dependency_type", data.get("type", "finish_to_start"))
            ),
            description=data.get("description"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "target_task_id": self.target_task_id,
            "dependency_type": self.dependency_type,
            "description": self.description,
        }


class TaskDAG:
    def __init__(
        self,
        tasks: list[TaskNode] | None = None,
        dependencies: list[DependencyEdge] | None = None,
    ) -> None:
        self.tasks: dict[str, TaskNode] = {}
        self.dependencies: list[DependencyEdge] = []
        for task in tasks or []:
            self.add_task(task)
        for edge in dependencies or []:
            self.add_dependency(edge)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskDAG":
        payload = data.get("task_dag", data)
        dag = cls()
        for task_data in payload.get("tasks", []) or []:
            dag.add_task(TaskNode.from_dict(task_data))
        for edge_data in payload.get("dependencies", []) or []:
            dag.add_dependency(DependencyEdge.from_dict(edge_data))
        return dag

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "dependencies": [edge.to_dict() for edge in self.dependencies],
        }

    def clone(self) -> "TaskDAG":
        return TaskDAG.from_dict(deepcopy(self.to_dict()))

    def reset_statuses(self) -> None:
        for task in self.tasks.values():
            task.status = PENDING

    def task_states(self) -> dict[str, str]:
        return {
            task_id: task.status
            for task_id, task in sorted(self.tasks.items())
        }

    def find_tasks(
        self,
        task_type: str | None = None,
        object_id: str | None = None,
        assigned_robot: str | None = None,
    ) -> list[TaskNode]:
        matches: list[TaskNode] = []
        for task in self.tasks.values():
            if task_type is not None and task.task_type != task_type:
                continue
            if object_id is not None and task.object_id != object_id:
                continue
            if (
                assigned_robot is not None
                and task.assigned_robot is not None
                and task.assigned_robot != assigned_robot
            ):
                continue
            matches.append(task)
        return matches

    def add_task(self, task: TaskNode | dict[str, Any]) -> None:
        node = TaskNode.from_dict(task) if isinstance(task, dict) else task
        if node.task_id in self.tasks:
            raise ValueError(f"Duplicate task_id: {node.task_id}")
        self.tasks[node.task_id] = node

    def add_dependency(self, edge: DependencyEdge | dict[str, Any]) -> None:
        dependency = DependencyEdge.from_dict(edge) if isinstance(edge, dict) else edge
        self.dependencies.append(dependency)

    def get_predecessors(self, task_id: str) -> list[TaskNode]:
        self._require_task(task_id)
        return [
            self.tasks[edge.source_task_id]
            for edge in self.dependencies
            if edge.target_task_id == task_id and edge.source_task_id in self.tasks
        ]

    def get_successors(self, task_id: str) -> list[TaskNode]:
        self._require_task(task_id)
        return [
            self.tasks[edge.target_task_id]
            for edge in self.dependencies
            if edge.source_task_id == task_id and edge.target_task_id in self.tasks
        ]

    def is_ready(self, task_id: str) -> bool:
        task = self._require_task(task_id)
        if task.status in {RUNNING, COMPLETED, FAILED, BLOCKED}:
            return False
        return all(
            predecessor.status == COMPLETED
            for predecessor in self.get_predecessors(task_id)
        )

    def mark_running(self, task_id: str) -> None:
        task = self._require_task(task_id)
        if not self.is_ready(task_id):
            raise ValueError(f"Task is not ready: {task_id}")
        self._set_status(task, RUNNING)

    def mark_completed(self, task_id: str) -> None:
        task = self._require_task(task_id)
        self._set_status(task, COMPLETED)

    def mark_failed(self, task_id: str) -> None:
        task = self._require_task(task_id)
        self._set_status(task, FAILED)

    def ready_tasks(self) -> list[TaskNode]:
        return [task for task in self.tasks.values() if self.is_ready(task.task_id)]

    def blocked_tasks(self) -> list[TaskNode]:
        blocked: list[TaskNode] = []
        for task in self.tasks.values():
            if task.status == BLOCKED:
                blocked.append(task)
            elif task.status in {PENDING, READY} and not self.is_ready(task.task_id):
                blocked.append(task)
        return blocked

    def completed_tasks(self) -> list[TaskNode]:
        return [task for task in self.tasks.values() if task.status == COMPLETED]

    def topological_sort(self) -> list[str]:
        indegree = {task_id: 0 for task_id in self.tasks}
        successors: dict[str, list[str]] = {task_id: [] for task_id in self.tasks}
        for edge in self.dependencies:
            if edge.source_task_id not in self.tasks or edge.target_task_id not in self.tasks:
                continue
            indegree[edge.target_task_id] += 1
            successors[edge.source_task_id].append(edge.target_task_id)

        queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
        ordered: list[str] = []
        while queue:
            task_id = queue.popleft()
            ordered.append(task_id)
            for successor in sorted(successors[task_id]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if len(ordered) != len(self.tasks):
            raise ValueError("Task DAG contains a cycle")
        return ordered

    def critical_path_length(self) -> int:
        order = self.topological_sort()
        distance = {task_id: 0 for task_id in order}
        for task_id in order:
            for successor in self.get_successors(task_id):
                distance[successor.task_id] = max(
                    distance[successor.task_id],
                    distance[task_id] + 1,
                )
        return max(distance.values(), default=0)

    def has_dependency_path(self, source_task_id: str, target_task_id: str) -> bool:
        self._require_task(source_task_id)
        self._require_task(target_task_id)
        stack = [source_task_id]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target_task_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(successor.task_id for successor in self.get_successors(current))
        return False

    def _require_task(self, task_id: str) -> TaskNode:
        key = str(task_id)
        if key not in self.tasks:
            raise KeyError(f"Unknown task_id: {key}")
        return self.tasks[key]

    def _set_status(self, task: TaskNode, status: str) -> None:
        ensure_transition_allowed(task.status, status)
        task.status = normalize_status(status)
