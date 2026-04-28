import pytest

from src.dag.dag_validator import validate_task_dag
from src.dag.task_dag import DependencyEdge, TaskDAG, TaskNode


def test_valid_pick_before_place_dag():
    dag = pick_place_dag()

    result = validate_task_dag(
        dag,
        scenario={"robots": {"robot_0": {"start": [0, 0]}}},
    )

    assert result.valid is True
    assert result.errors == []


def test_missing_dependency_endpoint_is_reported():
    dag = TaskDAG(tasks=[TaskNode("a", "nav")])
    dag.add_dependency(DependencyEdge("a", "missing"))

    result = validate_task_dag(dag)

    assert result.valid is False
    assert any("missing target_task_id" in error for error in result.errors)


def test_cycle_is_reported():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {"task_id": "a", "task_type": "nav"},
                {"task_id": "b", "task_type": "nav"},
            ],
            "dependencies": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "a"},
            ],
        }
    )

    result = validate_task_dag(dag)

    assert result.valid is False
    assert any("cycle" in error for error in result.errors)


def test_place_without_matching_pick_dependency_is_reported():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {"task_id": "pick_obj_0", "task_type": "pick", "object_id": "obj_0"},
                {"task_id": "place_obj_0", "task_type": "place", "object_id": "obj_0"},
            ],
            "dependencies": [],
        }
    )

    result = validate_task_dag(dag)

    assert result.valid is False
    assert any("must depend on a pick task" in error for error in result.errors)


def test_place_accepts_transitive_pick_dependency():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {"task_id": "pick_obj_0", "task_type": "pick", "object_id": "obj_0"},
                {"task_id": "inspect_obj_0", "task_type": "nav"},
                {"task_id": "place_obj_0", "task_type": "place", "object_id": "obj_0"},
            ],
            "dependencies": [
                {"source": "pick_obj_0", "target": "inspect_obj_0"},
                {"source": "inspect_obj_0", "target": "place_obj_0"},
            ],
        }
    )

    result = validate_task_dag(dag)

    assert result.valid is True


def test_assembly_or_final_check_without_predecessor_is_reported():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {"task_id": "final_check", "task_type": "final_check"},
                {"task_id": "assembly", "task_type": "assembly"},
            ],
            "dependencies": [],
        }
    )

    result = validate_task_dag(dag)

    assert result.valid is False
    assert any("final_check task final_check" in error for error in result.errors)
    assert any("assembly task assembly" in error for error in result.errors)


def test_invalid_assigned_robot_is_reported():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {
                    "task_id": "pick_obj_0",
                    "task_type": "pick",
                    "assigned_robot": "robot_9",
                }
            ],
            "dependencies": [],
        }
    )

    result = validate_task_dag(dag, robot_ids=["robot_0"])

    assert result.valid is False
    assert any("assigned_robot robot_9" in error for error in result.errors)


def test_duplicate_task_ids_are_rejected_by_parser():
    with pytest.raises(ValueError):
        TaskDAG.from_dict(
            {
                "tasks": [
                    {"task_id": "a", "task_type": "nav"},
                    {"task_id": "a", "task_type": "recovery"},
                ],
                "dependencies": [],
            }
        )


def pick_place_dag() -> TaskDAG:
    return TaskDAG.from_dict(
        {
            "tasks": [
                {
                    "task_id": "pick_obj_0",
                    "task_type": "pick",
                    "object_id": "obj_0",
                    "assigned_robot": "robot_0",
                },
                {
                    "task_id": "place_obj_0",
                    "task_type": "place",
                    "object_id": "obj_0",
                    "assigned_robot": "robot_0",
                },
                {"task_id": "final_check", "task_type": "final_check"},
            ],
            "dependencies": [
                {"source": "pick_obj_0", "target": "place_obj_0"},
                {"source": "place_obj_0", "target": "final_check"},
            ],
        }
    )
