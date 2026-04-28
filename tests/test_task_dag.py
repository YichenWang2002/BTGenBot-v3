import pytest

from src.dag.task_dag import DependencyEdge, TaskDAG, TaskNode


def test_add_tasks_dependencies_and_queries():
    dag = TaskDAG()
    dag.add_task(TaskNode("pick_obj_0", "pick", object_id="obj_0"))
    dag.add_task(TaskNode("place_obj_0", "place", object_id="obj_0"))
    dag.add_dependency(DependencyEdge("pick_obj_0", "place_obj_0"))

    assert [task.task_id for task in dag.get_predecessors("place_obj_0")] == [
        "pick_obj_0"
    ]
    assert [task.task_id for task in dag.get_successors("pick_obj_0")] == [
        "place_obj_0"
    ]
    assert dag.is_ready("pick_obj_0") is True
    assert dag.is_ready("place_obj_0") is False


def test_status_transitions_and_task_sets():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {"task_id": "pick_obj_0", "task_type": "pick"},
                {"task_id": "place_obj_0", "task_type": "place"},
            ],
            "dependencies": [
                {"source": "pick_obj_0", "target": "place_obj_0"},
            ],
        }
    )

    assert [task.task_id for task in dag.ready_tasks()] == ["pick_obj_0"]
    assert [task.task_id for task in dag.blocked_tasks()] == ["place_obj_0"]
    dag.mark_running("pick_obj_0")
    dag.mark_completed("pick_obj_0")

    assert [task.task_id for task in dag.ready_tasks()] == ["place_obj_0"]
    assert [task.task_id for task in dag.completed_tasks()] == ["pick_obj_0"]
    dag.mark_failed("place_obj_0")
    assert dag.tasks["place_obj_0"].status == "failed"


def test_mark_running_rejects_blocked_task():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {"task_id": "a", "task_type": "nav"},
                {"task_id": "b", "task_type": "nav"},
            ],
            "dependencies": [{"source": "a", "target": "b"}],
        }
    )

    with pytest.raises(ValueError):
        dag.mark_running("b")


def test_topological_sort_and_critical_path_length():
    dag = TaskDAG.from_dict(
        {
            "tasks": [
                {"task_id": "a", "task_type": "nav"},
                {"task_id": "b", "task_type": "recovery"},
                {"task_id": "c", "task_type": "final_check"},
            ],
            "dependencies": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        }
    )

    assert dag.topological_sort() == ["a", "b", "c"]
    assert dag.critical_path_length() == 2


def test_from_dict_accepts_task_dag_yaml_shape():
    dag = TaskDAG.from_dict(
        {
            "task_dag": {
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
                ],
                "dependencies": [
                    {
                        "source": "pick_obj_0",
                        "target": "place_obj_0",
                        "type": "finish_to_start",
                    }
                ],
            }
        }
    )

    assert sorted(dag.tasks) == ["pick_obj_0", "place_obj_0"]
    assert dag.dependencies[0].source_task_id == "pick_obj_0"
    assert dag.dependencies[0].target_task_id == "place_obj_0"
    assert dag.dependencies[0].dependency_type == "finish_to_start"


def test_duplicate_task_ids_are_rejected():
    dag = TaskDAG()
    dag.add_task(TaskNode("a", "nav"))

    with pytest.raises(ValueError):
        dag.add_task(TaskNode("a", "nav"))
