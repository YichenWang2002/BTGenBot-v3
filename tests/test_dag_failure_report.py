from src.env.scenario_loader import load_scenario_data
from src.repair.failure_report import build_failure_report


def test_failure_report_includes_dag_failure_types():
    scenario = build_dag_scenario()
    report = build_failure_report(
        scenario,
        assignment=[{"robot_id": "robot_0", "task_ids": ["task_obj_0"], "reason": "test"}],
        metrics={
            "dag_violation_count": 2,
            "dependency_wait_count": 1,
            "timeout": False,
        },
        trace=[],
        static_validation={
            "valid": False,
            "errors": [
                "robot_0: blocked task place_obj_0 executed without dependency check",
                "robot_0: invalid DAG task_id for WaitForDependency: unknown_task",
                "robot_1: task pick_obj_0 assigned to robot_0 but executed in this robot tree",
                "robot_0: final_check task final_check is not ready",
            ],
        },
    )

    assert report["status"] == "failed"
    assert "blocked_task_executed" in report["failure_types"]
    assert "missing_dependency_wait" in report["failure_types"]
    assert "invalid_task_id" in report["failure_types"]
    assert "wrong_assigned_robot_for_task" in report["failure_types"]
    assert "final_check_not_ready" in report["failure_types"]
    assert any(
        detail["type"] == "dependency_not_satisfied" for detail in report["details"]
    )


def build_dag_scenario():
    return load_scenario_data(
        {
            "name": "failure_report_dag",
            "map": {"width": 6, "height": 4, "obstacles": []},
            "robots": {
                "robot_0": {"start": [1, 1], "goal": None},
                "robot_1": {"start": [1, 2], "goal": None},
            },
            "objects": {
                "obj_0": {
                    "position": [2, 1],
                    "target_position": [3, 1],
                    "held_by": None,
                    "status": "available",
                }
            },
            "pickplace": {
                "enabled": True,
                "allow_reassignment": False,
                "pick_fail_prob": 0.0,
                "place_fail_prob": 0.0,
                "tasks": [
                    {
                        "task_id": "task_obj_0",
                        "object_id": "obj_0",
                        "pickup_position": [2, 1],
                        "drop_position": [3, 1],
                        "assigned_robot": "robot_0",
                        "pick_task_id": "pick_obj_0",
                        "place_task_id": "place_obj_0",
                    }
                ],
            },
            "task": {
                "type": "dag_pickplace",
                "assignments": {"robot_0": ["task_obj_0"], "robot_1": []},
            },
            "task_dag": {
                "tasks": [
                    {
                        "task_id": "pick_obj_0",
                        "task_type": "pick",
                        "object_id": "obj_0",
                        "assigned_robot": "robot_0",
                        "expected_action_id": "PickObject",
                    },
                    {
                        "task_id": "place_obj_0",
                        "task_type": "place",
                        "object_id": "obj_0",
                        "assigned_robot": "robot_0",
                        "expected_action_id": "PlaceObject",
                    },
                    {
                        "task_id": "final_check",
                        "task_type": "final_check",
                        "assigned_robot": "robot_0",
                        "expected_action_id": "FinalCheck",
                    },
                ],
                "dependencies": [
                    {"source": "pick_obj_0", "target": "place_obj_0"},
                    {"source": "place_obj_0", "target": "final_check"},
                ],
            },
            "max_steps": 20,
            "render": False,
        }
    )
