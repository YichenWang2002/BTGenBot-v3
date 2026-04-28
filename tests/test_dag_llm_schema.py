from src.env.scenario_loader import load_scenario_data
from src.llm.schema import validate_llm_output, validate_bt_xml
from src.repair.critic import static_validate


def test_schema_accepts_dag_actions_and_task_ids():
    scenario = build_dag_scenario()
    payload = {
        "robot_trees": {
            "robot_0": (
                '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence>'
                '<Action ID="WaitForDependency" task_id="pick_obj_0"/>'
                '<Action ID="RequestResource" resource_id="obj_0" resource_type="object"/>'
                '<Action ID="PickObject" object_id="obj_0" task_id="pick_obj_0"/>'
                '<Action ID="WaitForDependency" task_id="place_obj_0"/>'
                '<Action ID="PlaceObject" object_id="obj_0" task_id="place_obj_0"/>'
                '<Action ID="ReleaseResource" resource_id="obj_0"/>'
                '<Action ID="WaitForDependency" task_id="final_check"/>'
                '<Action ID="FinalCheck" task_id="final_check"/>'
                "</Sequence></BehaviorTree></root>"
            ),
            "robot_1": (
                '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence>'
                '<Action ID="ReportStatus" status="done"/>'
                "</Sequence></BehaviorTree></root>"
            ),
        },
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["task_obj_0"], "reason": "test"}
        ],
        "coordination_notes": ["use DAG"],
        "assumptions": ["discrete grid"],
    }

    bt_result = validate_bt_xml(payload["robot_trees"]["robot_0"])
    assert bt_result.valid is True

    result = validate_llm_output(payload, scenario)

    assert result.valid is True
    assert result.errors == []


def test_static_critic_reports_dag_misuse():
    scenario = build_dag_scenario()
    payload = {
        "robot_trees": {
            "robot_0": (
                '<root><BehaviorTree ID="MainTree"><Sequence>'
                '<Action ID="PickObject" object_id="obj_0" task_id="pick_obj_0"/>'
                '<Action ID="PlaceObject" object_id="obj_0" task_id="place_obj_0"/>'
                '<Action ID="FinalCheck" task_id="final_check"/>'
                "</Sequence></BehaviorTree></root>"
            )
        },
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["task_obj_0"], "reason": "test"}
        ],
        "coordination_notes": ["use DAG"],
        "assumptions": ["discrete grid"],
    }

    result = static_validate(scenario, payload)

    assert result["valid"] is False
    assert result["dag_static_error_count"] >= 1
    assert result["missing_dependency_wait_count"] >= 1
    assert result["invalid_task_id_count"] == 0
    assert result["wrong_assigned_robot_for_task_count"] == 0
    assert any("dependency check" in error for error in result["errors"])


def test_static_critic_reports_wrong_robot_and_invalid_task_id():
    scenario = build_dag_scenario()
    payload = {
        "robot_trees": {
            "robot_1": (
                '<root><BehaviorTree ID="MainTree"><Sequence>'
                '<Action ID="WaitForDependency" task_id="unknown_task"/>'
                '<Action ID="PickObject" object_id="obj_0" task_id="pick_obj_0"/>'
                "</Sequence></BehaviorTree></root>"
            )
        },
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["task_obj_0"], "reason": "test"}
        ],
        "coordination_notes": ["use DAG"],
        "assumptions": ["discrete grid"],
    }

    result = static_validate(scenario, payload)

    assert result["valid"] is False
    assert result["invalid_task_id_count"] >= 1
    assert result["wrong_assigned_robot_for_task_count"] >= 1


def build_dag_scenario():
    return load_scenario_data(
        {
            "name": "schema_dag",
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
