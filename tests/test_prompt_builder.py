import yaml

from src.llm.prompt_builder import build_prompt


def test_prompt_builder_includes_schema_and_scenario(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "prompt_nav",
                "source": {"dataset_index": 1},
                "map": {"width": 5, "height": 4, "obstacles": [[2, 2]]},
                "robots": {"robot_0": {"start": [1, 1], "goal": None}},
                "waypoints": [{"id": "wp_0", "position": [4, 1]}],
                "task": {"type": "pure_navigation", "assignments": {"robot_0": ["wp_0"]}},
                "centralized_rule": True,
                "max_steps": 20,
                "render": False,
            }
        ),
        encoding="utf-8",
    )

    prompt = build_prompt(
        scenario_path,
        "centralized_rule_multi_bt",
        source_sample={"index": 1, "input": "Navigate."},
        previous_failure_report={"status": "failed", "failure_types": ["duplicate_task"]},
        previous_robot_trees={"robot_0": "<root/>"},
    )

    assert "Output JSON schema example" in prompt
    assert "prompt_nav" in prompt
    assert "wp_0" in prompt
    assert "centralized rule layer" in prompt
    assert "Do not use markdown" in prompt
    assert "failure_report" in prompt


def test_prompt_builder_includes_recovery_resource_example(tmp_path):
    scenario_path = tmp_path / "recovery.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "prompt_recovery",
                "map": {"width": 5, "height": 4, "obstacles": []},
                "robots": {"robot_0": {"start": [1, 1], "goal": None}},
                "waypoints": [{"id": "wp_0", "position": [4, 1]}],
                "recovery": {"enabled": True},
                "task": {"type": "navigation_recovery", "assignments": {"robot_0": ["wp_0"]}},
                "centralized_rule": True,
                "max_steps": 20,
                "render": False,
            }
        ),
        encoding="utf-8",
    )

    prompt = build_prompt(
        scenario_path,
        "centralized_rule_multi_bt",
        previous_failure_report={
            "status": "failed",
            "details": [
                {
                    "message": "robot_0: recovery action lacks recovery_zone RequestResource"
                }
            ],
        },
    )

    assert (
        '<Action ID="RequestResource" resource_id="recovery_zone_0" resource_type="recovery_zone"/>'
        in prompt
    )
    assert '<Action ID="ReleaseResource" resource_id="recovery_zone_0"/>' in prompt
    assert 'Replace any resource="recovery_zone"' in prompt


def test_prompt_builder_forbids_recovery_node(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "prompt_forbidden",
                "map": {"width": 5, "height": 4, "obstacles": []},
                "robots": {"robot_0": {"start": [1, 1], "goal": None}},
                "waypoints": [{"id": "wp_0", "position": [4, 1]}],
                "task": {"type": "pure_navigation", "assignments": {"robot_0": ["wp_0"]}},
                "centralized_rule": True,
                "max_steps": 20,
                "render": False,
            }
        ),
        encoding="utf-8",
    )

    prompt = build_prompt(scenario_path, "centralized_rule_multi_bt")

    assert "Forbidden XML tags" in prompt
    assert "RecoveryNode" in prompt
    assert "ReactiveFallback" in prompt
    assert "PipelineSequence" in prompt


def test_prompt_builder_includes_dag_task_table_and_rules(tmp_path):
    scenario_path = tmp_path / "dag_scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "prompt_dag_pickplace",
                "source": {"dataset_index": 2},
                "map": {"width": 6, "height": 4, "obstacles": []},
                "robots": {
                    "robot_0": {"start": [0, 0], "goal": None},
                    "robot_1": {"start": [0, 1], "goal": None},
                },
                "objects": {
                    "obj_0": {"position": [2, 0], "target_position": [3, 0], "status": "available"}
                },
                "pickplace": {
                    "enabled": True,
                    "tasks": [
                        {
                            "task_id": "task_obj_0",
                            "object_id": "obj_0",
                            "pickup_position": [2, 0],
                            "drop_position": [3, 0],
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
                "centralized_rule": True,
                "max_steps": 20,
                "render": False,
            }
        ),
        encoding="utf-8",
    )

    prompt = build_prompt(scenario_path, "llm_dag_with_dependency_prompt")

    assert "DAG Task Table" in prompt
    assert "task_id | task_type | object_id | assigned_robot | predecessors | expected_action_id" in prompt
    assert "WaitForDependency" in prompt
    assert "IsTaskReady" in prompt
    assert "final_check must only run after all required place tasks are completed" in prompt
    assert "Do not execute a task before all predecessor tasks are completed." in prompt
