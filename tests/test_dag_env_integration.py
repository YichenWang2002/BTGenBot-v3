from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data


def test_dag_blocks_place_before_pick_and_records_wait():
    scenario = load_scenario_data(dag_pickplace_scenario())
    env = build_env(scenario)
    env.reset()
    robot = env.state.robots["robot_0"]
    robot.position = [2, 1]
    robot.current_task = "task_obj_0"
    robot.task_state = "place"

    observation = env.step({"robot_0": {"type": "navigate"}})

    assert observation["metrics"]["dag_enabled"] is True
    assert observation["metrics"]["dependency_wait_count"] == 1
    assert observation["metrics"]["dag_violation_count"] == 0
    assert observation["metrics"]["dag_completed_task_count"] == 0
    assert env.task_dag is not None
    assert env.task_dag.is_ready("place_obj_0") is False
    assert observation["metrics"]["dag_blocked_task_count"] == 1
    assert env.metrics.trace[-1]["dag_events"][0]["event_type"] == "dependency_wait"


def test_pick_completion_makes_place_ready():
    scenario = load_scenario_data(dag_pickplace_scenario())
    env = build_env(scenario)
    env.reset()
    robot = env.state.robots["robot_0"]
    robot.position = [1, 1]
    robot.current_task = "task_obj_0"
    robot.task_state = "pick"

    observation = env.step({"robot_0": {"type": "navigate"}})

    assert env.task_dag is not None
    assert env.task_dag.tasks["pick_obj_0"].status == "completed"
    assert env.task_dag.is_ready("place_obj_0") is True
    assert observation["metrics"]["dag_completed_task_count"] == 1
    assert observation["metrics"]["dag_ready_task_count"] == 1


def test_place_completion_updates_dag_completed_count():
    scenario = load_scenario_data(dag_pickplace_scenario())
    env = build_env(scenario)
    env.reset()
    env.task_dag.mark_completed("pick_obj_0")
    robot = env.state.robots["robot_0"]
    obj = env.state.objects["obj_0"]
    robot.position = [2, 1]
    robot.current_task = "task_obj_0"
    robot.task_state = "place"
    robot.carrying_object = "obj_0"
    obj.held_by = "robot_0"
    obj.position = None
    obj.status = "held"

    observation = env.step({"robot_0": {"type": "navigate"}})

    assert env.task_dag.tasks["place_obj_0"].status == "completed"
    assert observation["metrics"]["dag_completed_task_count"] == 2


def test_no_task_dag_keeps_old_behavior_unblocked():
    data = dag_pickplace_scenario()
    data.pop("task_dag")
    scenario = load_scenario_data(data)
    env = build_env(scenario)
    env.reset()
    robot = env.state.robots["robot_0"]
    robot.position = [2, 1]
    robot.current_task = "task_obj_0"
    robot.task_state = "place"

    observation = env.step({"robot_0": {"type": "navigate"}})

    assert observation["metrics"]["dag_enabled"] is False
    assert observation["metrics"]["dependency_wait_count"] == 0
    assert observation["metrics"]["dag_violation_count"] == 0
    assert env.pickplace_tasks["task_obj_0"]["status"] == "placed"


def build_env(scenario):
    return MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=False,
        pickplace_config=scenario.pickplace,
        zones=scenario.zones,
        task_dag=scenario.task_dag,
    )


def dag_pickplace_scenario():
    return {
        "name": "dag_pickplace_env",
        "map": {"width": 5, "height": 4, "obstacles": []},
        "robots": {"robot_0": {"start": [1, 1], "goal": None}},
        "objects": {
            "obj_0": {
                "position": [1, 1],
                "target_position": [2, 1],
                "status": "available",
            }
        },
        "zones": {
            "pickup_zones": {"pickup_zone_0": {"cells": [[1, 1]]}},
            "drop_zones": {"drop_zone_0": {"cells": [[2, 1]]}},
        },
        "pickplace": {
            "enabled": True,
            "pick_fail_prob": 0.0,
            "place_fail_prob": 0.0,
            "allow_reassignment": False,
            "tasks": [
                {
                    "task_id": "task_obj_0",
                    "object_id": "obj_0",
                    "pickup_position": [1, 1],
                    "drop_position": [2, 1],
                    "assigned_robot": "robot_0",
                    "status": "assigned",
                    "attempts": 0,
                }
            ],
        },
        "task": {
            "type": "navigation_pickplace",
            "assignments": {"robot_0": ["task_obj_0"]},
        },
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
            "dependencies": [{"source": "pick_obj_0", "target": "place_obj_0"}],
        },
        "max_steps": 20,
        "render": False,
    }
