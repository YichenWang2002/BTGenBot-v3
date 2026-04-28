import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data


def test_corridor_lock_makes_second_robot_wait():
    scenario = load_scenario_data(corridor_scenario(enable_lock=True))
    env = build_env(scenario)
    env.reset()

    observation = env.step(
        {"robot_0": {"type": "wait"}, "robot_1": {"type": "wait"}}
    )

    assert observation["metrics"]["collision_count"] == 0
    assert observation["metrics"]["corridor_wait_count"] == 1


def test_corridor_lock_defaults_to_disabled():
    scenario = load_scenario_data(corridor_scenario(enable_lock=False))
    env = build_env(scenario)
    env.reset()

    observation = env.step(
        {"robot_0": {"type": "wait"}, "robot_1": {"type": "wait"}}
    )

    assert observation["metrics"]["corridor_wait_count"] == 0


def build_env(scenario):
    return MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=True,
        pickplace_config=scenario.pickplace,
        zones=scenario.zones,
        coordination_config=scenario.coordination,
    )


def corridor_scenario(enable_lock):
    coordination = (
        {
            "enable_corridor_lock": True,
            "corridor_cells": [[[0, 1], [1, 1], [2, 1]]],
        }
        if enable_lock
        else {}
    )
    return {
        "name": "corridor_lock_test",
        "map": {"width": 4, "height": 4, "obstacles": []},
        "robots": {
            "robot_0": {"start": [0, 1], "goal": None, "carrying_object": None},
            "robot_1": {"start": [2, 1], "goal": None, "carrying_object": None},
        },
        "objects": {
            "obj_0": {
                "position": [0, 1],
                "target_position": [2, 1],
                "held_by": None,
                "status": "available",
            }
        },
        "zones": {
            "pickup_zones": {"pickup_zone_0": {"cells": [[0, 1]]}},
            "drop_zones": {"drop_zone_0": {"cells": [[2, 1]]}},
        },
        "pickplace": {
            "enabled": True,
            "pick_fail_prob": 0.0,
            "place_fail_prob": 0.0,
            "lock_ttl": 8,
            "allow_reassignment": False,
            "seed": 0,
            "tasks": [
                {
                    "task_id": "task_obj_0",
                    "object_id": "obj_0",
                    "pickup_position": [0, 1],
                    "drop_position": [2, 1],
                    "assigned_robot": "robot_0",
                    "status": "assigned",
                    "attempts": 0,
                }
            ],
        },
        "task": {
            "type": "navigation_pickplace",
            "assignment_strategy": "nearest_robot",
            "baseline_type": "test",
            "assignments": {"robot_0": ["task_obj_0"], "robot_1": []},
        },
        "coordination": coordination,
        "centralized_rule": True,
        "max_steps": 10,
        "render": False,
    }
