import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data


def test_approach_zone_lock_makes_second_robot_wait():
    scenario = load_scenario_data(
        approach_zone_scenario(enable_lock=True, second_robot_task=True)
    )
    env = build_env(scenario)
    env.reset()

    observation = env.step()

    summary = observation["metrics"]
    assert summary["collision_count"] == 0
    assert summary["approach_zone_wait_count"] == 1
    assert summary["approach_zone_denied_count"] == 1
    assert summary["rule_prevented_approach_conflict_count"] == 1


def test_approach_zone_lock_does_not_grant_pass_through_robot():
    scenario = load_scenario_data(approach_zone_scenario(enable_lock=True))
    env = build_env(scenario)
    env.reset()

    observation = env.step()

    locks = env.rule_manager.snapshot()["resource_locks"]["locks"]
    assert observation["metrics"]["approach_zone_wait_count"] == 1
    assert locks["pickup_approach_zone_1_1"]["owner"] == "robot_0"


def test_approach_zone_lock_defaults_to_disabled():
    scenario = load_scenario_data(
        approach_zone_scenario(enable_lock=False, second_robot_task=True)
    )
    env = build_env(scenario)
    env.reset()

    observation = env.step()

    assert observation["metrics"]["approach_zone_wait_count"] == 0


def test_approach_radius_zero_only_locks_target_cell():
    scenario = load_scenario_data(
        approach_zone_scenario(
            enable_lock=True,
            second_robot_task=True,
            coordination_overrides={"approach_radius": 0},
        )
    )
    env = build_env(scenario)
    env.reset()

    adjacent = env._approach_resources_for_robot_target(
        "robot_0", [0, 1], {"action_type": "navigate_to_pickup"}
    )
    target = env._approach_resources_for_robot_target(
        "robot_0", [1, 1], {"action_type": "navigate_to_pickup"}
    )

    assert adjacent == []
    assert target[0]["resource_id"] == "pickup_approach_zone_1_1"


def test_approach_lock_releases_after_pick_success():
    scenario = load_scenario_data(
        approach_zone_scenario(enable_lock=True, starts_at_pickup=True)
    )
    env = build_env(scenario)
    env.reset()

    env.step()
    observation = env.step()

    assert observation["metrics"]["pick_success_count"] == 1
    assert observation["metrics"]["approach_lock_hold_time"] >= 1
    locks = env.rule_manager.snapshot()["resource_locks"]["locks"]
    assert "pickup_approach_zone_1_1" not in locks


def test_approach_lock_aging_boosts_waiting_robot():
    scenario = load_scenario_data(
        approach_zone_scenario(
            enable_lock=True,
            second_robot_task=True,
            coordination_overrides={"max_wait_ticks": 1},
        )
    )
    env = build_env(scenario)
    env.reset()

    observation = env.step()

    assert observation["metrics"]["approach_lock_starvation_count"] == 1
    locks = env.rule_manager.snapshot()["resource_locks"]["locks"]
    assert locks["pickup_approach_zone_1_1"]["owner"] == "robot_1"


def test_approach_lock_wait_can_trigger_reassignment():
    scenario = load_scenario_data(
        approach_zone_scenario(
            enable_lock=True,
            second_robot_task=True,
            allow_reassignment=True,
            coordination_overrides={
                "max_wait_ticks": 99,
                "reassignment_wait_ticks": 1,
            },
        )
    )
    env = build_env(scenario)
    env.reset()

    observation = env.step()

    assert observation["metrics"]["approach_lock_reassignment_count"] == 1


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


def approach_zone_scenario(
    enable_lock,
    second_robot_task=False,
    starts_at_pickup=False,
    allow_reassignment=False,
    coordination_overrides=None,
):
    coordination = {"enable_approach_zone_lock": True, "approach_radius": 1} if enable_lock else {}
    coordination.update(coordination_overrides or {})
    objects = {
        "obj_0": {
            "position": [1, 1],
            "target_position": [3, 1],
            "held_by": None,
            "status": "available",
        }
    }
    tasks = [
        {
            "task_id": "task_obj_0",
            "object_id": "obj_0",
            "pickup_position": [1, 1],
            "drop_position": [3, 1],
            "assigned_robot": "robot_0",
            "status": "assigned",
            "attempts": 0,
        }
    ]
    assignments = {"robot_0": ["task_obj_0"], "robot_1": []}
    if second_robot_task:
        objects["obj_1"] = {
            "position": [1, 1],
            "target_position": [3, 2],
            "held_by": None,
            "status": "available",
        }
        tasks.append(
            {
                "task_id": "task_obj_1",
                "object_id": "obj_1",
                "pickup_position": [1, 1],
                "drop_position": [3, 2],
                "assigned_robot": "robot_1",
                "status": "assigned",
                "attempts": 0,
            }
        )
        assignments = {"robot_0": ["task_obj_0"], "robot_1": ["task_obj_1"]}
    return {
        "name": "approach_zone_test",
        "map": {"width": 4, "height": 4, "obstacles": []},
        "robots": {
            "robot_0": {
                "start": [1, 1] if starts_at_pickup else [0, 1],
                "goal": None,
                "carrying_object": None,
            },
            "robot_1": {"start": [2, 1], "goal": None, "carrying_object": None},
        },
        "objects": objects,
        "zones": {
            "pickup_zones": {"pickup_zone_0": {"cells": [[1, 1]]}},
            "drop_zones": {"drop_zone_0": {"cells": [[3, 1]]}},
        },
        "pickplace": {
            "enabled": True,
            "pick_fail_prob": 0.0,
            "place_fail_prob": 0.0,
            "lock_ttl": 8,
            "allow_reassignment": allow_reassignment,
            "seed": 0,
            "tasks": tasks,
        },
        "task": {
            "type": "navigation_pickplace",
            "assignment_strategy": "nearest_robot",
            "baseline_type": "test",
            "assignments": assignments,
        },
        "coordination": coordination,
        "centralized_rule": True,
        "max_steps": 10,
        "render": False,
    }
