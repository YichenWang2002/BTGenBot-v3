import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario, load_scenario_data


def run_until_done(env):
    env.reset()
    observation = env.observation([])
    while not env.done:
        observation = env.step()
    return observation


def test_default_coordination_flags_preserve_centralized_crossing():
    scenario = load_scenario("configs/scenarios/demo_conflict_crossing.yaml")
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=True,
    )

    observation = run_until_done(env)

    assert observation["metrics"]["success"] is True
    assert observation["metrics"]["collision_count"] == 0
    assert observation["metrics"]["rule_rejection_count"] == 1


def test_disabling_cell_reservation_allows_centralized_vertex_conflict():
    scenario = load_scenario("configs/scenarios/demo_conflict_crossing.yaml")
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=True,
        coordination_config={
            "enable_cell_reservation": False,
            "enable_edge_reservation": False,
        },
    )

    observation = run_until_done(env)

    assert observation["metrics"]["collision_count"] > 0
    assert observation["metrics"]["vertex_conflict_count"] > 0


def test_disabling_pickplace_resource_lock_records_resource_conflict():
    scenario = load_scenario_data(pickplace_conflict_scenario())
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=True,
        pickplace_config=scenario.pickplace,
        zones=scenario.zones,
        coordination_config={"enable_resource_lock": False},
    )
    env.reset()

    env.step()
    env.step()

    summary = env.metrics.summary()
    assert summary["lock_wait_count"] == 0
    assert summary["resource_conflict_count"] > 0
    assert summary["duplicate_pick_attempt_count"] > 0


def test_disabling_recovery_lock_records_recovery_conflict():
    scenario = load_scenario_data(recovery_conflict_scenario())
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=True,
        recovery_config=scenario.recovery,
        coordination_config={"enable_recovery_lock": False},
    )
    env.reset()

    env.step()

    assert env.metrics.summary()["recovery_conflict_count"] == 1


def pickplace_conflict_scenario():
    return {
        "name": "pickplace_resource_ablation",
        "map": {"width": 5, "height": 5, "obstacles": []},
        "robots": {
            "robot_0": {"start": [1, 1], "goal": None, "carrying_object": None},
            "robot_1": {"start": [1, 1], "goal": None, "carrying_object": None},
        },
        "objects": {
            "obj_0": {
                "position": [1, 1],
                "target_position": [3, 1],
                "held_by": None,
                "status": "available",
            }
        },
        "zones": {
            "pickup_zones": {"pickup_zone_0": {"cells": [[1, 1]]}},
            "drop_zones": {"drop_zone_0": {"cells": [[3, 1]]}},
        },
        "pickplace": {
            "enabled": True,
            "pick_fail_prob": 0.0,
            "place_fail_prob": 0.0,
            "lock_ttl": 8,
            "allow_reassignment": True,
            "seed": 0,
            "tasks": [
                {
                    "task_id": "task_obj_0",
                    "object_id": "obj_0",
                    "pickup_position": [1, 1],
                    "drop_position": [3, 1],
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
            "assignments": {"robot_0": ["task_obj_0"], "robot_1": ["task_obj_0"]},
        },
        "centralized_rule": True,
        "max_steps": 20,
        "render": False,
    }


def recovery_conflict_scenario():
    return {
        "name": "recovery_lock_ablation",
        "map": {"width": 8, "height": 8, "obstacles": []},
        "robots": {
            "robot_0": {"start": [1, 1], "goal": None},
            "robot_1": {"start": [1, 3], "goal": None},
        },
        "waypoints": [
            {"id": "wp_0", "position": [4, 1]},
            {"id": "wp_1", "position": [4, 3]},
        ],
        "recovery": {
            "enabled": True,
            "trigger_mode": "blocked_path",
            "recovery_zone_radius": 10,
            "recovery_lock_ttl": 5,
            "blocked_cells": [[2, 1], [2, 3]],
            "temporary_obstacles": [],
            "stuck_threshold": 3,
        },
        "task": {
            "type": "navigation_recovery",
            "num_waypoints": 2,
            "assignment_strategy": "nearest_robot",
            "baseline_type": "test",
            "assignments": {"robot_0": ["wp_0"], "robot_1": ["wp_1"]},
        },
        "centralized_rule": True,
        "max_steps": 20,
        "render": False,
    }
