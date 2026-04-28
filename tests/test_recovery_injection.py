from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data
from src.scenarios.recovery_injection import active_temporary_obstacles


def test_temporary_obstacle_blocks_navigation():
    scenario = load_scenario_data(recovery_scenario(temporary_only=True))
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=False,
        recovery_config=scenario.recovery,
    )
    env.reset()

    observation = env.step()

    assert observation["state"]["robots"]["robot_0"]["position"] == [1, 1]
    assert env.metrics.summary()["temporary_obstacle_hits"] == 1
    assert env.metrics.trace[0]["recovery_events"][0]["event_type"] == "recovery_start"


def test_clear_costmap_removes_temporary_obstacle_or_unblocks_zone():
    scenario = load_scenario_data(recovery_scenario())
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=False,
        recovery_config=scenario.recovery,
    )
    env.reset()

    env.step()

    assert (2, 1) in env.recovery_cleared_cells
    assert env._is_recovery_blocked([2, 1]) is False


def test_recovery_zone_lock_allows_only_one_robot():
    scenario = load_scenario_data(recovery_scenario(num_robots=2))
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=True,
        recovery_config=scenario.recovery,
    )
    env.reset()

    env.step()

    summary = env.metrics.summary()
    assert summary["recovery_conflict_count"] == 0
    assert summary["recovery_lock_wait_count"] == 1
    assert env.metrics.trace[0]["resource_locks"]["locks"]


def test_naive_recovery_can_record_conflict():
    scenario = load_scenario_data(recovery_scenario(num_robots=2))
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=False,
        recovery_config=scenario.recovery,
    )
    env.reset()

    env.step()

    assert env.metrics.summary()["recovery_conflict_count"] == 1


def test_centralized_recovery_reduces_conflict():
    naive = load_scenario_data(recovery_scenario(num_robots=2))
    centralized = load_scenario_data(recovery_scenario(num_robots=2))
    naive_env = MultiRobotEnv(
        naive.state,
        max_steps=naive.max_steps,
        render=False,
        scenario_name=naive.name,
        centralized_rule=False,
        recovery_config=naive.recovery,
    )
    centralized_env = MultiRobotEnv(
        centralized.state,
        max_steps=centralized.max_steps,
        render=False,
        scenario_name=centralized.name,
        centralized_rule=True,
        recovery_config=centralized.recovery,
    )
    naive_env.reset()
    centralized_env.reset()

    naive_env.step()
    centralized_env.step()

    assert (
        centralized_env.metrics.summary()["recovery_conflict_count"]
        <= naive_env.metrics.summary()["recovery_conflict_count"]
    )


def test_active_temporary_obstacles_helper():
    temporary = [{"cell": [2, 1], "appear_at": 1, "disappear_at": 3}]

    assert active_temporary_obstacles(temporary, 0) == set()
    assert active_temporary_obstacles(temporary, 1) == {(2, 1)}
    assert active_temporary_obstacles(temporary, 3) == set()


def recovery_scenario(num_robots=1, temporary_only=False):
    robots = {
        f"robot_{index}": {"start": [1, 1 + index * 2], "goal": None}
        for index in range(num_robots)
    }
    waypoints = [
        {"id": f"wp_{index}", "position": [4, 1 + index * 2]}
        for index in range(num_robots)
    ]
    blocked_cells = [] if temporary_only else [[2, 1 + index * 2] for index in range(num_robots)]
    temporary_obstacles = (
        [{"cell": [2, 1], "appear_at": 0, "disappear_at": 5}]
        if temporary_only
        else []
    )
    return {
        "name": "recovery_test",
        "map": {"width": 8, "height": 8, "obstacles": []},
        "robots": robots,
        "waypoints": waypoints,
        "recovery": {
            "enabled": True,
            "trigger_mode": "blocked_path",
            "recovery_zone_radius": 10,
            "recovery_lock_ttl": 5,
            "actions": ["ClearEntireCostmap", "Spin", "Wait", "BackUp"],
            "blocked_cells": blocked_cells,
            "temporary_obstacles": temporary_obstacles,
        },
        "task": {
            "type": "navigation_recovery",
            "num_waypoints": num_robots,
            "assignment_strategy": "round_robin",
            "baseline_type": "test",
            "assignments": {
                f"robot_{index}": [f"wp_{index}"] for index in range(num_robots)
            },
        },
        "centralized_rule": False,
        "max_steps": 30,
        "render": False,
    }

