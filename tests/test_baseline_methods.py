import os

import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.experiments.run_baseline_experiment import (
    build_method_scenario,
    run_simulator_method,
)


def test_single_robot_sequential_assigns_all_nav_waypoints():
    base_data = load_yaml("configs/generated/nav_easy/nav_easy_idx124_seed0_robots3.yaml")

    data = build_method_scenario(
        base_data,
        {"seed": 0},
        "single_robot_sequential",
        "nav",
    )

    assert sorted(data["robots"]) == ["robot_0"]
    assert data["centralized_rule"] is False
    assert data["task"]["assignments"]["robot_0"] == [
        waypoint["id"] for waypoint in data["waypoints"]
    ]


def test_pickplace_resource_ablation_sets_only_resource_lock_flag():
    base_data = load_yaml(
        "configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml"
    )

    data = build_method_scenario(
        base_data,
        {"seed": 0},
        "centralized_without_resource_lock",
        "pickplace",
    )

    assert data["centralized_rule"] is True
    assert data["coordination"]["enable_resource_lock"] is False
    assert data["coordination"]["enable_recovery_lock"] is True


def test_recovery_lock_ablation_sets_only_recovery_lock_flag():
    base_data = load_yaml(
        "configs/generated/recovery_medium/recovery_medium_idx184_seed0_robots3.yaml"
    )

    data = build_method_scenario(
        base_data,
        {"seed": 0},
        "centralized_without_recovery_lock",
        "recovery",
    )

    assert data["centralized_rule"] is True
    assert data["coordination"]["enable_resource_lock"] is True
    assert data["coordination"]["enable_recovery_lock"] is False


def test_prioritized_planning_resolves_crossing_without_collision(tmp_path):
    base_data = load_yaml("configs/scenarios/demo_conflict_crossing.yaml")
    base_data["source"] = {"dataset_index": 0}
    base_data["task"] = {
        "type": "navigation",
        "num_waypoints": 2,
        "assignment_strategy": "nearest_robot",
        "assignments": {},
    }
    base_data["waypoints"] = [
        {"id": "wp_0", "position": [3, 2]},
        {"id": "wp_1", "position": [2, 3]},
    ]
    base_data["robots"]["robot_0"]["goal"] = None
    base_data["robots"]["robot_1"]["goal"] = None
    base_data["task"]["assignments"] = {
        "robot_0": ["wp_0"],
        "robot_1": ["wp_1"],
    }

    row = run_simulator_method(
        run_id="prioritized_test",
        scenario_data=base_data,
        scenario_family="nav",
        method="prioritized_planning_multi_robot",
        trace_path=tmp_path / "trace.json",
        prioritized_planning=True,
    )

    assert row["success"] is True
    assert row["collision_count"] == 0
    assert row["total_robot_steps"] == 4


def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
