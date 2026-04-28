import json
import os

import yaml

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario_data
from src.experiments.run_pickplace_experiment import run_suite
from src.scenarios.pickplace_generator import generate_pickplace_scenarios


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def test_pick_requires_object_lock_when_centralized():
    env = build_pickplace_env(centralized_rule=True)
    env.reset()

    env.step()
    env.step()

    locks = env.metrics.trace[-1]["resource_locks"]["locks"]
    assert locks["obj_0"]["owner"] == "robot_0"
    assert locks["pickup_zone_0"]["owner"] == "robot_0"
    assert env.state.objects["obj_0"].held_by == "robot_0"


def test_two_robots_cannot_pick_same_object_with_centralized_rule():
    env = build_pickplace_env(centralized_rule=True, duplicate_assignment=True)
    env.reset()

    env.step()
    env.step()

    summary = env.metrics.summary()
    assert env.state.objects["obj_0"].held_by in {"robot_0", "robot_1"}
    assert summary["lock_wait_count"] >= 1
    assert summary["resource_conflict_count"] == 0
    assert summary["resource_request_denied_count"] >= 1
    assert summary["rule_prevented_resource_conflict_count"] >= 1
    assert summary["duplicate_pick_attempt_count"] == 0


def test_naive_records_duplicate_pick_attempt():
    env = build_pickplace_env(centralized_rule=False, duplicate_assignment=True)
    env.reset()

    env.step()
    env.step()

    summary = env.metrics.summary()
    assert summary["duplicate_pick_attempt_count"] >= 1
    assert summary["object_conflict_count"] >= 1


def test_place_requires_drop_zone_lock_when_centralized():
    env = build_pickplace_env(centralized_rule=True)
    env.reset()
    robot = env.state.robots["robot_0"]
    obj = env.state.objects["obj_0"]
    robot.position = [3, 1]
    robot.goal = [3, 1]
    robot.carrying_object = "obj_0"
    robot.current_task = "task_obj_0"
    robot.task_state = "place"
    obj.position = None
    obj.held_by = "robot_0"
    env.pickplace_tasks["task_obj_0"]["status"] = "picked"
    assert env.rule_manager is not None
    env.rule_manager.request_resource(
        "robot_1", "drop_zone_0", "drop_zone", env.state.timestep, 8, "test"
    )

    env.step({"robot_0": {"type": "navigate"}, "robot_1": {"type": "wait"}})

    summary = env.metrics.summary()
    assert summary["lock_wait_count"] >= 1
    assert summary["resource_conflict_count"] == 0
    assert summary["place_success_count"] == 0


def test_lock_wait_not_counted_as_resource_conflict():
    env = build_pickplace_env(centralized_rule=True, duplicate_assignment=True)
    env.reset()

    env.step()
    env.step()

    summary = env.metrics.summary()
    assert summary["lock_wait_count"] >= 1
    assert summary["resource_conflict_count"] == 0


def test_rule_prevented_conflict_counted_separately():
    env = build_pickplace_env(centralized_rule=True, duplicate_assignment=True)
    env.reset()

    env.step()
    env.step()

    summary = env.metrics.summary()
    assert summary["resource_request_denied_count"] >= 1
    assert summary["rule_prevented_resource_conflict_count"] >= 1


def test_pickplace_experiment_smoke_runs_one_scenario(tmp_path):
    profile_path, candidates_path = write_pickplace_profile_inputs(tmp_path)
    generated_dir = tmp_path / "generated"
    generate_pickplace_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=generated_dir,
        num_scenarios=1,
        num_robots=3,
        num_objects=3,
        seeds=[0],
        pick_fail_prob=0.0,
        object_unavailable_prob=0.0,
        candidates_out=tmp_path / "pickplace_candidates.json",
    )
    output_dir = tmp_path / "results"
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "pickplace_smoke_test",
                "scenario_manifest": str(generated_dir / "manifest.json"),
                "methods": [
                    "single_robot_pickplace",
                    "naive_multi_robot_pickplace",
                    "centralized_rule_multi_robot_pickplace",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path)

    assert {row["method"] for row in rows} == {
        "single_robot_pickplace",
        "naive_multi_robot_pickplace",
        "centralized_rule_multi_robot_pickplace",
    }
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 3
    assert all("pick_success_count" in row for row in raw_rows)


def build_pickplace_env(
    centralized_rule: bool, duplicate_assignment: bool = False
) -> MultiRobotEnv:
    assignments = {"robot_0": ["task_obj_0"]}
    robots = {
        "robot_0": {"start": [1, 1], "goal": None, "carrying_object": None},
        "robot_1": {"start": [1, 1], "goal": None, "carrying_object": None},
    }
    if duplicate_assignment:
        assignments["robot_1"] = ["task_obj_0"]
    else:
        robots["robot_1"]["start"] = [4, 4]
    data = {
        "name": "pickplace_lock_test",
        "map": {"width": 5, "height": 5, "obstacles": []},
        "robots": robots,
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
            "assignments": assignments,
        },
        "centralized_rule": centralized_rule,
        "max_steps": 20,
        "render": False,
    }
    scenario = load_scenario_data(data)
    return MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=scenario.centralized_rule,
        pickplace_config=scenario.pickplace,
        zones=scenario.zones,
    )


def write_pickplace_profile_inputs(tmp_path):
    profile_path = tmp_path / "profile.json"
    candidates_path = tmp_path / "candidates.json"
    sample = {
        "index": 8,
        "input": "Pick the bottle and place it.",
        "has_navigation": True,
        "has_pick_place": True,
        "has_operation": True,
        "parse_error": None,
        "xml_node_count": 20,
        "action_condition_names": ["Pick", "Place", "Gripper"],
        "matched_pick_place_keywords": ["Pick", "Place"],
        "matched_operation_keywords": ["Bottle", "Object"],
    }
    profile_path.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")
    candidates_path.write_text(
        json.dumps({"pick_place": [{"index": 8}]}),
        encoding="utf-8",
    )
    return profile_path, candidates_path
