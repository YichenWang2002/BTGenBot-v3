import csv
import hashlib
import json
from pathlib import Path

import yaml

from src.env.scenario_loader import load_scenario_data
from src.experiments.run_pickplace_experiment import run_suite
from src.scenarios.dag_pickplace_generator import generate_dag_pickplace_scenarios


def test_dag_pickplace_generator_writes_manifest_and_valid_dag(tmp_path):
    manifest_path = write_source_manifest(tmp_path, seeds=[0, 1])

    manifest = generate_dag_pickplace_scenarios(
        manifest_path=manifest_path,
        out_dir=tmp_path / "dag_pickplace",
        num_scenarios=2,
        seeds=[0, 1],
    )

    assert len(manifest["scenarios"]) == 2
    assert (tmp_path / "dag_pickplace" / "manifest.json").exists()
    first = yaml.safe_load(
        Path(manifest["scenarios"][0]["scenario_path"]).read_text(encoding="utf-8")
    )
    second = yaml.safe_load(
        Path(manifest["scenarios"][1]["scenario_path"]).read_text(encoding="utf-8")
    )

    assert first["name"].startswith("dag_pickplace_")
    assert first["task"]["type"] == "dag_pickplace"
    assert first["task"]["cooperative_dependency"] is True
    assert second["task"]["cooperative_dependency"] is False
    assert first["pickplace"]["tasks"][0]["pick_task_id"] == "pick_obj_0"
    assert first["pickplace"]["tasks"][0]["place_task_id"] == "place_obj_0"

    dag_tasks = {task["task_id"]: task for task in first["task_dag"]["tasks"]}
    assert dag_tasks["pick_obj_0"]["expected_action_id"] == "PickObject"
    assert dag_tasks["place_obj_0"]["expected_action_id"] == "PlaceObject"
    assert dag_tasks["final_check"]["expected_action_id"] == "FinalCheck"

    dependencies = {
        (dependency["source"], dependency["target"])
        for dependency in first["task_dag"]["dependencies"]
    }
    assert ("pick_obj_0", "place_obj_0") in dependencies
    assert ("place_obj_0", "final_check") in dependencies
    assert ("place_obj_0", "place_obj_2") in dependencies
    assert ("place_obj_1", "place_obj_2") in dependencies

    loaded = load_scenario_data(first)
    assert loaded.task_dag is not None
    assert len(loaded.task_dag.tasks) == 7


def test_dag_pickplace_generator_does_not_modify_dataset(tmp_path):
    dataset_path = Path("dataset/bt_dataset.json")
    before = sha256(dataset_path) if dataset_path.exists() else None
    manifest_path = write_source_manifest(tmp_path, seeds=[0])

    generate_dag_pickplace_scenarios(
        manifest_path=manifest_path,
        out_dir=tmp_path / "dag_pickplace",
        num_scenarios=1,
        seeds=[0],
    )

    if before is not None:
        assert sha256(dataset_path) == before


def test_dag_pickplace_smoke_suite_methods_output_dag_metrics(tmp_path):
    manifest_path = write_source_manifest(tmp_path, seeds=[0])
    dag_manifest = generate_dag_pickplace_scenarios(
        manifest_path=manifest_path,
        out_dir=tmp_path / "dag_pickplace",
        num_scenarios=1,
        seeds=[0],
    )
    output_dir = tmp_path / "results"
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "dag_pickplace_test",
                "scenario_manifest": str(tmp_path / "dag_pickplace" / "manifest.json"),
                "methods": [
                    "dag_enabled_centralized",
                    "ignore_dag_centralized",
                    "dag_enabled_without_approach_lock",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
                "summary_style": "dag_pickplace",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path)

    assert len(rows) == 3
    assert {row["method"] for row in rows} == {
        "dag_enabled_centralized",
        "ignore_dag_centralized",
        "dag_enabled_without_approach_lock",
    }
    assert all(row["dag_enabled"] for row in rows)
    assert all(row["dag_completed_task_count"] == 7 for row in rows)
    assert dag_manifest["scenarios"][0]["cooperative_dependency"] is True
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows
    for field in [
        "success_rate",
        "makespan",
        "dependency_wait_count",
        "dag_violation_count",
        "dag_completed_task_count",
        "collision_event_count",
        "actual_collision_count",
        "resource_conflict_count",
        "approach_zone_wait_count",
        "timeout_rate",
    ]:
        assert field in summary_rows[0]


def write_source_manifest(tmp_path, seeds):
    scenarios = []
    for seed in seeds:
        scenario = base_pickplace_scenario(seed)
        scenario_path = tmp_path / f"pickplace_hard_idx99_seed{seed}_robots3.yaml"
        scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
        scenarios.append(
            {
                "scenario_path": str(scenario_path),
                "dataset_index": 99,
                "seed": seed,
                "num_robots": 3,
                "num_objects": 3,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"scenarios": scenarios}), encoding="utf-8")
    return manifest_path


def base_pickplace_scenario(seed):
    return {
        "name": f"pickplace_hard_idx99_seed{seed}_robots3",
        "source": {"dataset_index": 99, "task_family": "navigation_pickplace"},
        "map": {"width": 8, "height": 5, "obstacles": []},
        "robots": {
            "robot_0": {"start": [1, 1], "goal": None, "carrying_object": None},
            "robot_1": {"start": [1, 2], "goal": None, "carrying_object": None},
            "robot_2": {"start": [1, 3], "goal": None, "carrying_object": None},
        },
        "objects": {
            "obj_0": {
                "position": [4, 1],
                "target_position": [6, 1],
                "held_by": None,
                "status": "available",
            },
            "obj_1": {
                "position": [4, 2],
                "target_position": [6, 2],
                "held_by": None,
                "status": "available",
            },
            "obj_2": {
                "position": [4, 3],
                "target_position": [6, 3],
                "held_by": None,
                "status": "available",
            },
        },
        "zones": {
            "pickup_zones": {"pickup_zone_0": {"cells": [[4, 1], [4, 2], [4, 3]]}},
            "drop_zones": {"drop_zone_0": {"cells": [[6, 1], [6, 2], [6, 3]]}},
        },
        "pickplace": {
            "enabled": True,
            "pick_fail_prob": 0.0,
            "place_fail_prob": 0.0,
            "allow_reassignment": False,
            "lock_ttl": 8,
            "seed": seed,
            "tasks": [
                pickplace_task("obj_0", "robot_0", [4, 1], [6, 1]),
                pickplace_task("obj_1", "robot_1", [4, 2], [6, 2]),
                pickplace_task("obj_2", "robot_2", [4, 3], [6, 3]),
            ],
        },
        "task": {
            "type": "navigation_pickplace",
            "assignment_strategy": "nearest_robot",
            "baseline_type": "centralized_rule_multi_robot_pickplace",
            "assignments": {
                "robot_0": ["task_obj_0"],
                "robot_1": ["task_obj_1"],
                "robot_2": ["task_obj_2"],
            },
        },
        "centralized_rule": True,
        "max_steps": 80,
        "render": False,
    }


def pickplace_task(object_id, robot_id, pickup, drop):
    return {
        "task_id": f"task_{object_id}",
        "object_id": object_id,
        "pickup_position": pickup,
        "drop_position": drop,
        "assigned_robot": robot_id,
        "status": "assigned",
        "attempts": 0,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
