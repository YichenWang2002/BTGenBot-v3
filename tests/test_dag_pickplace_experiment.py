import csv
import hashlib
import json
from pathlib import Path

import yaml

from src.experiments.run_dag_pickplace_experiment import run_suite


def test_dag_pickplace_experiment_outputs_requested_artifacts(tmp_path):
    dataset_path = Path("dataset/bt_dataset.json")
    before_hash = sha256(dataset_path) if dataset_path.exists() else None
    manifest_path = write_manifest(tmp_path)
    output_dir = tmp_path / "results"
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "dag_pickplace_test",
                "scenario_manifest": str(manifest_path),
                "methods": [
                    "dag_enabled_centralized",
                    "ignore_dag_centralized",
                    "dag_enabled_without_approach_lock",
                    "no_dag_naive",
                    "oracle_topological_sequential",
                    "topological_sequential_baseline",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
                "coordination": {
                    "enable_cell_reservation": True,
                    "enable_edge_reservation": True,
                    "enable_resource_lock": True,
                    "enable_recovery_lock": True,
                    "enable_approach_zone_lock": True,
                    "approach_radius": 1,
                    "max_wait_ticks": 3,
                    "reassignment_wait_ticks": 8,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path)

    assert len(rows) == 6
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert len(list((output_dir / "traces").glob("*.json"))) == 6

    by_method = {row["method"]: row for row in rows}
    assert by_method["dag_enabled_centralized"]["dag_violation_count"] == 0
    assert by_method["ignore_dag_centralized"]["dag_violation_count"] > 0 or by_method[
        "ignore_dag_centralized"
    ]["final_check_early"]
    assert by_method["no_dag_naive"]["dag_enabled"] is False
    assert by_method["oracle_topological_sequential"]["dag_violation_count"] == 0
    assert by_method["topological_sequential_baseline"]["dag_violation_count"] == 0
    assert by_method["topological_sequential_baseline"]["first_uncompleted_task"] is None
    assert by_method["topological_sequential_baseline"]["final_check_completed"] is True
    assert (
        by_method["topological_sequential_baseline"]["makespan"]
        >= by_method["dag_enabled_centralized"]["makespan"]
    )

    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 6
    assert all(Path(row["trace_path"]).exists() for row in raw_rows)
    trace_payload = json.loads(
        Path(by_method["topological_sequential_baseline"]["trace_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert trace_payload["oracle_debug"]["final_check_completed"] is True

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows
    for field in [
        "runs",
        "success_rate",
        "makespan",
        "timeout_rate",
        "dependency_wait_count",
        "dag_violation_count",
        "dag_completed_task_count",
        "dag_task_completion_rate",
        "first_uncompleted_task_count",
        "resource_conflict_count",
        "collision_event_count",
        "actual_collision_count",
        "approach_zone_wait_count",
        "lock_wait_count",
    ]:
        assert field in summary_rows[0]

    if before_hash is not None:
        assert sha256(dataset_path) == before_hash


def write_manifest(tmp_path):
    scenario_path = tmp_path / "dag_pickplace_case.yaml"
    scenario_path.write_text(
        yaml.safe_dump(dag_pickplace_scenario(), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_path": str(scenario_path),
                        "dataset_index": 999,
                        "seed": 0,
                        "num_robots": 3,
                        "num_objects": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def dag_pickplace_scenario():
    return {
        "name": "dag_pickplace_case",
        "source": {"dataset_index": 999, "task_family": "dag_pickplace"},
        "map": {"width": 8, "height": 5, "obstacles": []},
        "robots": {
            "robot_0": {"start": [0, 0], "goal": None, "carrying_object": None},
            "robot_1": {"start": [0, 4], "goal": None, "carrying_object": None},
            "robot_2": {"start": [1, 3], "goal": None, "carrying_object": None},
        },
        "objects": {
            "obj_0": {
                "position": [5, 0],
                "target_position": [6, 0],
                "held_by": None,
                "status": "available",
            },
            "obj_1": {
                "position": [5, 4],
                "target_position": [6, 4],
                "held_by": None,
                "status": "available",
            },
            "obj_2": {
                "position": [1, 3],
                "target_position": [2, 3],
                "held_by": None,
                "status": "available",
            },
        },
        "zones": {
            "pickup_zones": {
                "pickup_zone_0": {"cells": [[5, 0]]},
                "pickup_zone_1": {"cells": [[5, 4]]},
                "pickup_zone_2": {"cells": [[1, 3]]},
            },
            "drop_zones": {
                "drop_zone_0": {"cells": [[6, 0]]},
                "drop_zone_1": {"cells": [[6, 4]]},
                "drop_zone_2": {"cells": [[2, 3]]},
            },
        },
        "pickplace": {
            "enabled": True,
            "pick_fail_prob": 0.0,
            "place_fail_prob": 0.0,
            "allow_reassignment": False,
            "lock_ttl": 8,
            "seed": 0,
            "tasks": [
                pickplace_task("obj_0", "robot_0", [5, 0], [6, 0]),
                pickplace_task("obj_1", "robot_1", [5, 4], [6, 4]),
                pickplace_task("obj_2", "robot_2", [1, 3], [2, 3]),
            ],
        },
        "task": {
            "type": "dag_pickplace",
            "assignment_strategy": "nearest_robot",
            "baseline_type": "dag_pickplace",
            "assignments": {
                "robot_0": ["task_obj_0"],
                "robot_1": ["task_obj_1"],
                "robot_2": ["task_obj_2"],
            },
            "cooperative_dependency": True,
        },
        "task_dag": {
            "tasks": [
                dag_task("pick_obj_0", "pick", "obj_0", "robot_0"),
                dag_task("place_obj_0", "place", "obj_0", "robot_0"),
                dag_task("pick_obj_1", "pick", "obj_1", "robot_1"),
                dag_task("place_obj_1", "place", "obj_1", "robot_1"),
                dag_task("pick_obj_2", "pick", "obj_2", "robot_2"),
                dag_task("place_obj_2", "place", "obj_2", "robot_2"),
                {
                    "task_id": "final_check",
                    "task_type": "final_check",
                    "assigned_robot": "robot_0",
                    "expected_action_id": "FinalCheck",
                },
            ],
            "dependencies": [
                {"source": "pick_obj_0", "target": "place_obj_0"},
                {"source": "pick_obj_1", "target": "place_obj_1"},
                {"source": "pick_obj_2", "target": "place_obj_2"},
                {"source": "place_obj_0", "target": "place_obj_2"},
                {"source": "place_obj_1", "target": "place_obj_2"},
                {"source": "place_obj_0", "target": "final_check"},
                {"source": "place_obj_1", "target": "final_check"},
                {"source": "place_obj_2", "target": "final_check"},
            ],
        },
        "centralized_rule": True,
        "max_steps": 120,
        "render": False,
    }


def pickplace_task(object_id, robot_id, pickup, drop):
    return {
        "task_id": f"task_{object_id}",
        "object_id": object_id,
        "pickup_position": pickup,
        "drop_position": drop,
        "assigned_robot": robot_id,
        "pick_task_id": f"pick_{object_id}",
        "place_task_id": f"place_{object_id}",
        "status": "assigned",
        "attempts": 0,
    }


def dag_task(task_id, task_type, object_id, robot_id):
    return {
        "task_id": task_id,
        "task_type": task_type,
        "object_id": object_id,
        "assigned_robot": robot_id,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
