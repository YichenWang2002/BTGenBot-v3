import csv
import json
import os

import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.experiments.run_pickplace_experiment import run_suite


def test_pickplace_coordination_ablation_outputs_requested_summary(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_path": "configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml",
                        "dataset_index": 45,
                        "seed": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "results"
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "pickplace_coordination_ablation_test",
                "scenario_manifest": str(manifest_path),
                "methods": [
                    "naive_multi_robot_pickplace",
                    "centralized_rule_multi_robot_pickplace",
                    "centralized_rule_with_approach_lock_pickplace",
                    "centralized_rule_with_approach_and_corridor_lock_pickplace",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
                "summary_style": "coordination_ablation",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path)

    assert len(rows) == 4
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows
    for field in [
        "success_rate",
        "makespan",
        "collision_event_count",
        "actual_collision_count",
        "collision_count",
        "motion_wait_count",
        "rule_prevented_motion_conflict_count",
        "resource_conflict_count",
        "approach_zone_wait_count",
        "corridor_wait_count",
        "rule_prevented_approach_conflict_count",
        "timeout_rate",
    ]:
        assert field in summary_rows[0]


def test_pickplace_approach_radius_sweep_outputs_new_metrics(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_path": "configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml",
                        "dataset_index": 45,
                        "seed": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "results"
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "pickplace_approach_radius_sweep_test",
                "scenario_manifest": str(manifest_path),
                "methods": [
                    "centralized_rule_multi_robot_pickplace",
                    "approach_radius_0",
                    "approach_radius_1",
                    "approach_radius_2",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
                "summary_style": "coordination_ablation",
                "coordination": {
                    "max_wait_ticks": 3,
                    "reassignment_wait_ticks": 8,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path)

    assert len(rows) == 4
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    methods = {row["method"] for row in summary_rows}
    assert {"approach_radius_0", "approach_radius_1", "approach_radius_2"} <= methods
    for field in [
        "collision_event_count",
        "actual_collision_count",
        "approach_zone_wait_count",
        "approach_lock_hold_time",
        "approach_lock_starvation_count",
        "approach_lock_reassignment_count",
    ]:
        assert field in summary_rows[0]
