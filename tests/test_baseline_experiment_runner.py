import csv
import json
import os

import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.experiments.run_baseline_experiment import run_suite


def test_baseline_runner_writes_requested_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    nav_manifest = write_manifest(
        tmp_path / "nav_manifest.json",
        "configs/generated/nav_easy/nav_easy_idx124_seed0_robots3.yaml",
    )
    recovery_manifest = write_manifest(
        tmp_path / "recovery_manifest.json",
        "configs/generated/recovery_medium/recovery_medium_idx184_seed0_robots3.yaml",
    )
    pickplace_manifest = write_manifest(
        tmp_path / "pickplace_manifest.json",
        "configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml",
    )
    output_dir = tmp_path / "baseline_results"
    suite_path = tmp_path / "baseline_suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "baseline_test",
                "output_dir": str(output_dir),
                "max_iters": 2,
                "families": [
                    {
                        "name": "nav",
                        "scenario_manifest": str(nav_manifest),
                        "max_scenarios": 1,
                        "methods": ["single_robot_sequential", "llm_only_multi_robot"],
                    },
                    {
                        "name": "recovery",
                        "scenario_manifest": str(recovery_manifest),
                        "max_scenarios": 1,
                        "methods": ["centralized_without_recovery_lock"],
                    },
                    {
                        "name": "pickplace",
                        "scenario_manifest": str(pickplace_manifest),
                        "max_scenarios": 1,
                        "methods": ["centralized_without_resource_lock"],
                    },
                ],
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
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["method"] for row in raw_rows} == {
        "single_robot_sequential",
        "llm_only_multi_robot",
        "centralized_without_recovery_lock",
        "centralized_without_resource_lock",
    }
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows
    for field in [
        "runs",
        "success_rate",
        "makespan",
        "total_robot_steps",
        "collision_count",
        "vertex_conflict_count",
        "edge_conflict_count",
        "resource_conflict_count",
        "recovery_conflict_count",
        "lock_wait_count",
        "rule_prevented_resource_conflict_count",
        "timeout_rate",
        "tasks_completed",
        "object_success_rate",
    ]:
        assert field in summary_rows[0]


def write_manifest(path, scenario_path):
    path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_path": scenario_path,
                        "dataset_index": 0,
                        "seed": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path
