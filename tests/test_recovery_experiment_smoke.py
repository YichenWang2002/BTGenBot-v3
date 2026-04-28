import json

import yaml

from src.experiments.run_recovery_experiment import run_suite
from src.scenarios.recovery_generator import generate_recovery_scenarios


def test_recovery_experiment_smoke_runs_one_scenario(tmp_path):
    profile_path, candidates_path = write_recovery_profile_inputs(tmp_path)
    generated_dir = tmp_path / "generated"
    generate_recovery_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=generated_dir,
        num_scenarios=1,
        num_robots=3,
        num_waypoints=3,
        seeds=[0],
        block_prob=1.0,
        candidates_out=tmp_path / "recovery_candidates.json",
    )
    output_dir = tmp_path / "results"
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "recovery_smoke_test",
                "scenario_manifest": str(generated_dir / "manifest.json"),
                "methods": [
                    "single_robot_recovery",
                    "naive_multi_robot_recovery",
                    "centralized_rule_multi_robot_recovery",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path)

    assert {row["method"] for row in rows} == {
        "single_robot_recovery",
        "naive_multi_robot_recovery",
        "centralized_rule_multi_robot_recovery",
    }
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 3
    assert all("recovery_attempts" in row for row in raw_rows)


def write_recovery_profile_inputs(tmp_path):
    profile_path = tmp_path / "profile.json"
    candidates_path = tmp_path / "candidates.json"
    sample = {
        "index": 32,
        "input": "Navigate with recovery.",
        "has_navigation": True,
        "has_recovery": True,
        "has_recovery_node": True,
        "parse_error": None,
        "xml_node_count": 11,
        "action_condition_names": ["RecoveryNode", "Wait", "NavigateToPose"],
        "matched_recovery_keywords": ["RecoveryNode", "Wait"],
    }
    profile_path.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")
    candidates_path.write_text(
        json.dumps({"recovery": [{"index": 32}]}), encoding="utf-8"
    )
    return profile_path, candidates_path
