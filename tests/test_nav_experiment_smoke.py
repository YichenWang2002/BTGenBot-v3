import json

import yaml

from src.experiments.run_nav_experiment import run_suite
from src.scenarios.nav_generator import generate_nav_scenarios


def test_nav_experiment_smoke_runs_one_scenario(tmp_path):
    profile_path, candidates_path = write_profile_inputs(tmp_path)
    generated_dir = tmp_path / "generated"
    generate_nav_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=generated_dir,
        num_scenarios=1,
        num_robots=3,
        num_waypoints=3,
        seeds=[0],
        pure_candidates_out=tmp_path / "nav_pure_candidates.json",
    )
    suite_path = tmp_path / "suite.yaml"
    output_dir = tmp_path / "results"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "nav_smoke_test",
                "scenario_manifest": str(generated_dir / "manifest.json"),
                "methods": [
                    "single_robot_sequential",
                    "naive_multi_robot",
                    "centralized_rule_multi_robot",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path)

    assert {row["method"] for row in rows} == {
        "single_robot_sequential",
        "naive_multi_robot",
        "centralized_rule_multi_robot",
    }
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 3
    assert all("trace_path" in row for row in raw_rows)


def write_profile_inputs(tmp_path):
    profile_path = tmp_path / "profile.json"
    candidates_path = tmp_path / "candidates.json"
    samples = [
        {
            "index": 10,
            "input": "Navigate to a goal.",
            "has_navigation": True,
            "has_recovery": False,
            "has_operation": False,
            "has_pick_place": False,
            "parse_error": None,
            "xml_node_count": 8,
        }
    ]
    profile_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    candidates_path.write_text(
        json.dumps(
            {
                "navigation": [
                    {"index": 10, "input": "Navigate to a goal.", "xml_node_count": 8}
                ]
            }
        ),
        encoding="utf-8",
    )
    return profile_path, candidates_path
