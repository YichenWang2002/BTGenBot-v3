import json
from pathlib import Path

import yaml

from src.scenarios.recovery_generator import generate_recovery_scenarios


def test_recovery_candidates_generated(tmp_path):
    profile_path, candidates_path = write_recovery_profile_inputs(tmp_path)
    candidates_out = tmp_path / "recovery_candidates.json"

    generate_recovery_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=tmp_path / "generated",
        num_scenarios=1,
        num_robots=3,
        num_waypoints=3,
        seeds=[0],
        block_prob=1.0,
        candidates_out=candidates_out,
    )

    payload = json.loads(candidates_out.read_text(encoding="utf-8"))
    assert payload["recovery_navigation"][0]["index"] == 32
    assert payload["recovery_navigation"][0]["has_recovery_node"] is True


def test_recovery_scenario_yaml_contains_recovery_config(tmp_path):
    profile_path, candidates_path = write_recovery_profile_inputs(tmp_path)
    out_dir = tmp_path / "generated"

    manifest = generate_recovery_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=out_dir,
        num_scenarios=1,
        num_robots=3,
        num_waypoints=3,
        seeds=[0],
        block_prob=1.0,
        candidates_out=tmp_path / "recovery_candidates.json",
    )

    scenario = yaml.safe_load(
        Path(manifest["scenarios"][0]["scenario_path"]).read_text(encoding="utf-8")
    )
    assert scenario["task"]["type"] == "navigation_recovery"
    assert scenario["recovery"]["enabled"] is True
    assert scenario["recovery"]["blocked_cells"]
    assert "ClearEntireCostmap" in scenario["recovery"]["actions"]


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

