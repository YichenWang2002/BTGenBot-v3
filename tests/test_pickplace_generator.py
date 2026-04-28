import json
from pathlib import Path

import yaml

from src.scenarios.pickplace_generator import generate_pickplace_scenarios


def test_pickplace_candidates_generated(tmp_path):
    profile_path, candidates_path = write_pickplace_profile_inputs(tmp_path)

    generate_pickplace_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=tmp_path / "generated",
        num_scenarios=1,
        num_robots=3,
        num_objects=2,
        seeds=[0],
        pick_fail_prob=0.0,
        object_unavailable_prob=0.0,
        candidates_out=tmp_path / "pickplace_candidates.json",
    )

    payload = json.loads((tmp_path / "pickplace_candidates.json").read_text())
    assert payload["pickplace"][0]["index"] == 8
    assert "Pick" in payload["pickplace"][0]["matched_pick_place_keywords"]


def test_pickplace_scenario_yaml_contains_objects_and_zones(tmp_path):
    profile_path, candidates_path = write_pickplace_profile_inputs(tmp_path)
    manifest = generate_pickplace_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=tmp_path / "generated",
        num_scenarios=1,
        num_robots=3,
        num_objects=4,
        seeds=[0],
        pick_fail_prob=0.0,
        object_unavailable_prob=0.0,
        candidates_out=tmp_path / "pickplace_candidates.json",
    )

    scenario_path = Path(manifest["scenarios"][0]["scenario_path"])
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))

    assert scenario["task"]["type"] == "navigation_pickplace"
    assert len(scenario["objects"]) == 4
    assert "pickup_zones" in scenario["zones"]
    assert "drop_zones" in scenario["zones"]
    assert scenario["pickplace"]["enabled"] is True
    assert scenario["pickplace"]["tasks"]


def write_pickplace_profile_inputs(tmp_path):
    profile_path = tmp_path / "profile.json"
    candidates_path = tmp_path / "candidates.json"
    sample = {
        "index": 8,
        "input": "Pick the bottle and place it on the table.",
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
