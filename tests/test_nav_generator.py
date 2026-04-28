import hashlib
import json
from pathlib import Path

import yaml

from src.scenarios.nav_generator import generate_nav_scenarios


def test_nav_generator_creates_manifest(tmp_path):
    profile_path, candidates_path = write_profile_inputs(tmp_path)
    out_dir = tmp_path / "generated"
    pure_out = tmp_path / "nav_pure_candidates.json"

    manifest = generate_nav_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=out_dir,
        num_scenarios=2,
        num_robots=3,
        num_waypoints=3,
        seeds=[0, 1],
        pure_candidates_out=pure_out,
    )

    assert len(manifest["scenarios"]) == 2
    assert (out_dir / "manifest.json").exists()
    assert pure_out.exists()
    scenario_path = Path(manifest["scenarios"][0]["scenario_path"])
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    assert scenario["task"]["type"] == "pure_navigation"
    assert len(scenario["robots"]) == 3
    assert len(scenario["waypoints"]) == 3


def test_nav_generator_does_not_modify_dataset(tmp_path):
    dataset_path = Path("dataset/bt_dataset.json")
    before = sha256(dataset_path)
    profile_path, candidates_path = write_profile_inputs(tmp_path)

    generate_nav_scenarios(
        profile_path=profile_path,
        candidates_path=candidates_path,
        out_dir=tmp_path / "generated",
        num_scenarios=1,
        num_robots=1,
        num_waypoints=2,
        seeds=[0],
        pure_candidates_out=tmp_path / "nav_pure_candidates.json",
    )

    assert sha256(dataset_path) == before


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
        },
        {
            "index": 11,
            "input": "Recover while navigating.",
            "has_navigation": True,
            "has_recovery": True,
            "has_operation": False,
            "has_pick_place": False,
            "parse_error": None,
            "xml_node_count": 8,
        },
    ]
    profile_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    candidates_path.write_text(
        json.dumps(
            {
                "navigation": [
                    {"index": 10, "input": "Navigate to a goal.", "xml_node_count": 8},
                    {
                        "index": 11,
                        "input": "Recover while navigating.",
                        "xml_node_count": 8,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return profile_path, candidates_path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

