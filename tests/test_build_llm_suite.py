import json
from pathlib import Path

import yaml

from src.repair.build_llm_suite import build_suite, main as build_suite_main
from src.repair.refinement_loop import run_suite


def test_build_llm_suite_selects_existing_paths(tmp_path):
    nav_manifest = write_manifest(tmp_path, "nav", existing=2, missing=1)
    recovery_manifest = write_manifest(tmp_path, "recovery", existing=1, missing=0)
    pickplace_manifest = write_manifest(tmp_path, "pickplace", existing=1, missing=0)

    suite, _ = build_suite(
        nav_manifest,
        recovery_manifest,
        pickplace_manifest,
        tmp_path / "suite.yaml",
        per_family=10,
        output_dir=tmp_path / "results",
    )

    paths = [Path(item["path"]) for item in suite["scenarios"]]
    assert paths
    assert all(path.exists() for path in paths)
    assert not any("missing" in str(path) for path in paths)


def test_build_llm_suite_counts_per_family(tmp_path):
    nav_manifest = write_manifest(tmp_path, "nav", existing=3, missing=0)
    recovery_manifest = write_manifest(tmp_path, "recovery", existing=3, missing=0)
    pickplace_manifest = write_manifest(tmp_path, "pickplace", existing=3, missing=0)

    _, summary = build_suite(
        nav_manifest,
        recovery_manifest,
        pickplace_manifest,
        tmp_path / "suite.yaml",
        per_family=2,
        output_dir=tmp_path / "results",
    )

    assert summary["selected_counts"] == {"nav": 2, "recovery": 2, "pickplace": 2}
    assert summary["total_scenarios"] == 6
    assert summary["total_runs_expected"] == 12


def test_build_llm_suite_writes_yaml(tmp_path):
    out = run_builder_cli(tmp_path)
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert payload["name"] == "llm_repair_small"
    assert payload["methods"] == ["llm_only_multi_robot", "centralized_rule_multi_bt"]
    assert payload["scenarios"][0]["path"]
    assert payload["scenarios"][0]["scenario_family"] in {"nav", "recovery", "pickplace"}


def test_build_llm_suite_writes_manifest_summary(tmp_path):
    out = run_builder_cli(tmp_path)
    summary_path = out.with_name("llm_repair_small_manifest_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["suite_path"] == str(out)
    assert summary["total_runs_expected"] == summary["total_scenarios"] * 2
    assert summary["scenario_paths"]


def test_refinement_loop_accepts_scenario_dict_format(tmp_path):
    suite_path = write_refinement_suite(tmp_path)

    rows = run_suite(
        suite_path,
        backend_name="mock",
        max_iters_override=3,
        output_dir_override=tmp_path / "override_results",
    )

    assert len(rows) == 1
    assert rows[0]["scenario_family"] == "nav"
    assert rows[0]["success"] is True


def test_scenario_family_in_raw_rows(tmp_path):
    suite_path = write_refinement_suite(tmp_path)
    output_dir = tmp_path / "override_results"

    run_suite(
        suite_path,
        backend_name="mock",
        max_iters_override=3,
        output_dir_override=output_dir,
    )

    raw_row = json.loads((output_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert raw_row["scenario_family"] == "nav"


def test_output_dir_override_for_backend(tmp_path):
    suite_path = write_refinement_suite(tmp_path)
    output_dir = tmp_path / "mock_results"

    run_suite(
        suite_path,
        backend_name="mock",
        max_iters_override=3,
        output_dir_override=output_dir,
    )

    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert not (tmp_path / "suite_results" / "raw.jsonl").exists()


def run_builder_cli(tmp_path):
    out = tmp_path / "llm_repair_small.yaml"
    nav_manifest = write_manifest(tmp_path, "nav", existing=2, missing=0)
    recovery_manifest = write_manifest(tmp_path, "recovery", existing=2, missing=0)
    pickplace_manifest = write_manifest(tmp_path, "pickplace", existing=2, missing=0)
    build_suite_main(
        [
            "--nav-manifest",
            str(nav_manifest),
            "--recovery-manifest",
            str(recovery_manifest),
            "--pickplace-manifest",
            str(pickplace_manifest),
            "--out",
            str(out),
            "--per-family",
            "1",
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )
    return out


def write_manifest(tmp_path, family, existing, missing):
    scenario_dir = tmp_path / family
    scenario_dir.mkdir()
    rows = []
    for index in range(existing):
        scenario_path = scenario_dir / f"{family}_{index}.yaml"
        write_nav_scenario(scenario_path)
        rows.append({"scenario_path": str(scenario_path), "dataset_index": index, "seed": index})
    for index in range(missing):
        rows.append(
            {
                "scenario_path": str(scenario_dir / f"missing_{index}.yaml"),
                "dataset_index": 100 + index,
                "seed": index,
            }
        )
    manifest_path = tmp_path / f"{family}_manifest.json"
    manifest_path.write_text(json.dumps({"scenarios": rows}), encoding="utf-8")
    return manifest_path


def write_refinement_suite(tmp_path):
    scenario_path = tmp_path / "nav_easy_case.yaml"
    write_nav_scenario(scenario_path)
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "llm_repair_dict_test",
                "scenarios": [{"path": str(scenario_path), "scenario_family": "nav"}],
                "methods": ["centralized_rule_multi_bt"],
                "output_dir": str(tmp_path / "suite_results"),
                "max_iters": 3,
            }
        ),
        encoding="utf-8",
    )
    return suite_path


def write_nav_scenario(path):
    path.write_text(
        yaml.safe_dump(
            {
                "name": "repair_nav",
                "source": {"dataset_index": 1},
                "map": {"width": 8, "height": 5, "obstacles": []},
                "robots": {
                    "robot_0": {"start": [1, 1], "goal": None},
                    "robot_1": {"start": [1, 3], "goal": None},
                },
                "waypoints": [
                    {"id": "wp_0", "position": [6, 1]},
                    {"id": "wp_1", "position": [6, 3]},
                ],
                "task": {
                    "type": "pure_navigation",
                    "assignments": {"robot_0": ["wp_0"], "robot_1": ["wp_1"]},
                },
                "centralized_rule": True,
                "max_steps": 30,
                "render": False,
            }
        ),
        encoding="utf-8",
    )
