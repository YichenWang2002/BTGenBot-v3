import csv
import json
from pathlib import Path

import yaml

from src.experiments.run_safety_ablation_experiment import (
    METHOD_ORDER,
    generate_mock_response,
    parse_family_filter,
    run_suite,
)


def test_safety_ablation_runner_writes_artifacts(tmp_path):
    suite_path = write_small_suite(tmp_path)
    output_dir = tmp_path / "results"

    rows = run_suite(
        suite_path,
        backend_name="mock",
        output_dir_override=output_dir,
        max_scenarios_per_family=1,
    )

    assert len(rows) == 15
    assert (output_dir / "raw.jsonl").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.md").exists()
    assert list((output_dir / "traces").glob("*.json"))

    summary_rows = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert summary_rows
    methods = {row["method"] for row in summary_rows}
    assert set(METHOD_ORDER) <= methods
    scopes = {row["summary_scope"] for row in summary_rows}
    assert {"overall", "by_family"} <= scopes
    for field in [
        "runs",
        "success_rate",
        "avg_num_iters",
        "xml_valid_rate",
        "assignment_valid_rate",
        "resource_request_valid_rate",
        "provider_error_rate",
        "backend",
        "model",
        "collision_event_count",
        "actual_collision_count",
        "approach_zone_wait_count",
        "rule_prevented_motion_conflict_count",
        "rule_prevented_resource_conflict_count",
        "rule_prevented_approach_conflict_count",
    ]:
        assert field in summary_rows[0]


def test_safety_ablation_rejects_unknown_backend(tmp_path):
    suite_path = write_small_suite(tmp_path)

    try:
        run_suite(suite_path, backend_name="unknown-backend")
    except ValueError as exc:
        assert "Unknown LLM backend" in str(exc)
    else:
        raise AssertionError("unknown backend should be rejected")


def test_safety_ablation_full_method_records_approach_field(tmp_path):
    suite_path = write_small_suite(tmp_path)
    output_dir = tmp_path / "results"

    rows = run_suite(
        suite_path,
        backend_name="mock",
        output_dir_override=output_dir,
        max_scenarios_per_family=1,
        max_runs=15,
    )

    full_rows = [row for row in rows if row["method"] == "llm_rule_with_repair_full"]
    assert full_rows
    assert all("approach_zone_wait_count" in row for row in full_rows)


def test_safety_ablation_openai_compatible_backend_records_model(
    tmp_path, monkeypatch
):
    suite_path = write_small_suite(tmp_path)
    output_dir = tmp_path / "results"

    class FakeOpenAIBackend:
        model = "fake-deepseek"

        def generate(self, prompt: str) -> str:
            return generate_mock_response(prompt, repaired=True)

    monkeypatch.setattr(
        "src.experiments.run_safety_ablation_experiment.build_backend",
        lambda name: FakeOpenAIBackend(),
    )

    rows = run_suite(
        suite_path,
        backend_name="openai-compatible",
        output_dir_override=output_dir,
        max_scenarios_per_family=1,
        max_runs=5,
    )

    assert rows
    assert {row["backend"] for row in rows} == {"openai-compatible"}
    assert {row["model"] for row in rows} == {"fake-deepseek"}

    raw_row = json.loads((output_dir / "raw.jsonl").read_text().splitlines()[0])
    assert raw_row["backend"] == "openai-compatible"
    assert raw_row["model"] == "fake-deepseek"

    summary_rows = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert summary_rows
    assert {"overall", "by_family"} <= {row["summary_scope"] for row in summary_rows}
    assert {row["backend"] for row in summary_rows} == {"openai-compatible"}
    assert {row["model"] for row in summary_rows} == {"fake-deepseek"}


def test_safety_ablation_openai_provider_error_is_sanitized(
    tmp_path, monkeypatch
):
    suite_path = write_small_suite(tmp_path)
    output_dir = tmp_path / "results"
    secret = "sk-test-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    class FailingOpenAIBackend:
        model = "fake-deepseek"

        def generate(self, prompt: str) -> str:
            raise RuntimeError(f"provider rejected key {secret}")

    monkeypatch.setattr(
        "src.experiments.run_safety_ablation_experiment.build_backend",
        lambda name: FailingOpenAIBackend(),
    )

    rows = run_suite(
        suite_path,
        backend_name="openai-compatible",
        output_dir_override=output_dir,
        max_scenarios_per_family=1,
        max_runs=1,
    )

    assert len(rows) == 1
    assert rows[0]["provider_error"] is True
    assert secret not in rows[0]["error_message"]
    assert "***" in rows[0]["error_message"]
    assert Path(rows[0]["trace_path"]).exists()

    summary_rows = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert summary_rows[0]["provider_error_rate"] == "1.0"


def test_safety_ablation_balanced_check_selects_one_scenario_per_family(tmp_path):
    suite_path = write_balanced_suite(tmp_path)
    output_dir = tmp_path / "results"

    rows = run_suite(
        suite_path,
        backend_name="mock",
        output_dir_override=output_dir,
        balanced_check=1,
    )

    assert len(rows) == 15
    families = {row["scenario_family"] for row in rows}
    assert families == {"nav", "recovery", "pickplace"}
    scenarios_by_family: dict[str, set[str]] = {}
    for row in rows:
        scenarios_by_family.setdefault(row["scenario_family"], set()).add(row["scenario_name"])
    assert all(len(names) == 1 for names in scenarios_by_family.values())
    methods_by_family: dict[str, set[str]] = {}
    for row in rows:
        methods_by_family.setdefault(row["scenario_family"], set()).add(row["method"])
    assert all(methods == set(METHOD_ORDER) for methods in methods_by_family.values())


def test_safety_ablation_family_filter_limits_selected_families(tmp_path):
    suite_path = write_balanced_suite(tmp_path)
    output_dir = tmp_path / "results"

    rows = run_suite(
        suite_path,
        backend_name="mock",
        output_dir_override=output_dir,
        balanced_check=1,
        families={"nav", "pickplace"},
    )

    assert len(rows) == 10
    assert {row["scenario_family"] for row in rows} == {"nav", "pickplace"}


def test_parse_family_filter():
    assert parse_family_filter(None) is None
    assert parse_family_filter("") is None
    assert parse_family_filter("nav,recovery,pickplace") == {"nav", "recovery", "pickplace"}
    assert parse_family_filter(" nav , pickplace ") == {"nav", "pickplace"}


def write_small_suite(tmp_path: Path) -> Path:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "safety_ablation_test",
                "methods": METHOD_ORDER,
                "families": [
                    {
                        "name": "nav",
                        "scenario_manifest": "configs/generated/nav_easy/manifest.json",
                        "max_scenarios": 1,
                    },
                    {
                        "name": "recovery",
                        "scenario_manifest": "configs/generated/recovery_medium/manifest.json",
                        "max_scenarios": 1,
                    },
                    {
                        "name": "pickplace",
                        "scenario_manifest": "configs/generated/pickplace_hard/manifest.json",
                        "max_scenarios": 1,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite_path


def write_balanced_suite(tmp_path: Path) -> Path:
    suite_path = tmp_path / "balanced_suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "safety_ablation_balanced_test",
                "methods": METHOD_ORDER,
                "families": [
                    {
                        "name": "nav",
                        "scenario_manifest": "configs/generated/nav_easy/manifest.json",
                        "max_scenarios": 3,
                    },
                    {
                        "name": "recovery",
                        "scenario_manifest": "configs/generated/recovery_medium/manifest.json",
                        "max_scenarios": 3,
                    },
                    {
                        "name": "pickplace",
                        "scenario_manifest": "configs/generated/pickplace_hard/manifest.json",
                        "max_scenarios": 3,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite_path
