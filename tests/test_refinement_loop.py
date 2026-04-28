import json
import csv
from pathlib import Path

import yaml

from src.llm.generate_multi_bt import inject_llm_assignment
from src.llm.backends.mock import MockLLMBackend
from src.llm.output_parser import parse_llm_json
from src.llm.prompt_builder import build_prompt
from src.repair.refinement_loop import run_suite


def test_mock_backend_first_fails_second_repairs(tmp_path):
    scenario_path = write_nav_scenario(tmp_path)
    backend = MockLLMBackend()
    prompt = build_prompt(scenario_path, "centralized_rule_multi_bt")

    first = parse_llm_json(backend.generate(prompt))
    second = parse_llm_json(
        backend.generate(
            build_prompt(
                scenario_path,
                "centralized_rule_multi_bt",
                previous_failure_report={
                    "status": "failed",
                    "failure_types": ["duplicate_task"],
                },
                previous_robot_trees=first["robot_trees"],
            )
        )
    )

    first_tasks = [task for row in first["assignment"] for task in row["task_ids"]]
    second_tasks = [task for row in second["assignment"] for task in row["task_ids"]]
    assert first_tasks.count("wp_0") > 1
    assert second_tasks.count("wp_0") == 1
    assert second_tasks.count("wp_1") == 1


def test_refinement_loop_saves_iteration_artifacts(tmp_path):
    suite_path = write_suite(tmp_path)

    rows = run_suite(suite_path, backend_name="mock", max_iters_override=3)

    assert len(rows) == 1
    run_dir = tmp_path / "results" / "runs" / rows[0]["run_id"]
    assert (run_dir / "iter_0" / "prompt.txt").exists()
    assert (run_dir / "iter_0" / "raw_response.txt").exists()
    assert (run_dir / "iter_0" / "parsed_response.json").exists()
    assert (run_dir / "iter_0" / "static_validation.json").exists()
    assert (run_dir / "iter_0" / "failure_report.json").exists()
    assert (run_dir / "iter_0" / "metrics.json").exists()
    assert (run_dir / "iter_0" / "trace.json").exists()
    assert (tmp_path / "results" / "raw.jsonl").exists()
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (tmp_path / "results" / "summary.md").exists()


def test_refinement_loop_mock_eventually_succeeds(tmp_path):
    suite_path = write_suite(tmp_path)

    rows = run_suite(suite_path, backend_name="mock", max_iters_override=3)

    assert rows[0]["success"] is True
    assert rows[0]["num_iters"] == 2
    assert rows[0]["final_failure_types"] == []
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert raw_rows[0]["success"] is True


def test_refinement_loop_max_runs(tmp_path):
    suite_path = write_suite(tmp_path, methods=["llm_only_multi_robot", "centralized_rule_multi_bt"])

    rows = run_suite(suite_path, backend_name="mock", max_iters_override=3, max_runs=1)

    assert len(rows) == 1
    raw_rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_rows) == 1


def test_refinement_loop_provider_error_creates_raw_row(tmp_path, monkeypatch):
    suite_path = write_suite(tmp_path)
    configure_openai_env(monkeypatch)
    monkeypatch.setattr("src.repair.refinement_loop.build_backend", lambda _: FailingBackend())

    rows = run_suite(suite_path, backend_name="openai-compatible", max_iters_override=3)

    assert len(rows) == 1
    assert rows[0]["success"] is False
    assert rows[0]["provider_error"] is True
    assert rows[0]["final_failure_types"] == ["provider_error"]
    assert "sk-secret" not in rows[0]["error_message"]
    raw_row = json.loads((tmp_path / "results" / "raw.jsonl").read_text(encoding="utf-8"))
    assert raw_row["provider_error"] is True
    assert raw_row["xml_valid"] is False
    run_dir = tmp_path / "results" / "runs" / rows[0]["run_id"] / "iter_0"
    report = json.loads((run_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert report["failure_types"] == ["provider_error"]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "raw_response.txt").exists()
    assert (run_dir / "static_validation.json").exists()


def test_refinement_loop_provider_error_does_not_abort_by_default(tmp_path, monkeypatch):
    suite_path = write_suite(tmp_path, methods=["llm_only_multi_robot", "centralized_rule_multi_bt"])
    configure_openai_env(monkeypatch)
    monkeypatch.setattr("src.repair.refinement_loop.build_backend", lambda _: FailingBackend())

    rows = run_suite(suite_path, backend_name="openai-compatible", max_iters_override=3)

    assert len(rows) == 2
    assert all(row["provider_error"] for row in rows)


def test_refinement_loop_fail_fast_aborts(tmp_path, monkeypatch):
    suite_path = write_suite(tmp_path, methods=["llm_only_multi_robot", "centralized_rule_multi_bt"])
    configure_openai_env(monkeypatch)
    monkeypatch.setattr("src.repair.refinement_loop.build_backend", lambda _: FailingBackend())

    rows = run_suite(
        suite_path,
        backend_name="openai-compatible",
        max_iters_override=3,
        fail_fast=True,
    )

    assert len(rows) == 1
    assert rows[0]["provider_error"] is True


def test_summary_includes_provider_error_rate(tmp_path, monkeypatch):
    suite_path = write_suite(tmp_path)
    configure_openai_env(monkeypatch)
    monkeypatch.setattr("src.repair.refinement_loop.build_backend", lambda _: FailingBackend())

    run_suite(suite_path, backend_name="openai-compatible", max_iters_override=3)

    summary_rows = list(csv.DictReader((tmp_path / "results" / "summary.csv").open()))
    assert summary_rows
    assert "provider_error_count" in summary_rows[0]
    assert "provider_error_rate" in summary_rows[0]
    assert any(row["provider_error_rate"] == "1" or row["provider_error_rate"] == "1.0" for row in summary_rows)


def test_approach_radius_method_uses_centralized_prompt_schema():
    prompt = build_prompt(
        "configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml",
        "centralized_rule_with_approach_radius_1",
    )

    assert "Strict XML schema for centralized_rule_multi_bt" in prompt


def test_approach_radius_method_injects_coordination_without_changing_default():
    scenario_data = yaml.safe_load(
        Path(
            "configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml"
        ).read_text(encoding="utf-8")
    )
    payload = {
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["task_obj_0"]},
            {"robot_id": "robot_1", "task_ids": ["task_obj_1"]},
            {"robot_id": "robot_2", "task_ids": ["task_obj_2"]},
        ]
    }

    default_data = inject_llm_assignment(
        scenario_data, payload, "centralized_rule_multi_bt"
    )
    approach_data = inject_llm_assignment(
        scenario_data, payload, "centralized_rule_with_approach_radius_1"
    )

    assert default_data["centralized_rule"] is True
    assert default_data.get("coordination", {}).get("enable_approach_zone_lock") is None
    assert approach_data["centralized_rule"] is True
    assert approach_data["coordination"] == {
        "enable_cell_reservation": True,
        "enable_edge_reservation": True,
        "enable_resource_lock": True,
        "enable_recovery_lock": True,
        "enable_approach_zone_lock": True,
        "approach_radius": 1,
        "max_wait_ticks": 3,
        "reassignment_wait_ticks": 8,
    }


def test_llm_pickplace_approach_summary_and_artifacts(tmp_path):
    output_dir = tmp_path / "results"
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "llm_pickplace_approach_v2_test",
                "scenarios": [
                    {
                        "path": "configs/generated/pickplace_hard/pickplace_hard_idx45_seed0_robots3.yaml",
                        "scenario_family": "pickplace",
                    }
                ],
                "methods": [
                    "centralized_rule_multi_bt",
                    "centralized_rule_with_approach_radius_1",
                ],
                "summary_style": "llm_pickplace_approach_v2",
                "output_dir": str(output_dir),
                "max_iters": 1,
            }
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path, backend_name="mock")

    assert len(rows) == 2
    summary_rows = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert summary_rows
    for field in [
        "success_rate",
        "avg_num_iters",
        "collision_count",
        "collision_event_count",
        "actual_collision_count",
        "resource_conflict_count",
        "approach_zone_wait_count",
        "approach_lock_hold_time",
        "approach_lock_starvation_count",
        "approach_lock_reassignment_count",
        "timeout_rate",
        "provider_error_rate",
    ]:
        assert field in summary_rows[0]
    run_dir = output_dir / "runs" / rows[0]["run_id"] / "iter_0"
    for artifact in [
        "prompt.txt",
        "raw_response.txt",
        "parsed_response.json",
        "static_validation.json",
        "failure_report.json",
        "metrics.json",
        "trace.json",
    ]:
        assert (run_dir / artifact).exists()


def test_refinement_loop_dag_llm_mock_suite_from_manifest(tmp_path):
    scenario_path = tmp_path / "dag_mock_scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "dag_mock_case",
                "source": {"dataset_index": 7, "task_family": "dag_pickplace"},
                "map": {"width": 6, "height": 4, "obstacles": []},
                "robots": {"robot_0": {"start": [1, 1], "goal": None}},
                "objects": {
                    "obj_0": {
                        "position": [2, 1],
                        "target_position": [3, 1],
                        "held_by": None,
                        "status": "available",
                    }
                },
                "zones": {
                    "pickup_zones": {"pickup_zone_0": {"cells": [[2, 1]]}},
                    "drop_zones": {"drop_zone_0": {"cells": [[3, 1]]}},
                },
                "pickplace": {
                    "enabled": True,
                    "pick_fail_prob": 0.0,
                    "place_fail_prob": 0.0,
                    "allow_reassignment": False,
                    "lock_ttl": 8,
                    "seed": 0,
                    "tasks": [
                        {
                            "task_id": "task_obj_0",
                            "object_id": "obj_0",
                            "pickup_position": [2, 1],
                            "drop_position": [3, 1],
                            "assigned_robot": "robot_0",
                            "status": "assigned",
                            "attempts": 0,
                            "pick_task_id": "pick_obj_0",
                            "place_task_id": "place_obj_0",
                        }
                    ],
                },
                "task": {
                    "type": "dag_pickplace",
                    "assignments": {"robot_0": ["task_obj_0"]},
                    "baseline_type": "dag_pickplace",
                },
                "task_dag": {
                    "tasks": [
                        {
                            "task_id": "pick_obj_0",
                            "task_type": "pick",
                            "object_id": "obj_0",
                            "assigned_robot": "robot_0",
                            "expected_action_id": "PickObject",
                        },
                        {
                            "task_id": "place_obj_0",
                            "task_type": "place",
                            "object_id": "obj_0",
                            "assigned_robot": "robot_0",
                            "expected_action_id": "PlaceObject",
                        },
                        {
                            "task_id": "final_check",
                            "task_type": "final_check",
                            "assigned_robot": "robot_0",
                            "expected_action_id": "FinalCheck",
                        },
                    ],
                    "dependencies": [
                        {"source": "pick_obj_0", "target": "place_obj_0"},
                        {"source": "place_obj_0", "target": "final_check"},
                    ],
                },
                "centralized_rule": False,
                "max_steps": 20,
                "render": False,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_path": str(scenario_path),
                        "dataset_index": 7,
                        "seed": 0,
                        "num_robots": 1,
                        "num_objects": 1,
                        "scenario_family": "dag_pickplace",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    output_dir = tmp_path / "results"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "dag_llm_mock_smoke_test",
                "scenario_manifest": str(manifest_path),
                "methods": [
                    "llm_dag_without_dependency_prompt",
                    "llm_dag_with_dependency_prompt",
                    "llm_dag_with_dependency_prompt_and_centralized_rules",
                ],
                "max_scenarios": 1,
                "output_dir": str(output_dir),
                "max_iters": 3,
                "summary_style": "dag_llm_mock",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rows = run_suite(suite_path, backend_name="mock", max_iters_override=3)

    assert len(rows) == 3
    assert any(row["success"] for row in rows)
    assert any(
        row["method"] == "llm_dag_without_dependency_prompt" and row["success"]
        for row in rows
    )
    summary_rows = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert summary_rows
    for field in [
        "runs",
        "success_rate",
        "avg_num_iters",
        "xml_valid_rate",
        "assignment_valid_rate",
        "dag_static_error_count",
        "missing_dependency_wait_count",
        "invalid_task_id_count",
        "dag_violation_count",
        "dependency_wait_count",
        "dag_task_completion_rate",
        "resource_conflict_count",
        "collision_event_count",
        "actual_collision_count",
        "timeout_rate",
    ]:
        assert field in summary_rows[0]


class FailingBackend:
    model = "minimax-m2.5-free"
    provider_profile = "opencode-zen"
    api_key = "sk-secret"

    def generate(self, prompt):
        raise RuntimeError("provider failed with key sk-secret")


def configure_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://opencode.ai/zen/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_MODEL", "minimax-m2.5-free")
    monkeypatch.setenv("OPENAI_PROVIDER_PROFILE", "opencode-zen")


def write_suite(tmp_path, methods=None):
    scenario_path = write_nav_scenario(tmp_path)
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "llm_repair_test",
                "scenarios": [str(scenario_path)],
                "methods": methods or ["centralized_rule_multi_bt"],
                "output_dir": str(tmp_path / "results"),
                "max_iters": 3,
            }
        ),
        encoding="utf-8",
    )
    return suite_path


def write_nav_scenario(tmp_path):
    scenario_path = tmp_path / "nav.yaml"
    scenario_path.write_text(
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
    return scenario_path
