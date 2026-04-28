import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.audit_trace_collisions import (
    audit_runs_dir,
    audit_trace_payload,
    summarize_batch_rows,
    write_batch_csv,
)


class AuditTraceCollisionsTest(unittest.TestCase):
    def test_no_conflict_trace(self):
        result = audit_trace_payload(
            {
                "metrics": {"collision_count": 0},
                "trace": [
                    {
                        "timestep": 0,
                        "robot_positions": {
                            "robot_0": [0, 0],
                            "robot_1": [1, 0],
                        },
                    },
                    {
                        "timestep": 1,
                        "robot_positions": {
                            "robot_0": [0, 1],
                            "robot_1": [1, 1],
                        },
                    },
                ],
            }
        )

        self.assertEqual(result["audited_collision_count"], 0)
        self.assertEqual(result["audited_vertex_conflict_count"], 0)
        self.assertEqual(result["audited_edge_swap_conflict_count"], 0)
        self.assertTrue(result["match"])

    def test_vertex_conflict_trace(self):
        result = audit_trace_payload(
            [
                {
                    "timestep": 3,
                    "robot_positions": {
                        "robot_0": [2, 2],
                        "robot_1": {"position": [2, 2]},
                        "robot_2": {"x": 4, "y": 5},
                    },
                }
            ]
        )

        self.assertEqual(result["audited_vertex_conflict_count"], 1)
        self.assertEqual(result["audited_collision_count"], 1)
        self.assertEqual(result["first_conflict_timestep"], 3)
        self.assertEqual(result["conflict_examples"][0]["robots"], ["robot_0", "robot_1"])

    def test_edge_swap_conflict_trace(self):
        result = audit_trace_payload(
            {
                "timesteps": [
                    {
                        "timestep": 0,
                        "robot_positions": [
                            {"robot_id": "robot_0", "position": [0, 0]},
                            {"robot_id": "robot_1", "position": [1, 0]},
                        ],
                    },
                    {
                        "timestep": 1,
                        "robot_positions": [
                            {"robot_id": "robot_0", "position": [1, 0]},
                            {"robot_id": "robot_1", "position": [0, 0]},
                        ],
                    },
                ]
            }
        )

        self.assertEqual(result["audited_vertex_conflict_count"], 0)
        self.assertEqual(result["audited_edge_swap_conflict_count"], 1)
        self.assertEqual(result["audited_collision_count"], 1)
        self.assertEqual(result["first_conflict_timestep"], 1)
        self.assertEqual(result["conflict_examples"][0]["type"], "edge_swap")

    def test_reported_and_audited_mismatch(self):
        result = audit_trace_payload(
            {
                "collision_count": 0,
                "steps": [
                    {
                        "timestep": 0,
                        "robot_positions": {
                            "robot_0": [2, 2],
                            "robot_1": [2, 2],
                        },
                    }
                ],
            }
        )

        self.assertEqual(result["reported_collision_count"], 0)
        self.assertEqual(result["audited_collision_count"], 1)
        self.assertFalse(result["match"])

    def test_batch_audit_writes_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "case_centralized_rule_with_approach_radius_1"
            iter_dir = run_dir / "iter_0"
            iter_dir.mkdir(parents=True)
            (iter_dir / "trace.json").write_text(
                json.dumps(
                    {
                        "scenario_name": "case",
                        "metrics": {"success": True, "collision_count": 0},
                        "trace": [
                            {
                                "timestep": 0,
                                "robot_positions": {
                                    "robot_0": [0, 0],
                                    "robot_1": [1, 0],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (iter_dir / "metrics.json").write_text(
                json.dumps({"success": True, "collision_count": 0}),
                encoding="utf-8",
            )

            rows = audit_runs_dir(root / "runs")
            out = root / "collision_audit.csv"
            write_batch_csv(out, rows)
            with out.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(csv_rows[0]["method"], "centralized_rule_with_approach_radius_1")
        self.assertEqual(csv_rows[0]["scenario_name"], "case")
        self.assertEqual(csv_rows[0]["iteration"], "0")
        self.assertEqual(csv_rows[0]["is_final_trace"], "True")
        self.assertTrue(csv_rows[0]["trace_file"].endswith("trace.json"))
        self.assertTrue(csv_rows[0]["metrics_file"].endswith("metrics.json"))
        self.assertTrue(csv_rows[0]["reported_collision_source"].endswith("metrics.json"))
        self.assertEqual(csv_rows[0]["audited_collision_count"], "0")
        self.assertEqual(csv_rows[0]["match"], "True")

    def test_final_only_uses_raw_jsonl_final_trace_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "case_centralized_rule_multi_bt"
            iter_0 = run_dir / "iter_0"
            iter_1 = run_dir / "iter_1"
            iter_0.mkdir(parents=True)
            iter_1.mkdir(parents=True)
            for iteration_dir, collision_count in ((iter_0, 1), (iter_1, 0)):
                (iteration_dir / "trace.json").write_text(
                    json.dumps(
                        {
                            "scenario_name": "case",
                            "trace": [
                                {
                                    "timestep": 0,
                                    "robot_positions": {
                                        "robot_0": [0, 0],
                                        "robot_1": [1, 0],
                                    },
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                (iteration_dir / "metrics.json").write_text(
                    json.dumps({"success": True, "collision_count": collision_count}),
                    encoding="utf-8",
                )
            final_trace_path = iter_1 / "trace.json"
            (root / "raw.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "final_trace_path": str(final_trace_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            all_rows = audit_runs_dir(root / "runs")
            final_rows = audit_runs_dir(root / "runs", final_only=True)

        self.assertEqual(len(all_rows), 2)
        self.assertEqual(len(final_rows), 1)
        self.assertEqual(final_rows[0]["iteration"], 1)
        self.assertTrue(final_rows[0]["is_final_trace"])
        self.assertEqual(final_rows[0]["reported_collision_count"], 0)

    def test_missing_metrics_reports_unknown_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            iter_dir = root / "runs" / "case_centralized_rule_multi_bt" / "iter_0"
            iter_dir.mkdir(parents=True)
            (iter_dir / "trace.json").write_text(
                json.dumps(
                    {
                        "scenario_name": "case",
                        "trace": [
                            {
                                "timestep": 0,
                                "robot_positions": {
                                    "robot_0": [0, 0],
                                    "robot_1": [1, 0],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = audit_runs_dir(root / "runs")

        self.assertEqual(rows[0]["reported_collision_count"], None)
        self.assertEqual(rows[0]["reported_collision_source"], "unknown")
        self.assertEqual(rows[0]["match"], None)

    def test_summary_counts_matches_by_method(self):
        rows = [
            {
                "run_dir": "run_a",
                "trace_file": "trace_a",
                "method": "centralized_rule_multi_bt",
                "reported_collision_count": 0,
                "audited_collision_count": 0,
                "match": True,
            },
            {
                "run_dir": "run_b",
                "trace_file": "trace_b",
                "method": "centralized_rule_with_approach_radius_1",
                "reported_collision_count": 0,
                "audited_collision_count": 0,
                "match": True,
            },
            {
                "run_dir": "run_c",
                "trace_file": "trace_c",
                "method": "centralized_rule_multi_bt",
                "reported_collision_count": 1,
                "audited_collision_count": 0,
                "match": False,
            },
        ]

        summary = summarize_batch_rows(rows)

        self.assertEqual(summary["total_traces_audited"], 3)
        self.assertEqual(summary["total_runs"], 3)
        self.assertEqual(summary["baseline_method_matches"], 1)
        self.assertEqual(summary["approach_method_matches"], 1)
        self.assertEqual(summary["mismatch_count"], 1)
        self.assertEqual(summary["mismatch_examples"][0]["trace_file"], "trace_c")


if __name__ == "__main__":
    unittest.main()
