import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.paper_tables import build_paper_rows, read_audit_actual_by_method


class PaperTablesTest(unittest.TestCase):
    def test_build_paper_rows_uses_audit_actual_collisions(self):
        rows = [
            {
                "method": "centralized_rule_multi_bt",
                "collision_count": "51.6",
                "collision_event_count": "42.0",
                "approach_zone_wait_count": "0.0",
            },
            {
                "method": "centralized_rule_with_approach_radius_1",
                "collision_count": "0.0",
                "approach_zone_wait_count": "55.5",
            },
        ]

        paper_rows = build_paper_rows(
            rows,
            {
                "centralized_rule_multi_bt": 0.0,
                "centralized_rule_with_approach_radius_1": 0.0,
            },
        )

        self.assertEqual(paper_rows[0]["collision_event_count"], "42.0")
        self.assertEqual(paper_rows[0]["actual_collision_count"], "0")
        self.assertEqual(paper_rows[0]["approach_zone_wait_count"], "0.0")
        self.assertEqual(paper_rows[1]["actual_collision_count"], "0")
        self.assertEqual(paper_rows[1]["approach_zone_wait_count"], "55.5")

    def test_read_audit_actual_by_method_averages_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["method", "audited_collision_count"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method": "centralized_rule_multi_bt",
                        "audited_collision_count": "0",
                    }
                )
                writer.writerow(
                    {
                        "method": "centralized_rule_multi_bt",
                        "audited_collision_count": "2",
                    }
                )

            actual_by_method = read_audit_actual_by_method(path)

        self.assertEqual(actual_by_method["centralized_rule_multi_bt"], 1.0)


if __name__ == "__main__":
    unittest.main()
