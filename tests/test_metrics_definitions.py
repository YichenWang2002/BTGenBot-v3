import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.metrics import MetricsRecorder


class MetricsDefinitionsTest(unittest.TestCase):
    def test_summary_exposes_collision_event_and_actual_counts(self):
        metrics = MetricsRecorder()
        metrics.record_step(
            timestep=0,
            robot_positions={"robot_0": [0, 0], "robot_1": [0, 0]},
            actions={},
            collisions=[{"type": "vertex"}],
            status={},
            moved_robot_ids=[],
        )

        summary = metrics.summary()

        self.assertEqual(summary["collision_count"], 1)
        self.assertEqual(summary["collision_event_count_legacy"], 1)
        self.assertEqual(summary["collision_event_count"], 1)
        self.assertEqual(summary["actual_collision_count"], 1)
        self.assertEqual(summary["vertex_conflict_event_count"], 1)
        self.assertEqual(summary["audited_vertex_conflict_count"], 1)

    def test_actual_collision_count_includes_edge_swaps(self):
        metrics = MetricsRecorder()
        metrics.record_step(
            timestep=0,
            robot_positions={"robot_0": [0, 0], "robot_1": [1, 0]},
            actions={},
            collisions=[],
            status={},
            moved_robot_ids=[],
        )
        metrics.record_step(
            timestep=1,
            robot_positions={"robot_0": [1, 0], "robot_1": [0, 0]},
            actions={},
            collisions=[],
            status={},
            moved_robot_ids=[],
        )

        summary = metrics.summary()

        self.assertEqual(summary["collision_event_count"], 0)
        self.assertEqual(summary["actual_collision_count"], 1)
        self.assertEqual(summary["audited_edge_swap_conflict_count"], 1)

    def test_safety_guarantor_motion_and_resource_metrics(self):
        metrics = MetricsRecorder()
        metrics.centralized_rule_enabled = True
        metrics.record_step(
            timestep=0,
            robot_positions={"robot_0": [0, 0], "robot_1": [1, 0]},
            actions={},
            collisions=[],
            status={},
            moved_robot_ids=[],
            rule_events=[
                {
                    "allowed": False,
                    "event_type": "move_request",
                    "safety_layer": "motion",
                    "conflict_type": "cell_conflict",
                },
                {
                    "allowed": False,
                    "event_type": "resource_request",
                    "safety_layer": "semantic_resource",
                    "conflict_type": "object_lock_denied",
                },
            ],
        )

        summary = metrics.summary()

        self.assertEqual(summary["motion_wait_count"], 1)
        self.assertEqual(summary["rule_prevented_motion_conflict_count"], 1)
        self.assertEqual(summary["object_lock_denied_count"], 1)
        self.assertEqual(summary["resource_request_denied_count"], 1)


if __name__ == "__main__":
    unittest.main()
