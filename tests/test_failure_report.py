from src.env.scenario_loader import load_scenario_data
from src.repair.critic import static_validate
from src.repair.failure_report import build_failure_report


def test_failure_report_detects_collision():
    scenario = build_nav_scenario()

    report = build_failure_report(
        scenario,
        assignment=[
            {"robot_id": "robot_0", "task_ids": ["wp_0"], "reason": "test"},
            {"robot_id": "robot_1", "task_ids": ["wp_1"], "reason": "test"},
        ],
        metrics={"collision_count": 1, "edge_conflict_count": 0, "timeout": False},
        trace=[],
        static_validation={"valid": True},
    )

    assert report["status"] == "failed"
    assert "collision" in report["failure_types"]


def test_failure_report_detects_duplicate_task():
    scenario = build_nav_scenario()

    report = build_failure_report(
        scenario,
        assignment=[
            {"robot_id": "robot_0", "task_ids": ["wp_0"], "reason": "test"},
            {"robot_id": "robot_1", "task_ids": ["wp_0"], "reason": "test"},
        ],
        metrics={},
        trace=[],
        static_validation={"valid": False, "errors": ["duplicate task assignment"]},
    )

    assert "duplicate_task" in report["failure_types"]
    assert any(detail["type"] == "duplicate_task" for detail in report["details"])


def test_critic_detects_unassigned_task():
    scenario = build_nav_scenario()
    payload = {
        "robot_trees": {
            "robot_0": "<root><BehaviorTree ID=\"MainTree\"><Sequence/></BehaviorTree></root>",
            "robot_1": "<root><BehaviorTree ID=\"MainTree\"><Sequence/></BehaviorTree></root>",
        },
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["wp_0"], "reason": "test"},
            {"robot_id": "robot_1", "task_ids": [], "reason": "test"},
        ],
    }

    result = static_validate(scenario, payload)

    assert result["assignment_valid"] is False
    assert any("unassigned" in error for error in result["errors"])


def test_critic_reports_invalid_resource_attribute():
    scenario = build_recovery_scenario()
    payload = {
        "robot_trees": {
            "robot_0": (
                '<root><BehaviorTree ID="MainTree"><Sequence>'
                '<Action ID="RequestResource" resource="recovery_zone"/>'
                '<Action ID="ClearCostmap"/>'
                "</Sequence></BehaviorTree></root>"
            )
        },
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["wp_0"], "reason": "test"}
        ],
    }

    result = static_validate(scenario, payload)

    assert result["resource_valid"] is False
    assert result["invalid_resource_request_count"] == 1
    assert "not resource='recovery_zone'" in result["errors"][0]
    assert "resource_id=\"recovery_zone_0\"" in result["suggested_xml_fix"]


def build_nav_scenario():
    return load_scenario_data(
        {
            "name": "failure_nav",
            "map": {"width": 6, "height": 4, "obstacles": []},
            "robots": {
                "robot_0": {"start": [1, 1], "goal": None},
                "robot_1": {"start": [1, 2], "goal": None},
            },
            "waypoints": [
                {"id": "wp_0", "position": [4, 1]},
                {"id": "wp_1", "position": [4, 2]},
            ],
            "task": {
                "type": "pure_navigation",
                "assignments": {"robot_0": ["wp_0"], "robot_1": ["wp_1"]},
            },
            "max_steps": 20,
            "render": False,
        }
    )


def build_recovery_scenario():
    return load_scenario_data(
        {
            "name": "failure_recovery",
            "map": {"width": 6, "height": 4, "obstacles": []},
            "robots": {"robot_0": {"start": [1, 1], "goal": None}},
            "waypoints": [{"id": "wp_0", "position": [4, 1]}],
            "recovery": {"enabled": True},
            "task": {
                "type": "navigation_recovery",
                "assignments": {"robot_0": ["wp_0"]},
            },
            "max_steps": 20,
            "render": False,
        }
    )
