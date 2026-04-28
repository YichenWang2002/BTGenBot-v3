from src.env.scenario_loader import load_scenario_data
from src.llm.schema import validate_llm_output


def test_schema_accepts_valid_llm_output():
    scenario = build_nav_scenario()
    payload = {
        "robot_trees": {
            "robot_0": '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence><Action ID="NavigateToWaypoint" waypoint_id="wp_0"/></Sequence></BehaviorTree></root>',
            "robot_1": '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence><Action ID="NavigateToWaypoint" waypoint_id="wp_1"/></Sequence></BehaviorTree></root>',
        },
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["wp_0"], "reason": "nearest"},
            {"robot_id": "robot_1", "task_ids": ["wp_1"], "reason": "nearest"},
        ],
        "coordination_notes": ["use centralized rules"],
        "assumptions": ["discrete grid"],
    }

    result = validate_llm_output(payload, scenario)

    assert result.valid is True
    assert result.errors == []


def test_schema_rejects_missing_robot_trees():
    scenario = build_nav_scenario()
    payload = {
        "robot_trees": {
            "robot_0": '<root><BehaviorTree ID="MainTree"><Sequence/></BehaviorTree></root>',
        },
        "assignment": [
            {"robot_id": "robot_0", "task_ids": ["wp_0"], "reason": "nearest"},
            {"robot_id": "robot_1", "task_ids": ["wp_1"], "reason": "nearest"},
        ],
        "coordination_notes": [],
        "assumptions": [],
    }

    result = validate_llm_output(payload, scenario)

    assert result.valid is False
    assert any("missing active robots" in error for error in result.errors)


def build_nav_scenario():
    return load_scenario_data(
        {
            "name": "schema_nav",
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
