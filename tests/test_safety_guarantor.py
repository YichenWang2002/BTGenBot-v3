from src.coordination.rules import CentralizedRuleManager
from src.coordination.safety_guarantor import SafetyGuarantor


def test_motion_allows_free_cell_and_reserves_it():
    guard = SafetyGuarantor(grid_width=5, grid_height=5)

    decision = guard.check_action(
        "robot_0",
        {
            "action_id": "move_request",
            "from_cell": [0, 0],
            "to_cell": [1, 0],
            "timestep": 0,
        },
    )

    assert decision.allowed is True
    assert decision.safety_layer == "motion"
    assert guard.reservation_table.get_cell_owner([1, 0], 1) == "robot_0"


def test_motion_denies_cell_conflict():
    guard = SafetyGuarantor(grid_width=5, grid_height=5)
    guard.check_action(
        "robot_0",
        {
            "action_id": "move_request",
            "from_cell": [0, 0],
            "to_cell": [1, 0],
            "timestep": 0,
        },
    )

    decision = guard.check_action(
        "robot_1",
        {
            "action_id": "move_request",
            "from_cell": [2, 0],
            "to_cell": [1, 0],
            "timestep": 0,
        },
    )

    assert decision.allowed is False
    assert decision.safety_layer == "motion"
    assert decision.conflict_type == "cell_conflict"
    assert decision.converted_to_wait is True
    assert decision.details["owner"] == "robot_0"


def test_motion_denies_edge_conflict():
    guard = SafetyGuarantor(grid_width=5, grid_height=5)
    guard.check_action(
        "robot_0",
        {
            "action_id": "move_request",
            "from_cell": [0, 0],
            "to_cell": [1, 0],
            "timestep": 0,
        },
    )

    decision = guard.check_action(
        "robot_1",
        {
            "action_id": "move_request",
            "from_cell": [1, 0],
            "to_cell": [0, 0],
            "timestep": 0,
        },
    )

    assert decision.allowed is False
    assert decision.safety_layer == "motion"
    assert decision.conflict_type == "edge_conflict"
    assert decision.details["owner"] == "robot_0"


def test_semantic_resource_denial_types():
    guard = SafetyGuarantor(grid_width=5, grid_height=5)

    for resource_type, conflict_type in [
        ("object", "object_lock_denied"),
        ("pickup_zone", "pickup_zone_denied"),
        ("drop_zone", "drop_zone_denied"),
        ("recovery_zone", "recovery_zone_denied"),
    ]:
        resource_id = f"{resource_type}_0"
        guard.check_action(
            "robot_0",
            {
                "action_id": "resource_request",
                "resource_id": resource_id,
                "resource_type": resource_type,
                "timestep": 0,
                "ttl": 5,
                "reason": "test",
            },
        )
        decision = guard.check_action(
            "robot_1",
            {
                "action_id": "resource_request",
                "resource_id": resource_id,
                "resource_type": resource_type,
                "timestep": 1,
                "ttl": 5,
                "reason": "test",
            },
        )

        assert decision.allowed is False
        assert decision.safety_layer == "semantic_resource"
        assert decision.conflict_type == conflict_type
        assert decision.resource_id == resource_id


def test_approach_zone_denial_is_manipulation_area():
    guard = SafetyGuarantor(grid_width=5, grid_height=5)
    guard.check_action(
        "robot_0",
        {
            "action_id": "approach_zone_request",
            "resource_id": "pickup_approach_zone_1_1",
            "timestep": 0,
            "ttl": 5,
            "reason": "pickup_approach_zone",
        },
    )

    decision = guard.check_action(
        "robot_1",
        {
            "action_id": "approach_zone_request",
            "resource_id": "pickup_approach_zone_1_1",
            "timestep": 1,
            "ttl": 5,
            "reason": "pickup_approach_zone",
        },
    )

    assert decision.allowed is False
    assert decision.safety_layer == "manipulation_area"
    assert decision.conflict_type == "approach_zone_denied"
    assert decision.converted_to_wait is True


def test_centralized_rule_events_include_safety_fields():
    manager = CentralizedRuleManager(grid_width=5, grid_height=5)
    manager.request_move("robot_0", [0, 0], [1, 0], 0)
    result = manager.request_move("robot_1", [2, 0], [1, 0], 0)
    event = manager.rule_events[-1]

    assert result["conflict_type"] == "vertex"
    assert event["safety_layer"] == "motion"
    assert event["conflict_type"] == "cell_conflict"
    assert event["decision"] == "converted_to_wait"
    assert event["converted_to_wait"] is True
    assert event["action_id"] == "move_request"
