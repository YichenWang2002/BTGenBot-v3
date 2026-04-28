from src.coordination.rules import CentralizedRuleManager


def test_request_move_allows_free_cell():
    manager = CentralizedRuleManager(grid_width=5, grid_height=5)

    result = manager.request_move("robot_0", [0, 0], [1, 0], 0)

    assert result["allowed"] is True
    assert manager.reservation_table.get_cell_owner([1, 0], 1) == "robot_0"


def test_request_move_rejects_reserved_cell():
    manager = CentralizedRuleManager(grid_width=5, grid_height=5)
    manager.request_move("robot_0", [0, 0], [1, 0], 0)

    result = manager.request_move("robot_1", [2, 0], [1, 0], 0)

    assert result["allowed"] is False
    assert result["conflict_type"] == "vertex"
    assert result["owner"] == "robot_0"


def test_request_move_rejects_edge_swap():
    manager = CentralizedRuleManager(grid_width=5, grid_height=5)
    manager.request_move("robot_0", [0, 0], [1, 0], 0)

    result = manager.request_move("robot_1", [1, 0], [0, 0], 0)

    assert result["allowed"] is False
    assert result["conflict_type"] == "edge"
    assert result["owner"] == "robot_0"


def test_rule_events_logged():
    manager = CentralizedRuleManager(grid_width=5, grid_height=5, obstacles={(2, 2)})

    result = manager.request_move("robot_0", [1, 2], [2, 2], 0)

    assert result["allowed"] is False
    assert manager.rule_events[-1]["event_type"] == "move_request"
    assert manager.rule_events[-1]["allowed"] is False
    assert manager.rule_events[-1]["reason"] == "obstacle"
    assert manager.rule_events[-1]["from_cell"] == [1, 2]
    assert manager.rule_events[-1]["to_cell"] == [2, 2]


def test_deadlock_detection():
    manager = CentralizedRuleManager(grid_width=3, grid_height=3, obstacles={(1, 1)})
    manager.request_move("robot_0", [0, 1], [1, 1], 0)
    manager.request_move("robot_1", [2, 1], [1, 1], 0)

    assert manager.detect_deadlock(window_size=2) is True

