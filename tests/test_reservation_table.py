from src.coordination.reservation_table import ReservationTable


def test_cell_reservation_conflict():
    table = ReservationTable()

    first = table.reserve_cell("robot_0", [3, 4], 5)
    second = table.reserve_cell("robot_1", [3, 4], 5)

    assert first["allowed"] is True
    assert second["allowed"] is False
    assert second["conflict_type"] == "vertex"
    assert second["owner"] == "robot_0"


def test_edge_swap_conflict():
    table = ReservationTable()

    reserve = table.reserve_edge("robot_0", [1, 1], [2, 1], 3)
    conflict = table.is_edge_conflict("robot_1", [2, 1], [1, 1], 3)

    assert reserve["allowed"] is True
    assert conflict["allowed"] is False
    assert conflict["conflict_type"] == "edge"
    assert conflict["owner"] == "robot_0"


def test_release_cell():
    table = ReservationTable()
    table.reserve_cell("robot_0", [3, 4], 5)

    released = table.release_cell("robot_0", [3, 4], 5)

    assert released["allowed"] is True
    assert table.is_cell_reserved([3, 4], 5) is False
    assert table.get_cell_owner([3, 4], 5) is None

