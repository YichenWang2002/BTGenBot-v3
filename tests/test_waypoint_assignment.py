from src.scenarios.waypoint_assignment import assign_waypoints


def test_waypoint_assignment_round_robin():
    waypoints = [{"id": f"wp_{index}", "position": [index, 1]} for index in range(5)]

    assignments = assign_waypoints("round_robin", ["robot_0", "robot_1"], waypoints)

    assert assignments == {
        "robot_0": ["wp_0", "wp_2", "wp_4"],
        "robot_1": ["wp_1", "wp_3"],
    }


def test_waypoint_assignment_nearest_robot():
    waypoints = [
        {"id": "wp_0", "position": [9, 1]},
        {"id": "wp_1", "position": [9, 5]},
    ]
    starts = {"robot_0": [1, 1], "robot_1": [1, 5]}

    assignments = assign_waypoints(
        "nearest_robot", ["robot_0", "robot_1"], waypoints, robot_starts=starts
    )

    assert assignments == {"robot_0": ["wp_0"], "robot_1": ["wp_1"]}


def test_single_robot_sequential_assignment():
    waypoints = [{"id": f"wp_{index}", "position": [index, 1]} for index in range(3)]

    assignments = assign_waypoints(
        "sequential_single_robot", ["robot_0", "robot_1"], waypoints
    )

    assert assignments == {"robot_0": ["wp_0", "wp_1", "wp_2"], "robot_1": []}

