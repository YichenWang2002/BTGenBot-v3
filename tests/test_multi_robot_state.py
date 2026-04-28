from src.env.multi_robot_state import MultiRobotState, ObjectState, RobotState


def test_state_tracks_default_robot_and_reset():
    state = MultiRobotState.from_single_robot(
        grid_width=5,
        grid_height=4,
        start=[1, 1],
        goal=[3, 1],
        obstacles=[[2, 2]],
    )

    assert state.timestep == 0
    assert state.get_robot().robot_id == "robot_0"
    assert state.is_obstacle([2, 2])
    assert state.is_in_bounds([4, 3])
    assert not state.is_in_bounds([5, 3])

    robot = state.get_robot()
    robot.position = [3, 1]
    robot.status = "success"
    state.timestep = 9

    state.reset()

    assert state.timestep == 0
    assert state.get_robot().position == [1, 1]
    assert state.get_robot().status == "running"
    assert state.get_robot().path == [[1, 1]]


def test_state_serializes_robots_and_objects():
    state = MultiRobotState(
        timestep=0,
        grid_width=6,
        grid_height=6,
        obstacles={(0, 0)},
        robots={
            "robot_0": RobotState(
                robot_id="robot_0",
                position=[1, 1],
                goal=[2, 1],
                status="running",
                path=[[1, 1]],
            )
        },
        objects={
            "box": ObjectState(
                object_id="box",
                position=[3, 3],
                target_position=[4, 4],
            )
        },
    )

    payload = state.to_dict()

    assert payload["robots"]["robot_0"]["position"] == [1, 1]
    assert payload["objects"]["box"]["target_position"] == [4, 4]

