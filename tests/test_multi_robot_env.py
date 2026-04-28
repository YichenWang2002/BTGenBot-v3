import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.env.multi_robot_env import MultiRobotEnv
from src.env.multi_robot_state import MultiRobotState, RobotState
from src.env.scenario_loader import load_scenario


def run_until_done(env: MultiRobotEnv) -> dict:
    env.reset()
    observation = env.observation([])
    while not env.done:
        observation = env.step()
    return observation


def test_single_robot_navigation_succeeds():
    scenario = load_scenario("configs/scenarios/demo_1robot_nav.yaml")
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
    )

    observation = run_until_done(env)

    assert observation["metrics"]["success"] is True
    assert observation["metrics"]["makespan"] == 7
    assert observation["metrics"]["total_robot_steps"] == 7
    assert observation["metrics"]["collision_count"] == 0
    assert scenario.state.robots["robot_0"].position == [8, 1]


def test_three_robot_navigation_succeeds_without_collisions():
    scenario = load_scenario("configs/scenarios/demo_3robots_nav.yaml")
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
    )

    observation = run_until_done(env)

    assert observation["metrics"]["success"] is True
    assert observation["metrics"]["makespan"] == 7
    assert observation["metrics"]["total_robot_steps"] == 21
    assert observation["metrics"]["collision_count"] == 0
    assert scenario.state.robots["robot_0"].position == [8, 1]
    assert scenario.state.robots["robot_1"].position == [8, 3]
    assert scenario.state.robots["robot_2"].position == [8, 5]


def test_obstacle_collision_is_recorded_and_robot_stays_put():
    state = MultiRobotState.from_single_robot(
        grid_width=4,
        grid_height=4,
        start=[1, 1],
        goal=[3, 1],
        obstacles=[[2, 1]],
    )
    env = MultiRobotEnv(state, max_steps=5, render=False)
    env.reset()

    observation = env.step({"robot_0": {"type": "move", "direction": "east"}})

    assert observation["metrics"]["collision_count"] == 1
    assert observation["collisions"][0]["type"] == "obstacle"
    assert observation["collisions"][0]["robot_ids"] == ["robot_0"]
    assert state.robots["robot_0"].position == [1, 1]


def test_vertex_collision_records_involved_robot_ids():
    state = MultiRobotState(
        timestep=0,
        grid_width=5,
        grid_height=5,
        robots={
            "robot_0": RobotState(
                robot_id="robot_0",
                position=[1, 2],
                goal=[2, 2],
                status="running",
                path=[[1, 2]],
            ),
            "robot_1": RobotState(
                robot_id="robot_1",
                position=[3, 2],
                goal=[2, 2],
                status="running",
                path=[[3, 2]],
            ),
        },
    )
    env = MultiRobotEnv(state, max_steps=5, render=False)
    env.reset()

    observation = env.step()

    assert observation["metrics"]["collision_count"] == 1
    assert observation["collisions"][0]["type"] == "vertex"
    assert observation["collisions"][0]["robot_ids"] == ["robot_0", "robot_1"]
    assert state.robots["robot_0"].position == [1, 2]
    assert state.robots["robot_1"].position == [3, 2]
    assert env.metrics.trace[0]["collisions"][0]["position"] == [2, 2]


def test_legacy_default_robot_action_without_robot_id():
    state = MultiRobotState.from_single_robot(
        grid_width=4,
        grid_height=4,
        start=[1, 1],
        goal=[2, 1],
    )
    env = MultiRobotEnv(state, max_steps=5, render=False)
    env.reset()

    env.move_one_step(direction="east")

    assert state.robots["robot_0"].position == [2, 1]
    assert env.metrics.summary()["success"] is True


def test_centralized_rule_rejects_conflict_and_records_trace_fields():
    scenario = load_scenario("configs/scenarios/demo_conflict_crossing.yaml")
    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=False,
        scenario_name=scenario.name,
        centralized_rule=True,
    )

    observation = run_until_done(env)
    first_trace = env.metrics.trace[0]

    assert observation["metrics"]["success"] is True
    assert observation["metrics"]["collision_count"] == 0
    assert observation["metrics"]["rule_rejection_count"] == 1
    assert observation["metrics"]["wait_count"] == 1
    assert first_trace["rule_events"]
    assert "reservations" in first_trace
    assert "resource_locks" in first_trace
