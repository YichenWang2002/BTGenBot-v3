from src.env.scenario_loader import load_scenario


def test_load_demo_1robot_scenario():
    scenario = load_scenario("configs/scenarios/demo_1robot_nav.yaml")

    assert scenario.name == "demo_1robot_nav"
    assert scenario.num_robots == 1
    assert scenario.state.grid_width == 10
    assert scenario.state.grid_height == 7
    assert scenario.state.robots["robot_0"].position == [1, 1]
    assert scenario.state.robots["robot_0"].goal == [8, 1]
    assert scenario.max_steps == 30
    assert scenario.render is False


def test_load_demo_3robots_scenario():
    scenario = load_scenario("configs/scenarios/demo_3robots_nav.yaml")

    assert scenario.name == "demo_3robots_nav"
    assert scenario.num_robots == 3
    assert sorted(scenario.state.robots) == ["robot_0", "robot_1", "robot_2"]
    assert scenario.state.robots["robot_2"].position == [1, 5]
    assert scenario.state.robots["robot_2"].goal == [8, 5]
    assert scenario.max_steps == 40


def test_load_scenario_data_parses_optional_task_dag():
    from src.env.scenario_loader import load_scenario_data

    scenario = load_scenario_data(
        {
            "name": "dag_loader_test",
            "map": {"width": 5, "height": 5, "obstacles": []},
            "robots": {"robot_0": {"start": [0, 0], "goal": None}},
            "task_dag": {
                "tasks": [
                    {
                        "task_id": "pick_obj_0",
                        "task_type": "pick",
                        "object_id": "obj_0",
                        "assigned_robot": "robot_0",
                    },
                    {
                        "task_id": "place_obj_0",
                        "task_type": "place",
                        "object_id": "obj_0",
                        "assigned_robot": "robot_0",
                    },
                ],
                "dependencies": [
                    {"source": "pick_obj_0", "target": "place_obj_0"},
                ],
            },
        }
    )

    assert scenario.task_dag is not None
    assert sorted(scenario.task_dag.tasks) == ["pick_obj_0", "place_obj_0"]


def test_load_scenario_without_task_dag_keeps_none():
    scenario = load_scenario("configs/scenarios/demo_1robot_nav.yaml")

    assert scenario.task_dag is None
