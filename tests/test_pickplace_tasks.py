from src.scenarios.pickplace_tasks import (
    assign_tasks_nearest_robot,
    build_pickplace_tasks,
    reassign_failed_task,
)


def test_build_pickplace_tasks():
    objects = {
        "obj_0": {"position": [2, 1], "target_position": [4, 1]},
        "obj_1": {"position": [2, 3], "target_position": [4, 3]},
    }

    tasks = build_pickplace_tasks(objects)

    assert [task["task_id"] for task in tasks] == ["task_obj_0", "task_obj_1"]
    assert tasks[0]["pickup_position"] == [2, 1]
    assert tasks[0]["drop_position"] == [4, 1]
    assert tasks[0]["status"] == "pending"


def test_assign_tasks_nearest_robot():
    tasks = build_pickplace_tasks(
        {
            "obj_0": {"position": [2, 1], "target_position": [4, 1]},
            "obj_1": {"position": [2, 5], "target_position": [4, 5]},
        }
    )
    robots = {
        "robot_0": {"start": [1, 1]},
        "robot_1": {"start": [1, 5]},
    }

    assigned = assign_tasks_nearest_robot(tasks, robots)

    assert assigned[0]["assigned_robot"] == "robot_0"
    assert assigned[1]["assigned_robot"] == "robot_1"
    assert all(task["status"] == "assigned" for task in assigned)


def test_failed_pick_can_be_reassigned():
    task = {
        "task_id": "task_obj_0",
        "object_id": "obj_0",
        "pickup_position": [5, 1],
        "drop_position": [7, 1],
        "assigned_robot": "robot_0",
        "status": "failed",
        "attempts": 1,
    }
    robots = {
        "robot_0": {"start": [1, 1]},
        "robot_1": {"start": [4, 1]},
    }

    reassigned = reassign_failed_task(task, robots, "robot_0")

    assert reassigned["assigned_robot"] == "robot_1"
    assert reassigned["status"] == "reassigned"
