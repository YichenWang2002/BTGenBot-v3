from src.coordination.resource_manager import ResourceManager


def test_acquire_release():
    manager = ResourceManager()

    acquired = manager.acquire("obj_0", "robot_0", "object", 0, 5, "pick")
    released = manager.release("obj_0", "robot_0")

    assert acquired["allowed"] is True
    assert manager.is_locked("obj_0") is False
    assert released["allowed"] is True
    assert released["next_owner"] is None


def test_lock_conflict():
    manager = ResourceManager()
    manager.acquire("obj_0", "robot_0", "object", 0, 5, "pick")

    conflict = manager.acquire("obj_0", "robot_1", "object", 1, 5, "pick")

    assert conflict["allowed"] is False
    assert conflict["conflict_type"] == "resource"
    assert conflict["owner"] == "robot_0"
    assert conflict["next_owner"] == "robot_1"


def test_ttl_release():
    manager = ResourceManager()
    manager.acquire("charger", "robot_0", "charging_zone", 0, 2, "charge")

    assert manager.tick(1) == []
    released = manager.tick(2)

    assert released[0]["resource_id"] == "charger"
    assert released[0]["owner"] == "robot_0"
    assert manager.is_locked("charger") is False


def test_fifo_queue():
    manager = ResourceManager()
    manager.acquire("zone", "robot_0", "pickup_zone", 0, 10, "pickup")
    manager.acquire("zone", "robot_1", "pickup_zone", 1, 10, "pickup")
    manager.acquire("zone", "robot_2", "pickup_zone", 2, 10, "pickup")

    released = manager.release("zone", "robot_0")

    assert released["allowed"] is True
    assert released["next_owner"] == "robot_1"
    assert manager.get_owner("zone") == "robot_1"
    snapshot = manager.snapshot()
    assert snapshot["wait_queues"]["zone"][0]["robot_id"] == "robot_2"

