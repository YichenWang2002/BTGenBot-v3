from src.dataset.classifier import classify_texts


def test_classify_navigation_recovery_and_operation_keywords():
    result = classify_texts(
        [
            "<RecoveryNode><ComputePathToPose goal='{goal}' path='{path}'/>"
            "<Action ID='Pick'/><Condition ID='Bottle_found'/></RecoveryNode>"
        ]
    )

    assert result.has_navigation is True
    assert result.has_recovery is True
    assert result.has_operation is True
    assert result.has_recovery_node is True
    assert result.has_pick_place is True
    assert "ComputePathToPose" in result.navigation_keywords
    assert "RecoveryNode" in result.recovery_keywords
    assert "Pick" in result.operation_keywords


def test_classify_is_case_insensitive_for_lowercase_keywords():
    result = classify_texts(["robot follows a waypoint path to a goal"])

    assert result.has_navigation is True
    assert result.navigation_keywords == ["waypoint", "goal", "path"]
