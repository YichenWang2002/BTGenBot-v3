"""Keyword-based behavior tree sample classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


NAVIGATION_KEYWORDS = (
    "ComputePathToPose",
    "FollowPath",
    "NavigateToPose",
    "Go_point",
    "BaseMovement",
    "GetPoseFromWp",
    "ApproachObject",
    "waypoint",
    "goal",
    "path",
)

RECOVERY_KEYWORDS = (
    "RecoveryNode",
    "RetryUntilSuccesful",
    "ClearEntireCostmap",
    "Spin",
    "Wait",
    "GoalUpdated",
    "BackUp",
)

OPERATION_KEYWORDS = (
    "Pick",
    "Place",
    "Grasp",
    "Fetch",
    "Pour",
    "Drop",
    "Bottle",
    "Glass",
    "Object",
    "Gripper",
)

PICK_PLACE_KEYWORDS = ("Pick", "Place")


@dataclass(frozen=True)
class Classification:
    """Classification flags and matched keywords for one dataset sample."""

    has_navigation: bool
    has_recovery: bool
    has_operation: bool
    has_recovery_node: bool
    has_pick_place: bool
    navigation_keywords: list[str]
    recovery_keywords: list[str]
    operation_keywords: list[str]
    pick_place_keywords: list[str]


def classify_texts(texts: Iterable[str]) -> Classification:
    """Classify a sample from XML tags, node names, attributes, and raw XML."""

    haystack = "\n".join(text for text in texts if text).lower()
    navigation = _matched_keywords(haystack, NAVIGATION_KEYWORDS)
    recovery = _matched_keywords(haystack, RECOVERY_KEYWORDS)
    operation = _matched_keywords(haystack, OPERATION_KEYWORDS)
    pick_place = _matched_keywords(haystack, PICK_PLACE_KEYWORDS)

    return Classification(
        has_navigation=bool(navigation),
        has_recovery=bool(recovery),
        has_operation=bool(operation),
        has_recovery_node="recoverynode" in haystack,
        has_pick_place=bool(pick_place),
        navigation_keywords=navigation,
        recovery_keywords=recovery,
        operation_keywords=operation,
        pick_place_keywords=pick_place,
    )


def _matched_keywords(haystack: str, keywords: Iterable[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword.lower() in haystack]
