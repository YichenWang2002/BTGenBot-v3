"""Deterministic pick/place failure injection helpers."""

from __future__ import annotations

import hashlib


def should_fail(seed: int, key: str, probability: float) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < probability


def object_is_unavailable(seed: int, object_id: str, probability: float) -> bool:
    return should_fail(seed, f"unavailable:{object_id}", probability)


def pick_fails(seed: int, task_id: str, attempt: int, probability: float) -> bool:
    return should_fail(seed, f"pick:{task_id}:{attempt}", probability)


def place_fails(seed: int, task_id: str, attempt: int, probability: float) -> bool:
    return should_fail(seed, f"place:{task_id}:{attempt}", probability)

