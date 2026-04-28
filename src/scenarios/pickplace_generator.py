"""Generate navigation + pick/place resource arbitration scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.scenarios.pickplace_injection import object_is_unavailable
from src.scenarios.pickplace_tasks import (
    assign_tasks_nearest_robot,
    assignments_by_robot,
    build_pickplace_tasks,
)
from src.scenarios.nav_generator import summarize_input, write_json


PREFERRED_KEYWORDS = (
    "Pick",
    "Place",
    "Grasp",
    "Fetch",
    "Drop",
    "HoldingItem",
    "SeeItem",
    "GetPickObject",
    "GetPlacingPose",
    "PlaceSimple",
    "Gripper",
    "Bottle",
    "Glass",
    "Object",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-scenarios", type=int, default=20)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-objects", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--pick-fail-prob", type=float, default=0.15)
    parser.add_argument("--object-unavailable-prob", type=float, default=0.10)
    parser.add_argument(
        "--candidates-out",
        type=Path,
        default=Path("data/derived/pickplace_candidates.json"),
    )
    args = parser.parse_args(argv)
    manifest = generate_pickplace_scenarios(
        profile_path=args.profile,
        candidates_path=args.candidates,
        out_dir=args.out,
        num_scenarios=args.num_scenarios,
        num_robots=args.num_robots,
        num_objects=args.num_objects,
        seeds=args.seeds,
        pick_fail_prob=args.pick_fail_prob,
        object_unavailable_prob=args.object_unavailable_prob,
        candidates_out=args.candidates_out,
    )
    print(f"Generated pickplace scenarios: {len(manifest['scenarios'])}")
    print(f"Manifest: {args.out / 'manifest.json'}")
    return 0


def generate_pickplace_scenarios(
    profile_path: Path,
    candidates_path: Path,
    out_dir: Path,
    num_scenarios: int,
    num_robots: int,
    num_objects: int,
    seeds: list[int],
    pick_fail_prob: float,
    object_unavailable_prob: float,
    candidates_out: Path = Path("data/derived/pickplace_candidates.json"),
) -> dict[str, Any]:
    profile = read_json(profile_path)
    candidates_data = read_json(candidates_path)
    candidates = derive_pickplace_candidates(profile, candidates_data)
    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(candidates_out, {"pickplace": candidates})

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    count = 0
    for candidate in candidates:
        for seed in seeds:
            if count >= num_scenarios:
                break
            scenario = build_pickplace_scenario(
                candidate,
                seed,
                num_robots,
                num_objects,
                pick_fail_prob,
                object_unavailable_prob,
            )
            path = out_dir / f"{scenario['name']}.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(scenario, handle, sort_keys=False)
            rows.append(
                {
                    "scenario_path": str(path),
                    "dataset_index": candidate["index"],
                    "seed": seed,
                    "num_robots": num_robots,
                    "num_objects": num_objects,
                    "assignment_strategy": "nearest_robot",
                    "baseline_type": "centralized_rule_multi_robot_pickplace",
                    "pick_fail_prob": pick_fail_prob,
                    "object_unavailable_prob": object_unavailable_prob,
                }
            )
            count += 1
        if count >= num_scenarios:
            break
    manifest = {"scenarios": rows}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def derive_pickplace_candidates(
    profile: dict[str, Any], candidates_data: dict[str, Any]
) -> list[dict[str, Any]]:
    preferred_indexes = {int(row["index"]) for row in candidates_data.get("pick_place", [])}
    rows: list[dict[str, Any]] = []
    for sample in profile.get("samples", []):
        if not (sample.get("has_pick_place") or sample.get("has_operation")):
            continue
        keywords = list(sample.get("matched_pick_place_keywords") or [])
        op_keywords = list(sample.get("matched_operation_keywords") or [])
        names = list(sample.get("action_condition_names") or [])
        all_text = set(keywords + op_keywords + names)
        score = sum(1 for keyword in PREFERRED_KEYWORDS if keyword in all_text)
        rows.append(
            {
                "index": int(sample["index"]),
                "input": sample.get("input", ""),
                "xml_node_count": int(sample.get("xml_node_count") or 0),
                "action_condition_names": names,
                "matched_pick_place_keywords": keywords,
                "matched_operation_keywords": op_keywords,
                "_score": score + (2 if int(sample["index"]) in preferred_indexes else 0),
                "_parse_error": sample.get("parse_error") is not None,
            }
        )
    rows.sort(
        key=lambda row: (
            -row["_score"],
            row["_parse_error"],
            row["xml_node_count"] >= 100,
            row["xml_node_count"],
            row["index"],
        )
    )
    for row in rows:
        row.pop("_score", None)
        row.pop("_parse_error", None)
    return rows


def build_pickplace_scenario(
    candidate: dict[str, Any],
    seed: int,
    num_robots: int,
    num_objects: int,
    pick_fail_prob: float,
    object_unavailable_prob: float,
) -> dict[str, Any]:
    dataset_index = int(candidate["index"])
    name = f"pickplace_hard_idx{dataset_index}_seed{seed}_robots{num_robots}"
    robots = robot_specs(num_robots)
    objects = object_specs(num_objects, seed + dataset_index, object_unavailable_prob)
    tasks = assign_tasks_nearest_robot(build_pickplace_tasks(objects), robots)
    task_assignments = assignments_by_robot(tasks, sorted(robots))
    return {
        "name": name,
        "source": {
            "dataset_index": dataset_index,
            "task_family": "navigation_pickplace",
            "input_summary": summarize_input(candidate.get("input", "")),
        },
        "map": {"width": 14, "height": 10, "obstacles": [[6, 0], [6, 9]]},
        "robots": robots,
        "objects": objects,
        "zones": zone_specs(num_objects),
        "pickplace": {
            "enabled": True,
            "pick_fail_prob": pick_fail_prob,
            "place_fail_prob": 0.05,
            "object_temporarily_unavailable_prob": object_unavailable_prob,
            "lock_ttl": 8,
            "allow_reassignment": True,
            "seed": seed + dataset_index,
            "tasks": tasks,
        },
        "task": {
            "type": "navigation_pickplace",
            "assignment_strategy": "nearest_robot",
            "baseline_type": "centralized_rule_multi_robot_pickplace",
            "assignments": task_assignments,
        },
        "centralized_rule": True,
        "max_steps": 200,
        "render": False,
    }


def robot_specs(num_robots: int) -> dict[str, dict[str, Any]]:
    rows = [1, 4, 7, 2, 5, 8]
    return {
        f"robot_{index}": {
            "start": [1, rows[index % len(rows)]],
            "goal": None,
            "carrying_object": None,
        }
        for index in range(num_robots)
    }


def object_specs(num_objects: int, seed: int, unavailable_prob: float) -> dict[str, dict[str, Any]]:
    rows = [1, 4, 7, 2, 5, 8]
    objects: dict[str, dict[str, Any]] = {}
    for index in range(num_objects):
        object_id = f"obj_{index}"
        status = "unavailable" if object_is_unavailable(seed, object_id, unavailable_prob) else "available"
        objects[object_id] = {
            "position": [10, rows[index % len(rows)]],
            "target_position": [12, rows[index % len(rows)]],
            "held_by": None,
            "status": status,
        }
    return objects


def zone_specs(num_objects: int) -> dict[str, Any]:
    rows = [1, 4, 7, 2, 5, 8]
    active_rows = rows[: max(1, min(num_objects, len(rows)))]
    pickup_cells = []
    drop_cells = []
    for row in active_rows:
        pickup_cells.extend([[9, row], [10, row], [11, row]])
        drop_cells.extend([[12, row], [13, row]])
    pickup = {"pickup_zone_0": {"cells": pickup_cells}}
    drop = {"drop_zone_0": {"cells": drop_cells}}
    return {"pickup_zones": pickup, "drop_zones": drop}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
