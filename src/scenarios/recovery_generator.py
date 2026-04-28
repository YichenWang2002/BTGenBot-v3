"""Generate navigation + recovery benchmark scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.scenarios.nav_generator import (
    deterministic_waypoints,
    robot_start_positions,
    summarize_input,
    write_json,
)
from src.scenarios.recovery_injection import generate_recovery_config
from src.scenarios.waypoint_assignment import assign_waypoints


PREFERRED_RECOVERY_KEYWORDS = (
    "RecoveryNode",
    "ClearEntireCostmap",
    "Spin",
    "Wait",
    "BackUp",
    "GoalUpdated",
    "RetryUntilSuccesful",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-scenarios", type=int, default=20)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-waypoints", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--block-prob", type=float, default=0.35)
    parser.add_argument("--assignment-strategy", default="nearest_robot")
    parser.add_argument(
        "--candidates-out",
        type=Path,
        default=Path("data/derived/recovery_candidates.json"),
    )
    args = parser.parse_args(argv)

    manifest = generate_recovery_scenarios(
        profile_path=args.profile,
        candidates_path=args.candidates,
        out_dir=args.out,
        num_scenarios=args.num_scenarios,
        num_robots=args.num_robots,
        num_waypoints=args.num_waypoints,
        seeds=args.seeds,
        block_prob=args.block_prob,
        assignment_strategy=args.assignment_strategy,
        candidates_out=args.candidates_out,
    )
    print(f"Generated recovery scenarios: {len(manifest['scenarios'])}")
    print(f"Manifest: {args.out / 'manifest.json'}")
    return 0


def generate_recovery_scenarios(
    profile_path: Path,
    candidates_path: Path,
    out_dir: Path,
    num_scenarios: int,
    num_robots: int,
    num_waypoints: int,
    seeds: list[int],
    block_prob: float,
    assignment_strategy: str = "nearest_robot",
    candidates_out: Path = Path("data/derived/recovery_candidates.json"),
) -> dict[str, Any]:
    profile = read_json(profile_path)
    candidates_data = read_json(candidates_path)
    recovery_candidates = derive_recovery_candidates(profile, candidates_data)

    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(candidates_out, {"recovery_navigation": recovery_candidates})

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    scenario_count = 0
    for candidate in recovery_candidates:
        for seed in seeds:
            if scenario_count >= num_scenarios:
                break
            scenario = build_recovery_scenario(
                candidate,
                seed=seed,
                num_robots=num_robots,
                num_waypoints=num_waypoints,
                block_prob=block_prob,
                assignment_strategy=assignment_strategy,
            )
            scenario_path = out_dir / f"{scenario['name']}.yaml"
            with scenario_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(scenario, handle, sort_keys=False)
            manifest_rows.append(
                {
                    "scenario_path": str(scenario_path),
                    "dataset_index": candidate["index"],
                    "seed": seed,
                    "num_robots": num_robots,
                    "num_waypoints": num_waypoints,
                    "assignment_strategy": assignment_strategy,
                    "baseline_type": "centralized_rule_multi_robot_recovery",
                    "block_prob": block_prob,
                }
            )
            scenario_count += 1
        if scenario_count >= num_scenarios:
            break

    manifest = {"scenarios": manifest_rows}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def derive_recovery_candidates(
    profile: dict[str, Any], candidates_data: dict[str, Any]
) -> list[dict[str, Any]]:
    candidate_indexes = {
        int(row["index"]) for row in candidates_data.get("recovery", [])
    }
    rows: list[dict[str, Any]] = []
    for sample in profile.get("samples", []):
        if not sample.get("has_navigation") or not sample.get("has_recovery"):
            continue
        if candidate_indexes and int(sample["index"]) not in candidate_indexes:
            continue
        matched_keywords = list(sample.get("matched_recovery_keywords") or [])
        preferred_count = sum(
            1 for keyword in PREFERRED_RECOVERY_KEYWORDS if keyword in matched_keywords
        )
        rows.append(
            {
                "index": int(sample["index"]),
                "input": sample.get("input", ""),
                "xml_node_count": int(sample.get("xml_node_count") or 0),
                "action_condition_names": list(sample.get("action_condition_names") or []),
                "matched_recovery_keywords": matched_keywords,
                "has_recovery_node": bool(sample.get("has_recovery_node")),
                "_preferred_count": preferred_count,
                "_parse_error": sample.get("parse_error") is not None,
            }
        )

    rows.sort(
        key=lambda row: (
            not row["has_recovery_node"],
            -row["_preferred_count"],
            row["_parse_error"],
            row["xml_node_count"] >= 80,
            row["xml_node_count"],
            row["index"],
        )
    )
    for row in rows:
        row.pop("_preferred_count", None)
        row.pop("_parse_error", None)
    return rows


def build_recovery_scenario(
    candidate: dict[str, Any],
    seed: int,
    num_robots: int,
    num_waypoints: int,
    block_prob: float,
    assignment_strategy: str,
) -> dict[str, Any]:
    dataset_index = int(candidate["index"])
    name = f"recovery_medium_idx{dataset_index}_seed{seed}_robots{num_robots}"
    robot_starts = robot_start_positions(num_robots)
    waypoints = deterministic_waypoints(dataset_index, seed, num_waypoints)
    assignments = assign_waypoints(
        assignment_strategy,
        list(robot_starts),
        waypoints,
        robot_starts=robot_starts,
        seed=seed,
    )
    recovery_config = generate_recovery_config(
        robot_starts=robot_starts,
        seed=dataset_index + seed,
        block_prob=block_prob,
    )

    return {
        "name": name,
        "source": {
            "dataset_index": dataset_index,
            "task_family": "navigation_recovery",
            "input_summary": summarize_input(candidate.get("input", "")),
        },
        "map": {
            "width": 12,
            "height": 8,
            "obstacles": [[5, 0], [5, 7]],
        },
        "robots": {
            robot_id: {"start": start, "goal": None}
            for robot_id, start in robot_starts.items()
        },
        "waypoints": waypoints,
        "recovery": recovery_config,
        "task": {
            "type": "navigation_recovery",
            "num_waypoints": num_waypoints,
            "assignment_strategy": assignment_strategy,
            "baseline_type": "centralized_rule_multi_robot_recovery",
            "assignments": assignments,
        },
        "centralized_rule": True,
        "max_steps": max(90, num_waypoints * 50),
        "render": False,
    }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())

