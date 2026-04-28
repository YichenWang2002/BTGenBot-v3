"""Generate deterministic pure-navigation benchmark scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.scenarios.waypoint_assignment import assign_waypoints


DEFAULT_WIDTH = 12
DEFAULT_HEIGHT = 8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-scenarios", type=int, default=20)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--num-waypoints", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--assignment-strategy", default="nearest_robot")
    parser.add_argument("--baseline-type", default="centralized_rule_multi_robot")
    parser.add_argument(
        "--pure-candidates-out",
        type=Path,
        default=Path("data/derived/nav_pure_candidates.json"),
    )
    args = parser.parse_args(argv)

    manifest = generate_nav_scenarios(
        profile_path=args.profile,
        candidates_path=args.candidates,
        out_dir=args.out,
        num_scenarios=args.num_scenarios,
        num_robots=args.num_robots,
        num_waypoints=args.num_waypoints,
        seeds=args.seeds,
        assignment_strategy=args.assignment_strategy,
        baseline_type=args.baseline_type,
        pure_candidates_out=args.pure_candidates_out,
    )
    print(f"Generated scenarios: {len(manifest['scenarios'])}")
    print(f"Manifest: {args.out / 'manifest.json'}")
    return 0


def generate_nav_scenarios(
    profile_path: Path,
    candidates_path: Path,
    out_dir: Path,
    num_scenarios: int,
    num_robots: int,
    num_waypoints: int,
    seeds: list[int],
    assignment_strategy: str = "nearest_robot",
    baseline_type: str = "centralized_rule_multi_robot",
    pure_candidates_out: Path = Path("data/derived/nav_pure_candidates.json"),
) -> dict[str, Any]:
    profile = read_json(profile_path)
    candidates_data = read_json(candidates_path)
    pure_candidates = derive_pure_navigation_candidates(profile, candidates_data)

    pure_candidates_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(pure_candidates_out, {"pure_navigation": pure_candidates})

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    scenario_count = 0
    for candidate in pure_candidates:
        for seed in seeds:
            if scenario_count >= num_scenarios:
                break
            scenario = build_nav_scenario(
                candidate=candidate,
                seed=seed,
                num_robots=num_robots,
                num_waypoints=num_waypoints,
                assignment_strategy=assignment_strategy,
                baseline_type=baseline_type,
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
                    "baseline_type": baseline_type,
                }
            )
            scenario_count += 1
        if scenario_count >= num_scenarios:
            break

    manifest = {"scenarios": manifest_rows}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def derive_pure_navigation_candidates(
    profile: dict[str, Any], candidates_data: dict[str, Any]
) -> list[dict[str, Any]]:
    if candidates_data.get("pure_navigation"):
        return list(candidates_data["pure_navigation"])

    samples_by_index = {
        int(sample["index"]): sample for sample in profile.get("samples", [])
    }
    navigation_rows = candidates_data.get("navigation") or []
    derived: list[dict[str, Any]] = []
    for row in navigation_rows:
        sample = samples_by_index.get(int(row["index"]))
        if sample is None:
            continue
        if not sample.get("has_navigation"):
            continue
        if sample.get("has_recovery") or sample.get("has_pick_place"):
            continue
        if sample.get("parse_error") is not None:
            continue
        xml_node_count = int(sample.get("xml_node_count") or row.get("xml_node_count") or 0)
        derived.append(
            {
                "index": int(sample["index"]),
                "input": sample.get("input", row.get("input", "")),
                "xml_node_count": xml_node_count,
                "has_operation": bool(sample.get("has_operation")),
            }
        )

    derived.sort(
        key=lambda item: (
            item["has_operation"],
            item["xml_node_count"] >= 40,
            item["xml_node_count"],
            item["index"],
        )
    )
    return derived


def build_nav_scenario(
    candidate: dict[str, Any],
    seed: int,
    num_robots: int,
    num_waypoints: int,
    assignment_strategy: str,
    baseline_type: str,
) -> dict[str, Any]:
    dataset_index = int(candidate["index"])
    name = f"nav_easy_idx{dataset_index}_seed{seed}_robots{num_robots}"
    robot_starts = robot_start_positions(num_robots)
    waypoints = deterministic_waypoints(dataset_index, seed, num_waypoints)
    robot_ids = list(robot_starts)
    assignments = assign_waypoints(
        assignment_strategy,
        robot_ids,
        waypoints,
        robot_starts=robot_starts,
        seed=seed,
    )

    return {
        "name": name,
        "source": {
            "dataset_index": dataset_index,
            "task_family": "pure_navigation",
            "input_summary": summarize_input(str(candidate.get("input", ""))),
        },
        "map": {
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "obstacles": [[5, 0], [5, 7]],
        },
        "robots": {
            robot_id: {"start": start, "goal": None}
            for robot_id, start in robot_starts.items()
        },
        "waypoints": waypoints,
        "task": {
            "type": "pure_navigation",
            "num_waypoints": num_waypoints,
            "assignment_strategy": assignment_strategy,
            "baseline_type": baseline_type,
            "assignments": assignments,
        },
        "centralized_rule": baseline_type == "centralized_rule_multi_robot",
        "max_steps": max(50, num_waypoints * 25),
        "render": False,
    }


def robot_start_positions(num_robots: int) -> dict[str, list[int]]:
    starts: dict[str, list[int]] = {}
    usable_rows = [1, 3, 5, 6, 2, 4]
    for index in range(num_robots):
        y = usable_rows[index % len(usable_rows)]
        starts[f"robot_{index}"] = [1, y]
    return starts


def deterministic_waypoints(
    dataset_index: int, seed: int, num_waypoints: int
) -> list[dict[str, Any]]:
    rows = [1, 3, 5, 6, 2, 4]
    waypoints: list[dict[str, Any]] = []
    for index in range(num_waypoints):
        row_index = (index + seed + dataset_index) % len(rows)
        x = DEFAULT_WIDTH - 2 - ((dataset_index + seed + index) % 2)
        waypoints.append(
            {
                "id": f"wp_{index}",
                "position": [x, rows[row_index]],
            }
        )
    return waypoints


def summarize_input(text: str, limit: int = 160) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

