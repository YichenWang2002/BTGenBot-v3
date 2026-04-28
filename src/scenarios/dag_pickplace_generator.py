"""Generate DAG-augmented pick/place benchmark scenarios from an existing manifest."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)

    manifest = generate_dag_pickplace_scenarios(
        manifest_path=args.manifest,
        out_dir=args.out,
        num_scenarios=args.num_scenarios,
        seeds=args.seeds,
    )
    print(f"Generated DAG pick/place scenarios: {len(manifest['scenarios'])}")
    print(f"Manifest: {args.out / 'manifest.json'}")
    return 0


def generate_dag_pickplace_scenarios(
    manifest_path: Path,
    out_dir: Path,
    num_scenarios: int,
    seeds: list[int],
) -> dict[str, Any]:
    source_manifest = read_json(manifest_path)
    selected_rows = select_manifest_rows(
        source_manifest.get("scenarios", []),
        num_scenarios=num_scenarios,
        seeds=seeds,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(selected_rows):
        source_path = resolve_scenario_path(manifest_path, manifest_row["scenario_path"])
        base_scenario = read_yaml(source_path)
        cooperative_dependency = index % 2 == 0
        scenario = build_dag_pickplace_scenario(
            base_scenario,
            cooperative_dependency=cooperative_dependency,
        )
        scenario_path = out_dir / f"{scenario['name']}.yaml"
        with scenario_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(scenario, handle, sort_keys=False)

        dag = scenario["task_dag"]
        rows.append(
            {
                "scenario_path": str(scenario_path),
                "source_scenario_path": str(source_path),
                "dataset_index": manifest_row.get(
                    "dataset_index",
                    scenario.get("source", {}).get("dataset_index"),
                ),
                "seed": manifest_row.get("seed"),
                "num_robots": len(scenario.get("robots") or {}),
                "num_objects": len(scenario.get("objects") or {}),
                "assignment_strategy": scenario.get("task", {}).get(
                    "assignment_strategy", "nearest_robot"
                ),
                "baseline_type": "dag_pickplace",
                "dag_task_count": len(dag["tasks"]),
                "dag_dependency_count": len(dag["dependencies"]),
                "cooperative_dependency": bool(
                    scenario.get("task", {}).get("cooperative_dependency", False)
                ),
            }
        )

    manifest = {"scenarios": rows}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def select_manifest_rows(
    rows: list[dict[str, Any]], num_scenarios: int, seeds: list[int]
) -> list[dict[str, Any]]:
    allowed_seeds = {int(seed) for seed in seeds}
    selected = [
        row
        for row in rows
        if int(row.get("seed", -1)) in allowed_seeds
    ][:num_scenarios]
    if len(selected) < num_scenarios:
        raise ValueError(
            f"Requested {num_scenarios} scenarios, found {len(selected)} for seeds {sorted(allowed_seeds)}"
        )
    return selected


def build_dag_pickplace_scenario(
    base_scenario: dict[str, Any],
    cooperative_dependency: bool,
) -> dict[str, Any]:
    scenario = deepcopy(base_scenario)
    base_name = str(scenario.get("name", "pickplace"))
    scenario["name"] = dag_scenario_name(base_name)
    scenario.setdefault("source", {})
    scenario["source"]["task_family"] = "dag_pickplace"
    scenario["source"]["source_scenario_name"] = base_name

    tasks = [dict(task) for task in scenario.get("pickplace", {}).get("tasks", [])]
    dag = build_task_dag(tasks, cooperative_dependency=cooperative_dependency)
    scenario["pickplace"]["tasks"] = tasks
    scenario["task_dag"] = dag
    scenario.setdefault("task", {})
    scenario["task"]["type"] = "dag_pickplace"
    scenario["task"]["baseline_type"] = "dag_pickplace"
    scenario["task"]["cooperative_dependency"] = has_cooperative_dependency(dag)
    return scenario


def build_task_dag(
    pickplace_tasks: list[dict[str, Any]],
    cooperative_dependency: bool,
) -> dict[str, Any]:
    dag_tasks: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    object_ids: list[str] = []

    for task in sorted(pickplace_tasks, key=lambda row: str(row.get("object_id", ""))):
        object_id = str(task["object_id"])
        assigned_robot = task.get("assigned_robot")
        pick_task_id = f"pick_{object_id}"
        place_task_id = f"place_{object_id}"
        task["pick_task_id"] = pick_task_id
        task["place_task_id"] = place_task_id
        object_ids.append(object_id)
        dag_tasks.append(
            {
                "task_id": pick_task_id,
                "task_type": "pick",
                "object_id": object_id,
                "assigned_robot": assigned_robot,
                "expected_action_id": "PickObject",
            }
        )
        dag_tasks.append(
            {
                "task_id": place_task_id,
                "task_type": "place",
                "object_id": object_id,
                "assigned_robot": assigned_robot,
                "expected_action_id": "PlaceObject",
            }
        )
        dependencies.append(
            {
                "source": pick_task_id,
                "target": place_task_id,
                "type": "finish_to_start",
            }
        )

    final_robot = pickplace_tasks[0].get("assigned_robot") if pickplace_tasks else None
    dag_tasks.append(
        {
            "task_id": "final_check",
            "task_type": "final_check",
            "object_id": None,
            "assigned_robot": final_robot,
            "expected_action_id": "FinalCheck",
        }
    )
    for object_id in object_ids:
        dependencies.append(
            {
                "source": f"place_{object_id}",
                "target": "final_check",
                "type": "finish_to_start",
            }
        )

    if cooperative_dependency and len(object_ids) >= 3:
        dependencies.extend(
            [
                {
                    "source": f"place_{object_ids[0]}",
                    "target": f"place_{object_ids[2]}",
                    "type": "finish_to_start",
                    "description": "cooperative_dependency",
                },
                {
                    "source": f"place_{object_ids[1]}",
                    "target": f"place_{object_ids[2]}",
                    "type": "finish_to_start",
                    "description": "cooperative_dependency",
                },
            ]
        )

    return {"tasks": dag_tasks, "dependencies": dependencies}


def has_cooperative_dependency(dag: dict[str, Any]) -> bool:
    return any(
        dependency.get("description") == "cooperative_dependency"
        for dependency in dag.get("dependencies", [])
    )


def dag_scenario_name(base_name: str) -> str:
    if base_name.startswith("pickplace_hard_"):
        return "dag_pickplace_" + base_name.removeprefix("pickplace_hard_")
    if base_name.startswith("pickplace_"):
        return "dag_" + base_name
    return f"dag_pickplace_{base_name}"


def resolve_scenario_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    raise SystemExit(main())
