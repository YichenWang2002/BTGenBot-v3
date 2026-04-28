"""Run a multi-robot navigation scenario."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.env.multi_robot_env import MultiRobotEnv
from src.env.scenario_loader import load_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--trace-out", default=Path("runs/demo_multi_robot_trace.json"), type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", default=5, type=int)
    parser.add_argument("--pause-on-end", action="store_true")
    parser.add_argument("--cell-size", default=48, type=int)
    parser.add_argument(
        "--centralized-rule",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override scenario centralized_rule setting.",
    )
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    centralized_rule = scenario.centralized_rule
    if args.centralized_rule != "auto":
        centralized_rule = args.centralized_rule == "true"
    render = bool(scenario.render or args.render)
    if args.headless:
        render = False
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = MultiRobotEnv(
        scenario.state,
        max_steps=scenario.max_steps,
        render=render,
        cell_size=args.cell_size,
        render_fps=args.fps,
        scenario_name=scenario.name,
        centralized_rule=centralized_rule,
        recovery_config=scenario.recovery,
        pickplace_config=scenario.pickplace,
        zones=scenario.zones,
        coordination_config=scenario.coordination,
        task_dag=scenario.task_dag,
    )
    env.reset()

    try:
        while not env.done:
            env.step()
        if render and args.pause_on_end:
            env.render()
            env.pause_render_window()
    finally:
        env.close()

    trace_payload = env.metrics.to_trace_payload(
        scenario_name=scenario.name,
        num_robots=scenario.num_robots,
    )
    args.trace_out.parent.mkdir(parents=True, exist_ok=True)
    with args.trace_out.open("w", encoding="utf-8") as handle:
        json.dump(trace_payload, handle, indent=2)
        handle.write("\n")

    summary = env.metrics.summary()
    print(f"scenario name: {scenario.name}")
    print(f"num_robots: {scenario.num_robots}")
    print(f"centralized_rule_enabled: {summary['centralized_rule_enabled']}")
    print(f"success: {summary['success']}")
    print(f"makespan: {summary['makespan']}")
    print(f"total_robot_steps: {summary['total_robot_steps']}")
    print(f"collision_count: {summary['collision_count']}")
    print(f"vertex_conflict_count: {summary['vertex_conflict_count']}")
    print(f"edge_conflict_count: {summary['edge_conflict_count']}")
    print(f"rule_rejection_count: {summary['rule_rejection_count']}")
    print(f"resource_conflict_count: {summary['resource_conflict_count']}")
    print(f"resource_request_denied_count: {summary['resource_request_denied_count']}")
    print(
        "rule_prevented_resource_conflict_count: "
        f"{summary['rule_prevented_resource_conflict_count']}"
    )
    print(f"lock_wait_count: {summary['lock_wait_count']}")
    print(f"wait_count: {summary['wait_count']}")
    print(f"deadlock_count: {summary['deadlock_count']}")
    print(f"timeout: {summary['timeout']}")
    print(f"trace path: {args.trace_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
