"""Build LLM repair benchmark suites from generated scenario manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_METHODS = ["llm_only_multi_robot", "centralized_rule_multi_bt"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-manifest", required=True, type=Path)
    parser.add_argument("--recovery-manifest", required=True, type=Path)
    parser.add_argument("--pickplace-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="results/llm_repair_small_openai_compatible",
        type=Path,
    )
    args = parser.parse_args(argv)

    suite, summary = build_suite(
        nav_manifest=args.nav_manifest,
        recovery_manifest=args.recovery_manifest,
        pickplace_manifest=args.pickplace_manifest,
        out=args.out,
        per_family=args.per_family,
        output_dir=args.output_dir,
    )
    write_suite(args.out, suite)
    summary_path = manifest_summary_path(args.out)
    write_json(summary_path, summary)
    print(
        "LLM suite scenarios: "
        f"nav={summary['selected_counts']['nav']} "
        f"recovery={summary['selected_counts']['recovery']} "
        f"pickplace={summary['selected_counts']['pickplace']} "
        f"runs={summary['total_runs_expected']}"
    )
    print(f"Suite path: {args.out}")
    print(f"Manifest summary: {summary_path}")
    return 0


def build_suite(
    nav_manifest: Path,
    recovery_manifest: Path,
    pickplace_manifest: Path,
    out: Path,
    per_family: int = 10,
    output_dir: Path | str = Path("results/llm_repair_small_openai_compatible"),
) -> tuple[dict[str, Any], dict[str, Any]]:
    selections = {
        "nav": select_existing_scenarios(nav_manifest, "nav", per_family),
        "recovery": select_existing_scenarios(recovery_manifest, "recovery", per_family),
        "pickplace": select_existing_scenarios(pickplace_manifest, "pickplace", per_family),
    }
    scenarios = [
        {"path": item["scenario_path"], "scenario_family": family}
        for family in ("nav", "recovery", "pickplace")
        for item in selections[family]
    ]
    suite = {
        "name": "llm_repair_small",
        "output_dir": str(output_dir),
        "max_iters": 3,
        "methods": list(DEFAULT_METHODS),
        "scenarios": scenarios,
    }
    selected_counts = {family: len(items) for family, items in selections.items()}
    scenario_paths = [item["path"] for item in scenarios]
    summary = {
        "suite_path": str(out),
        "output_dir": str(output_dir),
        "per_family_requested": per_family,
        "selected_counts": selected_counts,
        "total_scenarios": len(scenarios),
        "total_runs_expected": len(scenarios) * len(DEFAULT_METHODS),
        "methods": list(DEFAULT_METHODS),
        "scenario_paths": scenario_paths,
    }
    return suite, summary


def select_existing_scenarios(
    manifest_path: Path, family: str, per_family: int
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    selected: list[dict[str, Any]] = []
    for item in manifest:
        scenario_path = item.get("scenario_path") if isinstance(item, dict) else str(item)
        if not scenario_path:
            continue
        if Path(scenario_path).exists():
            selected.append({"scenario_path": scenario_path, **(item if isinstance(item, dict) else {})})
        if len(selected) >= per_family:
            break
    if len(selected) < per_family:
        print(
            f"Warning: requested {per_family} {family} scenarios, "
            f"but only found {len(selected)} existing paths in {manifest_path}"
        )
    return selected


def load_manifest(path: Path) -> list[dict[str, Any] | str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        value = payload.get("scenarios")
        if isinstance(value, list):
            return value
        value = payload.get("items")
        if isinstance(value, list):
            return value
    raise ValueError(f"Unsupported manifest format: {path}")


def write_suite(path: Path, suite: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def manifest_summary_path(suite_path: Path) -> Path:
    return suite_path.with_name(f"{suite_path.stem}_manifest_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
