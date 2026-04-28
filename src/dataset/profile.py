"""Profile the BTGenBot behavior tree dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.dataset.classifier import classify_texts


CONTROL_TAGS = {
    "root",
    "BehaviorTree",
    "BehaviourTree",
    "TreeNodesModel",
    "Sequence",
    "SequenceStar",
    "ReactiveSequence",
    "PipelineSequence",
    "Fallback",
    "Selector",
    "Parallel",
    "Inverter",
    "Repeat",
    "RetryUntilSuccesful",
    "RateController",
    "DistanceController",
    "Root",
    "SubTree",
    "ForceSuccess",
    "ForceFailure",
    "AlwaysSuccess",
    "AlwaysFailure",
    "BlackboardCheckString",
    "BlackboardCheckInt",
    "BlackboardCheckDouble",
}

OPENING_TAG_RE = re.compile(r"<\s*(?!/|!|\?)([A-Za-z_][\w:.-]*)\b")
ACTION_CONDITION_TAG_RE = re.compile(
    r"<\s*(Action|Condition)\b(?P<attrs>[^>]*)>", re.IGNORECASE | re.DOTALL
)
ATTR_RE = re.compile(r"\b([A-Za-z_:][\w:.-]*)\s*=\s*([\"'])(.*?)\2", re.DOTALL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    dataset_path = resolve_dataset_path(args.dataset)
    records = load_dataset(dataset_path)
    profiles = [profile_sample(index, sample) for index, sample in enumerate(records)]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(dataset_path, profiles)
    profile_path = out_dir / "bt_dataset_profile.json"
    index_path = out_dir / "bt_dataset_index.csv"
    candidates_path = out_dir / "scenario_candidates.json"

    write_profile_json(profile_path, summary, profiles)
    write_index_csv(index_path, profiles)
    write_candidates_json(candidates_path, build_scenario_candidates(profiles))

    print_report(summary, profile_path, index_path, candidates_path)
    return 0


def resolve_dataset_path(requested_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    fallback = Path("dataset/bt_dataset.json")
    if fallback.exists() and requested_path.as_posix() == "data/bt_dataset.json":
        print(
            f"Dataset not found at {requested_path}; falling back to {fallback}.",
            file=sys.stderr,
        )
        return fallback

    raise FileNotFoundError(f"Dataset file not found: {requested_path}")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}")
    return data


def profile_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    input_text = str(sample.get("input", ""))
    output_xml = str(sample.get("output", ""))

    xml_info = analyze_xml(output_xml)
    classification = classify_texts(
        [
            output_xml,
            *xml_info["xml_tags"],
            *xml_info["action_condition_names"],
            *xml_info["attribute_values"],
        ]
    )

    return {
        "index": index,
        "input": input_text,
        "output_xml": output_xml,
        "xml_node_count": xml_info["xml_node_count"],
        "xml_tags": xml_info["xml_tags"],
        "action_condition_names": xml_info["action_condition_names"],
        "has_navigation": classification.has_navigation,
        "has_recovery": classification.has_recovery,
        "has_operation": classification.has_operation,
        "has_recovery_node": classification.has_recovery_node,
        "has_pick_place": classification.has_pick_place,
        "matched_navigation_keywords": classification.navigation_keywords,
        "matched_recovery_keywords": classification.recovery_keywords,
        "matched_operation_keywords": classification.operation_keywords,
        "matched_pick_place_keywords": classification.pick_place_keywords,
        "parse_error": xml_info["parse_error"],
    }


def analyze_xml(xml_text: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return analyze_xml_with_regex(xml_text, str(exc))

    tags: list[str] = []
    action_condition_names: list[str] = []
    attribute_values: list[str] = []
    node_count = 0

    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        node_count += 1
        tags.append(element.tag)
        attribute_values.extend(str(value) for value in element.attrib.values())
        maybe_name = action_condition_name(element)
        if maybe_name:
            action_condition_names.append(maybe_name)

    return {
        "xml_node_count": node_count,
        "xml_tags": sorted(set(tags)),
        "action_condition_names": sorted(set(action_condition_names)),
        "attribute_values": attribute_values,
        "parse_error": None,
    }


def analyze_xml_with_regex(xml_text: str, parse_error: str) -> dict[str, Any]:
    tags = OPENING_TAG_RE.findall(xml_text)
    action_condition_names: list[str] = []

    for match in ACTION_CONDITION_TAG_RE.finditer(xml_text):
        attrs = dict((key, value) for key, _, value in ATTR_RE.findall(match.group("attrs")))
        name = attrs.get("ID") or attrs.get("id") or attrs.get("name")
        if name:
            action_condition_names.append(name)

    attribute_values = [value for _, _, value in ATTR_RE.findall(xml_text)]

    return {
        "xml_node_count": len(tags),
        "xml_tags": sorted(set(tags)),
        "action_condition_names": sorted(set(action_condition_names)),
        "attribute_values": attribute_values,
        "parse_error": parse_error,
    }


def action_condition_name(element: ET.Element) -> str | None:
    if element.tag in {"Action", "Condition"}:
        return element.attrib.get("ID") or element.attrib.get("name")

    if len(list(element)) == 0 and element.tag not in CONTROL_TAGS:
        return element.tag

    return None


def build_summary(dataset_path: Path, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    recovery_node_indexes = [
        profile["index"] for profile in profiles if profile["has_recovery_node"]
    ]
    pick_place_indexes = [
        profile["index"] for profile in profiles if profile["has_pick_place"]
    ]

    return {
        "dataset_path": str(dataset_path),
        "sample_count": len(profiles),
        "parse_error_count": sum(1 for profile in profiles if profile["parse_error"]),
        "counts": {
            "navigation": sum(1 for profile in profiles if profile["has_navigation"]),
            "recovery": sum(1 for profile in profiles if profile["has_recovery"]),
            "operation": sum(1 for profile in profiles if profile["has_operation"]),
            "pick_place": len(pick_place_indexes),
        },
        "indexes": {
            "recovery_node": recovery_node_indexes,
            "pick_place": pick_place_indexes,
        },
    }


def build_scenario_candidates(profiles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "navigation": candidate_rows(profiles, "has_navigation"),
        "recovery": candidate_rows(profiles, "has_recovery"),
        "operation": candidate_rows(profiles, "has_operation"),
        "pick_place": candidate_rows(profiles, "has_pick_place"),
    }


def candidate_rows(profiles: list[dict[str, Any]], flag: str) -> list[dict[str, Any]]:
    return [
        {
            "index": profile["index"],
            "input": profile["input"],
            "xml_node_count": profile["xml_node_count"],
            "xml_tags": profile["xml_tags"],
            "action_condition_names": profile["action_condition_names"],
        }
        for profile in profiles
        if profile[flag]
    ]


def write_profile_json(
    path: Path, summary: dict[str, Any], profiles: list[dict[str, Any]]
) -> None:
    payload = {**summary, "samples": profiles}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_candidates_json(path: Path, candidates: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_index_csv(path: Path, profiles: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "input",
        "xml_node_count",
        "xml_tags",
        "action_condition_names",
        "has_navigation",
        "has_recovery",
        "has_operation",
        "has_recovery_node",
        "has_pick_place",
        "parse_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            writer.writerow(
                {
                    "index": profile["index"],
                    "input": profile["input"],
                    "xml_node_count": profile["xml_node_count"],
                    "xml_tags": ";".join(profile["xml_tags"]),
                    "action_condition_names": ";".join(profile["action_condition_names"]),
                    "has_navigation": profile["has_navigation"],
                    "has_recovery": profile["has_recovery"],
                    "has_operation": profile["has_operation"],
                    "has_recovery_node": profile["has_recovery_node"],
                    "has_pick_place": profile["has_pick_place"],
                    "parse_error": profile["parse_error"] or "",
                }
            )


def print_report(
    summary: dict[str, Any],
    profile_path: Path,
    index_path: Path,
    candidates_path: Path,
) -> None:
    counts = summary["counts"]
    indexes = summary["indexes"]
    print(f"Total samples: {summary['sample_count']}")
    print(
        "Counts: "
        f"navigation={counts['navigation']}, "
        f"recovery={counts['recovery']}, "
        f"operation={counts['operation']}, "
        f"pick_place={counts['pick_place']}"
    )
    print(f"Parse errors: {summary['parse_error_count']}")
    print(f"RecoveryNode sample indexes: {indexes['recovery_node']}")
    print(f"Pick/place sample indexes: {indexes['pick_place']}")
    print(f"Wrote profile JSON: {profile_path}")
    print(f"Wrote index CSV: {index_path}")
    print(f"Wrote scenario candidates JSON: {candidates_path}")


if __name__ == "__main__":
    raise SystemExit(main())
