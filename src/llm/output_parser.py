"""Parse and validate raw LLM responses."""

from __future__ import annotations

from typing import Any
import json

from src.llm.schema import ValidationResult, validate_llm_output


def parse_llm_json(raw_response: str) -> dict[str, Any]:
    stripped = raw_response.strip()
    parsed = parse_json_object_candidate(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def parse_json_object_candidate(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = strip_json_fence(text)
    if fenced != text:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    obj = extract_first_json_object(text)
    if obj is not None:
        return json.loads(obj)
    raise ValueError("Could not parse LLM response as a JSON object")


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return text
    closing = stripped.rfind("```")
    if closing <= first_newline:
        return text
    return stripped[first_newline + 1 : closing].strip()


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        candidate = balanced_json_object_from(text, start)
        if candidate is not None:
            return candidate
        start = text.find("{", start + 1)
    return None


def balanced_json_object_from(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return candidate
    return None


def parse_and_validate(raw_response: str, scenario: Any) -> tuple[dict[str, Any] | None, ValidationResult]:
    try:
        parsed = parse_llm_json(raw_response)
    except Exception as exc:
        return None, ValidationResult(False, [str(exc)])
    return parsed, validate_llm_output(parsed, scenario)
