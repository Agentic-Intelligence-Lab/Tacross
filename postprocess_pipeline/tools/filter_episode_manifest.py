#!/usr/bin/env python3
"""Filter an existing manifest using derived-artifact and JSON status gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_json_gate(value: str) -> tuple[str, str, str]:
    if ":" not in value or "=" not in value:
        raise argparse.ArgumentTypeError(
            "JSON gate must be RELATIVE_PATH:DOT.KEY=EXPECTED_VALUE"
        )
    relative, expression = value.split(":", 1)
    key, expected = expression.split("=", 1)
    if not relative or not key:
        raise argparse.ArgumentTypeError(
            "JSON gate must be RELATIVE_PATH:DOT.KEY=EXPECTED_VALUE"
        )
    return relative, key, expected


def nested_value(payload: object, dotted_key: str) -> object:
    value = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--json-equals", action="append", type=parse_json_gate, default=[])
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve()
    original = json.loads(args.input.expanduser().resolve().read_text(encoding="utf-8"))
    valid: list[str] = []
    rejected: list[dict[str, object]] = []
    for name in original.get("valid_episodes", []):
        episode = source / name
        reasons: list[str] = []
        for relative in args.required:
            if not (episode / relative).is_file():
                reasons.append(f"missing:{relative}")
        for relative, key, expected in args.json_equals:
            path = episode / relative
            if not path.is_file():
                reasons.append(f"missing:{relative}")
                continue
            try:
                actual = nested_value(json.loads(path.read_text(encoding="utf-8")), key)
            except (OSError, json.JSONDecodeError, KeyError):
                reasons.append(f"invalid_json_gate:{relative}:{key}")
                continue
            if str(actual) != expected:
                reasons.append(
                    f"json_gate_failed:{relative}:{key}={actual!s};expected={expected}"
                )
        if reasons:
            rejected.append({"episode": name, "reasons": reasons})
        else:
            valid.append(name)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(source),
        "parent_manifest": str(args.input.expanduser().resolve()),
        "gates": {
            "required_paths": args.required,
            "json_equals": [
                {"path": relative, "key": key, "expected": expected}
                for relative, key, expected in args.json_equals
            ],
        },
        "valid_episodes": valid,
        "excluded_episodes": rejected,
        "counts": {
            "input": len(original.get("valid_episodes", [])),
            "valid": len(valid),
            "excluded": len(rejected),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        "EPISODE_MANIFEST_FILTER_PASS "
        f"valid={len(valid)} excluded={len(rejected)} output={output}"
    )
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
