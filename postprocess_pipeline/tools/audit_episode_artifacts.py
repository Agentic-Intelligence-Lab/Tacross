#!/usr/bin/env python3
"""Audit per-Episode artifacts and produce an explicit training-ready list."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_stage(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("stage must be NAME=RELATIVE_PATH")
    name, relative = value.split("=", 1)
    if not name or not relative:
        raise argparse.ArgumentTypeError("stage must be NAME=RELATIVE_PATH")
    return name, relative


def csv_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    return len(pd.read_csv(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ik-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline-relative", required=True)
    parser.add_argument("--stage", action="append", type=parse_stage, default=[])
    parser.add_argument(
        "--fusion-metadata-relative",
        default="sync/gripper_hamer_robot_prior_metadata.json",
    )
    parser.add_argument("--ik-report-name", default="offline_ik_report.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
    source = args.source_root.expanduser().resolve()
    ik_root = args.ik_root.expanduser().resolve()
    episodes = []
    stage_counts: Counter[str] = Counter()
    for name in manifest["valid_episodes"]:
        episode = source / name
        timeline_path = episode / args.timeline_relative
        timeline_rows = csv_rows(timeline_path) if timeline_path.is_file() else None
        stages = {}
        for stage_name, relative in args.stage:
            path = episode / relative
            rows = csv_rows(path) if path.is_file() else None
            complete = path.is_file() and (
                rows is None or timeline_rows is None or rows == timeline_rows
            )
            stages[stage_name] = {
                "complete": complete,
                "path": str(path),
                "rows": rows,
                "expected_rows": timeline_rows if rows is not None else None,
            }
            if complete:
                stage_counts[stage_name] += 1

        fusion_path = episode / args.fusion_metadata_relative
        fusion_status = None
        if fusion_path.is_file():
            try:
                fusion_status = json.loads(fusion_path.read_text(encoding="utf-8")).get("status")
            except (OSError, json.JSONDecodeError):
                fusion_status = "INVALID_JSON"
        ik_path = ik_root / name / args.ik_report_name
        ik_status = None
        if ik_path.is_file():
            try:
                ik_status = json.loads(ik_path.read_text(encoding="utf-8")).get("status")
            except (OSError, json.JSONDecodeError):
                ik_status = "INVALID_JSON"
        ready = (
            bool(stages)
            and all(item["complete"] for item in stages.values())
            and fusion_status == "PASS"
            and ik_status == "PASS"
        )
        episodes.append(
            {
                "episode": name,
                "timeline_rows": timeline_rows,
                "stages": stages,
                "fusion_status": fusion_status,
                "ik_status": ik_status,
                "training_ready": ready,
            }
        )

    ready_names = [item["episode"] for item in episodes if item["training_ready"]]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.expanduser().resolve()),
        "source_root": str(source),
        "ik_root": str(ik_root),
        "requested_episode_count": len(episodes),
        "stage_complete_counts": dict(stage_counts),
        "training_ready_count": len(ready_names),
        "training_ready_episodes": ready_names,
        "rejected_episodes": [
            item["episode"] for item in episodes if not item["training_ready"]
        ],
        "episodes": episodes,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        "EPISODE_ARTIFACT_AUDIT "
        f"requested={len(episodes)} ready={len(ready_names)} output={output}"
    )
    complete = bool(episodes) and len(ready_names) == len(episodes)
    return 0 if complete or not args.strict else 3


if __name__ == "__main__":
    raise SystemExit(main())
