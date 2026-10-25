#!/usr/bin/env python3
"""Create a non-destructive Episode selection manifest.

This tool only inspects paths. It never deletes, moves, or renumbers source
Episodes. Dataset-specific required files are supplied on the command line.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

EPISODE_SUFFIX = re.compile(r"(\d+)$")


def episode_number(name: str) -> int | None:
    match = EPISODE_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-glob", default="episode_*")
    parser.add_argument("--first-episode", type=int)
    parser.add_argument("--last-episode", type=int)
    parser.add_argument(
        "--required",
        action="append",
        default=[],
        metavar="RELATIVE_PATH",
        help="Required Episode-relative file or directory; repeatable.",
    )
    args = parser.parse_args()

    if (args.first_episode is None) != (args.last_episode is None):
        parser.error("--first-episode and --last-episode must be used together")
    if args.first_episode is not None and args.last_episode < args.first_episode:
        parser.error("invalid inclusive Episode range")

    root = args.source_root.expanduser().resolve()
    discovered = sorted(path for path in root.glob(args.episode_glob) if path.is_dir())
    by_number = {
        number: path
        for path in discovered
        if (number := episode_number(path.name)) is not None
    }
    if args.first_episode is None:
        candidates: list[tuple[int | None, Path | None, str]] = [
            (episode_number(path.name), path, path.name) for path in discovered
        ]
    else:
        width = max(6, len(str(args.last_episode)))
        candidates = [
            (number, by_number.get(number), f"episode_{number:0{width}d}")
            for number in range(args.first_episode, args.last_episode + 1)
        ]

    valid: list[str] = []
    excluded: list[dict[str, object]] = []
    for number, episode, expected_name in candidates:
        reasons: list[str] = []
        if episode is None:
            reasons.append("missing_source_episode")
            name = expected_name
        else:
            name = episode.name
            for relative in args.required:
                if not (episode / relative).exists():
                    reasons.append(f"missing:{relative}")
        if reasons:
            excluded.append(
                {"episode": name, "source_episode": number, "reasons": reasons}
            )
        else:
            valid.append(name)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(root),
        "selection": {
            "episode_glob": args.episode_glob,
            "episode_range_inclusive": (
                [args.first_episode, args.last_episode]
                if args.first_episode is not None
                else None
            ),
            "required_paths": args.required,
            "source_episode_ids_preserved": True,
        },
        "valid_episodes": valid,
        "excluded_episodes": excluded,
        "counts": {
            "discovered": len(discovered),
            "requested": len(candidates),
            "valid": len(valid),
            "excluded": len(excluded),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        "EPISODE_MANIFEST_PASS "
        f"valid={len(valid)} excluded={len(excluded)} output={output}"
    )
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
