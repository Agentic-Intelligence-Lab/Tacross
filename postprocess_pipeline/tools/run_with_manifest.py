#!/usr/bin/env python3
"""Append selected Episode arguments from a manifest and run one command."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split our own options from the wrapped command at the first ``--``.

    ``argparse.REMAINDER`` stops treating options as options as soon as it sees
    a non-option token, which silently mangles wrapped commands that contain
    flags of their own.  Splitting explicitly keeps the wrapped argv verbatim.
    """
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-flag", default="--episode")
    parser.add_argument("--mode", choices=("repeat", "csv"), default="repeat")
    parser.add_argument("--dry-run", action="store_true")
    own, command = split_argv(sys.argv[1:])
    args = parser.parse_args(own)

    if not command:
        parser.error("a command is required after --")
    payload = json.loads(args.manifest.expanduser().resolve().read_text(encoding="utf-8"))
    episodes = payload.get("valid_episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("manifest contains no valid_episodes")
    if not all(isinstance(item, str) and item for item in episodes):
        raise RuntimeError("manifest valid_episodes must be non-empty strings")
    if args.mode == "repeat":
        for episode in episodes:
            command.extend((args.target_flag, episode))
    else:
        command.extend((args.target_flag, ",".join(episodes)))
    print("MANIFEST_COMMAND", shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
