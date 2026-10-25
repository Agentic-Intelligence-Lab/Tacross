#!/usr/bin/env python3
"""Run the existing post-acquisition tools as a reproducible staged pipeline.

The configuration is JSON and every command is an argv array.  Shell parsing is
deliberately not used: paths containing spaces remain safe and the exact argv
is preserved in the provenance report.  The default mode is a read-only dry
run; pass ``--execute`` to start processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(RuntimeError):
    """Raised when a pipeline configuration is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Run only this stage; repeat to select several stages.",
    )
    parser.add_argument("--from-stage", help="Start at this stage, inclusive.")
    parser.add_argument("--to-stage", help="Stop at this stage, inclusive.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute commands. Without this flag only a dry run is printed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a stage when every path listed in its produces field exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later stages after a command failure.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a config variable; repeatable. Useful for portable configs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be an object")
    return value


def expand_text(value: str, variables: dict[str, str]) -> str:
    """Substitute ``{name}`` placeholders, leaving other braces untouched.

    ``str.format_map`` raises on any stray brace, so a task instruction such as
    ``"put the {red} block away"`` or a Windows path with braces would break the
    whole run.  Only identifier-shaped placeholders are expanded here, and
    environment variables plus ``~`` are expanded afterwards.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise ConfigError(f"unknown template variable {name!r} in {value!r}")
        return variables[name]

    expanded = PLACEHOLDER.sub(replace, value)
    return os.path.expandvars(os.path.expanduser(expanded))


def parse_overrides(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set expects NAME=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"--set has an empty variable name: {item!r}")
        result[key] = value
    return result


def resolve_variables(
    raw: dict[str, Any],
    config_path: Path,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ConfigError("variables must be an object")
    values = {str(key): str(value) for key, value in raw.items()}
    values.setdefault("config_dir", str(config_path.parent.resolve()))
    values.update(overrides or {})
    # Several passes allow one variable to reference another while still
    # detecting unresolved names and cycles deterministically.
    for _ in range(len(values) + 2):
        changed = False
        for key, value in list(values.items()):
            try:
                expanded = expand_text(value, values)
            except ConfigError:
                continue
            if expanded != value:
                values[key] = expanded
                changed = True
        if not changed:
            break
    for key, value in values.items():
        try:
            values[key] = expand_text(value, values)
        except ConfigError as exc:
            raise ConfigError(f"cannot resolve variable {key!r}: {exc}") from exc
        if PLACEHOLDER.search(values[key]):
            raise ConfigError(
                f"variable {key!r} still contains a placeholder after resolution "
                f"(cyclic reference?): {values[key]!r}"
            )
    return values


def expand_list(values: Any, variables: dict[str, str], field: str) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
        raise ConfigError(f"{field} must be a list of strings")
    return [expand_text(value, variables) for value in values]


def normalized_stages(config: dict[str, Any], variables: dict[str, str]) -> list[dict[str, Any]]:
    raw_stages = config.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ConfigError("stages must be a non-empty list")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    # Accept either conventional root name so hand-written configs work.
    default_cwd = variables.get(
        "project_root", variables.get("repo_root", str(Path.cwd()))
    )
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise ConfigError(f"stage #{index + 1} must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"stage #{index + 1} has no valid name")
        if name in names:
            raise ConfigError(f"duplicate stage name: {name}")
        names.add(name)
        commands_raw = raw.get("commands")
        if commands_raw is None and "command" in raw:
            commands_raw = [raw["command"]]
        if not isinstance(commands_raw, list) or not commands_raw:
            raise ConfigError(f"stage {name!r} has no commands")
        if commands_raw and all(isinstance(x, str) for x in commands_raw):
            commands_raw = [commands_raw]
        commands = [
            expand_list(command, variables, f"stage {name!r} command")
            for command in commands_raw
        ]
        stage = {
            "name": name,
            "description": str(raw.get("description", "")),
            "commands": commands,
            "cwd": expand_text(str(raw.get("cwd", default_cwd)), variables),
            "requires": expand_list(raw.get("requires", []), variables, f"stage {name!r} requires"),
            "produces": expand_list(raw.get("produces", []), variables, f"stage {name!r} produces"),
            "env": {
                str(key): expand_text(str(value), variables)
                for key, value in dict(raw.get("env", {})).items()
            },
            "accept_returncodes": [int(value) for value in raw.get("accept_returncodes", [0])],
        }
        if not stage["accept_returncodes"]:
            raise ConfigError(f"stage {name!r} accept_returncodes cannot be empty")
        result.append(stage)
    return result


def select_stages(stages: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    names = [stage["name"] for stage in stages]
    requested = set(args.stage)
    unknown = requested - set(names)
    if unknown:
        raise ConfigError(f"unknown stage(s): {', '.join(sorted(unknown))}")
    start = 0
    end = len(stages)
    if args.from_stage:
        if args.from_stage not in names:
            raise ConfigError(f"unknown --from-stage: {args.from_stage}")
        start = names.index(args.from_stage)
    if args.to_stage:
        if args.to_stage not in names:
            raise ConfigError(f"unknown --to-stage: {args.to_stage}")
        end = names.index(args.to_stage) + 1
    if start >= end:
        raise ConfigError("--from-stage occurs after --to-stage")
    selected = stages[start:end]
    if requested:
        selected = [stage for stage in selected if stage["name"] in requested]
    if not selected:
        raise ConfigError("stage selection is empty")
    return selected


def quote_command(command: list[str]) -> str:
    return shlex.join(command)


def stage_ready(stage: dict[str, Any]) -> bool:
    outputs = stage["produces"]
    return bool(outputs) and all(Path(path).exists() for path in outputs)


def path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def missing_requirements(
    stage: dict[str, Any], satisfied: set[str] | None = None
) -> list[str]:
    """Requirements that are neither on disk nor produced earlier in this run.

    Without the ``satisfied`` set a cold start could never execute: every stage
    after the first one requires an artifact that a previous stage has not
    written yet, so the up-front preflight would abort the whole run.
    """
    satisfied = satisfied or set()
    return [
        path
        for path in stage["requires"]
        if not Path(path).exists() and path_key(path) not in satisfied
    ]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    if config.get("schema_version") != 1:
        raise ConfigError("only schema_version=1 is supported")
    variables = resolve_variables(
        config.get("variables", {}),
        config_path,
        parse_overrides(args.set),
    )
    stages = normalized_stages(config, variables)

    if args.list:
        print(f"pipeline: {config.get('name', config_path.stem)}")
        for index, stage in enumerate(stages, start=1):
            print(f"{index:02d}  {stage['name']:<20} {stage['description']}")
        return 0

    selected = select_stages(stages, args)
    print(f"pipeline={config.get('name', config_path.stem)} mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"config={config_path}")
    scheduled: set[str] = set()
    for stage in selected:
        print(f"\n[{stage['name']}] {stage['description']}")
        print(f"  cwd: {stage['cwd']}")
        missing = missing_requirements(stage, scheduled)
        if missing:
            for path in missing:
                print(f"  MISSING: {path}")
        if args.resume and stage_ready(stage):
            print("  SKIP: all declared outputs already exist")
            scheduled.update(path_key(path) for path in stage["produces"])
            continue
        for command in stage["commands"]:
            print(f"  $ {quote_command(command)}")
        scheduled.update(path_key(path) for path in stage["produces"])
        if missing and args.execute:
            print("  FAIL: required paths are missing", file=sys.stderr)
            return 2

    if not args.execute:
        print("\nDry run only. Add --execute after reviewing the commands.")
        return 0

    run_root = Path(
        expand_text(
            str(config.get("run_root", "{config_dir}/runs")),
            variables,
        )
    )
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{stamp}_{config_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": config.get("name", config_path.stem),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "variables": variables,
        "stages": [],
    }
    report_path = run_dir / "provenance.json"
    exit_code = 0
    for stage in selected:
        stage_report: dict[str, Any] = {
            "name": stage["name"],
            "description": stage["description"],
            "cwd": stage["cwd"],
            "commands": [],
        }
        report["stages"].append(stage_report)
        if args.resume and stage_ready(stage):
            stage_report["status"] = "SKIPPED_OUTPUTS_EXIST"
            write_report(report_path, report)
            continue
        missing = missing_requirements(stage)
        if missing:
            stage_report.update({"status": "FAILED_PREFLIGHT", "missing": missing})
            exit_code = 2
            write_report(report_path, report)
            if not args.continue_on_error:
                break
            continue
        stage_report["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        stage_clock = time.monotonic()
        env = os.environ.copy()
        env.update(stage["env"])
        stage_failed = False
        for command_index, command in enumerate(stage["commands"], start=1):
            log_path = run_dir / f"{stage['name']}_{command_index:02d}.log"
            print(f"\nRUN [{stage['name']}] {quote_command(command)}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"$ {quote_command(command)}\n")
                log.flush()
                process = subprocess.run(
                    command,
                    cwd=stage["cwd"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            command_report = {
                "argv": command,
                "log": str(log_path),
                "returncode": process.returncode,
            }
            stage_report["commands"].append(command_report)
            if process.returncode not in stage["accept_returncodes"]:
                stage_failed = True
                exit_code = process.returncode or 1
                print(f"FAIL [{stage['name']}] returncode={process.returncode} log={log_path}")
                break
            if process.returncode != 0:
                command_report["accepted_nonzero_returncode"] = True
        stage_report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        stage_report["duration_s"] = round(time.monotonic() - stage_clock, 3)
        stage_report["status"] = "FAILED" if stage_failed else "PASS"
        write_report(report_path, report)
        if stage_failed and not args.continue_on_error:
            break

    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["status"] = "PASS" if exit_code == 0 else "FAILED"
    write_report(report_path, report)
    print(f"\n{report['status']}: provenance={report_path}")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as error:
        print(f"CONFIG_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
