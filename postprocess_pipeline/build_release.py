#!/usr/bin/env python3
"""Build a deterministic, privacy-scanned source archive.

The builder works from an explicit allowlist, so new private experiments,
datasets, checkpoints, logs, and credentials are never picked up implicitly.
Every file is scanned for absolute personal paths, host addresses, and
credential-shaped strings before it is written to the archive. Archive
contents, entry order, and ZIP timestamps are deterministic for identical
inputs and version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# Directories copied recursively. Everything else must be listed explicitly.
INCLUDE_DIRS = (
    "postprocess_pipeline",
    "adapters",
    "examples",
    "docs",
)
INCLUDE_FILES = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "README.zh-CN.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-undistort.txt",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp", ".swo"}
# Generated demo output, and anything else that must never reach an archive.
EXCLUDED_RELATIVE = ("examples/demo_data",)

# Generic leak patterns. These match shapes rather than the author's own
# machine, so the check stays useful for contributors.
FORBIDDEN_PATTERNS = (
    ("posix_home_path", re.compile(r"/(?:home|Users|media|mnt)/[A-Za-z0-9._-]+/")),
    (
        "windows_user_path",
        re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+", re.I),
    ),
    (
        "ipv4_literal",
        re.compile(
            r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
    ),
    ("ssh_target", re.compile(r"\bssh\s+[A-Za-z0-9._-]+@[A-Za-z0-9._-]+")),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ),
    (
        "generic_api_key",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)"
            r"\s*[:=]\s*['\"][A-Za-z0-9/+_-]{16,}['\"]"
        ),
    ),
)

# This file necessarily contains the pattern text it searches for.
SCAN_EXEMPT = {"postprocess_pipeline/build_release.py"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release_files() -> list[Path]:
    paths = [PROJECT / name for name in INCLUDE_FILES]
    for directory in INCLUDE_DIRS:
        paths.extend(
            path
            for path in (PROJECT / directory).rglob("*")
            if path.is_file()
            and not any(
                part in EXCLUDED_PARTS for part in path.relative_to(PROJECT).parts
            )
            and path.suffix not in EXCLUDED_SUFFIXES
            and not path.relative_to(PROJECT)
            .as_posix()
            .startswith(EXCLUDED_RELATIVE)
        )
    unique = sorted(set(paths), key=lambda path: path.relative_to(PROJECT).as_posix())
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "release allowlist contains missing files: "
            + ", ".join(str(path.relative_to(PROJECT)) for path in missing)
        )
    return unique


def scan(relative: str, data: bytes) -> None:
    if relative in SCAN_EXEMPT:
        return
    text = data.decode("utf-8", errors="ignore")
    hits = sorted({name for name, pattern in FORBIDDEN_PATTERNS if pattern.search(text)})
    if hits:
        raise RuntimeError(f"privacy scan failed for {relative}: {hits}")


def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--name", default="embodied-postprocess-pipeline")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "dist")
    args = parser.parse_args()

    prefix = f"{args.name}-{args.version}"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{prefix}.zip"
    files = release_files()
    payloads: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(PROJECT).as_posix()
        data = path.read_bytes()
        scan(relative, data)
        payloads.append((relative, data))

    manifest = {
        "schema_version": 1,
        "name": args.name,
        "version": args.version,
        "contents": [
            {"path": relative, "sha256": sha256(data), "bytes": len(data)}
            for relative, data in payloads
        ],
        "excluded": (
            "datasets, model weights, logs, caches, credentials, and third-party "
            "source/weights"
        ),
    }
    manifest_data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relative, data in payloads:
            add_bytes(archive, f"{prefix}/{relative}", data)
        add_bytes(archive, f"{prefix}/RELEASE_MANIFEST.json", manifest_data)

    archive_digest = sha256(archive_path.read_bytes())
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{archive_digest}  {archive_path.name}\n", encoding="utf-8")
    print("OPEN_PIPELINE_RELEASE_PASS")
    print(f"  files: {len(payloads)}")
    print(f"  archive: {archive_path}")
    print(f"  sha256: {archive_digest}")
    print(f"  checksum: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
