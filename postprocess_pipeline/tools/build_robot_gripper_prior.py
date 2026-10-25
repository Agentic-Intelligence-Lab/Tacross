#!/usr/bin/env python3
"""Build a robust 1-DoF gripper prior from LeRobot state feedback."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EPISODE_NUMBER = re.compile(r"episode_(\d+)\.parquet$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="robot_gripper_prior")
    parser.add_argument("--first-episode", type=int)
    parser.add_argument("--last-episode", type=int)
    parser.add_argument("--state-column", default="observation.state")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--state-index", type=int, default=-1)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--profile-points", type=int, default=301)
    parser.add_argument("--task-scope", default="user-selected same-task demonstrations")
    args = parser.parse_args()

    if (args.first_episode is None) != (args.last_episode is None):
        parser.error("--first-episode and --last-episode must be used together")
    if args.profile_points < 2:
        parser.error("--profile-points must be >= 2")

    dataset = args.dataset.expanduser().resolve()
    parquet_files = sorted(dataset.glob("data/**/episode_*.parquet"))
    selected: list[tuple[int, Path]] = []
    for path in parquet_files:
        match = EPISODE_NUMBER.search(path.name)
        if not match:
            continue
        number = int(match.group(1))
        if args.first_episode is not None and not (
            args.first_episode <= number <= args.last_episode
        ):
            continue
        selected.append((number, path))
    if not selected:
        raise RuntimeError("no matching Episode parquet files")

    grid = np.linspace(0.0, 1.0, args.profile_points)
    profiles: list[np.ndarray] = []
    values_all: list[np.ndarray] = []
    episode_rows = []
    for number, path in selected:
        frame = pd.read_parquet(
            path, columns=[args.state_column, args.timestamp_column]
        )
        states = np.stack(frame[args.state_column].to_numpy()).astype(float)
        if states.ndim != 2 or len(states) < 2:
            raise RuntimeError(f"invalid state array in {path}: shape={states.shape}")
        index = args.state_index if args.state_index >= 0 else states.shape[1] + args.state_index
        if not 0 <= index < states.shape[1]:
            raise RuntimeError(f"invalid state array/index in {path}")
        values = states[:, index]
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite gripper feedback in {path}")
        timestamps = pd.to_numeric(frame[args.timestamp_column], errors="raise").to_numpy(float)
        phase = np.linspace(0.0, 1.0, len(values))
        profiles.append(np.interp(grid, phase, values))
        values_all.append(values)
        episode_rows.append(
            {
                "episode": number,
                "parquet": str(path),
                "frames": len(values),
                "duration_s": float(timestamps[-1] - timestamps[0]),
                "start": float(values[0]),
                "end": float(values[-1]),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )

    profile_array = np.stack(profiles)
    all_values = np.concatenate(values_all)
    quantiles = {
        f"q{int(round(q * 100)):02d}": float(np.quantile(all_values, q))
        for q in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / f"{args.output_prefix}.csv"
    pd.DataFrame(
        {
            "normalized_phase": grid,
            **{
                f"p{int(q * 100):02d}": np.quantile(profile_array, q, axis=0)
                for q in (0.10, 0.25, 0.50, 0.75, 0.90)
            },
        }
    ).to_csv(profile_path, index=False, float_format="%.9f")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset": str(dataset),
        "source_episode_range_inclusive": (
            [args.first_episode, args.last_episode]
            if args.first_episode is not None
            else [selected[0][0], selected[-1][0]]
        ),
        "source_episode_count": len(selected),
        "task_scope": args.task_scope,
        "signal": f"{args.state_column}[{args.state_index}] actual feedback",
        "source_fps": args.source_fps,
        "total_source_frames": int(sum(len(values) for values in values_all)),
        "global_feedback_quantiles": quantiles,
        "fusion_anchors": {
            "open_typical": quantiles["q05"],
            "closed_typical": quantiles["q95"],
            "hard_min": quantiles["q01"],
            "hard_max": quantiles["q99"],
            "explanation": "visual closure maps to q05..q95 and clips to q01..q99",
        },
        "normalized_phase_profile": {
            "points": args.profile_points,
            "csv": str(profile_path),
            "columns": ["normalized_phase", "p10", "p25", "p50", "p75", "p90"],
        },
        "episodes": episode_rows,
    }
    metadata_path = output / f"{args.output_prefix}.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        "ROBOT_GRIPPER_PRIOR_PASS "
        f"episodes={len(selected)} range={quantiles['q01']:.4f}..{quantiles['q99']:.4f} "
        f"metadata={metadata_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
