#!/usr/bin/env python3
"""Generate a synthetic Ego recording plus robot demonstrations for the demo.

Nothing here is a real recording. The point is to produce files with the exact
directory layout and column names the pipeline expects, so the orchestration and
the quality gates can be exercised on a laptop without cameras, a GPU, ffmpeg, or
any external model.

Two deliberate simplifications:

* No ``.avi``/``.mp4`` assets are written. Every stage before the packer reads
  only the timestamp CSVs, so the demo never needs a decoder. A real recording
  does need the videos -- see ``examples/README.md``.
* Episode 3 is generated with heavy hand-detection dropout so it fails the
  gripper-fusion gate. That is the interesting case: the pipeline should carry it
  through the timeline, then reject it before IK.

Usage::

    python examples/make_demo_dataset.py --output demo_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAMERA_FPS = 30.0
TACTILE_FPS = 100.0
DURATION_S = 12.0
STATE_DIM = 8
ROBOT_FPS = 30.0

# name, camera start offset, tactile start offset, detection fraction
EPISODES = (
    ("episode_000001", 0.000, 0.010, 0.93),
    ("episode_000002", 0.033, 0.004, 0.88),
    ("episode_000003", 0.017, 0.020, 0.45),  # fails the fusion gate on purpose
)
ROBOT_EPISODES = 4


def write_trigger_log(path: Path, master_start_ns: int, gopro_ack_ns: int) -> None:
    payload = {
        "schema_version": 1,
        "start": {
            "master_start_ns": master_start_ns,
            "results": {
                "camera1": {"ack_ns": master_start_ns + 1_000_000},
                "camera3": {"ack_ns": master_start_ns + 2_000_000},
                "gopro": {"ack_ns": gopro_ack_ns},
                "tactile": {"ack_ns": master_start_ns + 3_000_000},
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def camera_timestamps(frames: int, start_ns: int) -> pd.DataFrame:
    offsets_ns = np.rint(np.arange(frames) * 1e9 / CAMERA_FPS).astype(np.int64)
    return pd.DataFrame(
        {
            "frame": np.arange(frames, dtype=np.int64),
            "device_frame_number": np.arange(10_000, 10_000 + frames, dtype=np.int64),
            "perf_counter_ns": start_ns + offsets_ns,
        }
    )


def tactile_timestamps(frames: int, start_ns: int) -> pd.DataFrame:
    offsets_ns = np.rint(np.arange(frames) * 1e9 / TACTILE_FPS).astype(np.int64)
    return pd.DataFrame(
        {
            "frame": np.arange(frames, dtype=np.int64),
            "sensor_frame_id": np.arange(frames, dtype=np.int64) % 128,
            "host_receive_monotonic_ns": start_ns + offsets_ns,
        }
    )


def build_episode(root: Path, name: str, camera_offset_s: float, tactile_offset_s: float) -> dict:
    episode = root / name
    for relative in ("rgb", "tactile", "sync"):
        (episode / relative).mkdir(parents=True, exist_ok=True)

    master_start_ns = 1_700_000_000_000_000_000
    camera_frames = int(round(DURATION_S * CAMERA_FPS))
    tactile_frames = int(round(DURATION_S * TACTILE_FPS))

    for camera, extra_offset in (("camera1", 0.0), ("camera3", 0.004)):
        start_ns = master_start_ns + int(round((camera_offset_s + extra_offset) * 1e9))
        camera_timestamps(camera_frames, start_ns).to_csv(
            episode / f"rgb/{camera}_timestamps.csv", index=False
        )
    tactile_timestamps(
        tactile_frames, master_start_ns + int(round(tactile_offset_s * 1e9))
    ).to_csv(episode / "tactile/tactile_timestamps.csv", index=False)
    write_trigger_log(
        episode / "sync/trigger_log.json",
        master_start_ns,
        master_start_ns + int(round(camera_offset_s * 1e9)),
    )
    return {
        "episode": name,
        "camera_frames": camera_frames,
        "tactile_frames": tactile_frames,
        "master_start_ns": master_start_ns,
    }


def build_robot_dataset(root: Path) -> list[dict]:
    """Write a tiny LeRobot-style dataset with 8-D state and a 1-DoF gripper."""
    written = []
    rng = np.random.default_rng(7)
    for index in range(ROBOT_EPISODES):
        frames = 180
        phase = np.linspace(0.0, 1.0, frames)
        # Grasp between 35% and 55% of the demonstration, release near the end.
        closure = np.clip(
            np.minimum((phase - 0.35) / 0.20, (0.92 - phase) / 0.12), 0.0, 1.0
        )
        gripper = 0.152 + 0.315 * closure
        arm = np.column_stack(
            [
                0.02 * np.sin(2.0 * np.pi * phase + offset)
                + 0.05 * index * 0.01
                for offset in np.linspace(0.0, 1.4, STATE_DIM - 1)
            ]
        )
        states = np.column_stack([arm, gripper])
        states = states + rng.normal(0.0, 2e-4, size=states.shape)

        directory = root / "data/chunk-000"
        directory.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {
                "timestamp": (np.arange(frames) / ROBOT_FPS).astype(np.float64),
                "observation.state": list(states.astype(np.float32)),
                "action": list(states.astype(np.float32)),
            }
        )
        path = directory / f"episode_{index:06d}.parquet"
        frame.to_parquet(path, index=False)
        written.append({"episode": path.stem, "path": str(path)})
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "demo_data",
        help="Destination root; ego/ and robot/ are created below it.",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    ego_root = output / "ego"
    robot_root = output / "robot"
    ego_root.mkdir(parents=True, exist_ok=True)
    robot_root.mkdir(parents=True, exist_ok=True)

    episodes = [
        build_episode(ego_root, name, camera_offset, tactile_offset)
        for name, camera_offset, tactile_offset, _ in EPISODES
    ]
    robot = build_robot_dataset(robot_root)

    manifest = {
        "schema_version": 1,
        "generator": "examples/make_demo_dataset.py",
        "synthetic": True,
        "ego_root": str(ego_root),
        "robot_root": str(robot_root),
        "episodes": [
            {**episode, "detection_fraction": detection}
            for episode, (_, _, _, detection) in zip(episodes, EPISODES, strict=True)
        ],
        "robot_episodes": robot,
    }
    (output / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"DEMO_DATASET_PASS episodes={len(episodes)} robot={len(robot)} root={output}")
    for episode in episodes:
        print(
            f"  {episode['episode']}: camera_frames={episode['camera_frames']} "
            f"tactile_frames={episode['tactile_frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
