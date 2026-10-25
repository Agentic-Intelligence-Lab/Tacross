#!/usr/bin/env python3
"""Build a camera/tactile-only 15 Hz timeline for Ego recordings without Rokoko.

The script is offline-only.  It preserves the native recordings and writes
derived alignment CSV/JSON files under each Episode's ``sync`` directory.
All timestamped sources use causal zero-order hold at a shared 15 Hz clock.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_FPS = 15.0


@dataclass
class Source:
    times_s: np.ndarray
    extras: dict[str, np.ndarray]

    @property
    def start_s(self) -> float:
        return float(self.times_s[0])

    @property
    def end_s(self) -> float:
        return float(self.times_s[-1])


def camera_ns(frame: pd.DataFrame) -> np.ndarray:
    for column in ("perf_counter_ns", "host_receive_monotonic_ns"):
        if column in frame:
            return frame[column].to_numpy(np.int64)
    raise KeyError("camera timestamp CSV has no host monotonic timestamp")


def tactile_ns(frame: pd.DataFrame) -> np.ndarray:
    if "host_receive_monotonic_ns" in frame:
        return frame["host_receive_monotonic_ns"].to_numpy(np.int64)
    if "monotonic_time_s" in frame:
        return np.rint(frame["monotonic_time_s"].to_numpy(float) * 1e9).astype(
            np.int64
        )
    raise KeyError("tactile timestamp CSV has no host monotonic timestamp")


def probe_video(path: Path) -> tuple[float, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or "ffprobe failed"
        raise RuntimeError(f"cannot probe {path}: {detail}")
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    if fps <= 0:
        raise RuntimeError(f"invalid video fps for {path}: {fps}")
    count = int(stream.get("nb_frames") or 0)
    if count <= 0:
        count = int(round(float(payload["format"]["duration"]) * fps))
    if count <= 0:
        raise RuntimeError(f"video contains no frames: {path}")
    return fps, count


def causal_index(times_s: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(times_s, target_s, side="right") - 1
    return np.clip(indices, 0, len(times_s) - 1).astype(np.int64)


def age_ms(target_s: np.ndarray, selected_s: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, (target_s - selected_s) * 1000.0)


def process_episode(
    episode: Path,
    *,
    camera2_fps: float | None = None,
    camera2_frame_count: int | None = None,
    use_camera2: bool = True,
) -> dict:
    trigger = json.loads((episode / "sync/trigger_log.json").read_text())
    master_start_ns = int(trigger["start"]["master_start_ns"])
    start_results = trigger["start"]["results"]

    camera_frames: dict[str, pd.DataFrame] = {
        camera: pd.read_csv(episode / f"rgb/{camera}_timestamps.csv")
        for camera in ("camera1", "camera3")
    }
    tactile_frame = pd.read_csv(episode / "tactile/tactile_timestamps.csv")
    camera_times = {
        camera: (camera_ns(frame).astype(np.float64) - master_start_ns) / 1e9
        for camera, frame in camera_frames.items()
    }
    tactile_times = (
        tactile_ns(tactile_frame).astype(np.float64) - master_start_ns
    ) / 1e9

    if use_camera2:
        if camera2_fps is not None and camera2_frame_count is not None:
            gopro_fps = float(camera2_fps)
            gopro_count = int(camera2_frame_count)
        else:
            gopro_fps, gopro_count = probe_video(episode / "rgb/camera2_gopro.mp4")
        gopro_start_s = (
            int(start_results["gopro"]["ack_ns"]) - master_start_ns
        ) / 1e9
        gopro_times = gopro_start_s + np.arange(gopro_count, dtype=float) / gopro_fps

    sources = {
        "camera1": Source(
            camera_times["camera1"],
            {
                "source_frame": camera_frames["camera1"].get(
                    "frame", pd.Series(np.arange(len(camera_frames["camera1"])))
                ).to_numpy(np.int64),
                "device_frame_number": camera_frames["camera1"][
                    "device_frame_number"
                ].to_numpy(np.int64),
            },
        ),
        "camera3": Source(
            camera_times["camera3"],
            {
                "source_frame": camera_frames["camera3"].get(
                    "frame", pd.Series(np.arange(len(camera_frames["camera3"])))
                ).to_numpy(np.int64),
                "device_frame_number": camera_frames["camera3"][
                    "device_frame_number"
                ].to_numpy(np.int64),
            },
        ),
        "tactile": Source(
            tactile_times,
            {
                "source_frame": tactile_frame["frame"].to_numpy(np.int64),
                "sensor_frame_id": tactile_frame["sensor_frame_id"].to_numpy(
                    np.int64
                ),
            },
        ),
    }
    if use_camera2:
        sources["camera2"] = Source(gopro_times, {})

    if any(len(source.times_s) == 0 for source in sources.values()):
        raise RuntimeError("one or more sources contain zero timestamps")

    shared_start = max(source.start_s for source in sources.values())
    shared_end = min(source.end_s for source in sources.values())
    if shared_end <= shared_start:
        bounds = {
            name: [source.start_s, source.end_s] for name, source in sources.items()
        }
        raise RuntimeError(
            f"non-positive shared window {shared_start:.6f}->{shared_end:.6f}; "
            f"source_bounds={bounds}"
        )

    frame_count = int(round((shared_end - shared_start) * OUTPUT_FPS)) + 1
    timestamp_s = np.arange(frame_count, dtype=float) / OUTPUT_FPS
    master_time_s = shared_start + timestamp_s
    rows: dict[str, np.ndarray] = {
        "output_frame_index": np.arange(frame_count, dtype=np.int64),
        "timestamp_s": timestamp_s,
        "master_time_s": master_time_s,
        "master_time_ns": master_start_ns
        + np.rint(master_time_s * 1e9).astype(np.int64),
    }
    age_summary = {}
    for name, source in sources.items():
        indices = causal_index(source.times_s, master_time_s)
        ages = age_ms(master_time_s, source.times_s[indices])
        rows[f"{name}_frame_index"] = indices
        rows[f"{name}_age_ms"] = ages
        for key, values in source.extras.items():
            rows[f"{name}_{key}"] = values[indices]
        age_summary[name] = {
            "mean": float(np.mean(ages)),
            "p99": float(np.quantile(ages, 0.99)),
            "max": float(np.max(ages)),
        }
    rows["camera2_source_time_s"] = (
        sources["camera2"].times_s[rows["camera2_frame_index"]] - gopro_start_s
    ) if use_camera2 else np.full(frame_count, np.nan)

    timeline = pd.DataFrame(rows)
    sync = episode / "sync"
    timeline_path = sync / "unified_no_rokoko_15hz_timeline.csv"
    metadata_path = sync / "unified_no_rokoko_15hz_metadata.json"
    timeline.to_csv(timeline_path, index=False, float_format="%.9f")
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "episode": episode.name,
        "rokoko_present": False,
        "output_fps": OUTPUT_FPS,
        "selection": "causal zero-order hold at shared camera/tactile clock",
        "shared_start_from_master_s": shared_start,
        "shared_end_from_master_s": shared_end,
        "output_frame_count": frame_count,
        "output_duration_s": float(timestamp_s[-1]),
        "source_bounds_from_master_s": {
            name: [source.start_s, source.end_s] for name, source in sources.items()
        },
        "source_counts": {name: len(source.times_s) for name, source in sources.items()},
        "source_age_ms": age_summary,
        "gopro": (
            {"fps": gopro_fps, "frame_count": gopro_count} if use_camera2 else None
        ),
        "outputs": {"timeline": str(timeline_path.relative_to(episode))},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--episode", action="append")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument(
        "--no-camera2",
        action="store_true",
        help="Do not use the GoPro view at all (recordings without camera2_gopro.mp4).",
    )
    parser.add_argument(
        "--camera2-fps",
        type=float,
        default=None,
        help="Skip ffprobe and use this GoPro frame rate; requires --camera2-frame-count.",
    )
    parser.add_argument(
        "--camera2-frame-count",
        type=int,
        default=None,
        help="Skip ffprobe and use this GoPro frame count; requires --camera2-fps.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional batch-summary destination; defaults to <root>/unified_no_rokoko_15hz_summary.json",
    )
    args = parser.parse_args()
    if (args.camera2_fps is None) != (args.camera2_frame_count is None):
        parser.error("--camera2-fps and --camera2-frame-count must be used together")
    if args.no_camera2 and args.camera2_fps is not None:
        parser.error("--no-camera2 cannot be combined with GoPro overrides")
    root = args.root.expanduser().resolve()
    episodes = (
        [root / name for name in args.episode]
        if args.episode
        else sorted(path for path in root.glob("episode_*") if path.is_dir())
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(root),
        "episodes": [],
    }
    summary_path = (
        args.summary_path.expanduser().resolve()
        if args.summary_path is not None
        else root / "unified_no_rokoko_15hz_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        try:
            metadata = process_episode(
                episode,
                camera2_fps=args.camera2_fps,
                camera2_frame_count=args.camera2_frame_count,
                use_camera2=not args.no_camera2,
            )
            entry = {
                "episode": episode.name,
                "status": "PASS",
                "frames": metadata["output_frame_count"],
                "duration_s": metadata["output_duration_s"],
            }
            print(
                f"NO_ROKOKO_TIMELINE_PASS {episode.name} "
                f"frames={entry['frames']} duration={entry['duration_s']:.3f}s",
                flush=True,
            )
        except Exception as error:
            entry = {"episode": episode.name, "status": "FAIL", "error": repr(error)}
            print(f"NO_ROKOKO_TIMELINE_FAIL {episode.name}: {error}", flush=True)
            if not args.skip_failed:
                summary["episodes"].append(entry)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                raise
        summary["episodes"].append(entry)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    passed = sum(item["status"] == "PASS" for item in summary["episodes"])
    failed = len(summary["episodes"]) - passed
    print(f"NO_ROKOKO_TIMELINE_COMPLETE passed={passed} failed={failed}")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
