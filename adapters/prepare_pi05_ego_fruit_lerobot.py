#!/usr/bin/env python3
"""Pack quality-passed tas5 Ego fruit IK trajectories as Pi0.5 LeRobot v3."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
EGO_DROPPER_ROOT = Path(
    os.environ.get(
        "EGO_DROPPER_ROOT",
        str(REPO_ROOT / "external_workspace"),
    )
)
HUMANEGO_PIPELINE = Path(
    os.environ.get(
        "HUMANEGO_PIPELINE_ROOT",
        str(EGO_DROPPER_ROOT / "third_party/HumanEgo/ego_dropper"),
    )
)

DEFAULT_TASK = "Pick the object and place it in the target container."
DEFAULT_SOURCE = REPO_ROOT / "data/ego_raw"
DEFAULT_IK = REPO_ROOT / "artifacts/ego_ik"
DEFAULT_REFERENCE = REPO_ROOT / "data/robot_reference_lerobot"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/ego_hamer_ik_lerobot"


@dataclass(frozen=True)
class Backend:
    """Symbols imported from the optional HumanEgo LeRobot toolkit.

    The packer deliberately does not vendor that toolkit: it is a separate
    project with its own license and release cadence. Resolving it lazily keeps
    ``--help`` working on a machine without it, and gives a clear error message
    instead of an ``ImportError`` traceback.
    """

    camera_keys: tuple[str, ...]
    camera_sources: tuple[tuple[str, str, str], ...]
    smooth_actions: Callable[..., Any]
    limit_output_action_steps: Callable[..., Any]
    decode_selected_frames: Callable[..., Any]
    resolve_decode_hwaccel: Callable[..., Any]
    output_features: Callable[..., Any]
    augment_quantile_stats_local: Callable[..., Any]
    quantile_audit: Callable[..., Any]


def load_backend() -> Backend:
    """Import the HumanEgo toolkit selected by HUMANEGO_PIPELINE_ROOT."""
    if str(HUMANEGO_PIPELINE) not in sys.path:
        sys.path.insert(0, str(HUMANEGO_PIPELINE))
    try:
        from prepare_act_dataset_3cam_tactile import (  # noqa: PLC0415
            CAMERA_KEYS,
            limit_output_action_steps,
            smooth_actions,
        )
        from prepare_ego_dropper_lerobot_v3 import (  # noqa: PLC0415
            CAMERA_SOURCES,
            decode_selected_frames,
            resolve_decode_hwaccel,
        )
        from prepare_vla_base import (  # noqa: PLC0415
            augment_quantile_stats_local,
            output_features,
            quantile_audit,
        )
    except ImportError as error:
        raise SystemExit(
            "the LeRobot packer needs the HumanEgo toolkit on PYTHONPATH.\n"
            f"  looked in: {HUMANEGO_PIPELINE}\n"
            "  set HUMANEGO_PIPELINE_ROOT to its ego_dropper directory, or use\n"
            "  another packer adapter that implements ADAPTER_CONTRACTS.md.\n"
            f"  original error: {error}"
        ) from error
    return Backend(
        camera_keys=tuple(CAMERA_KEYS),
        camera_sources=tuple(CAMERA_SOURCES),
        smooth_actions=smooth_actions,
        limit_output_action_steps=limit_output_action_steps,
        decode_selected_frames=decode_selected_frames,
        resolve_decode_hwaccel=resolve_decode_hwaccel,
        output_features=output_features,
        augment_quantile_stats_local=augment_quantile_stats_local,
        quantile_audit=quantile_audit,
    )



def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ready_pairs(
    source_root: Path,
    ik_root: Path,
    *,
    require_gripper_fusion_pass: bool = False,
) -> list[tuple[Path, Path]]:
    summary = read_json(ik_root / "batch_summary.json")
    passed = set()
    for entry in summary["episodes"]:
        name = entry["episode"]
        report_path = ik_root / name / "offline_ik_report.json"
        if entry["status"] not in {"PASS", "SKIP_READY"} or not report_path.is_file():
            continue
        report = read_json(report_path)
        if report.get("status") != "PASS":
            continue
        if require_gripper_fusion_pass:
            fusion_path = source_root / name / "sync/gripper_hamer_robot_prior_metadata.json"
            if not fusion_path.is_file() or read_json(fusion_path).get("status") != "PASS":
                continue
        passed.add(name)
    return [(source_root / name, ik_root / name) for name in sorted(passed)]


def decode_selected_frames_ffmpeg(
    path: Path,
    wanted_indices: list[int],
    *,
    output_height: int,
    output_width: int,
) -> list[np.ndarray]:
    """Fast batch decode for the high-frame-rate HEVC GoPro stream."""
    wanted = [int(value) for value in wanted_indices]
    unique = sorted(set(wanted))
    select_expression = "+".join(f"eq(n\\,{index})" for index in unique)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-i",
        str(path),
        "-vf",
        f"select='{select_expression}',scale={output_width}:{output_height}",
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
    frame_bytes = output_height * output_width * 3
    if len(result.stdout) % frame_bytes:
        raise RuntimeError(f"ffmpeg returned partial RGB frame for {path}")
    decoded = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        -1, output_height, output_width, 3
    )
    if len(decoded) != len(unique):
        raise RuntimeError(
            f"ffmpeg decoded {len(decoded)}/{len(unique)} selected frames from {path}"
        )
    by_index = {index: np.ascontiguousarray(decoded[i]) for i, index in enumerate(unique)}
    return [by_index[index] for index in wanted]


def process_episode(
    episode_dir: Path,
    ik_dir: Path,
    *,
    backend: Backend,
    timeline_relative_path: str,
    height: int,
    width: int,
    smooth_window: int,
    lead_frames: int,
    max_arm_step_deg: float,
    max_gripper_step: float,
    decode_workers: int,
    decode_hwaccel: str | None,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]], float, dict]:
    report = read_json(ik_dir / "offline_ik_report.json")
    planned = pd.read_csv(ik_dir / "planned_joints_source_15hz.csv")
    timeline = pd.read_csv(episode_dir / timeline_relative_path)
    frame_indices = planned["output_frame_index"].to_numpy(np.int64)
    if len(frame_indices) < 2 or frame_indices.min() < 0 or frame_indices.max() >= len(timeline):
        raise RuntimeError(f"{episode_dir.name}: planned frames outside timeline")

    states = np.column_stack(
        [
            planned[[f"joint{i}" for i in range(1, 8)]].to_numpy(np.float32),
            planned["gripper_1dof"].to_numpy(np.float32),
        ]
    )
    smoothed = backend.smooth_actions(states.copy(), smooth_window)
    actions = np.vstack(
        [smoothed[min(i + lead_frames, len(smoothed) - 1)] for i in range(len(smoothed))]
    )
    actions = backend.limit_output_action_steps(actions, max_arm_step_deg, max_gripper_step)

    jobs: list[tuple[Path, list[int]]] = []
    for _, relative_video, timeline_column in backend.camera_sources:
        video = episode_dir / relative_video
        indices = [int(timeline.iloc[int(i)][timeline_column]) for i in frame_indices]
        jobs.append((video, indices))
    decode_pyav = partial(
        backend.decode_selected_frames,
        output_height=height,
        output_width=width,
        hwaccel=decode_hwaccel,
    )

    def decode_job(item: tuple[Path, list[int]]) -> list[np.ndarray]:
        video, indices = item
        if video.name == "camera2_gopro.mp4":
            return decode_selected_frames_ffmpeg(
                video,
                indices,
                output_height=height,
                output_width=width,
            )
        return decode_pyav(video, indices)

    with ThreadPoolExecutor(max_workers=min(3, max(1, decode_workers))) as pool:
        cameras = list(pool.map(decode_job, jobs))

    duration = float(planned["timestamp_s"].iloc[-1] - planned["timestamp_s"].iloc[0])
    metadata = {
        "source_frames": report["path"]["source_frames"],
        "stereo_valid_fraction": report["path"]["valid_fraction_after_edge_trim"],
        "interpolated_frames": report["path"]["interpolated_frames"],
        "largest_internal_gap_frames": report["path"]["largest_internal_gap_frames"],
        "max_joint_step_deg_before_action_limit": report["ik_source_15hz"][
            "max_joint_waypoint_step_deg"
        ],
    }
    return states, actions, cameras, duration, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--ik-root", type=Path, default=DEFAULT_IK)
    parser.add_argument(
        "--timeline-relative-path",
        default="sync/unified_15hz_timeline.csv",
        help="Episode-relative synchronized 15 Hz timeline CSV",
    )
    parser.add_argument("--reference-dataset", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-repo-id", default="local/tas5_fruit_ego_pi05_v1_3cam_8d_15hz")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--domain", default="ego_hamer_stereo_ik")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--resize-height", type=int, default=224)
    parser.add_argument("--resize-width", type=int, default=224)
    parser.add_argument("--smooth-action-window", type=int, default=5)
    parser.add_argument("--action-lead-frames", type=int, default=3)
    parser.add_argument("--max-output-arm-step-deg", type=float, default=2.5)
    parser.add_argument("--max-output-gripper-step", type=float, default=0.05)
    parser.add_argument("--decode-workers", type=int, default=3)
    parser.add_argument("--decode-backend", choices=("auto", "pyav", "nvdec"), default="auto")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--encoder-threads", type=int, default=8)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-gripper-fusion-pass",
        action="store_true",
        help="Exclude Episodes whose HaMeR/Revo2 fusion metadata is absent or failed.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="finish stats/summary for an already encoded dataset without decoding videos again",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    ik_root = args.ik_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    backend = load_backend()
    pairs = ready_pairs(
        source_root,
        ik_root,
        require_gripper_fusion_pass=args.require_gripper_fusion_pass,
    )
    if args.start_offset < 0:
        raise ValueError("--start-offset cannot be negative")
    pairs = pairs[args.start_offset :]
    if args.max_episodes > 0:
        pairs = pairs[: args.max_episodes]
    if not pairs:
        raise SystemExit("no quality-passed Ego IK episodes")
    if args.finalize_only:
        progress = output_root / "preprocess_summary.in_progress.json"
        if not progress.is_file():
            raise FileNotFoundError(f"missing interrupted-run summary: {progress}")
        summary = read_json(progress)
        summary["total_episodes"] = len(summary["written_episodes"])
        summary["total_frames"] = sum(
            item["output_frames"] for item in summary["written_episodes"]
        )
        if summary["total_episodes"] != len(pairs):
            raise RuntimeError(
                f"dataset is incomplete: encoded {summary['total_episodes']}/{len(pairs)} episodes"
            )
        result = backend.augment_quantile_stats_local(args.output_repo_id, output_root, overwrite=True)
        result["audit"] = backend.quantile_audit(output_root)
        summary["quantiles"] = result
        write_json(output_root / "preprocess_summary.json", summary)
        progress.unlink(missing_ok=True)
        if result["audit"]["status"] != "PASS":
            raise SystemExit("Ego Pi0.5 quantile audit failed")
        print(
            f"EGO_PI05_DATASET_PASS episodes={summary['total_episodes']} "
            f"frames={summary['total_frames']} state=8D action=8D cameras=3x224 tactile=OFF"
        )
        return 0
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_root}")
        shutil.rmtree(output_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    reference_info = read_json(args.reference_dataset / "meta/info.json")
    dataset = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=output_root,
        fps=args.fps,
        robot_type=reference_info.get("robot_type", "nero_revo2"),
        features=backend.output_features(
            reference_info,
            state_mode="robot_only",
            fps=args.fps,
            height=args.resize_height,
            width=args.resize_width,
        ),
        use_videos=True,
        video_backend="pyav",
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        batch_encoding_size=1,
    )

    summary = {
        "schema_version": 1,
        "policy_family": "pi05",
        "normalization_mode": "QUANTILES",
        "task": args.task,
        "domain": args.domain,
        "source_root": str(source_root),
        "ik_root": str(ik_root),
        "output_root": str(output_root),
        "selected_source_episodes": [episode.name for episode, _ in pairs],
        "state_action": "8D absolute Nero joints + normalized Revo2 grasp",
        "tactile": "excluded",
        "written_episodes": [],
        "source_to_output_episode": {},
    }
    progress = output_root / "preprocess_summary.in_progress.json"
    write_json(progress, summary)
    decode_hwaccel = backend.resolve_decode_hwaccel(args.decode_backend)
    for output_episode, (episode, ik_dir) in enumerate(pairs):
        states, actions, cameras, duration, meta = process_episode(
            episode,
            ik_dir,
            backend=backend,
            timeline_relative_path=args.timeline_relative_path,
            height=args.resize_height,
            width=args.resize_width,
            smooth_window=args.smooth_action_window,
            lead_frames=args.action_lead_frames,
            max_arm_step_deg=args.max_output_arm_step_deg,
            max_gripper_step=args.max_output_gripper_step,
            decode_workers=args.decode_workers,
            decode_hwaccel=decode_hwaccel,
        )
        for frame_index in range(len(states)):
            dataset.add_frame(
                {
                    "observation.state": states[frame_index],
                    "action": actions[frame_index],
                    backend.camera_keys[0]: cameras[0][frame_index],
                    backend.camera_keys[1]: cameras[1][frame_index],
                    backend.camera_keys[2]: cameras[2][frame_index],
                    "task": args.task,
                }
            )
        dataset.save_episode(parallel_encoding=True)
        item = {
            "source_episode": episode.name,
            "output_episode": output_episode,
            "output_frames": len(states),
            "duration_s": duration,
            **meta,
        }
        summary["written_episodes"].append(item)
        summary["source_to_output_episode"][episode.name] = output_episode
        write_json(progress, summary)
        print(f"EGO_PACK_PASS {episode.name} -> {output_episode} frames={len(states)}", flush=True)

    # save_episode(parallel_encoding=True) may still be flushing the final
    # parquet/video shard here.  Finalize before reading those shards to build
    # normalization statistics; otherwise a fast reader can observe a partial
    # parquet footer.
    dataset.finalize()
    summary["total_episodes"] = len(summary["written_episodes"])
    summary["total_frames"] = sum(x["output_frames"] for x in summary["written_episodes"])
    result = backend.augment_quantile_stats_local(args.output_repo_id, output_root, overwrite=True)
    result["audit"] = backend.quantile_audit(output_root)
    summary["quantiles"] = result
    write_json(output_root / "preprocess_summary.json", summary)
    progress.unlink(missing_ok=True)
    if result["audit"]["status"] != "PASS":
        raise SystemExit("Ego Pi0.5 quantile audit failed")
    print(
        f"EGO_PI05_DATASET_PASS episodes={summary['total_episodes']} "
        f"frames={summary['total_frames']} state=8D action=8D cameras=3x224 tactile=OFF"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
