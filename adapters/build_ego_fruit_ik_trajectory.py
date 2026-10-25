#!/usr/bin/env python3
"""Build offline Nero IK trajectories for the tas5 fruit Ego recordings.

This script never connects to hardware.  It converts metric cam1/cam3 stereo
wrist displacement into a continuous 7-DoF Nero trajectory from a demonstrated
robot start pose, and maps the Ego hand closure into the robot gripper range.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE_ROOT = Path(
    os.environ.get(
        "EGO_DROPPER_ROOT",
        str(REPO_ROOT / "external_workspace"),
    )
)
HUMANEGO_PIPELINE = Path(
    os.environ.get(
        "HUMANEGO_PIPELINE_ROOT",
        str(WORKSPACE_ROOT / "third_party/HumanEgo/ego_dropper"),
    )
)
DEFAULT_IK_BACKEND = HUMANEGO_PIPELINE / "build_ego_ik_task2_trajectory.py"
DEFAULT_SOURCE = REPO_ROOT / "data/ego_raw"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/ego_ik"
DEFAULT_START = REPO_ROOT / "calibration/demonstrated_robot_start.json"
DEFAULT_CALIBRATION = REPO_ROOT / "calibration/camera_to_robot_base.json"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/episode_manifest.json"
STEREO_NAME = "T_hand_cam_stereo_open_pipeline_15hz.csv"

# Example cross-domain calibration from task-level q01/q99 anchors:
# Ego [0.175, 0.675] -> robot [0.189, 0.457]. Override on the command line for
# a different gripper or task; these values are specific to the Nero/Revo2 run.
GRIPPER_SCALE = 0.536
GRIPPER_OFFSET = 0.0952
GRIPPER_MIN = 0.15
GRIPPER_MAX = 0.47


def load_task2_module(script: Path):
    script = script.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("ego_task2_ik_base", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invalid_gap_lengths(valid: np.ndarray) -> list[int]:
    gaps: list[int] = []
    start: int | None = None
    for index, ok in enumerate(valid.tolist()):
        if not ok and start is None:
            start = index
        elif ok and start is not None:
            gaps.append(index - start)
            start = None
    if start is not None:
        gaps.append(len(valid) - start)
    return gaps


def load_pose_targets(
    stereo_csv: Path,
    calibration: dict,
    start_pose: np.ndarray,
    *,
    min_valid_fraction: float,
    max_gap_frames: int,
    smooth_window: int,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    frame = pd.read_csv(stereo_csv)
    xyz = frame[["position_x", "position_y", "position_z"]].to_numpy(float)
    valid = frame["valid"].to_numpy(bool) & np.isfinite(xyz).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        raise RuntimeError("fewer than two valid stereo wrist frames")

    first, last = int(valid_indices[0]), int(valid_indices[-1])
    selected = frame.iloc[first : last + 1].copy().reset_index(drop=True)
    selected_valid = valid[first : last + 1]
    gaps = invalid_gap_lengths(selected_valid)
    valid_fraction = float(selected_valid.mean())
    max_gap = max(gaps, default=0)
    if valid_fraction < min_valid_fraction:
        raise RuntimeError(
            f"stereo valid fraction {valid_fraction:.3f} < {min_valid_fraction:.3f}"
        )
    if max_gap > max_gap_frames:
        raise RuntimeError(f"largest internal stereo gap {max_gap} > {max_gap_frames}")

    positions_cam1 = (
        selected[["position_x", "position_y", "position_z"]]
        .interpolate(method="linear", limit_direction="both")
        .to_numpy(float)
    )
    window = min(int(smooth_window), len(positions_cam1))
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        positions_cam1 = savgol_filter(
            positions_cam1, window_length=window, polyorder=2, axis=0, mode="interp"
        )

    t_cam1_from_base = np.asarray(calibration["T_cam1_from_base"], dtype=float)
    r_base_from_cam1 = np.linalg.inv(t_cam1_from_base)[:3, :3]
    displacement_base = (positions_cam1 - positions_cam1[0]) @ r_base_from_cam1.T
    positions_base = start_pose[:3] + displacement_base
    poses = np.column_stack(
        [positions_base, np.tile(start_pose[3:6], (len(positions_base), 1))]
    )
    cart_steps = np.linalg.norm(np.diff(positions_base, axis=0), axis=1)
    report = {
        "source_frames": [first, last],
        "source_frame_count": int(len(selected)),
        "source_duration_s": float(
            selected["timestamp_s"].iloc[-1] - selected["timestamp_s"].iloc[0]
        ),
        "raw_valid_frames": int(selected_valid.sum()),
        "valid_fraction_after_edge_trim": valid_fraction,
        "interpolated_frames": int((~selected_valid).sum()),
        "largest_internal_gap_frames": int(max_gap),
        "largest_internal_gap_s": float(max_gap / 15.0),
        "smoothing": f"Savitzky-Golay window={window}, polyorder=2",
        "translation_scale": 1.0,
        "span_base_xyz_mm": (1000.0 * np.ptp(positions_base, axis=0)).tolist(),
        "max_relative_translation_mm": float(
            1000.0 * np.linalg.norm(displacement_base, axis=1).max()
        ),
        "max_cartesian_step_mm": float(1000.0 * cart_steps.max())
        if len(cart_steps)
        else 0.0,
        "orientation_policy": "hold demonstrated locked-start flange orientation",
    }
    return selected, poses, report


def map_gripper(
    values: np.ndarray,
    *,
    scale: float = GRIPPER_SCALE,
    offset: float = GRIPPER_OFFSET,
    minimum: float = GRIPPER_MIN,
    maximum: float = GRIPPER_MAX,
) -> np.ndarray:
    return np.clip(offset + scale * values, minimum, maximum)


def process_episode(
    episode_dir: Path,
    output_dir: Path,
    task2,
    calibration: dict,
    start: dict,
    args: argparse.Namespace,
) -> dict:
    stereo_csv = episode_dir / "sync/humanego_min" / args.stereo_name
    gripper_csv = episode_dir / args.gripper_relative_path
    if not stereo_csv.is_file() or not gripper_csv.is_file():
        raise FileNotFoundError(f"missing stereo or gripper source for {episode_dir.name}")

    start_pose = np.asarray(start["start_flange_pose"], dtype=float)
    selected, poses, path_report = load_pose_targets(
        stereo_csv,
        calibration,
        start_pose,
        min_valid_fraction=float(args.min_valid_fraction),
        max_gap_frames=int(args.max_gap_frames),
        smooth_window=int(args.smooth_window),
    )
    first, last = path_report["source_frames"]
    gripper_frame = pd.read_csv(gripper_csv)
    raw_gripper = gripper_frame.iloc[first : last + 1]["gripper_position"].to_numpy(
        float
    )
    if len(raw_gripper) != len(poses) or not np.isfinite(raw_gripper).all():
        raise RuntimeError("gripper/timeline length mismatch or non-finite values")
    gripper = raw_gripper.copy() if args.gripper_is_robot else map_gripper(
        raw_gripper,
        scale=float(args.gripper_scale),
        offset=float(args.gripper_offset),
        minimum=float(args.gripper_min),
        maximum=float(args.gripper_max),
    )

    start_joints = np.radians(np.asarray(start["start_joints_deg"], dtype=float))
    model = task2.load_urdf(task2.DEFAULT_URDF)
    lower, upper = task2.arm_limits(model)
    joints, ik_report = task2.solve_joint_waypoints(poses, start_joints, lower, upper)

    errors: list[str] = []
    if ik_report["first_target_joint_delta_deg"] > 0.5:
        errors.append("first target is not locked to demonstrated start")
    if ik_report["max_joint_waypoint_step_deg"] > float(args.max_joint_step_deg):
        errors.append("joint continuity gate exceeded")
    if ik_report["max_fk_translation_residual_mm"] > 8.0:
        errors.append("FK translation residual exceeds 8 mm")
    if ik_report["max_fk_rotation_residual_deg"] > float(
        args.max_fk_rotation_residual_deg
    ):
        errors.append(
            f"FK rotation residual exceeds {args.max_fk_rotation_residual_deg:g} degrees"
        )
    if ik_report["max_joint_excursion_deg"] > float(args.max_joint_excursion_deg):
        errors.append(f"joint excursion exceeds {args.max_joint_excursion_deg:g} degrees")
    if ik_report["minimum_joint_limit_margin_deg"] < float(
        args.min_joint_limit_margin_deg
    ):
        errors.append(
            f"joint-limit margin below {args.min_joint_limit_margin_deg:g} degrees"
        )
    if errors:
        raise RuntimeError("; ".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "planned_joints_source_15hz.csv"
    report_json = output_dir / "offline_ik_report.json"
    pd.DataFrame(
        {
            "output_frame_index": selected["output_frame_index"].to_numpy(np.int64),
            "timestamp_s": selected["timestamp_s"].to_numpy(float),
            **{f"joint{i + 1}": joints[:, i] for i in range(7)},
            "gripper_1dof": gripper,
            "gripper_1dof_raw": raw_gripper,
        }
    ).to_csv(output_csv, index=False, float_format="%.9f")

    path_report["gripper_mapping"] = (
        {
            "formula": "identity; source already fused in demonstrated Revo2 feedback scale",
            "source": str(gripper_csv),
            "raw_min_max": [float(raw_gripper.min()), float(raw_gripper.max())],
            "mapped_min_max": [float(gripper.min()), float(gripper.max())],
        }
        if args.gripper_is_robot
        else {
            "formula": (
                f"clip({args.gripper_offset:g} + {args.gripper_scale:g} * ego_gripper, "
                f"{args.gripper_min:g}, {args.gripper_max:g})"
            ),
            "source_q01_q99": [0.175, 0.675],
            "target_robot_q01_q99": [0.189, 0.457],
            "raw_min_max": [float(raw_gripper.min()), float(raw_gripper.max())],
            "mapped_min_max": [float(gripper.min()), float(gripper.max())],
        }
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "robot_execution": "FORBIDDEN_IN_THIS_SCRIPT",
        "episode": episode_dir.name,
        "task": start["task"],
        "source_stereo": str(stereo_csv),
        "source_gripper": str(gripper_csv),
        "start_contract": str(args.start_contract.resolve()),
        "locked_start_joints_deg": start["start_joints_deg"],
        "locked_start_flange_pose": start["start_flange_pose"],
        "path": path_report,
        "ik_source_15hz": ik_report,
        "outputs": {
            "planned_joints_source_15hz_csv": str(output_csv),
            "report_json": str(report_json),
        },
        "status": "PASS",
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--start-contract", type=Path, default=DEFAULT_START)
    parser.add_argument(
        "--start-current-episode",
        type=int,
        default=-1,
        help="Diagnostic override: derive the locked start from frame 0 of a current compacted robot episode.",
    )
    parser.add_argument(
        "--robot-raw-root",
        type=Path,
        default=REPO_ROOT / "recordings/nero_revo2_3cam_tactile_raw_30hz",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--ik-backend-script",
        type=Path,
        default=DEFAULT_IK_BACKEND,
        help="Robot-specific URDF/IK backend implementing the adapter calls.",
    )
    parser.add_argument("--stereo-name", default=STEREO_NAME)
    parser.add_argument(
        "--gripper-relative-path",
        type=Path,
        default=Path("sync/gripper_1dof_15hz.csv"),
    )
    parser.add_argument(
        "--gripper-is-robot",
        action="store_true",
        help="Use gripper_position directly because it is already in demonstrated Revo2 scale.",
    )
    parser.add_argument(
        "--gripper-scale",
        type=float,
        default=GRIPPER_SCALE,
        help="Scale of the Ego-to-robot 1-DoF gripper affine map.",
    )
    parser.add_argument(
        "--gripper-offset",
        type=float,
        default=GRIPPER_OFFSET,
        help="Offset of the Ego-to-robot 1-DoF gripper affine map.",
    )
    parser.add_argument(
        "--gripper-min",
        type=float,
        default=GRIPPER_MIN,
        help="Lower clip bound for the mapped robot gripper command.",
    )
    parser.add_argument(
        "--gripper-max",
        type=float,
        default=GRIPPER_MAX,
        help="Upper clip bound for the mapped robot gripper command.",
    )
    parser.add_argument("--episodes", default="", help="Comma-separated episode names")
    parser.add_argument("--min-valid-fraction", type=float, default=0.70)
    parser.add_argument("--max-gap-frames", type=int, default=50)
    parser.add_argument("--smooth-window", type=int, default=21)
    parser.add_argument("--max-joint-step-deg", type=float, default=5.5)
    parser.add_argument("--max-fk-rotation-residual-deg", type=float, default=5.0)
    parser.add_argument("--max-joint-excursion-deg", type=float, default=90.0)
    parser.add_argument("--min-joint-limit-margin-deg", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    args = parser.parse_args()
    if args.gripper_max <= args.gripper_min:
        parser.error("--gripper-max must be greater than --gripper-min")

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    start = json.loads(args.start_contract.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    requested = [x.strip() for x in args.episodes.split(",") if x.strip()]
    candidates = requested or list(manifest["valid_episodes"])
    task2 = load_task2_module(args.ik_backend_script)
    if args.start_current_episode >= 0:
        parquet = (
            args.robot_raw_root.expanduser().resolve()
            / "data/chunk-000"
            / f"episode_{args.start_current_episode:06d}.parquet"
        )
        frame0 = np.asarray(
            pd.read_parquet(parquet, columns=["observation.state"]).iloc[0][
                "observation.state"
            ],
            dtype=float,
        )
        start = {
            **start,
            "selection": f"diagnostic frame-0 override from current robot Episode {args.start_current_episode}",
            "start_joints_deg": np.degrees(frame0[:7]).tolist(),
            "start_flange_pose": task2.fk_from_mdh(
                list(task2.get_mdh("nero")), frame0[:7].tolist()
            ),
            "start_gripper": float(frame0[7]),
        }

    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "start_contract": str(args.start_contract.resolve()),
        "quality_gates": {
            "min_valid_fraction": args.min_valid_fraction,
            "max_gap_frames": args.max_gap_frames,
            "max_joint_step_deg": args.max_joint_step_deg,
        },
        "gripper_mapping": {
            "mode": "identity" if args.gripper_is_robot else "ego_affine",
            "scale": None if args.gripper_is_robot else args.gripper_scale,
            "offset": None if args.gripper_is_robot else args.gripper_offset,
            "clip": None if args.gripper_is_robot else [args.gripper_min, args.gripper_max],
        },
        "episodes": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.json"
    for name in candidates:
        destination = output_root / name
        report_path = destination / "offline_ik_report.json"
        prior_report = None
        if report_path.is_file() and not args.overwrite:
            try:
                prior_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior_report = None
        if prior_report is not None and prior_report.get("status") == "PASS":
            entry = {
                "episode": name,
                "status": "SKIP_READY",
                "frames": prior_report.get("path", {}).get("source_frame_count"),
                "output": str(destination),
            }
            print(f"IK_SKIP_READY {name}", flush=True)
        else:
            try:
                report = process_episode(
                    source_root / name,
                    destination,
                    task2,
                    calibration,
                    start,
                    args,
                )
                entry = {
                    "episode": name,
                    "status": "PASS",
                    "frames": report["path"]["source_frame_count"],
                    "valid_fraction": report["path"]["valid_fraction_after_edge_trim"],
                    "max_gap_frames": report["path"]["largest_internal_gap_frames"],
                    "max_joint_step_deg": report["ik_source_15hz"][
                        "max_joint_waypoint_step_deg"
                    ],
                    "output": str(destination),
                }
                print(
                    f"IK_PASS {name} frames={entry['frames']} "
                    f"valid={entry['valid_fraction']:.1%} "
                    f"max_step={entry['max_joint_step_deg']:.2f}deg",
                    flush=True,
                )
            except Exception as error:
                entry = {"episode": name, "status": "FAIL", "error": repr(error)}
                print(f"IK_FAIL {name}: {error!r}", flush=True)
                if not args.skip_failed:
                    summary["episodes"].append(entry)
                    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                    raise
        summary["episodes"].append(entry)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    passed = sum(e["status"] in {"PASS", "SKIP_READY"} for e in summary["episodes"])
    failed = sum(e["status"] == "FAIL" for e in summary["episodes"])
    print(f"EGO_FRUIT_IK_COMPLETE passed={passed} failed={failed} total={len(candidates)}")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
