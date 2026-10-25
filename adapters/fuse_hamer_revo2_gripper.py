#!/usr/bin/env python3
"""Fuse HaMeR hand closure with a demonstrated Revo2 fruit-task prior.

The direct observation is built from scale-invariant HaMeR 21-joint geometry.
The robot prior supplies executable amplitude anchors and a weak normalized-time
fallback for missed/occluded hands.  This script is offline-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_SOURCE = REPO_ROOT / "data/ego_raw"
DEFAULT_PRIOR = REPO_ROOT / "artifacts/robot_gripper_prior.json"


def robust_unit(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-8:
        raise RuntimeError(f"invalid feature anchors: {low}, {high}")
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def rate_limit(values: np.ndarray, maximum_step: float) -> np.ndarray:
    output = values.copy()
    for index in range(1, len(output)):
        output[index] = np.clip(
            output[index],
            output[index - 1] - maximum_step,
            output[index - 1] + maximum_step,
        )
    for index in range(len(output) - 2, -1, -1):
        output[index] = np.clip(
            output[index],
            output[index + 1] - maximum_step,
            output[index + 1] + maximum_step,
        )
    return output


def smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    filtered = (
        pd.Series(values)
        .interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1)
        .median()
        .to_numpy(float)
    )
    width = min(window, len(filtered) if len(filtered) % 2 else len(filtered) - 1)
    if width >= 5:
        filtered = savgol_filter(filtered, width, 2, mode="interp")
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--episodes", default="")
    parser.add_argument("--min-detected-fraction", type=float, default=0.60)
    parser.add_argument("--hamer-weight", type=float, default=0.80)
    parser.add_argument("--interpolated-hamer-weight", type=float, default=0.35)
    parser.add_argument("--max-step", type=float, default=0.018)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.source_root.expanduser().resolve()
    prior_path = args.prior.expanduser().resolve()
    prior = json.loads(prior_path.read_text())
    profile_path = Path(prior["normalized_phase_profile"]["csv"])
    if not profile_path.is_file():
        portable_profile = prior_path.parent / profile_path.name
        if not portable_profile.is_file():
            raise FileNotFoundError(
                "robot gripper phase profile is unavailable at either "
                f"{profile_path} or {portable_profile}"
            )
        profile_path = portable_profile
    profile = pd.read_csv(profile_path)
    requested = {item.strip() for item in args.episodes.split(",") if item.strip()}

    candidates = []
    for episode in sorted(path for path in root.glob("episode_*") if path.is_dir()):
        if requested and episode.name not in requested:
            continue
        feature_path = episode / "sync/humanego_min/hamer_hand_features_15hz.csv"
        feature_cam3_path = (
            episode / "sync/humanego_min_cam3/hamer_hand_features_15hz.csv"
        )
        timeline_path = episode / "sync/unified_no_rokoko_15hz_timeline.csv"
        if feature_path.is_file() and timeline_path.is_file():
            feature = pd.read_csv(feature_path)
            feature_cam3 = (
                pd.read_csv(feature_cam3_path) if feature_cam3_path.is_file() else None
            )
            timeline = pd.read_csv(timeline_path)
            if len(feature) != len(timeline):
                print(
                    f"GRIPPER_FUSION_SKIP {episode.name}: feature/timeline "
                    f"length={len(feature)}/{len(timeline)}",
                    flush=True,
                )
                continue
            if feature_cam3 is not None and len(feature_cam3) != len(timeline):
                print(
                    f"GRIPPER_FUSION_SKIP {episode.name}: camera3/timeline "
                    f"length={len(feature_cam3)}/{len(timeline)}",
                    flush=True,
                )
                continue
            candidates.append((episode, feature, feature_cam3))
    if not candidates:
        raise RuntimeError("no complete HaMeR feature/timeline pairs found")

    feature_columns = (
        "pinch_aperture_norm",
        "finger_reach_norm",
        "finger_bend_mean_rad",
    )
    anchors = {}
    for camera_name, feature_index in (("camera1", 1), ("camera3", 2)):
        pooled = {}
        for column in feature_columns:
            arrays = []
            for candidate in candidates:
                feature = candidate[feature_index]
                if feature is None:
                    continue
                mask = feature["hamer_detected"].to_numpy(bool)
                values = feature[column].to_numpy(float)
                arrays.append(values[mask & np.isfinite(values)])
            if arrays:
                pooled[column] = np.concatenate(arrays)
        if pooled:
            anchors[camera_name] = {
                column: {
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                    "count": int(len(values)),
                }
                for column, values in pooled.items()
            }

    robot = prior["fusion_anchors"]
    open_value = float(robot["open_typical"])
    closed_value = float(robot["closed_typical"])
    hard_min = float(robot["hard_min"])
    hard_max = float(robot["hard_max"])
    prior_phase = profile["normalized_phase"].to_numpy(float)
    prior_median = profile["p50"].to_numpy(float)

    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(root),
        "robot_prior": str(prior_path),
        "robot_episode_range": prior["source_episode_range_inclusive"],
        "robot_anchors": robot,
        "hamer_global_feature_anchors": anchors,
        "fusion": {
            "feature_weights": {
                "pinch_aperture_norm": 0.45,
                "finger_reach_norm": 0.25,
                "finger_bend_mean_rad": 0.30,
            },
            "camera_weights_when_both_valid": {"camera1": 0.60, "camera3": 0.40},
            "hamer_weight_detected": args.hamer_weight,
            "hamer_weight_interpolated": args.interpolated_hamer_weight,
            "max_step_per_15hz_frame": args.max_step,
        },
        "episodes": [],
    }
    def camera_closure(feature: pd.DataFrame, camera_name: str):
        detected = feature["hamer_detected"].to_numpy(bool)
        valid = feature["hamer_valid"].to_numpy(bool)
        pinch = feature["pinch_aperture_norm"].to_numpy(float)
        reach = feature["finger_reach_norm"].to_numpy(float)
        bend = feature["finger_bend_mean_rad"].to_numpy(float)
        finite = np.isfinite(pinch) & np.isfinite(reach) & np.isfinite(bend)
        camera_anchors = anchors[camera_name]
        closure_pinch = 1.0 - robust_unit(
            pinch, camera_anchors["pinch_aperture_norm"]["q05"],
            camera_anchors["pinch_aperture_norm"]["q95"]
        )
        closure_reach = 1.0 - robust_unit(
            reach, camera_anchors["finger_reach_norm"]["q05"],
            camera_anchors["finger_reach_norm"]["q95"]
        )
        closure_bend = robust_unit(
            bend, camera_anchors["finger_bend_mean_rad"]["q05"],
            camera_anchors["finger_bend_mean_rad"]["q95"]
        )
        closure = np.clip(
            smooth(0.45 * closure_pinch + 0.25 * closure_reach + 0.30 * closure_bend),
            0.0,
            1.0,
        )
        reliable = detected & valid & finite
        return closure, reliable, detected, valid, (pinch, reach, bend)

    for episode, feature, feature_cam3 in candidates:
        output_path = episode / "sync/gripper_1dof_15hz.csv"
        diagnostic_path = episode / "sync/gripper_hamer_robot_prior_15hz.csv"
        metadata_path = episode / "sync/gripper_hamer_robot_prior_metadata.json"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output_path}")

        closure1, reliable1, detected1, valid1, geometry1 = camera_closure(
            feature, "camera1"
        )
        if feature_cam3 is not None and "camera3" in anchors:
            closure3, reliable3, detected3, valid3, _ = camera_closure(
                feature_cam3, "camera3"
            )
        else:
            closure3 = np.full(len(feature), np.nan)
            reliable3 = detected3 = valid3 = np.zeros(len(feature), dtype=bool)
        direct_weight1 = reliable1.astype(float) * 0.60
        direct_weight3 = reliable3.astype(float) * 0.40
        direct_weight_sum = direct_weight1 + direct_weight3
        closure = np.divide(
            direct_weight1 * closure1 + direct_weight3 * np.nan_to_num(closure3),
            direct_weight_sum,
            out=closure1.copy(),
            where=direct_weight_sum > 0,
        )
        direct_reliable = reliable1 | reliable3
        detected_fraction = float(np.mean(direct_reliable))
        status = "PASS" if detected_fraction >= args.min_detected_fraction else "FAIL"
        hamer_gripper = open_value + closure * (closed_value - open_value)

        phase = np.linspace(0.0, 1.0, len(feature))
        phase_prior = np.interp(phase, prior_phase, prior_median)
        hamer_weight = np.where(
            direct_reliable,
            args.hamer_weight,
            np.where(np.isfinite(closure), args.interpolated_hamer_weight, 0.0),
        )
        fused = hamer_weight * hamer_gripper + (1.0 - hamer_weight) * phase_prior
        fused = np.clip(smooth(fused), hard_min, hard_max)
        fused = rate_limit(fused, float(args.max_step))

        output = pd.DataFrame(
            {
                "output_frame_index": feature["output_frame_index"].to_numpy(np.int64),
                "timestamp_s": feature["timestamp_s"].to_numpy(float),
                "gripper_position": fused,
            }
        )
        diagnostic = output.assign(
            hamer_closure_score=closure,
            hamer_gripper_mapped=hamer_gripper,
            robot_phase_prior=phase_prior,
            hamer_weight=hamer_weight,
            hamer_detected=direct_reliable.astype(np.uint8),
            camera1_detected=detected1.astype(np.uint8),
            camera1_valid=valid1.astype(np.uint8),
            camera1_closure_score=closure1,
            camera3_detected=detected3.astype(np.uint8),
            camera3_valid=valid3.astype(np.uint8),
            camera3_closure_score=closure3,
            pinch_aperture_norm=geometry1[0],
            finger_reach_norm=geometry1[1],
            finger_bend_mean_rad=geometry1[2],
        )
        output.to_csv(output_path, index=False, float_format="%.9f")
        diagnostic.to_csv(diagnostic_path, index=False, float_format="%.9f")
        episode_metadata = {
            "schema_version": 1,
            "episode": episode.name,
            "status": status,
            "frame_count": len(feature),
            "detected_fraction": detected_fraction,
            "mapped_min_max": [float(np.min(fused)), float(np.max(fused))],
            "sources": {
                "hamer_features_camera1": "sync/humanego_min/hamer_hand_features_15hz.csv",
                "hamer_features_camera3": "sync/humanego_min_cam3/hamer_hand_features_15hz.csv",
                "robot_prior": str(prior_path),
            },
            "outputs": {
                "training_gripper": str(output_path.relative_to(episode)),
                "diagnostics": str(diagnostic_path.relative_to(episode)),
            },
        }
        metadata_path.write_text(json.dumps(episode_metadata, indent=2) + "\n")
        summary["episodes"].append(episode_metadata)
        print(
            f"GRIPPER_FUSION_{status} {episode.name} frames={len(feature)} "
            f"detected={detected_fraction:.1%} range={fused.min():.3f}..{fused.max():.3f}",
            flush=True,
        )

    summary_path = root / "gripper_hamer_robot_prior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    passed = sum(item["status"] == "PASS" for item in summary["episodes"])
    print(
        f"GRIPPER_FUSION_COMPLETE passed={passed} "
        f"failed={len(summary['episodes']) - passed} total={len(summary['episodes'])}"
    )
    return 0 if passed == len(summary["episodes"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
