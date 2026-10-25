#!/usr/bin/env python3
"""Simulate the hand-pose and stereo stages for the synthetic demo.

In a real run this output comes from HaMeR (or MediaPipe/MANO) on two camera
views, followed by stereo triangulation. Here it is generated from a parametric
reach-and-grasp trajectory so the downstream fusion and filtering stages have
something plausible to consume.

The values are **not** a model of hand-pose accuracy. What is realistic is the
*shape* of the data the contract requires: one row per timeline row, a detection
and a validity flag, scale-invariant closure features, and a metric wrist track
with explicit gaps.

Usage::

    python examples/simulate_hand_pose.py --ego-root demo_data/ego
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_FILE = "sync/humanego_min/hamer_hand_features_15hz.csv"
FEATURE_FILE_CAM3 = "sync/humanego_min_cam3/hamer_hand_features_15hz.csv"
STEREO_FILE = "sync/humanego_min/T_hand_cam_stereo_open_pipeline_15hz.csv"
TIMELINE_FILE = "sync/unified_no_rokoko_15hz_timeline.csv"

HOME = np.array([0.34, -0.12, 0.28])
TARGET = np.array([0.46, 0.05, 0.19])
LIFT = np.array([0.0, 0.0, 0.09])


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def closure_profile(phase: np.ndarray) -> np.ndarray:
    """Grasp between 38% and 58% of the episode, release between 80% and 90%."""
    closing = smoothstep((phase - 0.38) / 0.20)
    opening = smoothstep((phase - 0.80) / 0.10)
    return np.clip(closing - opening, 0.0, 1.0)


def wrist_track(phase: np.ndarray) -> np.ndarray:
    reach = smoothstep(phase / 0.38)
    lift = smoothstep((phase - 0.55) / 0.20) - smoothstep((phase - 0.86) / 0.08)
    return (
        HOME[None, :]
        + (TARGET - HOME)[None, :] * reach[:, None]
        + LIFT[None, :] * lift[:, None]
    )


def features_for(closure: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Closure decreases the pinch aperture and reach, and increases finger bend."""
    noise = lambda scale: rng.normal(0.0, scale, size=closure.shape)  # noqa: E731
    return pd.DataFrame(
        {
            "pinch_aperture_norm": 0.061 - 0.049 * closure + noise(0.0015),
            "finger_reach_norm": 0.092 - 0.071 * closure + noise(0.0020),
            "finger_bend_mean_rad": 0.14 + 1.05 * closure + noise(0.030),
        }
    )


def process_episode(episode: Path, detection_fraction: float, seed: int) -> dict:
    timeline = pd.read_csv(episode / TIMELINE_FILE)
    rows = len(timeline)
    phase = np.linspace(0.0, 1.0, rows)
    rng = np.random.default_rng(seed)

    closure = closure_profile(phase)
    wrist = wrist_track(phase)

    detected = rng.random(rows) < detection_fraction
    # Detection failures cluster rather than scattering, as they do in practice
    # when the hand leaves the frame or is occluded by the object.
    for start in rng.integers(0, max(1, rows - 8), size=max(1, rows // 90)):
        if rng.random() < 0.5:
            detected[start : start + rng.integers(3, 9)] = False

    frame_index = timeline["output_frame_index"].to_numpy(np.int64)
    timestamp_s = timeline["timestamp_s"].to_numpy(float)

    for relative, camera_shift in ((FEATURE_FILE, 0.0), (FEATURE_FILE_CAM3, 0.0015)):
        features = features_for(closure + camera_shift * rng.normal(size=rows), rng)
        features["hamer_detected"] = detected.astype(np.uint8)
        # A frame can be detected yet fail the joint-quality check.
        features["hamer_valid"] = (
            detected & (rng.random(rows) > 0.02)
        ).astype(np.uint8)
        features["wrist_x_m"] = wrist[:, 0]
        features["wrist_y_m"] = wrist[:, 1]
        features["wrist_z_m"] = wrist[:, 2]
        features["output_frame_index"] = frame_index
        features["timestamp_s"] = timestamp_s
        features.loc[~detected, ["pinch_aperture_norm", "finger_reach_norm",
                                 "finger_bend_mean_rad"]] = np.nan
        path = episode / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        features[
            [
                "output_frame_index",
                "timestamp_s",
                "hamer_detected",
                "hamer_valid",
                "pinch_aperture_norm",
                "finger_reach_norm",
                "finger_bend_mean_rad",
                "wrist_x_m",
                "wrist_y_m",
                "wrist_z_m",
            ]
        ].to_csv(path, index=False, float_format="%.9f")

    stereo = pd.DataFrame(
        {
            "output_frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "position_x": wrist[:, 0] + rng.normal(0.0, 3e-4, rows),
            "position_y": wrist[:, 1] + rng.normal(0.0, 3e-4, rows),
            "position_z": wrist[:, 2] + rng.normal(0.0, 3e-4, rows),
            "valid": detected.astype(np.uint8),
        }
    )
    stereo_path = episode / STEREO_FILE
    stereo_path.parent.mkdir(parents=True, exist_ok=True)
    stereo.to_csv(stereo_path, index=False, float_format="%.9f")

    return {
        "episode": episode.name,
        "rows": rows,
        "detection_fraction": float(detected.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-root", type=Path, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="demo_manifest.json from make_demo_dataset.py; defaults to <ego-root>/../demo_manifest.json",
    )
    args = parser.parse_args()

    ego_root = args.ego_root.expanduser().resolve()
    spec_path = (
        args.spec.expanduser().resolve()
        if args.spec is not None
        else ego_root.parent / "demo_manifest.json"
    )
    if not spec_path.is_file():
        raise SystemExit(
            f"missing {spec_path}; run examples/make_demo_dataset.py first "
            "or pass --spec"
        )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    fractions = {
        item["episode"]: float(item.get("detection_fraction", 0.9))
        for item in spec.get("episodes", [])
    }

    results = []
    for index, episode in enumerate(sorted(p for p in ego_root.glob("episode_*") if p.is_dir())):
        if not (episode / TIMELINE_FILE).is_file():
            print(f"SIMULATE_SKIP {episode.name}: no timeline yet")
            continue
        result = process_episode(
            episode, fractions.get(episode.name, 0.9), seed=1000 + index
        )
        results.append(result)
        print(
            f"SIMULATE_HAND_POSE_PASS {result['episode']} rows={result['rows']} "
            f"detected={result['detection_fraction']:.1%}"
        )

    summary_path = ego_root / "simulated_hand_pose_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "synthetic": True,
                "source": "examples/simulate_hand_pose.py",
                "episodes": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SIMULATE_HAND_POSE_COMPLETE episodes={len(results)} summary={summary_path}")
    return 0 if results else 3


if __name__ == "__main__":
    raise SystemExit(main())
