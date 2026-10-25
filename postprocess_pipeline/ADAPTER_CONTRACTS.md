# Adapter contracts

The orchestration core is robot-, task-, and dataset-independent. Algorithms
that depend on a particular hand model, robot, or storage format are external
commands selected in the JSON configuration.

## Episode layout adapter

An episode discovery/validation command must write a JSON manifest containing:

```json
{
  "schema_version": 1,
  "source_root": "/absolute/path",
  "valid_episodes": ["episode_000001"],
  "excluded_episodes": [
    {"episode": "episode_000002", "reasons": ["missing:rgb/camera1.avi"]}
  ]
}
```

Source Episode identifiers must remain stable. The pipeline represents
selection through manifests and must not delete or compact source Episodes.

## View undistortion adapter

An undistortion adapter removes lens distortion from one camera view. It is the
only adapter allowed to rewrite pixels, and it must write them to a **new** path:
the source video is opened read-only and must remain byte-identical.

It consumes a calibration file describing one camera:

```json
{
  "cameras": {
    "camera1": {
      "model": "fisheye",
      "image_size": [1920, 1080],
      "K": [[700.0, 0.0, 960.0], [0.0, 700.0, 540.0], [0.0, 0.0, 1.0]],
      "D": [0.12, -0.03, 0.008, -0.001]
    }
  }
}
```

`model` is `pinhole` (also `radtan`, `plumb_bob`, `rational_polynomial`, `brown`)
or `fisheye` (also `equidistant`, `kb`, `kannala_brandt`). `D` is `k1, k2, p1,
p2[, k3[, k4, k5, k6]]` for pinhole and `k1..k4` for fisheye. `K` and `D` may
also be spelled `camera_matrix` / `dist_coeffs`; the image size may be
`image_size: [w, h]` or `image_width` + `image_height`. A flat single-camera file
with no `cameras` wrapper is accepted and answers to any requested camera name; a
file that names its cameras must be asked for the right one.

It writes, per Episode, a report containing:

- `status: PASS|FAIL` and the source/output geometry, frame count and fps;
- the resolved `model`, `alpha`, `K`, `D` and the resulting `new_matrix`;
- the intrinsics rescale factor when the video resolution differs from the
  calibration, and the warning that says so;
- `in_frame_fraction` and `map_monotonic`, the two numbers behind the decision.

A fold in the remap (`map_monotonic: false`) must fail the Episode: it means the
coefficients do not describe this lens, and the output is torn rather than merely
wrong. The adapter must never silently substitute a different camera's intrinsics
for the one requested.

## Hand-pose backend

A hand-pose backend consumes synchronized image indices and emits one row per
timeline row. The minimum output columns are:

- `output_frame_index`, `timestamp_s`;
- a validity/detection flag;
- wrist position and orientation, or 2-D joint observations from which stereo
  triangulation can recover a metric wrist pose;
- scale-invariant hand-closure features when a gripper label is required.

HaMeR/HumanEgo is one possible backend, not a dependency of the orchestration
core. A MediaPipe, MANO, or custom backend may implement the same contract.

## Gripper-prior adapter

The adapter consumes demonstrated robot feedback rather than commanded hand
targets. It must publish:

- robust open/closed anchors;
- hard clipping bounds;
- an optional normalized-time profile for temporary visual occlusion;
- the exact source dataset and source Episode identifiers.

The phase profile is a weak fallback. It must not replace a valid visual hand
observation.

## Robot retargeting / IK adapter

The IK command receives a metric wrist trajectory, a real demonstrated robot
start contract, calibration, and a gripper trajectory. It writes one directory
per Episode plus `batch_summary.json`.

Each Episode report must include `status: PASS|FAIL` and the following checks:

- valid observation fraction and maximum interpolation gap;
- joint limits and minimum limit margin;
- maximum per-frame joint step and whole-trajectory excursion;
- forward-kinematics translation/orientation residual;
- input/output frame counts and source Episode identifier.

The orchestrator only treats a report explicitly marked `PASS` as packable.

## Dataset packer adapter

The packer consumes only quality-passed Episode reports. It must preserve a
source-to-output Episode map and write:

- dataset metadata and feature schema;
- state/action normalization statistics;
- a preprocessing summary containing all smoothing, action lead, clipping,
  resize, frame-rate, and codec parameters;
- a final numerical/video decoding audit.

Normalization statistics belong in metadata. Raw physical state/action values
should not be destructively replaced merely to prepare the dataset.
