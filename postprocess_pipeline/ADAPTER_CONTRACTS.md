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
