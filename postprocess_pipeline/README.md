# Orchestration core

This directory is the dataset-, task-, and robot-independent part of the project:
the stage runner, the public config schema, the manifest/audit tools, and the
adapter contracts. It requires **only the Python standard library**; the tabular
tools add NumPy, pandas, and PyArrow.

It does not depend on tas5-1, Nero, Revo2, Rokoko, HaMeR, or HumanEgo. Those are
integrations selected by configuration. `configs/ego_hamer_stereo_ik.template.json`
demonstrates the full no-Rokoko path while keeping hand-pose and robot-IK
implementations replaceable.

For the project overview, the demo, and licensing, start at the
[root README](../README.md).

## Layout

```text
pipeline.py                     generic stage runner
pipeline.schema.json            public JSON config schema
tools/
  build_episode_manifest.py     non-destructive episode selection
  filter_episode_manifest.py    derived-artifact and status quality gates
  run_with_manifest.py          pass stable episode IDs to any adapter
  build_robot_gripper_prior.py  robust prior from robot feedback
  audit_episode_artifacts.py    per-episode training-readiness gate
configs/
  ego_hamer_stereo_ik.template.json   portable template, full path
  ego_no_rokoko_tas5_1.example.json   real-world example, no Rokoko
  ego_task2_rokoko.example.json       real-world example, with Rokoko
  robot_qc.example.json               quality-check only, no Ego data
tests/
ADAPTER_CONTRACTS.md            interfaces for pose, IK, and packer backends
THIRD_PARTY.md                  dependency and weight licensing checklist
build_release.py                deterministic, privacy-scanned source archive
```

The common Ego flow:

```text
raw multi-view recording
  -> immutable Episode manifest
  -> causal synchronized timeline
  -> hand-pose backend on two views
  -> metric stereo wrist trajectory
  -> visual closure + demonstrated robot gripper prior
  -> robot-specific calibration and IK adapter
  -> per-Episode quality audit
  -> policy-specific LeRobot packer + normalization statistics
```

Stage-by-stage inputs, outputs, and gates: [`docs/PIPELINE_STAGES.md`](../docs/PIPELINE_STAGES.md).
Design rationale: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

The runner never uses shell evaluation. Paths containing spaces are safe, and the
exact argv for every subprocess is written to provenance.

## Quick start

```bash
python -m pip install -r requirements.txt
```

Hand-pose, robot SDK, IK, and policy packer dependencies are installed by their
respective adapters and are intentionally not bundled here.

List the stages in the portable template:

```bash
python postprocess_pipeline/pipeline.py \
  --config postprocess_pipeline/configs/ego_hamer_stereo_ik.template.json \
  --list
```

Override paths and parameters without editing Python source:

```bash
python postprocess_pipeline/pipeline.py \
  --config postprocess_pipeline/configs/ego_hamer_stereo_ik.template.json \
  --set source_root=/datasets/my_ego_recordings \
  --set robot_dataset=/datasets/my_robot_demonstrations \
  --set humanego_root=/opt/HumanEgo/ego_dropper \
  --set stereo_calibration=/calibration/stereo.json \
  --set base_calibration=/calibration/camera_to_base.json \
  --set start_contract=/calibration/demonstrated_start.json \
  --set episode_first=1 \
  --set episode_last=120
```

That command is a dry run. It prints missing requirements and the exact commands
but does not modify data. After reviewing it, add `--execute`.

```bash
python postprocess_pipeline/pipeline.py \
  --config postprocess_pipeline/configs/ego_hamer_stereo_ik.template.json \
  --set source_root=/datasets/my_ego_recordings \
  --set robot_dataset=/datasets/my_robot_demonstrations \
  --execute
```

Run one stage or a stage interval:

```bash
# One stage
python postprocess_pipeline/pipeline.py --config CONFIG.json \
  --stage manifest_raw --execute

# A contiguous range
python postprocess_pipeline/pipeline.py --config CONFIG.json \
  --from-stage hamer_camera1 --to-stage audit --execute

# Skip a stage only when all declared outputs exist
python postprocess_pipeline/pipeline.py --config CONFIG.json \
  --resume --execute
```

Every real run creates a timestamped directory below `run_root` containing:

- stdout/stderr logs for each subprocess;
- the exact argv, working directory, accepted return codes, and return value;
- the resolved variable values;
- config SHA-256, Python version, executable, platform, timestamps, per-stage
  durations, and final status.

## Config format

The public schema is `pipeline.schema.json`. A minimal configuration is:

```json
{
  "schema_version": 1,
  "name": "my-pipeline",
  "variables": {
    "project_root": "{config_dir}/../..",
    "source_root": "{project_root}/data/raw",
    "artifact_root": "{project_root}/artifacts/my-run"
  },
  "run_root": "{artifact_root}/runs",
  "stages": [
    {
      "name": "validate",
      "cwd": "{project_root}",
      "requires": ["{source_root}"],
      "produces": ["{artifact_root}/manifest.json"],
      "command": [
        "python", "validate.py", "{source_root}",
        "--output", "{artifact_root}/manifest.json"
      ]
    }
  ]
}
```

Variables may reference other variables. `config_dir` is supplied automatically.
CLI `--set NAME=VALUE` values override the JSON variables. Only
identifier-shaped `{placeholders}` are expanded, so JSON snippets and paths
containing other braces pass through unchanged; an unknown variable name is still
reported as an error instead of becoming a literal path.

Use `accept_returncodes` only when a tool intentionally uses a non-zero status to
mean "completed with rejected Episodes". For example:

```json
"accept_returncodes": [0, 3]
```

Rejected Episodes still have to be excluded by an audit before packing.

## Data invariants

- Raw Episode identifiers are immutable; selection is represented by manifests.
- Time alignment must state whether it is causal and which modalities define the
  common valid interval.
- Three cameras are views of one trajectory, not three times the task duration.
- Weak-perspective hand-model depth must not be presented as metric depth; metric
  translation requires stereo or another calibrated depth source.
- Gripper phase priors are only occlusion fallbacks. Valid visual closure
  observations remain the primary signal.
- Robot reset/start state must come from a real demonstrated pose, not an
  arithmetic mean that may be dynamically or kinematically unsafe.
- Only Episodes explicitly passing pose, fusion, IK, and final decode/numerical
  audits may enter the packed dataset.
- Physical state/action values remain physical in storage. Normalization
  statistics are stored as dataset metadata for the training loader.
- Source-to-output Episode mappings and all preprocessing parameters must be
  retained.

## Adding a backend

Implement an executable command that follows the relevant interface in
[`ADAPTER_CONTRACTS.md`](ADAPTER_CONTRACTS.md), then point a stage at that
executable. No changes to `pipeline.py` are required.

Typical replaceable components are:

- hand-pose backend: HaMeR, MANO, MediaPipe, or a custom estimator;
- metric reconstruction: stereo, RGB-D, motion capture, or calibrated multi-view;
- robot adapter: Nero, Franka, UR, xArm, or another URDF/solver;
- packer: ACT, Pi0/Pi0.5, Diffusion Policy, or another LeRobot-compatible schema.

## Testing

```bash
python -m pytest
```

The suite covers variable resolution, manifest immutability, status gates,
preflight scheduling, and the adapter CLIs. CI additionally runs the synthetic
demo on Linux, macOS, and Windows.

## Licensing

The runner and original tools are LGPL-3.0; see [`../LICENSE`](../LICENSE).

The HumanEgo copy used during development is PolyForm Noncommercial 1.0.0. It
must remain an optional external dependency. See
[`THIRD_PARTY.md`](THIRD_PARTY.md) and [`../NOTICE`](../NOTICE) before publishing
code or weights.

Do not publish private server addresses, credentials, hardware serial numbers,
absolute personal paths, or recordings without a suitable data license and
participant consent.

## Building a release archive

The release builder uses an explicit file allowlist, excludes data/model/cache
artifacts, scans for leaked paths and credential shapes, and writes a SHA-256
checksum:

```bash
python postprocess_pipeline/build_release.py --version 0.1.0
```

The ZIP is written below `dist/`. Given identical source files and version,
archive contents and ZIP timestamps are deterministic.
