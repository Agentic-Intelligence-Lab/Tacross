# Stage reference

Twelve stages, in the order the portable template runs them. "Produces" is what
`--resume` checks; "gate" is what can stop the run.

| # | Stage | Tool | Consumes | Produces | Gate |
| --- | --- | --- | --- | --- | --- |
| 1 | `manifest_raw` | `tools/build_episode_manifest.py` | raw episode directories | `manifest_raw.json` | exit 3 if no episode is valid |
| 2 | `sync_15hz` | `adapters/build_ego_no_rokoko_15hz_timeline.py` | manifest, camera/tactile timestamp CSVs | per-episode `sync/unified_no_rokoko_15hz_timeline.csv` | non-positive shared window aborts that episode |
| 3 | `manifest_timeline_ready` | `tools/build_episode_manifest.py` | manifest, timelines | `manifest_timeline_ready.json` | exit 3 if empty |
| 4 | `robot_gripper_prior` | `tools/build_robot_gripper_prior.py` | robot demonstration dataset | `robot_gripper_prior.json` + `.csv` | non-finite gripper feedback aborts |
| 5 | `hamer_camera1` | *external* hand-pose backend | timeline, manifest | `sync/humanego_min/hamer_hand_features_15hz.csv` | backend-defined |
| 6 | `hamer_camera3` | *external* hand-pose backend | timeline, manifest | `sync/humanego_min_cam3/hamer_hand_features_15hz.csv` | backend-defined |
| 7 | `stereo_wrist` | *external* stereo backend | two feature files, calibration | `T_hand_cam_stereo_<label>_15hz.csv` | epipolar and reprojection thresholds |
| 8 | `fuse_gripper` | `adapters/fuse_hamer_revo2_gripper.py` | features, prior, timeline | `sync/gripper_1dof_15hz.csv` + metadata | `detected_fraction < 0.60` → `status: FAIL`, exit 3 |
| 9 | `manifest_ik_ready` | `tools/filter_episode_manifest.py` | manifest, stereo + gripper artifacts | `manifest_ik_ready.json` | exit 3 if empty |
| 10 | `robot_ik` | `adapters/build_ego_fruit_ik_trajectory.py` | stereo wrist, gripper, calibration, start pose | `planned_joints_source_15hz.csv`, `offline_ik_report.json` | joint step, FK residual, limit margin, excursion |
| 11 | `audit` | `tools/audit_episode_artifacts.py` | manifest, per-episode artifacts, IK reports | `pipeline_audit.json` | `training_ready` per episode |
| 12 | `pack_lerobot` | `adapters/prepare_pi05_ego_fruit_lerobot.py` | audit-passed episodes, reference dataset | LeRobot dataset + `preprocess_summary.json` | quantile audit, decode audit |

## Why the order matters

**Manifest before anything.** Stages 1 and 3 both call the same tool with
different `--required` sets. Stage 1 asks "does this recording structurally
exist?"; stage 3 asks "does it have a timeline?". Splitting the question keeps a
structural failure distinguishable from a processing failure in the audit.

**The prior is built from robot data, not from the hand.** Stage 4 reads
`observation.state` from real teleoperation and takes robust quantiles. It never
looks at the Ego recording, so the anchors cannot drift toward whatever the hand
detector happened to output.

**Filtering happens before IK.** Stage 9 is the cheap gate: a missing stereo file
or a failed fusion is rejected before an IK solve spends minutes on it.

**Audit is separate from packing.** Stage 11 writes an explicit
`training_ready_episodes` list; stage 12 consumes only that list. If you want to
train on a subset, edit the audit input, not the packer.

## Return codes

Tools use exit code 3 for "completed, but some episodes were rejected". This is
deliberately distinct from 0 (all good) and 1 (crash), and configs accept it with:

```json
"accept_returncodes": [0, 3]
```

A rejected episode is never silently dropped: it appears in the manifest's
`excluded_episodes` with machine-readable reasons, and in the audit's
`rejected_episodes`.

## What `--resume` skips

A stage is skipped when **every** path in its `produces` list exists. Stages that
declare no outputs — the hand-pose, stereo, and fusion stages, whose outputs are
per-episode and therefore not known up front — always re-run. That is intentional:
re-running a fusion is cheap, re-running a 200-episode IK batch is not.

## Timeline semantics

`sync_15hz` builds a **causal** clock: for each 15 Hz tick it takes the most
recent sample at or before that time (zero-order hold), never a future one. Every
row records the source frame index and the age in milliseconds for each modality,
so a downstream stage can reject a row whose data is stale.

The common valid interval is the intersection of all participating sources:

```text
shared_start = max(start of each source)
shared_end   = min(end   of each source)
```

Three cameras are three views of one trajectory. They do not extend the
duration, and the timeline length is derived from the intersection, not from the
sum of the sources.

## Gripper fusion weights

| Signal | Weight | When |
| --- | --- | --- |
| Visual closure, camera1 | 0.60 | frame detected and valid |
| Visual closure, camera3 | 0.40 | frame detected and valid |
| Visual closure, interpolated | 0.35 | detection gap, features interpolated |
| Robot phase prior | 1 − visual weight | occlusion fallback only |

Feature weights inside one camera's closure score: pinch aperture 0.45, finger
reach 0.25, finger bend 0.30. Output is smoothed, clipped to the prior's
`hard_min`/`hard_max`, then rate-limited to 0.018 per 15 Hz frame.

All of these are CLI flags on the adapter; the table documents the defaults.
