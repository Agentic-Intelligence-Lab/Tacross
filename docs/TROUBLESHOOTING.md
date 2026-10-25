# Troubleshooting

## `CONFIG_ERROR: unknown template variable 'x'`

A stage or variable references `{x}` but no variable of that name exists. Only
identifier-shaped placeholders are expanded; other braces are left alone. Run
with `--list` to confirm the config parses, then compare the name against the
`variables` block.

## `variable 'a' still contains a placeholder after resolution`

Two variables reference each other (`a = {b}`, `b = {a}`). Break the cycle.

## The plan reports `MISSING` for a path a previous stage should create

Paths are compared after `normcase`/`normpath`, but they are still strings. If one
stage writes `{artifact_root}/manifest.json` and another requires
`{artifact_root}/../artifacts/manifest.json`, the runner cannot know they are the
same file. Define a variable once and reference it everywhere.

## `--resume` re-runs a stage every time

Stages without a `produces` list always re-run; there is nothing to check. This is
intentional for per-episode stages. Add `produces` only if the stage writes a
single, predictable artifact.

## `ffprobe failed` / `cannot probe ...`

`sync_15hz` probes the GoPro view with `ffprobe` to recover its frame rate and
count. If ffmpeg is not installed, or the recording has no GoPro view, either:

```bash
# recording without a third camera
--no-camera2

# known frame rate and length, no probe
--camera2-fps 60 --camera2-frame-count 18000
```

## `non-positive shared window 12.400000->11.900000`

The timestamped sources do not overlap: the latest start is after the earliest
end. Usually a camera started late, or a timestamp column is in the wrong clock
domain. The error prints the bounds of every source — compare them against
`sync/trigger_log.json`.

## `the LeRobot packer needs the HumanEgo toolkit on PYTHONPATH`

The packer adapter does not vendor that toolkit. Point it at your checkout:

```bash
export HUMANEGO_PIPELINE_ROOT=/path/to/HumanEgo/ego_dropper
```

The config template does this for you with the stage `env` block. If you do not
have the toolkit, write a packer that implements the contract — the packer is the
most replaceable component in the pipeline.

## `first target is not locked to demonstrated start`

The IK adapter anchors the trajectory to a real demonstrated pose. If frame 0 of
your stereo track is far from that pose, the solve is refused rather than
silently snapping. Check `--start-contract` and the calibration, or use
`--start-current-episode` to derive the start from a robot episode's frame 0
(diagnostic only — do not ship a dataset built that way).

## `stereo valid fraction ... < 0.700`

Too few frames triangulated successfully. Raise `--max-epipolar-error-px` /
`--max-reprojection-error-px` in the stereo stage, or drop the episode. Lowering
`--min-valid-fraction` past ~0.6 mostly interpolates, which hides the problem
instead of fixing it.

## `joint continuity gate exceeded`

The IK solution jumps between adjacent waypoints, usually near a singularity or a
wrist flip. Options: increase `--smooth-window` on the wrist track, reduce the
motion range, or reject the episode. Do not raise `--max-joint-step-deg` without
checking the resulting trajectory by hand.

## Windows: `python3` is not recognised

Windows installs the interpreter as `python`. The example configs use `python`
for this reason. The upstream template used `python3`, which is POSIX-only; if you
copied it, change the first element of every `command` array, or set it once:

```bash
python postprocess_pipeline/pipeline.py --config CONFIG.json \
  --set python=python3
```

## A run produced `PASS` but the dataset looks empty

`pack_lerobot` consumes the audit's `training_ready_episodes`. If every episode
failed a gate, the audit is honest about it and the packer has nothing to write.
Read `pipeline_audit.json` → `rejected_episodes`, then the per-episode
`offline_ik_report.json` and `gripper_hamer_robot_prior_metadata.json`.

## `UNDISTORT_CONFIG_ERROR: no block for camera 'camera1'`

The calibration file names its cameras and `camera1` is not one of them. The
error lists what is available. Either pass the right `--camera`, or wrap a
single-camera file so it can answer for any name:

```json
{"cameras": {"camera1": {"model": "fisheye", "image_size": [1920, 1080],
                          "K": [[...]], "D": [...]}}}
```

A flat file with no `cameras` wrapper answers to any requested name, because
there is only one camera in it and no name to get wrong. A file that names its
cameras never substitutes one for another.

## `the rectification map folds back along the optical axes`

The distortion coefficients do not describe this lens: `r → r(1 + k1r² + k2r⁴ + …)`
stops being monotonic inside the frame, so the dewarped image tears or mirrors
instead of bending. Check, in this order:

1. **Model.** A fisheye lens described as `pinhole` (or the reverse) folds almost
   immediately. `model` must be `fisheye`/`equidistant`/`kb` for an equidistant
   lens.
2. **Coefficient order.** Pinhole is `k1, k2, p1, p2[, k3]`; fisheye is `k1..k4`.
   Swapping tangential for radial terms looks plausible and folds the map.
3. **Resolution.** Intrinsics are rescaled automatically when the video matches
   the calibration's aspect ratio, and the factor is recorded in the report as
   `calibration.rescale_factor`. A factor far from 1.0 usually means the
   calibration was made at a different capture mode.

Run with `--dry-run` to see the resolved matrix, `D`, `new_matrix` and
`in_frame_fraction` without writing a single frame.

## `only 62.0% of the output samples inside the source frame`

`alpha` is too high for this calibration: the output includes regions the source
never recorded, so they come out black. Lower `--alpha` (0 keeps only valid
pixels), or accept it deliberately and lower `--min-in-frame`. `alpha=0` sits at
100% by construction, so a low fraction there points at a wrong principal point
rather than at `alpha`.

## The dewarped video exists but the dataset still uses the raw one

By design. `rgb/<camera>_undistorted.mp4` is a *derived* view and no stage reads
it implicitly. Point the packer or the hand-pose backend at it explicitly, and
keep `sync/undistort_report.json` with the episode so the parameters used stay
recoverable.
