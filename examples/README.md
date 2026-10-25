# Examples

## Synthetic end-to-end demo

No cameras, no GPU, no ffmpeg, no model weights. Only NumPy, pandas, and PyArrow.

```bash
python -m pip install -r requirements.txt
python examples/make_demo_dataset.py
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json --execute
```

Expected outcome:

```text
3 episodes pass the structural manifest
3 timelines built at 15 Hz (180 rows, ~12 s)
robot gripper prior: open 0.152, closed 0.467
fuse_gripper: episode_000001 PASS (94.4% detected)
              episode_000002 PASS (92.2% detected)
              episode_000003 FAIL (51.1% detected)
manifest_ik_ready: 2 valid, 1 excluded
```

Episode 3 is generated with heavy detection dropout on purpose. The pipeline
carries it through the timeline, rejects it at the fusion gate, and records the
reason in the IK-ready manifest:

```json
{"episode": "episode_000003",
 "reasons": ["json_gate_failed:sync/gripper_hamer_robot_prior_metadata.json:status=FAIL;expected=PASS"]}
```

That is the point of the demo: the gate fires, and the rejection is auditable.

## What the fixtures represent

| File | Stands in for | Produced by |
| --- | --- | --- |
| `rgb/camera{1,3}_timestamps.csv` | host monotonic capture timestamps | `make_demo_dataset.py` |
| `tactile/tactile_timestamps.csv` | tactile sensor timestamps | `make_demo_dataset.py` |
| `sync/trigger_log.json` | hardware start-trigger handshake | `make_demo_dataset.py` |
| `robot/data/**/episode_*.parquet` | teleoperated demonstrations (8-D state) | `make_demo_dataset.py` |
| `sync/humanego_min*/hamer_hand_features_15hz.csv` | HaMeR output on two views | `simulate_hand_pose.py` |
| `sync/humanego_min/T_hand_cam_stereo_*.csv` | stereo-triangulated wrist track | `simulate_hand_pose.py` |

Everything downstream of those files is the real implementation.

## What the demo deliberately does not do

- **No video assets.** Every stage before the packer reads only the timestamp
  CSVs, so the demo needs no decoder. A real recording does need
  `rgb/camera1.avi`, `rgb/camera2_gopro.mp4`, and `rgb/camera3.avi`; add them to
  the `manifest_raw` stage's `--required` list.
- **No IK or packing.** Both need backends that are not vendored (robot URDF +
  solver, and the LeRobot packing helpers). The demo stops at
  `manifest_ik_ready`, which is exactly where a real run hands off to them.
- **No accuracy claim.** `simulate_hand_pose.py` generates a parametric
  reach-and-grasp track. It exercises the *shape* of the contract, not HaMeR's
  precision. Do not quote its numbers.

## Running the demo's stages individually

```bash
# what would run?
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json --list

# one stage
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json \
  --stage fuse_gripper --execute

# everything after the prior
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json \
  --from-stage fuse_gripper --execute

# skip stages whose declared outputs exist
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json \
  --resume --execute
```

Delete `examples/demo_data/` to start over; it is regenerable and gitignored.

## Using a specific interpreter

The demo config uses `python` as the interpreter. To run the stages under a
different environment without editing JSON:

```bash
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json \
  --set python=/path/to/venv/bin/python --execute
```
