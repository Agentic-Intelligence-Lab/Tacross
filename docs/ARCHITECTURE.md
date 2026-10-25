# Architecture

## Two layers, one contract

```text
┌──────────────────────────────────────────────────────────────┐
│ orchestration layer (this repository, stdlib only)           │
│                                                              │
│  pipeline.py ── resolves config, orders stages, runs argv,   │
│                 writes provenance, enforces preflight        │
│                                                              │
│  tools/ ─────── manifests, filtering, priors, audit          │
└───────────────────────────┬──────────────────────────────────┘
                            │  argv arrays, JSON artifacts
┌───────────────────────────▼──────────────────────────────────┐
│ adapter layer (replaceable)                                  │
│                                                              │
│  hand pose  │ stereo │ gripper fusion │ robot IK │ packer    │
└──────────────────────────────────────────────────────────────┘
```

The orchestration layer knows nothing about HaMeR, Nero, Revo2, or LeRobot. It
knows about *stages*: an argv array, a working directory, required paths,
produced paths, environment overrides, and accepted return codes.

That is why swapping a backend is a configuration change, not a code change. The
interface each backend must satisfy is [`ADAPTER_CONTRACTS.md`](../postprocess_pipeline/ADAPTER_CONTRACTS.md).

## Why argv arrays instead of shell strings

```json
"command": ["python", "tool.py", "{source_root}", "--output", "{out}"]
```

- A path containing a space, a quote, or `$(...)` cannot be reinterpreted by a
  shell. Injection is structurally impossible, not filtered against.
- The exact argv that ran is written verbatim to `provenance.json`, so a run can
  be reproduced or audited without guessing how a string was tokenized.
- Windows and POSIX behave the same way, because no shell is involved.

The trade-off is that pipes, redirection, and globs are not available. Wrap them
in a script if you need them.

## State lives in files, not in the runner

The runner holds no state between invocations. Everything it needs is on disk:

| Artifact | Written by | Meaning |
| --- | --- | --- |
| `manifest_*.json` | manifest tools | which episodes are in, which are out, and why |
| `*_metadata.json` | adapters | parameters and status for one episode |
| `offline_ik_report.json` | IK adapter | quality gates for one episode |
| `pipeline_audit.json` | audit tool | the training-ready list |
| `provenance.json` | runner | what actually executed, and with which variables |

This is what makes `--resume`, re-running a subset, and post-hoc auditing
possible. It is also why source data never has to be modified: selection is
always an artifact, never a directory rename.

## Failure model

| Situation | Behaviour |
| --- | --- |
| Config invalid (unknown variable, cycle, bad stage) | `CONFIG_ERROR`, exit 2, nothing runs |
| Required path missing at plan time | stage marked `FAILED_PREFLIGHT`, run aborts unless `--continue-on-error` |
| Subprocess returns an unaccepted code | stage `FAILED`, later stages skipped unless `--continue-on-error` |
| Subprocess returns an accepted non-zero code (3) | stage `PASS`, flag `accepted_nonzero_returncode` recorded |
| Episode rejected by a quality gate | not an error: recorded in the manifest with reasons |

Preflight accounts for artifacts that earlier stages in the same run will
produce, so a cold start works. Without that, every stage after the first would
report its input as missing before the pipeline had a chance to create it.

## Provenance

Every real run creates `run_root/<timestamp>_<config>/` containing:

```text
provenance.json                     resolved variables, config SHA-256,
                                    interpreter, platform, per-stage status
                                    and duration
<stage>_01.log                      stdout+stderr of each subprocess
```

`provenance.json` is rewritten after every stage, so a crash mid-run still leaves
a usable record of what completed.

## Extending

**A new hand-pose backend** needs to emit one row per timeline row with
`output_frame_index`, `timestamp_s`, a detection flag, and either a wrist pose or
2-D joints. Point `hamer_camera1` / `hamer_camera3` at it.

**A new robot** needs an IK adapter that accepts a metric wrist trajectory, a
start contract, and calibration, and writes `offline_ik_report.json` with
`status` plus the joint/FK checks. Point `robot_ik` at it.

**A new policy format** needs a packer that consumes only audit-passed episodes
and writes state/action statistics as metadata. Point `pack_lerobot` at it.

None of these require touching `pipeline.py`.
