# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Wrist-view undistortion** (`adapters/undistort_wrist_view.py`, stage 5). Reads
  camera intrinsics from JSON, supports pinhole/radtan and fisheye/equidistant
  models, and re-encodes one camera view into `rgb/<camera>_undistorted.mp4`
  without touching the source recording. Gates on a folded remap, the blank-border
  fraction, and a frame-count match. `--camera` is configurable, `--alpha` follows
  the OpenCV convention, and the resolved `K`/`D`/`new_K` are recorded per episode
  in `sync/undistort_report.json`. Optional: needs `opencv-python-headless` and
  ffmpeg, installable with `pip install '.[undistort]'`.
- `configs/wrist_undistort.example.json` runs the dewarp stage on its own, for
  retrofitting an existing dataset.
- Synthetic end-to-end demo (`examples/`) that runs the real pipeline with no
  cameras, GPU, ffmpeg, or external model, and deliberately fails one episode so
  the quality gate is observable.
- `docs/ARCHITECTURE.md`, `docs/PIPELINE_STAGES.md`, `docs/TROUBLESHOOTING.md`.
- CI across Linux, macOS, and Windows on Python 3.10–3.13, plus a dedicated job
  that installs OpenCV and ffmpeg so the undistortion tests cannot silently skip.

### Changed

- Adapter scripts moved from the repository root into `adapters/`. Config
  templates reference an `adapter_root` variable, so relocation is a one-line
  change.
- `build_release.py` now scans for generic leak *shapes* (home directories, IPv4
  literals, SSH targets, key and token formats) instead of a fixed list of
  personal strings, and builds a cross-platform `.zip` rather than a
  Windows-suffixed archive.
- Example configs use `python` instead of `python3`; Windows does not ship a
  `python3` shim.

### Fixed

- `--execute` aborted on a cold start: the up-front preflight reported artifacts
  as missing before the stages that produce them had run. Preflight now accounts
  for paths scheduled by earlier stages in the same invocation.
- `prepare_pi05_ego_fruit_lerobot.py` failed at import time without the HumanEgo
  toolkit, so `--help` crashed. The toolkit is now resolved lazily with an
  actionable error message.
- `run_with_manifest.py` used `argparse.REMAINDER`, which silently mangled
  wrapped commands containing flags. Options and the wrapped command are now
  split explicitly at `--`.
- `pipeline.py` expanded config values with `str.format_map`, which raised on any
  literal brace in a task instruction or path. Only identifier-shaped
  placeholders are expanded now, and unresolved or cyclic variables are reported
  instead of silently left in place.
- `build_ego_fruit_ik_trajectory.py` hard-coded the Ego-to-robot gripper affine
  map. Scale, offset, and clip bounds are now CLI flags with the previous values
  as defaults, and the values used are recorded in the report and batch summary.
- `sync_15hz` required `ffprobe` even for recordings without a GoPro view.
  `--no-camera2` and explicit `--camera2-fps`/`--camera2-frame-count` overrides
  remove the dependency.
- Adapter default paths resolved relative to `adapters/` after the move, which
  pointed them one directory too deep. They now resolve against the repository
  root.

## [0.1.0] - 2026-10-25

### Added

- Generic JSON-configured stage runner with dry-run default, stage/range
  selection, resume, `--continue-on-error`, `--set` variable overrides, and
  per-stage provenance.
- Non-destructive episode manifest tooling: structural discovery, derived-artifact
  filtering, JSON status gates.
- Causal 15 Hz camera/tactile timeline with per-modality frame index and sample
  age.
- Robust 1-DoF gripper prior from demonstrated robot feedback (quantile anchors,
  hard clip bounds, normalized-phase profile).
- Hand-closure fusion across two views with an explicit occlusion fallback and a
  detection-fraction gate.
- Offline robot retargeting and IK with joint-step, FK-residual, joint-limit
  margin, and excursion gates.
- Per-episode artifact audit producing an explicit training-ready list.
- LeRobot v3 packer with quantile normalization statistics and a decode audit.
- Adapter contracts for hand pose, stereo, gripper prior, IK, and packing.
- Deterministic release builder with privacy scanning and SHA-256 checksums.

[Unreleased]: https://github.com/fangdiyu/embodied-postprocess-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fangdiyu/embodied-postprocess-pipeline/releases/tag/v0.1.0
