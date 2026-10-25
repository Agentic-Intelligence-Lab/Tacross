#!/usr/bin/env python3
"""Dewarp one camera view into a new video without touching the source Episode.

The adapter is calibration-driven.  It reads camera intrinsics plus a distortion
model from JSON, builds a rectification map once per run, and transcodes the
selected view frame by frame through ffmpeg.  The source recording stays
byte-identical; the dewarped view is written next to it as a derived artifact
together with a JSON report that records every parameter used.

Typical use is a wide-angle wrist camera whose barrel distortion hurts both the
hand-pose backend and the packed training images::

    python3 adapters/undistort_wrist_view.py data/ego_raw \\
        --camera camera1 --calibration calibration/wrist_camera1.json \\
        --alpha 0.0 --skip-failed

Exit codes match the other adapters: 0 when every Episode passed, 3 when at
least one Episode was rejected, 2 when the configuration itself is unusable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPORT_RELATIVE = "sync/undistort_report.json"
DEFAULT_ALPHA = 0.0
DEFAULT_CRF = 18
DEFAULT_CODEC = "libx264"
DEFAULT_PRESET = "medium"
DEFAULT_MIN_IN_FRAME = 0.5
VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".webm", ".MP4", ".AVI")

PINHOLE_MODELS = {
    "pinhole",
    "radtan",
    "plumb_bob",
    "rational_polynomial",
    "brown",
    "opencv",
    "standard",
}
FISHEYE_MODELS = {"fisheye", "equidistant", "kb", "kannala_brandt", "fisheye624"}


class CalibrationError(RuntimeError):
    """Raised when the calibration file cannot describe the requested camera."""


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics for exactly one camera."""

    name: str
    model: str
    matrix: np.ndarray
    distortion: np.ndarray
    width: int
    height: int
    model_source: str = "file"


def _as_matrix(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3):
        return array
    if array.size == 9:
        return array.reshape(3, 3)
    raise CalibrationError(f"{field_name} must be a 3x3 matrix or 9 numbers, got {array.shape}")


def _as_distortion(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _pick(block: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in block and block[name] is not None:
            return block[name]
    return None


def _normalise_model(raw: Any) -> tuple[str, str]:
    if raw is None:
        # A four-parameter fisheye model and a four-parameter pinhole model are
        # indistinguishable without a label, so fall back to the common case and
        # say so in the report rather than guessing silently.
        return "pinhole", "default"
    text = str(raw).strip().lower().replace("-", "_")
    if text in FISHEYE_MODELS:
        return "fisheye", "file"
    if text in PINHOLE_MODELS:
        return "pinhole", "file"
    raise CalibrationError(
        f"unknown distortion model {raw!r}; expected one of "
        f"{sorted(PINHOLE_MODELS | FISHEYE_MODELS)}"
    )


def _image_size(block: dict[str, Any], name: str) -> tuple[int, int]:
    size = _pick(block, "image_size", "resolution", "size")
    if size is not None:
        values = list(size)
        if len(values) != 2:
            raise CalibrationError(f"{name}: image_size must be [width, height]")
        return int(values[0]), int(values[1])
    width = _pick(block, "image_width", "width")
    height = _pick(block, "image_height", "height")
    if width is None or height is None:
        raise CalibrationError(
            f"{name}: missing image size; add image_size:[w,h] or image_width/image_height"
        )
    return int(width), int(height)


def _parse_block(name: str, block: dict[str, Any]) -> CameraCalibration:
    matrix = _pick(block, "K", "camera_matrix", "intrinsics", "intrinsic_matrix")
    if matrix is None:
        raise CalibrationError(f"{name}: missing intrinsic matrix (K or camera_matrix)")
    distortion = _as_distortion(_pick(block, "D", "distortion", "dist_coeffs", "distortion_coefficients"))
    model, model_source = _normalise_model(
        _pick(block, "model", "distortion_model", "camera_model")
    )
    width, height = _image_size(block, name)
    return CameraCalibration(
        name=name,
        model=model,
        matrix=_as_matrix(matrix, field_name=f"{name}.K"),
        distortion=distortion,
        width=width,
        height=height,
        model_source=model_source,
    )


def _camera_blocks(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], bool]:
    """Collect every camera block the file offers, keyed by camera name.

    The second return value says whether the file describes one unnamed camera
    (a flat ``{K, D, image_size}`` object).  Only that shape is allowed to answer
    to any requested camera name; a file that names its cameras must be asked for
    the right one, because silently dewarping a different view with the wrong
    intrinsics is not recoverable downstream.
    """
    cameras = payload.get("cameras")
    if isinstance(cameras, dict):
        return {str(key): value for key, value in cameras.items() if isinstance(value, dict)}, False
    if isinstance(cameras, list):
        blocks = {}
        for index, entry in enumerate(cameras):
            if not isinstance(entry, dict):
                continue
            blocks[str(entry.get("name") or f"camera{index}")] = entry
        return blocks, False

    # A flat single-camera file.
    if _pick(payload, "K", "camera_matrix") is not None:
        return {str(payload.get("name") or "camera"): payload}, payload.get("name") is None

    # Per-camera top-level keys, e.g. {"camera1": {...}, "camera3": {...}}.
    blocks = {}
    for key, value in payload.items():
        if isinstance(value, dict) and _pick(value, "K", "camera_matrix") is not None:
            blocks[str(key)] = value
    if blocks:
        return blocks, False

    # Stereo files that wrap a pair under left/right.
    for wrapper in ("stereo", "pair", "calibration"):
        nested = payload.get(wrapper)
        if isinstance(nested, dict):
            found, flat = _camera_blocks(nested)
            if found:
                return found, flat
    return {}, False


def load_calibration(path: Path, camera: str) -> CameraCalibration:
    """Read the calibration file and return the block that describes ``camera``."""
    if not path.is_file():
        raise CalibrationError(f"calibration file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CalibrationError(f"{path} must contain a JSON object")

    blocks, unnamed = _camera_blocks(payload)
    if not blocks:
        raise CalibrationError(
            f"{path}: no camera block found; expected a 'cameras' map or a flat "
            "{K, D, image_size} object"
        )

    if camera in blocks:
        return _parse_block(camera, blocks[camera])
    folded = {key.lower(): key for key in blocks}
    if camera.lower() in folded:
        key = folded[camera.lower()]
        return _parse_block(key, blocks[key])
    if unnamed and len(blocks) == 1:
        only = next(iter(blocks))
        calibration = _parse_block(camera, blocks[only])
        return CameraCalibration(
            name=camera,
            model=calibration.model,
            matrix=calibration.matrix,
            distortion=calibration.distortion,
            width=calibration.width,
            height=calibration.height,
            model_source=f"{calibration.model_source} (single-camera file)",
        )
    raise CalibrationError(
        f"{path}: no block for camera {camera!r}; available: {sorted(blocks)}"
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass
class Rectification:
    """Everything needed to remap one camera at one output size."""

    map_x: np.ndarray
    map_y: np.ndarray
    new_matrix: np.ndarray
    width: int
    height: int
    source_width: int
    source_height: int
    in_frame_fraction: float
    monotonic: bool = True
    crop: tuple[int, int, int, int] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def source_frame_bytes(self) -> int:
        return self.source_width * self.source_height * 3


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "the undistortion adapter needs OpenCV.\n"
            "  install it with: pip install opencv-python-headless>=4.8\n"
            "  or:              pip install 'embodied-postprocess-pipeline[undistort]'\n"
            f"  original error: {error}"
        ) from error
    return cv2


def _even(value: int) -> int:
    """yuv420p needs even dimensions; shave a single pixel when necessary."""
    return value - (value % 2)


def map_is_monotonic(map_x: np.ndarray, map_y: np.ndarray, tolerance: float = 1e-3) -> bool:
    """Check that the remap never folds back along the two optical axes.

    A fold is what a wrong distortion model looks like numerically: the map stops
    being invertible and the dewarped frame tears or mirrors.  Sampling the centre
    row and column is enough to catch it without a full Jacobian sweep.

    Only samples that land inside the source frame are judged: with ``alpha=1``
    the border of the output is extrapolated by definition, and OpenCV's iterative
    polynomial inversion is free to wander there.
    """
    if map_x.size == 0:
        return True
    return _monotonic_in_bounds(map_x[map_x.shape[0] // 2, :], map_x.shape[1], tolerance) and (
        _monotonic_in_bounds(map_y[:, map_y.shape[1] // 2], map_y.shape[0], tolerance)
    )


def _monotonic_in_bounds(values: np.ndarray, limit: int, tolerance: float) -> bool:
    inside = values[(values >= 0.0) & (values <= limit - 1)]
    if inside.size < 2:
        return True
    return bool(np.all(np.diff(inside) >= -tolerance))


def build_rectification(
    calibration: CameraCalibration,
    *,
    alpha: float = DEFAULT_ALPHA,
    output_size: tuple[int, int] | None = None,
    scale: float = 1.0,
    crop_to_valid: bool = False,
) -> Rectification:
    """Build the remap tables for one camera.

    ``alpha`` follows the OpenCV convention: 0 keeps only valid pixels, 1 keeps
    every source pixel at the cost of a black border.  For fisheye intrinsics it
    is passed as the ``balance`` argument.
    """
    cv2 = _require_cv2()
    warnings: list[str] = []

    width, height = output_size or (calibration.width, calibration.height)
    if scale != 1.0:
        if output_size is not None:
            raise ValueError("pass either output_size or scale, not both")
        width, height = int(round(width * scale)), int(round(height * scale))
    width, height = _even(width), _even(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid output size {width}x{height}")

    source_size = (calibration.width, calibration.height)
    matrix = calibration.matrix.astype(np.float64).copy()
    distortion = calibration.distortion.astype(np.float64).copy()
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}")

    if calibration.model == "fisheye":
        coefficients = distortion[:4] if distortion.size >= 4 else np.zeros(4)
        if distortion.size and distortion.size != 4:
            warnings.append(
                f"fisheye expects 4 coefficients, {distortion.size} were given; "
                "the first four are used"
            )
        coefficients = coefficients.reshape(4, 1)
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            matrix,
            coefficients,
            source_size,
            np.eye(3),
            balance=alpha,
            new_size=(width, height),
        )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            matrix, coefficients, np.eye(3), new_matrix, (width, height), cv2.CV_32FC1
        )
    else:
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            matrix, distortion, source_size, alpha, (width, height)
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrix, distortion, None, new_matrix, (width, height), cv2.CV_32FC1
        )

    # Half a pixel of slack: an identity map lands on 639.9999 rather than 639
    # and must still count as inside.
    inside = (
        (map_x >= -0.5)
        & (map_x <= source_size[0] - 0.5)
        & (map_y >= -0.5)
        & (map_y <= source_size[1] - 0.5)
    )
    fraction = float(inside.mean()) if inside.size else 0.0
    monotonic = map_is_monotonic(map_x, map_y)
    if not monotonic:
        warnings.append(
            "the rectification map folds back along the optical axes; the distortion "
            "model or its coefficients almost certainly do not match this lens"
        )

    crop = None
    if crop_to_valid:
        rows = np.flatnonzero(inside.any(axis=1))
        columns = np.flatnonzero(inside.any(axis=0))
        if rows.size and columns.size:
            top, bottom = int(rows[0]), int(rows[-1]) + 1
            left, right = int(columns[0]), int(columns[-1]) + 1
            crop = (left, top, right - left, bottom - top)
            map_x = map_x[top:bottom, left:right]
            map_y = map_y[top:bottom, left:right]
            width, height = _even(crop[2]), _even(crop[3])
            map_x = map_x[:height, :width]
            map_y = map_y[:height, :width]
        else:
            warnings.append("no valid pixels to crop to; the full frame is kept")

    return Rectification(
        map_x=map_x,
        map_y=map_y,
        new_matrix=new_matrix,
        width=width,
        height=height,
        source_width=calibration.width,
        source_height=calibration.height,
        in_frame_fraction=fraction,
        monotonic=monotonic,
        crop=crop,
        warnings=warnings,
    )


def rescale_matrix(
    calibration: CameraCalibration, width: int, height: int
) -> tuple[CameraCalibration, float]:
    """Scale intrinsics when the video resolution differs from the calibration."""
    if (width, height) == (calibration.width, calibration.height):
        return calibration, 1.0
    factor_x = width / calibration.width
    factor_y = height / calibration.height
    if abs(factor_x - factor_y) > 1e-6:
        raise CalibrationError(
            f"{calibration.name}: video is {width}x{height} but calibration is "
            f"{calibration.width}x{calibration.height}; the aspect ratio differs, "
            "so intrinsics cannot be rescaled safely"
        )
    matrix = calibration.matrix.copy()
    matrix[0, :] *= factor_x
    matrix[1, :] *= factor_y
    return (
        CameraCalibration(
            name=calibration.name,
            model=calibration.model,
            matrix=matrix,
            distortion=calibration.distortion,
            width=width,
            height=height,
            model_source=calibration.model_source,
        ),
        factor_x,
    )


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #


def require_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise SystemExit(
            "the undistortion adapter shells out to ffmpeg for decoding and encoding.\n"
            "  install ffmpeg and make sure it is on PATH"
        )
    return binary


def probe_video(path: Path) -> dict[str, Any]:
    """Read the geometry of a video, preferring ffprobe over OpenCV.

    ffprobe is authoritative and matches what the rest of the pipeline uses, but
    it is not always installed next to ffmpeg.  OpenCV is already a dependency of
    this adapter, so it makes a usable fallback; the source is recorded in the
    report so a surprising frame count can be traced to the probe that produced it.
    """
    binary = shutil.which("ffprobe")
    if binary:
        return _probe_with_ffprobe(binary, path)
    return _probe_with_opencv(path)


def _probe_with_ffprobe(binary: str, path: Path) -> dict[str, Any]:
    command = [
        binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or "ffprobe failed"
        raise RuntimeError(f"cannot probe {path}: {detail}")
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = map(int, str(stream["avg_frame_rate"]).split("/"))
    fps = numerator / denominator if denominator else 0.0
    if fps <= 0:
        raise RuntimeError(f"invalid frame rate for {path}")
    frames = int(stream.get("nb_frames") or 0)
    if frames <= 0:
        frames = int(round(float(payload["format"]["duration"]) * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(fps, 6),
        "frames": frames,
        "probe": "ffprobe",
    }


def _probe_with_opencv(path: Path) -> dict[str, Any]:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {path}; install ffprobe for a detailed reason")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"cannot read geometry from {path}")
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frames": max(frames, 0),
        "probe": "opencv",
    }


def transcode(
    source: Path,
    destination: Path,
    rectification: Rectification,
    *,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
) -> int:
    """Remap every frame of ``source`` and encode it to ``destination``.

    Returns the number of frames written.  ffmpeg does the decode and encode so
    that any container the source uses keeps working; only the remap happens in
    Python.  stderr goes to temporary files rather than pipes: a full pipe would
    block the child process mid-frame.
    """
    cv2 = _require_cv2()
    ffmpeg = require_ffmpeg()
    width, height = rectification.width, rectification.height
    frame_bytes = rectification.source_frame_bytes

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real extension: ffmpeg picks the muxer from it, and a ".partial"
    # tail would leave it with nothing to infer from.
    partial = destination.with_name(f"{destination.stem}.partial{destination.suffix}")

    with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
        decoder = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(source),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=decode_log,
        )
        encoder = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                codec,
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                str(partial),
            ],
            stdin=subprocess.PIPE,
            stderr=encode_log,
        )

        written = 0
        broken_pipe: str | None = None
        try:
            assert decoder.stdout is not None and encoder.stdin is not None
            while True:
                buffer = _read_exactly(decoder.stdout, frame_bytes)
                if buffer is None:
                    break
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    rectification.source_height, rectification.source_width, 3
                )
                remapped = cv2.remap(
                    frame,
                    rectification.map_x,
                    rectification.map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                encoder.stdin.write(remapped.tobytes())
                written += 1
            encoder.stdin.close()
            encoder.wait()
            decoder.wait()
        except BrokenPipeError:
            broken_pipe = "the encoder stopped accepting frames"
        finally:
            for process in (decoder, encoder):
                if process.poll() is None:
                    process.kill()
                    process.wait()

        decode_log.seek(0)
        encode_log.seek(0)
        decoder_error = decode_log.read().decode("utf-8", "replace").strip()
        encoder_error = encode_log.read().decode("utf-8", "replace").strip()

    if broken_pipe:
        raise RuntimeError(f"ffmpeg failed to encode {destination.name}: {broken_pipe}\n{encoder_error}")
    if decoder.returncode:
        raise RuntimeError(f"ffmpeg failed to decode {source}: {decoder_error}")
    if encoder.returncode:
        raise RuntimeError(f"ffmpeg failed to encode {destination.name}: {encoder_error}")
    if written == 0:
        raise RuntimeError(f"no frames decoded from {source}")

    partial.replace(destination)
    return written


def _read_exactly(stream: Any, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# --------------------------------------------------------------------------- #
# Episode processing
# --------------------------------------------------------------------------- #


def resolve_source_video(episode: Path, camera: str, template: str | None) -> Path:
    if template:
        return episode / template.format(camera=camera)
    for suffix in VIDEO_SUFFIXES:
        candidate = episode / f"rgb/{camera}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{episode.name}: no rgb/{camera}.* video found")


def process_episode(
    episode: Path,
    *,
    calibration: CameraCalibration,
    calibration_path: Path,
    camera: str,
    video_relative: str | None,
    output_relative: str,
    alpha: float,
    scale: float,
    output_size: tuple[int, int] | None,
    crop_to_valid: bool,
    codec: str,
    crf: int,
    preset: str,
    min_in_frame: float,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source = resolve_source_video(episode, camera, video_relative)
    destination = episode / output_relative.format(camera=camera)
    geometry = probe_video(source)

    warnings: list[str] = []
    scaled, factor = rescale_matrix(calibration, geometry["width"], geometry["height"])
    if factor != 1.0:
        warnings.append(
            f"video is {geometry['width']}x{geometry['height']} but calibration is "
            f"{calibration.width}x{calibration.height}; intrinsics scaled by {factor:.6f}"
        )
    if scaled.model_source == "default":
        warnings.append(
            "the calibration file declares no distortion model; pinhole was assumed"
        )

    rectification = build_rectification(
        scaled,
        alpha=alpha,
        output_size=output_size,
        scale=scale,
        crop_to_valid=crop_to_valid,
    )
    warnings.extend(rectification.warnings)

    in_frame_ok = rectification.in_frame_fraction >= min_in_frame
    if not in_frame_ok:
        warnings.append(
            f"only {rectification.in_frame_fraction:.1%} of the output samples inside "
            f"the source frame (threshold {min_in_frame:.0%}); alpha={alpha} may be too high"
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "episode": episode.name,
        "camera": camera,
        "source": {"path": source.relative_to(episode).as_posix(), **geometry},
        "output": {
            "path": destination.relative_to(episode).as_posix(),
            "width": rectification.width,
            "height": rectification.height,
            "codec": codec,
            "crf": crf,
            "preset": preset,
            "frames": None,
        },
        "calibration": {
            "path": str(calibration_path),
            "camera": calibration.name,
            "model": scaled.model,
            "model_source": scaled.model_source,
            "alpha": alpha,
            "matrix": scaled.matrix.tolist(),
            "distortion": scaled.distortion.tolist(),
            "new_matrix": rectification.new_matrix.tolist(),
            "rescale_factor": factor,
        },
        "geometry": {
            "in_frame_fraction": round(rectification.in_frame_fraction, 6),
            "map_monotonic": rectification.monotonic,
            "crop": list(rectification.crop) if rectification.crop else None,
        },
        "checks": {
            "in_frame_fraction_ok": in_frame_ok,
            "map_monotonic": rectification.monotonic,
            "frame_count_match": None,
        },
        "warnings": warnings,
    }

    if dry_run:
        report["status"] = "DRY_RUN"
        return report

    if destination.exists() and not overwrite:
        existing = _existing_report(episode)
        if existing and existing.get("status") == "PASS":
            report["status"] = "SKIPPED_EXISTS"
            report["output"]["frames"] = existing.get("output", {}).get("frames")
            return report
        raise RuntimeError(
            f"{destination.relative_to(episode)} already exists; pass --overwrite to replace it"
        )

    written = transcode(
        source,
        destination,
        rectification,
        fps=geometry["fps"],
        codec=codec,
        crf=crf,
        preset=preset,
    )
    frame_count_ok = written == geometry["frames"] if geometry["frames"] else True
    report["output"]["frames"] = written
    report["checks"]["frame_count_match"] = frame_count_ok
    if not frame_count_ok:
        warnings.append(
            f"decoded {written} frames but the container declares {geometry['frames']}"
        )

    report["status"] = (
        "PASS" if (in_frame_ok and frame_count_ok and rectification.monotonic) else "FAIL"
    )
    return report


def _existing_report(episode: Path) -> dict[str, Any] | None:
    path = episode / REPORT_RELATIVE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1280x720") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="Root directory containing episode_* folders")
    parser.add_argument("--episode", action="append", help="Episode name; repeatable")
    parser.add_argument("--skip-failed", action="store_true", help="Keep going when an Episode fails")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="JSON file with the camera intrinsics and distortion model",
    )
    parser.add_argument("--camera", default="camera1", help="Camera name to dewarp (default: camera1)")
    parser.add_argument(
        "--model",
        choices=("auto", "pinhole", "fisheye"),
        default="auto",
        help="Override the model declared in the calibration file (default: auto)",
    )
    parser.add_argument(
        "--video-relative",
        default=None,
        help="Source video template relative to the Episode; default searches rgb/<camera>.*",
    )
    parser.add_argument(
        "--output-relative",
        default="rgb/{camera}_undistorted.mp4",
        help="Output video template relative to the Episode",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="0 keeps only valid pixels, 1 keeps every source pixel (default: 0)",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Output resolution factor")
    parser.add_argument("--size", type=_parse_size, default=None, help="Explicit output WIDTHxHEIGHT")
    parser.add_argument(
        "--crop-to-valid",
        action="store_true",
        help="Crop the output to the bounding box of valid pixels",
    )
    parser.add_argument("--codec", default=DEFAULT_CODEC, help=f"ffmpeg encoder (default: {DEFAULT_CODEC})")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help=f"Quality, lower is better (default: {DEFAULT_CRF})")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help=f"x264 preset (default: {DEFAULT_PRESET})")
    parser.add_argument(
        "--min-in-frame",
        type=float,
        default=DEFAULT_MIN_IN_FRAME,
        help="Reject an Episode when less than this fraction of the output is valid",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output videos")
    parser.add_argument("--dry-run", action="store_true", help="Resolve everything, write nothing")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Batch summary destination; defaults to <root>/undistort_summary.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.size is not None and args.scale != 1.0:
        parser.error("--size and --scale are mutually exclusive")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    calibration_path = args.calibration.expanduser().resolve()

    try:
        calibration = load_calibration(calibration_path, args.camera)
    except CalibrationError as error:
        print(f"UNDISTORT_CONFIG_ERROR: {error}", file=sys.stderr)
        return 2

    if args.model != "auto":
        calibration = CameraCalibration(
            name=calibration.name,
            model=args.model,
            matrix=calibration.matrix,
            distortion=calibration.distortion,
            width=calibration.width,
            height=calibration.height,
            model_source="command line",
        )

    episodes = (
        [root / name for name in args.episode]
        if args.episode
        else sorted(path for path in root.glob("episode_*") if path.is_dir())
    )
    if not episodes:
        print(f"UNDISTORT_CONFIG_ERROR: no episodes under {root}", file=sys.stderr)
        return 2

    summary_path = (
        args.summary_path.expanduser().resolve()
        if args.summary_path is not None
        else root / "undistort_summary.json"
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(root),
        "camera": args.camera,
        "calibration": str(calibration_path),
        "model": calibration.model,
        "alpha": args.alpha,
        "dry_run": args.dry_run,
        "episodes": [],
    }

    for episode in episodes:
        try:
            report = process_episode(
                episode,
                calibration=calibration,
                calibration_path=calibration_path,
                camera=args.camera,
                video_relative=args.video_relative,
                output_relative=args.output_relative,
                alpha=args.alpha,
                scale=args.scale,
                output_size=args.size,
                crop_to_valid=args.crop_to_valid,
                codec=args.codec,
                crf=args.crf,
                preset=args.preset,
                min_in_frame=args.min_in_frame,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                report_path = episode / REPORT_RELATIVE
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            entry = {
                "episode": episode.name,
                "status": report["status"],
                "output": report["output"]["path"],
                "in_frame_fraction": report["geometry"]["in_frame_fraction"],
                "warnings": report["warnings"],
            }
            print(
                f"UNDISTORT_{report['status']} {episode.name} "
                f"in_frame={report['geometry']['in_frame_fraction']:.3f} "
                f"out={report['output']['width']}x{report['output']['height']}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - one bad Episode must not hide the rest
            entry = {"episode": episode.name, "status": "FAIL", "error": repr(error)}
            print(f"UNDISTORT_FAIL {episode.name}: {error}", flush=True)
            if not args.skip_failed:
                summary["episodes"].append(entry)
                _write_summary(summary_path, summary)
                raise
        summary["episodes"].append(entry)
        _write_summary(summary_path, summary)

    passed = sum(item["status"] in {"PASS", "DRY_RUN", "SKIPPED_EXISTS"} for item in summary["episodes"])
    failed = len(summary["episodes"]) - passed
    print(f"UNDISTORT_COMPLETE passed={passed} failed={failed} summary={summary_path}")
    return 0 if failed == 0 else 3


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
