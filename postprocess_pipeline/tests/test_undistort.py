"""Tests for the wrist-view undistortion adapter.

The calibration parser and the intrinsics rescaling are pure numpy, so they run
everywhere.  The geometry tests need OpenCV and the end-to-end video test also
needs ffmpeg; both are skipped when the dependency is missing instead of failing
the whole suite.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ADAPTER_PATH = Path(__file__).resolve().parents[2] / "adapters" / "undistort_wrist_view.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("undistort_wrist_view", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves the defining module through sys.modules while the
    # class body executes, so the module has to be registered first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = load_adapter()


def has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def has_ffmpeg() -> bool:
    # ffprobe is optional: the adapter falls back to OpenCV for geometry.
    return bool(shutil.which("ffmpeg"))


# The dedicated CI job installs OpenCV and ffmpeg and sets this, so a missing
# dependency there is a failure rather than a silent skip.
if os.environ.get("UNDISTORT_REQUIRE_DEPS") == "1" and not (has_cv2() and has_ffmpeg()):
    raise RuntimeError(
        "UNDISTORT_REQUIRE_DEPS=1 but OpenCV or ffmpeg is missing; "
        "the undistortion tests would have been skipped"
    )


PINHOLE_BLOCK = {
    "model": "pinhole",
    "image_size": [1280, 720],
    "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
    "D": [-0.28, 0.09, 0.0004, -0.0002, -0.01],
}

FISHEYE_BLOCK = {
    "model": "fisheye",
    "image_size": [1920, 1080],
    "camera_matrix": [[700.0, 0.0, 960.0], [0.0, 700.0, 540.0], [0.0, 0.0, 1.0]],
    "dist_coeffs": [0.12, -0.03, 0.008, -0.001],
}


class CalibrationParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, payload: dict) -> Path:
        path = self.tmp / f"calibration_{len(list(self.tmp.iterdir()))}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_flat_single_camera_file(self) -> None:
        path = self.write(PINHOLE_BLOCK)
        calibration = adapter.load_calibration(path, "camera1")
        self.assertEqual(calibration.model, "pinhole")
        self.assertEqual((calibration.width, calibration.height), (1280, 720))
        self.assertAlmostEqual(calibration.matrix[0, 0], 900.0)
        self.assertEqual(calibration.distortion.size, 5)

    def test_nested_cameras_map(self) -> None:
        path = self.write({"cameras": {"camera1": PINHOLE_BLOCK, "camera3": FISHEYE_BLOCK}})
        left = adapter.load_calibration(path, "camera1")
        right = adapter.load_calibration(path, "camera3")
        self.assertEqual(left.model, "pinhole")
        self.assertEqual(right.model, "fisheye")
        self.assertEqual(right.width, 1920)

    def test_camera_list_form(self) -> None:
        path = self.write(
            {"cameras": [{"name": "wrist", **PINHOLE_BLOCK}, {"name": "third", **FISHEYE_BLOCK}]}
        )
        self.assertEqual(adapter.load_calibration(path, "wrist").model, "pinhole")
        self.assertEqual(adapter.load_calibration(path, "third").model, "fisheye")

    def test_top_level_per_camera_keys(self) -> None:
        path = self.write({"camera1": PINHOLE_BLOCK, "camera3": FISHEYE_BLOCK})
        self.assertEqual(adapter.load_calibration(path, "camera3").model, "fisheye")

    def test_single_camera_file_answers_to_any_name(self) -> None:
        path = self.write(FISHEYE_BLOCK)
        calibration = adapter.load_calibration(path, "wrist_cam")
        self.assertEqual(calibration.model, "fisheye")
        self.assertIn("single-camera file", calibration.model_source)

    def test_named_single_camera_is_not_substituted(self) -> None:
        # A file that names its camera must not silently answer for another one.
        path = self.write({"cameras": {"wrist": PINHOLE_BLOCK}})
        with self.assertRaises(adapter.CalibrationError) as caught:
            adapter.load_calibration(path, "camera1")
        self.assertIn("wrist", str(caught.exception))

    def test_camera_name_is_case_insensitive(self) -> None:
        path = self.write({"cameras": {"Camera1": PINHOLE_BLOCK}})
        self.assertEqual(adapter.load_calibration(path, "camera1").width, 1280)

    def test_missing_camera_lists_the_available_ones(self) -> None:
        path = self.write({"cameras": {"camera1": PINHOLE_BLOCK, "camera3": FISHEYE_BLOCK}})
        with self.assertRaises(adapter.CalibrationError) as caught:
            adapter.load_calibration(path, "camera2_gopro")
        message = str(caught.exception)
        self.assertIn("camera1", message)
        self.assertIn("camera3", message)

    def test_unknown_model_is_rejected(self) -> None:
        block = dict(PINHOLE_BLOCK, model="omni_radtan")
        path = self.write({"cameras": {"camera1": block}})
        with self.assertRaises(adapter.CalibrationError):
            adapter.load_calibration(path, "camera1")

    def test_missing_intrinsics_is_rejected(self) -> None:
        path = self.write({"cameras": {"camera1": {"image_size": [640, 480], "D": [0, 0]}}})
        with self.assertRaises(adapter.CalibrationError):
            adapter.load_calibration(path, "camera1")

    def test_missing_image_size_is_rejected(self) -> None:
        block = {key: value for key, value in PINHOLE_BLOCK.items() if key != "image_size"}
        path = self.write({"cameras": {"camera1": block}})
        with self.assertRaises(adapter.CalibrationError):
            adapter.load_calibration(path, "camera1")

    def test_absent_model_falls_back_to_pinhole_and_says_so(self) -> None:
        block = {key: value for key, value in PINHOLE_BLOCK.items() if key != "model"}
        path = self.write({"cameras": {"camera1": block}})
        calibration = adapter.load_calibration(path, "camera1")
        self.assertEqual(calibration.model, "pinhole")
        self.assertEqual(calibration.model_source, "default")

    def test_non_json_file_is_reported_clearly(self) -> None:
        path = Path(tempfile.mkdtemp()) / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(adapter.CalibrationError) as caught:
            adapter.load_calibration(path, "camera1")
        self.assertIn("not valid JSON", str(caught.exception))


class RescaleTest(unittest.TestCase):
    def calibration(self) -> object:
        return adapter.CameraCalibration(
            name="camera1",
            model="pinhole",
            matrix=np.array(PINHOLE_BLOCK["K"], dtype=np.float64),
            distortion=np.array(PINHOLE_BLOCK["D"], dtype=np.float64),
            width=1280,
            height=720,
        )

    def test_matching_resolution_is_untouched(self) -> None:
        scaled, factor = adapter.rescale_matrix(self.calibration(), 1280, 720)
        self.assertEqual(factor, 1.0)
        self.assertTrue(np.array_equal(scaled.matrix, self.calibration().matrix))

    def test_half_resolution_scales_focal_length_and_principal_point(self) -> None:
        scaled, factor = adapter.rescale_matrix(self.calibration(), 640, 360)
        self.assertEqual(factor, 0.5)
        self.assertAlmostEqual(scaled.matrix[0, 0], 450.0)
        self.assertAlmostEqual(scaled.matrix[0, 2], 320.0)
        self.assertAlmostEqual(scaled.matrix[1, 2], 180.0)
        self.assertEqual((scaled.width, scaled.height), (640, 360))

    def test_aspect_change_is_refused(self) -> None:
        with self.assertRaises(adapter.CalibrationError):
            adapter.rescale_matrix(self.calibration(), 640, 480)


@unittest.skipUnless(has_cv2(), "OpenCV is not installed")
class GeometryTest(unittest.TestCase):
    def calibration(self, model: str, distortion: list[float]) -> object:
        return adapter.CameraCalibration(
            name="camera1",
            model=model,
            matrix=np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]]),
            distortion=np.array(distortion, dtype=np.float64),
            width=640,
            height=480,
        )

    def test_zero_distortion_pinhole_is_an_identity_map(self) -> None:
        rectification = adapter.build_rectification(self.calibration("pinhole", [0.0] * 5))
        ys, xs = np.mgrid[0:480, 0:640]
        self.assertTrue(np.allclose(rectification.map_x, xs, atol=1e-3))
        self.assertTrue(np.allclose(rectification.map_y, ys, atol=1e-3))
        self.assertAlmostEqual(rectification.in_frame_fraction, 1.0, places=3)

    def test_zero_distortion_fisheye_is_not_identity_but_keeps_the_centre(self) -> None:
        # An equidistant lens with zero coefficients still is not a pinhole, so
        # the dewarped frame is rescaled; only the optical centre must not move.
        rectification = adapter.build_rectification(self.calibration("fisheye", [0.0] * 4))
        self.assertAlmostEqual(rectification.map_x[240, 320], 320.0, places=3)
        self.assertAlmostEqual(rectification.map_y[240, 320], 240.0, places=3)
        self.assertGreater(rectification.in_frame_fraction, 0.99)
        self.assertNotAlmostEqual(rectification.new_matrix[0, 0], 300.0, places=1)

    def test_pinhole_alpha_trades_valid_area_for_field_of_view(self) -> None:
        calibration = self.calibration("pinhole", [-0.35, 0.12, 0.0, 0.0, -0.02])
        tight = adapter.build_rectification(calibration, alpha=0.0)
        loose = adapter.build_rectification(calibration, alpha=1.0)
        # alpha=0 keeps only pixels that exist in the source, so nothing is blank.
        self.assertGreater(tight.in_frame_fraction, loose.in_frame_fraction)
        self.assertAlmostEqual(tight.in_frame_fraction, 1.0, places=3)
        self.assertLess(loose.in_frame_fraction, 1.0)

    def test_fisheye_balance_trades_valid_area_for_field_of_view(self) -> None:
        calibration = self.calibration("fisheye", [0.2, -0.05, 0.01, -0.002])
        tight = adapter.build_rectification(calibration, alpha=0.0)
        loose = adapter.build_rectification(calibration, alpha=1.0)
        self.assertGreater(tight.in_frame_fraction, loose.in_frame_fraction)
        # balance=0 zooms in, so the focal length grows relative to the raw one.
        self.assertLess(tight.new_matrix[0, 0], 300.0)

    def test_a_wrong_model_folds_the_map(self) -> None:
        # k1 = -0.5 makes r -> r(1 + k1 r^2) turn over inside the frame, which is
        # what a mismatch between the lens and the coefficients looks like.
        rectification = adapter.build_rectification(
            self.calibration("pinhole", [-0.5, 0.0, 0.0, 0.0, 0.0]), alpha=1.0
        )
        self.assertFalse(rectification.monotonic)
        self.assertTrue(any("folds back" in warning for warning in rectification.warnings))

    def test_a_realistic_barrel_lens_is_not_flagged(self) -> None:
        rectification = adapter.build_rectification(
            self.calibration("pinhole", [-0.35, 0.12, 0.0, 0.0, -0.02]), alpha=1.0
        )
        self.assertTrue(rectification.monotonic, rectification.warnings)

    def test_scale_changes_the_output_resolution(self) -> None:
        rectification = adapter.build_rectification(
            self.calibration("pinhole", [0.0] * 5), scale=0.5
        )
        self.assertEqual((rectification.width, rectification.height), (320, 240))
        self.assertEqual(rectification.map_x.shape, (240, 320))

    def test_odd_output_size_is_shaved_to_even(self) -> None:
        rectification = adapter.build_rectification(
            self.calibration("pinhole", [0.0] * 5), output_size=(641, 481)
        )
        self.assertEqual((rectification.width, rectification.height), (640, 480))

    def test_crop_to_valid_shrinks_the_frame(self) -> None:
        calibration = self.calibration("pinhole", [-0.35, 0.12, 0.0, 0.0, -0.02])
        full = adapter.build_rectification(calibration, alpha=0.0)
        cropped = adapter.build_rectification(calibration, alpha=0.0, crop_to_valid=True)
        self.assertIsNotNone(cropped.crop)
        self.assertLessEqual(cropped.width * cropped.height, full.width * full.height)
        self.assertEqual(cropped.map_x.shape, (cropped.height, cropped.width))

    def test_alpha_outside_the_unit_interval_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            adapter.build_rectification(self.calibration("pinhole", [0.0] * 5), alpha=1.5)

    def test_output_size_and_scale_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            adapter.build_rectification(
                self.calibration("pinhole", [0.0] * 5), output_size=(320, 240), scale=0.5
            )


@unittest.skipUnless(has_cv2() and has_ffmpeg(), "needs OpenCV and ffmpeg")
class VideoRoundTripTest(unittest.TestCase):
    """A straight scene line must come back straight after dewarping."""

    width = 640
    height = 480
    matrix = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.22, -0.06, 0.012, -0.002], dtype=np.float64)
    line_row = 350

    def setUp(self) -> None:
        import cv2

        self.cv2 = cv2
        self.tmp = Path(tempfile.mkdtemp())
        ideal = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ideal[self.line_row - 1 : self.line_row + 2, :, :] = 255

        # Warp the ideal image the way the lens would: for every distorted
        # pixel, sample the ideal image at its undistorted position.
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        points = np.stack([xs, ys], axis=-1).astype(np.float64).reshape(-1, 1, 2)
        undistorted = cv2.fisheye.undistortPoints(
            points, self.matrix, self.distortion.reshape(4, 1), P=self.matrix
        )
        map_x = undistorted[..., 0].reshape(self.height, self.width).astype(np.float32)
        map_y = undistorted[..., 1].reshape(self.height, self.width).astype(np.float32)
        self.distorted = cv2.remap(
            ideal, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        self.episode = self.tmp / "episode_000001"
        (self.episode / "rgb").mkdir(parents=True)
        self.source = self.episode / "rgb" / "camera1.avi"
        self.write_video(self.source, self.distorted)

        self.calibration_path = self.tmp / "wrist.json"
        self.calibration_path.write_text(
            json.dumps(
                {
                    "cameras": {
                        "camera1": {
                            "model": "fisheye",
                            "image_size": [self.width, self.height],
                            "K": self.matrix.tolist(),
                            "D": self.distortion.tolist(),
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_video(self, path: Path, frame: np.ndarray, frames: int = 6) -> None:
        command = [
            shutil.which("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            "15",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        result = subprocess.run(command, input=frame.tobytes() * frames, capture_output=True)
        if result.returncode:
            self.skipTest(f"ffmpeg cannot encode h264 here: {result.stderr.decode()}")

    def read_first_frame(self, path: Path) -> np.ndarray:
        capture = self.cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok, f"cannot decode {path}")
        return frame

    def line_spread(self, frame: np.ndarray) -> tuple[float, int]:
        """Return how far the bright line wanders vertically, and over how many columns."""
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        rows = []
        for column in range(gray.shape[1]):
            strip = gray[:, column]
            if strip.max() < 128:
                continue
            rows.append(int(np.argmax(strip)))
        if len(rows) < 50:
            return float("inf"), len(rows)
        return float(np.ptp(rows)), len(rows)

    def test_report_and_video(self) -> None:
        report = adapter.process_episode(
            self.episode,
            calibration=adapter.load_calibration(self.calibration_path, "camera1"),
            calibration_path=self.calibration_path,
            camera="camera1",
            video_relative=None,
            output_relative="rgb/{camera}_undistorted.mp4",
            alpha=0.0,
            scale=1.0,
            output_size=None,
            crop_to_valid=False,
            codec="libx264",
            crf=12,
            preset="veryfast",
            min_in_frame=0.2,
            overwrite=True,
            dry_run=False,
        )
        self.assertEqual(report["status"], "PASS", report["warnings"])
        self.assertTrue(report["checks"]["frame_count_match"])
        self.assertEqual(report["output"]["frames"], 6)
        self.assertEqual(report["calibration"]["model"], "fisheye")

        output = self.episode / report["output"]["path"]
        self.assertTrue(output.is_file())
        self.assertTrue((self.episode / "rgb" / "camera1.avi").is_file())

        before, columns_before = self.line_spread(self.distorted)
        after, columns_after = self.line_spread(self.read_first_frame(output))
        self.assertGreater(columns_before, 400)
        self.assertGreater(columns_after, 200)
        self.assertGreater(before, 10.0, "the synthetic lens should bend the line")
        self.assertLess(after, before * 0.15, f"line still bent: {after:.1f}px vs {before:.1f}px")

    def test_dry_run_writes_nothing(self) -> None:
        report = adapter.process_episode(
            self.episode,
            calibration=adapter.load_calibration(self.calibration_path, "camera1"),
            calibration_path=self.calibration_path,
            camera="camera1",
            video_relative=None,
            output_relative="rgb/{camera}_undistorted.mp4",
            alpha=0.0,
            scale=1.0,
            output_size=None,
            crop_to_valid=False,
            codec="libx264",
            crf=12,
            preset="veryfast",
            min_in_frame=0.2,
            overwrite=True,
            dry_run=True,
        )
        self.assertEqual(report["status"], "DRY_RUN")
        self.assertFalse((self.episode / "rgb" / "camera1_undistorted.mp4").exists())

    def test_existing_output_is_not_replaced_without_overwrite(self) -> None:
        target = self.episode / "rgb" / "camera1_undistorted.mp4"
        target.write_bytes(b"sentinel")
        with self.assertRaises(RuntimeError) as caught:
            adapter.process_episode(
                self.episode,
                calibration=adapter.load_calibration(self.calibration_path, "camera1"),
                calibration_path=self.calibration_path,
                camera="camera1",
                video_relative=None,
                output_relative="rgb/{camera}_undistorted.mp4",
                alpha=0.0,
                scale=1.0,
                output_size=None,
                crop_to_valid=False,
                codec="libx264",
                crf=12,
                preset="veryfast",
                min_in_frame=0.2,
                overwrite=False,
                dry_run=False,
            )
        self.assertIn("--overwrite", str(caught.exception))
        self.assertEqual(target.read_bytes(), b"sentinel")

    def test_missing_video_is_a_clean_error(self) -> None:
        empty = self.tmp / "episode_000009"
        (empty / "rgb").mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            adapter.process_episode(
                empty,
                calibration=adapter.load_calibration(self.calibration_path, "camera1"),
                calibration_path=self.calibration_path,
                camera="camera1",
                video_relative=None,
                output_relative="rgb/{camera}_undistorted.mp4",
                alpha=0.0,
                scale=1.0,
                output_size=None,
                crop_to_valid=False,
                codec="libx264",
                crf=12,
                preset="veryfast",
                min_in_frame=0.2,
                overwrite=True,
                dry_run=False,
            )


class CliTest(unittest.TestCase):
    def test_size_parser(self) -> None:
        self.assertEqual(adapter._parse_size("1280x720"), (1280, 720))
        with self.assertRaises(argparse.ArgumentTypeError):
            adapter._parse_size("1280")
        with self.assertRaises(argparse.ArgumentTypeError):
            adapter._parse_size("0x720")

    def test_bad_calibration_exits_with_two(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            bad = tmp / "bad.json"
            bad.write_text(json.dumps({"cameras": {"other": PINHOLE_BLOCK}}), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER_PATH),
                    str(tmp),
                    "--calibration",
                    str(bad),
                    "--camera",
                    "camera1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("UNDISTORT_CONFIG_ERROR", result.stderr)
            self.assertIn("other", result.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
