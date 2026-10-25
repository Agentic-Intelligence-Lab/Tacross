from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from postprocess_pipeline.pipeline import (
    ConfigError,
    expand_text,
    resolve_variables,
)

PROJECT = Path(__file__).resolve().parents[2]
PIPELINE = PROJECT / "postprocess_pipeline/pipeline.py"
TOOLS = PROJECT / "postprocess_pipeline/tools"
WRITE_ARG = "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"


def run_pipeline(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PIPELINE), "--config", str(config), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


class ExpansionTests(unittest.TestCase):
    def test_only_identifier_placeholders_are_expanded(self) -> None:
        variables = {"root": "/data"}
        self.assertEqual(expand_text("{root}/raw", variables), "/data/raw")
        # Non-identifier braces are literal: a JSON snippet, a Windows path with
        # a version directory, a shell-style parameter expansion.
        self.assertEqual(expand_text('{"a": 1} in {root}', variables), '{"a": 1} in /data')
        self.assertEqual(expand_text("{2024}/log", variables), "{2024}/log")

    def test_unknown_variable_is_an_error(self) -> None:
        # A typo in a variable name must not silently become a literal path.
        with self.assertRaises(ConfigError):
            expand_text("{missing}/x", {})


    def test_cyclic_variables_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "c.json"
            with self.assertRaises(ConfigError) as context:
                resolve_variables({"a": "{b}", "b": "{a}"}, config)
            self.assertIn("placeholder", str(context.exception))


class StageRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "work").mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_config(self, stages: list[dict], **extra: object) -> Path:
        payload = {
            "schema_version": 1,
            "name": "test",
            "variables": {"root": "{config_dir}/work"},
            "run_root": "{root}/runs",
            "stages": stages,
            **extra,
        }
        path = self.root / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_cold_start_chains_stage_outputs(self) -> None:
        """A later stage may require an artifact an earlier stage produces.

        Regression: the up-front preflight used to abort the run before the
        producing stage had a chance to execute.
        """
        config = self.write_config(
            [
                {
                    "name": "first",
                    "cwd": "{root}",
                    "requires": ["{root}"],
                    "produces": ["{root}/a.txt"],
                    "command": [sys.executable, "-c", WRITE_ARG, "{root}/a.txt", "a"],
                },
                {
                    "name": "second",
                    "cwd": "{root}",
                    "requires": ["{root}/a.txt"],
                    "produces": ["{root}/b.txt"],
                    "command": [sys.executable, "-c", WRITE_ARG, "{root}/b.txt", "b"],
                },
            ]
        )
        result = run_pipeline(config, "--execute")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue((self.root / "work/a.txt").is_file())
        self.assertTrue((self.root / "work/b.txt").is_file())

    def test_missing_requirement_aborts_before_running(self) -> None:
        config = self.write_config(
            [
                {
                    "name": "needs_input",
                    "cwd": "{root}",
                    "requires": ["{root}/absent.txt"],
                    "command": [sys.executable, "-c", "print('should not run')"],
                }
            ]
        )
        result = run_pipeline(config, "--execute")
        self.assertEqual(result.returncode, 2)
        self.assertIn("FAIL: required paths are missing", result.stderr)

    def test_dry_run_writes_no_run_directory(self) -> None:
        config = self.write_config(
            [
                {
                    "name": "noop",
                    "cwd": "{root}",
                    "command": [sys.executable, "-c", "print('x')"],
                }
            ]
        )
        result = run_pipeline(config)
        self.assertEqual(result.returncode, 0)
        self.assertIn("DRY-RUN", result.stdout)
        self.assertFalse((self.root / "work/runs").exists())

    def test_resume_skips_stage_with_existing_outputs(self) -> None:
        marker = self.root / "work/marker.txt"
        marker.write_text("pre-existing", encoding="utf-8")
        config = self.write_config(
            [
                {
                    "name": "already_done",
                    "cwd": "{root}",
                    "produces": ["{root}/marker.txt"],
                    "command": [sys.executable, "-c", WRITE_ARG, "{root}/marker.txt", "ran"],
                }
            ]
        )
        result = run_pipeline(config, "--resume", "--execute")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SKIP: all declared outputs already exist", result.stdout)
        report = json.loads(
            next((self.root / "work/runs").glob("*/provenance.json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["stages"][0]["status"], "SKIPPED_OUTPUTS_EXIST")
        self.assertEqual(marker.read_text(encoding="utf-8"), "pre-existing")

    def test_provenance_records_argv_and_variables(self) -> None:
        config = self.write_config(
            [
                {
                    "name": "noop",
                    "cwd": "{root}",
                    "command": [sys.executable, "-c", "print('x')"],
                }
            ]
        )
        result = run_pipeline(config, "--execute")
        self.assertEqual(result.returncode, 0, result.stderr)
        reports = list((self.root / "work/runs").glob("*/provenance.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["stages"][0]["commands"][0]["returncode"], 0)
        self.assertIn("config_sha256", report)
        self.assertIn("duration_s", report["stages"][0])

    def test_accepted_nonzero_returncode_is_not_a_failure(self) -> None:
        config = self.write_config(
            [
                {
                    "name": "partial",
                    "cwd": "{root}",
                    "accept_returncodes": [0, 3],
                    "command": [sys.executable, "-c", "raise SystemExit(3)"],
                }
            ]
        )
        result = run_pipeline(config, "--execute")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(
            next((self.root / "work/runs").glob("*/provenance.json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["stages"][0]["status"], "PASS")
        self.assertTrue(report["stages"][0]["commands"][0]["accepted_nonzero_returncode"])



class ToolTests(unittest.TestCase):
    def test_run_with_manifest_keeps_wrapped_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "m.json"
            manifest.write_text(
                json.dumps({"valid_episodes": ["episode_000001", "episode_000002"]}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "run_with_manifest.py"),
                    "--manifest",
                    str(manifest),
                    "--mode",
                    "csv",
                    "--target-flag=--episodes",
                    "--dry-run",
                    "--",
                    "echo",
                    "--not-our-flag",
                    "value",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--not-our-flag value", result.stdout)
            self.assertIn("--episodes episode_000001,episode_000002", result.stdout)

    def test_audit_requires_every_stage_and_a_passing_ik_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            ik_root = root / "ik"
            for name in ("episode_000001", "episode_000002"):
                sync = source / name / "sync"
                sync.mkdir(parents=True)
                (sync / "timeline.csv").write_text("a\n1\n2\n", encoding="utf-8")
                (sync / "stage.csv").write_text("a\n1\n2\n", encoding="utf-8")
                (sync / "fusion.json").write_text(json.dumps({"status": "PASS"}))
                report_dir = ik_root / name
                report_dir.mkdir(parents=True)
                status = "PASS" if name.endswith("1") else "FAIL"
                (report_dir / "offline_ik_report.json").write_text(
                    json.dumps({"status": status})
                )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"valid_episodes": ["episode_000001", "episode_000002"]})
            )
            output = root / "audit.json"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "audit_episode_artifacts.py"),
                    "--manifest",
                    str(manifest),
                    "--source-root",
                    str(source),
                    "--ik-root",
                    str(ik_root),
                    "--output",
                    str(output),
                    "--timeline-relative",
                    "sync/timeline.csv",
                    "--stage",
                    "hand=sync/stage.csv",
                    "--fusion-metadata-relative",
                    "sync/fusion.json",
                ],
                check=True,
                capture_output=True,
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["training_ready_episodes"], ["episode_000001"])
            self.assertEqual(payload["rejected_episodes"], ["episode_000002"])

    def test_gripper_prior_is_bounded_by_demonstrated_feedback(self) -> None:
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chunk = root / "robot/data/chunk-000"
            chunk.mkdir(parents=True)
            for index in range(2):
                frames = 60
                closure = np.linspace(0.0, 1.0, frames)
                states = np.column_stack(
                    [np.zeros((frames, 7)), 0.15 + 0.30 * closure]
                ).astype(np.float32)
                pd.DataFrame(
                    {
                        "timestamp": np.arange(frames) / 30.0,
                        "observation.state": list(states),
                    }
                ).to_parquet(chunk / f"episode_{index:06d}.parquet", index=False)
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "build_robot_gripper_prior.py"),
                    str(root / "robot"),
                    "--output-dir",
                    str(root / "out"),
                    "--output-prefix",
                    "prior",
                ],
                check=True,
                capture_output=True,
            )
            payload = json.loads((root / "out/prior.json").read_text())
            anchors = payload["fusion_anchors"]
            self.assertLess(anchors["open_typical"], anchors["closed_typical"])
            self.assertGreaterEqual(anchors["hard_min"], 0.149)
            self.assertLessEqual(anchors["hard_max"], 0.451)
            self.assertTrue((root / "out/prior.csv").is_file())


if __name__ == "__main__":
    unittest.main()
