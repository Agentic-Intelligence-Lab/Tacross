from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from postprocess_pipeline.pipeline import parse_overrides, resolve_variables

PROJECT = Path(__file__).resolve().parents[2]
TOOLS = PROJECT / "postprocess_pipeline/tools"


class PipelineTests(unittest.TestCase):
    def test_variable_override(self) -> None:
        config = PROJECT / "postprocess_pipeline/configs/ego_hamer_stereo_ik.template.json"
        values = resolve_variables(
            {"root": "{config_dir}/data", "source": "{root}/raw"},
            config,
            parse_overrides(["root=/tmp/public-test"]),
        )
        self.assertEqual(values["source"], "/tmp/public-test/raw")

    def test_manifest_preserves_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for number in (1, 3):
                episode = root / f"episode_{number:06d}"
                episode.mkdir()
                (episode / "ready.txt").write_text("ok")
            output = root / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "build_episode_manifest.py"),
                    str(root),
                    "--output",
                    str(output),
                    "--first-episode",
                    "1",
                    "--last-episode",
                    "3",
                    "--required",
                    "ready.txt",
                ],
                check=True,
            )
            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["valid_episodes"],
                ["episode_000001", "episode_000003"],
            )
            self.assertEqual(
                payload["excluded_episodes"][0]["episode"],
                "episode_000002",
            )

    def test_filter_uses_explicit_json_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["episode_000001", "episode_000002"]
            for index, name in enumerate(names):
                sync = root / name / "sync"
                sync.mkdir(parents=True)
                (sync / "artifact.csv").write_text("x\n1\n")
                (sync / "status.json").write_text(
                    json.dumps({"status": "PASS" if index == 0 else "FAIL"})
                )
            source_manifest = root / "input.json"
            source_manifest.write_text(json.dumps({"valid_episodes": names}))
            output = root / "output.json"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "filter_episode_manifest.py"),
                    "--input",
                    str(source_manifest),
                    "--source-root",
                    str(root),
                    "--output",
                    str(output),
                    "--required",
                    "sync/artifact.csv",
                    "--json-equals",
                    "sync/status.json:status=PASS",
                ],
                check=True,
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["valid_episodes"], ["episode_000001"])


if __name__ == "__main__":
    unittest.main()
