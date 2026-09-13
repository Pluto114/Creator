"""Verify the scaffold cannot masquerade as completed reconstruction work."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"


def run_cli(workdir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "creator_recon", *args],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


class ScaffoldCliTests(unittest.TestCase):
    def test_status_distinguishes_scaffold_from_a_completed_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_cli(Path(temporary), "status")
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertEqual(status["implementation_status"], "scaffold")
        self.assertEqual(status["status_kind"], "installation_capabilities")
        self.assertEqual(status["reconstruction"], "not_implemented")
        self.assertNotIn("run_id", status)

    def test_unimplemented_actions_never_write_artifacts(self) -> None:
        commands = [("run",), ("case", "create")]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            request = directory / "request.json"
            request.write_text(
                json.dumps({"kind": "reconstruct", "output_dir": str(directory / "output")}),
                encoding="utf-8",
            )
            original = request.read_bytes()
            for command in commands:
                with self.subTest(command=command):
                    result = run_cli(directory, *command, "--request", str(request))
                    self.assertEqual(result.returncode, 3, result.stderr)
                    self.assertIn("E_NOT_IMPLEMENTED", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(request.read_bytes(), original)
                    self.assertEqual(list(directory.iterdir()), [request])

    def test_malformed_request_is_a_failure_without_side_effects(self) -> None:
        malformed = ['{"kind":', "[]", '{"x": NaN}', '{"x": 1, "x": 2}']
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            request = directory / "request.json"
            for content in malformed:
                with self.subTest(content=content):
                    request.write_text(content, encoding="utf-8")
                    result = run_cli(directory, "run", "--request", str(request))
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("E_INPUT", result.stderr)
                    self.assertEqual(list(directory.iterdir()), [request])


if __name__ == "__main__":
    unittest.main()
