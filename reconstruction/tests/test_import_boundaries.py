"""The independent core must remain importable without host or model runtimes."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class ImportBoundaryTests(unittest.TestCase):
    def test_importing_all_core_modules_does_not_load_external_runtimes(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        script = """
import importlib
import pkgutil
import sys
import creator_recon
for module in pkgutil.walk_packages(creator_recon.__path__, creator_recon.__name__ + "."):
    importlib.import_module(module.name)
for forbidden in ("bpy", "torch", "depth_anything_3", "creator_eval"):
    assert forbidden not in sys.modules, forbidden
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
