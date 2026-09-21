"""Entry-point and evaluator isolation checks for foreground identity experiments."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_rod_foreground_identity import query_truth  # noqa: E402


class ForegroundRunnerTests(unittest.TestCase):
    def test_cli_starts_in_a_fresh_process(self):
        for filename in ("run_rod_foreground_identity.py", "audit_rod_foreground_identity.py"):
            with self.subTest(filename=filename):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / filename), "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--help", result.stdout)

    def test_alternate_query_uses_the_requested_object_not_the_default_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = [[0, 0, -1], [0, 0, 1]], [[.2, 0, -1], [.2, 0, 1]]
            (root / "geometry.json").write_text(json.dumps({"objects": [
                {"surface_id": 1, "world_centerline_endpoints": first},
                {"surface_id": 2, "world_centerline_endpoints": second}]}), encoding="utf-8")
            original = {"target": {"present": True, "endpoints": first}, "geometry_path": "geometry.json", "gap_segment": first}
            before = copy.deepcopy(original)
            result = query_truth({"target_selector": 2}, original, root)
            self.assertEqual(result["target"]["endpoints"], second)
            self.assertIsNone(result["gap_segment"])
            self.assertEqual(original, before)
            for invalid in (True, "2", 3):
                with self.subTest(selector=invalid), self.assertRaises(ValueError):
                    query_truth({"target_selector": invalid}, original, root)

    def test_declared_and_empty_queries_do_not_read_geometry(self):
        target = {"target": {"present": True, "endpoints": [[0, 0, 0], [0, 0, 1]]}, "gap_segment": [[0, 0, .4], [0, 0, .6]]}
        with patch("run_rod_foreground_identity.read_json", side_effect=AssertionError("unexpected truth read")):
            self.assertEqual(query_truth({"target_selector": "declared"}, target, Path("absent")), target)
            empty = query_truth({"target_selector": "empty"}, target, Path("absent"))
            self.assertFalse(empty["target"]["present"])
            self.assertIsNone(empty["gap_segment"])


if __name__ == "__main__":
    unittest.main()
