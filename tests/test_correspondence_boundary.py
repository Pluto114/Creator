"""Exercise the track inference read tripwire in a fresh interpreter."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CorrespondenceBoundaryTests(unittest.TestCase):
    def check_block(self, relative):
        # A subprocess keeps the permanent Python audit hook out of other tests.
        code = "\n".join([
            "import sys", "from pathlib import Path", f"sys.path.insert(0, {str(ROOT / 'scripts')!r})",
            "from run_foreground_tracks import reject_model_open", "sys.addaudithook(reject_model_open)",
            "try:", f"    Path({str(ROOT / relative)!r}).read_bytes()", "except PermissionError:", "    pass",
            "else:", "    raise AssertionError('forbidden read reached the filesystem')",
        ])
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_truth_and_evaluation_reads_are_blocked(self):
        for path in ("data/eval_gt/no-such-artifact.json", "data/evaluation/no-such-artifact.json"):
            with self.subTest(path=path):
                self.check_block(path)

    def test_model_geometry_reads_are_blocked(self):
        for path in (".runtime/nonexistent/prediction.npz", ".runtime/nonexistent/geometry.json"):
            with self.subTest(path=path):
                self.check_block(path)


if __name__ == "__main__":
    unittest.main()
