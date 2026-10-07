"""Mock-only contracts for transparent CI execution and public diagnostics."""

import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_ci_unittests as runner  # noqa: E402


class CiUnittestRunnerTests(unittest.TestCase):
    def test_all_successful_patterns_execute_once_in_original_order(self):
        patterns = ["test_one*.py", "test_two.py", "test_three.py"]
        result = SimpleNamespace(returncode=0, stdout=b"normal\n", stderr=b"test ... ok\n")
        out, err = io.StringIO(), io.StringIO()
        with patch.object(runner.subprocess, "run", return_value=result) as execute, \
                patch.object(runner.sys, "stdout", out), patch.object(runner.sys, "stderr", err):
            self.assertEqual(runner.run_patterns(patterns, Path("checkout")), 0)
        self.assertEqual(execute.call_count, 3)
        for call, pattern in zip(execute.call_args_list, patterns):
            self.assertEqual(call.args[0], [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"])
            self.assertEqual(call.kwargs, dict(cwd=Path("checkout"), capture_output=True, check=False))
        self.assertEqual(out.getvalue().count("normal\n"), 3)
        self.assertEqual(err.getvalue(), "test ... ok\n"*3)
        self.assertNotIn("::error::", out.getvalue())

    def test_first_nonzero_is_returned_without_retry_or_later_patterns(self):
        success = SimpleNamespace(returncode=0, stdout=b"", stderr=b"OK\n")
        failed = SimpleNamespace(returncode=7, stdout=b"original stdout\n", stderr=b"original stderr\nFAILED (errors=1)\n")
        out, err = io.StringIO(), io.StringIO()
        with patch.object(runner.subprocess, "run", side_effect=[success, failed]) as execute, \
                patch.object(runner.sys, "stdout", out), patch.object(runner.sys, "stderr", err):
            self.assertEqual(runner.run_patterns(["first.py", "failed.py", "never.py"]), 7)
        self.assertEqual(execute.call_count, 2)
        self.assertIn("original stdout\n", out.getvalue())
        self.assertEqual(err.getvalue(), "OK\noriginal stderr\nFAILED (errors=1)\n")
        self.assertIn("::error::unittest pattern=failed.py; exit=7%0A", out.getvalue())

    def test_annotation_escapes_percent_cr_and_lf_without_command_injection(self):
        self.assertEqual(runner.annotation_escape("10%\r\n::warning::x"), "10%25%0D%0A::warning::x")

    def test_annotation_contains_last_traceback_and_failed_summary(self):
        trace = b"old noise\nTraceback (most recent call last):\n  File test.py, line 2\nValueError: 50% mismatch\n\nFAILED (errors=1)\n"
        result = SimpleNamespace(returncode=1, stdout=b"", stderr=trace)
        out, err = io.StringIO(), io.StringIO()
        with patch.object(runner.subprocess, "run", return_value=result), \
                patch.object(runner.sys, "stdout", out), patch.object(runner.sys, "stderr", err):
            self.assertEqual(runner.run_patterns(["test_bad.py"]), 1)
        annotation = out.getvalue().split("::error::", 1)[1]
        self.assertIn("Traceback (most recent call last):%0A", annotation)
        self.assertIn("ValueError: 50%25 mismatch", annotation)
        self.assertIn("FAILED (errors=1)", annotation)
        self.assertEqual(err.getvalue(), trace.decode())


if __name__ == "__main__":
    unittest.main()
