"""Check that the paired support experiment fails on invalid or mixed artifacts."""
import copy
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rod_candidate_ablation import canonical_hash
from run_rod_support_recheck import score_record, verify_final_support


class SupportRecheckTests(unittest.TestCase):
    def test_standalone_cli_imports_without_relying_on_another_runner(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_rod_support_recheck.py"
        completed = subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("prepare", completed.stdout)

    def test_final_support_gate_accepts_threshold_and_rejects_short_or_duplicate_rows(self):
        frame = {"observations": {"rows": [{"y": y} for y in (0, 50, 100, 150)]},
                 "pool": {"candidates": [{"support_rows": 4, "row_matches": [[i, 0, 0] for i in range(4)]}]}}
        config = {"minimum_rows": 4, "minimum_y_span": 150}
        self.assertEqual(verify_final_support([frame], config), 1)
        bad = copy.deepcopy(frame)
        bad["observations"]["rows"][-1]["y"] = 149
        with self.assertRaisesRegex(ValueError, "support contract"):
            verify_final_support([bad], config)
        bad = copy.deepcopy(frame)
        bad["pool"]["candidates"][0]["row_matches"][2] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "support contract"):
            verify_final_support([bad], config)

    def test_scoring_keeps_empty_output_as_zero_recovery_and_checks_shared_hash(self):
        association = {"state": "ambiguous", "selected": None, "search_complete": True,
                       "unique_hypothesis_count": 2, "attempted_combination_count": 10}
        sha = canonical_hash(association)
        record = {"case_id": "a", "search_offset_px": 0, "cap": 8,
                  "association": association, "association_sha256": sha, "association_seconds": 1,
                  "variants": [{"identity_offset_px": 0, "association_sha256": sha,
                                "guarded": {"state": "ambiguous"},
                                "finite": {"state": "rejected", "segments": [], "rejection_reasons": ["ambiguous"]}}]}
        truth = {"target": {"present": True, "endpoints": [[0, 0, 0], [0, 0, 1]]}}
        policy = {"curve_tolerance_m": .025, "curve_spacing_m": .002}
        row = score_record(record, truth, policy)[0]
        self.assertEqual(row["curve_metrics"]["recovery_fraction"], 0)
        self.assertIsNone(row["curve_metrics"]["precision_fraction"])
        record["variants"][0]["association_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "share"):
            score_record(record, truth, policy)


if __name__ == "__main__":
    unittest.main()
