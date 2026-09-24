"""Generator/contract checks before seeing any new reader challenge results."""
import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.readout_background_controls import generate, rotation_xyz  # noqa: E402


class BackgroundControlProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/readout_background_challenges_v1.json").read_text(encoding="utf-8"))
        cls.cases = list(generate(cls.config["generation"]))

    def test_predeclared_counts_and_finite_geometry(self):
        self.assertEqual(len(self.cases), 36)
        self.assertEqual(Counter(case[3]["expectation"] for case in self.cases), {"positive": 16, "negative": 8, "unidentifiable": 12})
        for points, truth, gap, labels in self.cases:
            self.assertEqual(points.shape[1:], (3,))
            self.assertEqual(truth.shape[1:], (2, 3))
            self.assertTrue(np.isfinite(points).all() and np.isfinite(truth).all())
            if labels["expectation"] == "positive":
                self.assertGreater(len(truth), 0)
            if gap is not None:
                self.assertGreater(np.linalg.norm(gap[1] - gap[0]), .2)

    def test_same_points_support_opposite_semantic_stories(self):
        grouped = {}
        for points, truth, _, labels in self.cases:
            pair = labels["ambiguity_pair"]
            if pair is not None:
                grouped.setdefault(pair, []).append((points, truth, labels["interpretation"]))
        self.assertEqual(len(grouped), 6)
        for records in grouped.values():
            self.assertEqual(len(records), 2)
            np.testing.assert_array_equal(records[0][0], records[1][0])
            self.assertNotEqual(records[0][2], records[1][2])
            self.assertNotEqual(len(records[0][1]), len(records[1][1]))

    def test_transforms_and_determinism_preserve_input_geometry(self):
        repeated = list(generate(self.config["generation"]))
        for first, second in zip(self.cases, repeated):
            np.testing.assert_array_equal(first[0], second[0])
        for placement in self.config["generation"]["placements"]:
            rotation = rotation_xyz(placement["euler_degrees"])
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_defaults_remain_unchanged_and_ambiguity_has_no_pass_rule(self):
        self.assertEqual(self.config["readout"], {})
        self.assertEqual(self.config["readers"], ["transverse_split", "robust_sections"])
        self.assertEqual(self.config["qualification"], {"positive_minimum_recovery": .9, "positive_minimum_precision": .9, "negative_maximum_predicted_length_m": 0.})
        self.assertIn("not included in pass/fail denominators", self.config["interpretation_limits"]["unidentifiable"])

    def test_inference_tripwire_blocks_truth_and_evaluation_paths(self):
        for directory in ("data/eval_gt", "data/evaluation"):
            code = "import sys; sys.path.insert(0, 'scripts'); from run_readout_background_challenges import reject_truth_open; sys.addaudithook(reject_truth_open); open(" + repr(str(ROOT / directory / "sentinel-does-not-exist.json")) + ")"
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PermissionError", result.stderr)
            self.assertIn("forbidden during readout inference", result.stderr)


if __name__ == "__main__":
    unittest.main()
