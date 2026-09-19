"""Boundary and scoring checks for the procedural Blender identity run."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rod_identity_blender import classify, line_metrics, locations  # noqa: E402, I001


POLICY = {"line_distance_tolerance_world_m": 0.025, "angle_tolerance_deg": 2.0}


def accepted_line(x=0.0):
    return {
        "state": "accepted",
        "selected": {
            "model": {"anchor": [x, 0.0, 0.0], "direction": [0.0, 0.0, 1.0]}
        },
    }


def truth(*, present=True, ambiguous=False):
    return {
        "target": {
            "present": present,
            "identity_ambiguous": ambiguous,
            "endpoints": [[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]],
        }
    }


class RodIdentityRunnerTests(unittest.TestCase):
    def test_run_id_cannot_escape_workspace(self):
        for run_id in ("../escape", "D:/elsewhere", "", "."):
            with self.assertRaises(ValueError):
                locations(run_id)

    def test_empty_scene_acceptance_is_a_false_accept(self):
        label, metrics = classify(accepted_line(), truth(present=False), POLICY)
        self.assertEqual(label, "false_accept_empty")
        self.assertIsNone(metrics)

    def test_refusal_has_no_fake_zero_error(self):
        label, metrics = classify(
            {"state": "rejected", "selected": None}, truth(), POLICY
        )
        self.assertEqual(label, "target_refused")
        self.assertIsNone(metrics)

    def test_line_distance_and_identity_ambiguity(self):
        metrics = line_metrics(
            accepted_line(0.075)["selected"], truth()["target"]["endpoints"]
        )
        self.assertAlmostEqual(metrics["target_to_line_distance_p95_m"], 0.075)
        label, _ = classify(accepted_line(), truth(ambiguous=True), POLICY)
        self.assertEqual(label, "unsupported_identity_accept")
        label, _ = classify(
            {"state": "ambiguous", "selected": None}, truth(ambiguous=True), POLICY
        )
        self.assertEqual(label, "safe_identity_ambiguity")


if __name__ == "__main__":
    unittest.main()
