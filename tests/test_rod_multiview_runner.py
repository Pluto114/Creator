"""Boundary and scoring tests for the frozen multi-view control runner."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rod_multiview_candidates import (  # noqa: E402, I001
    classify,
    line_metrics,
    locations,
    locations_from_run_id,
)


POLICY = {"line_distance_tolerance_world": 0.03, "angle_tolerance_deg": 2.0}


def truth(*, present=True, ambiguous=False):
    return {
        "target": {"present": present, "identity_ambiguous": ambiguous},
        "target_endpoints_world": [[0.0, -1.0, 6.0], [0.0, 1.0, 6.0]],
    }


def accepted_line(x=0):
    return {
        "state": "accepted",
        "selected": {"model": {"anchor": [x, 0, 6], "direction": [0, 1, 0]}},
    }


class MultiViewRunnerTests(unittest.TestCase):
    def test_run_id_cannot_escape_workspace(self):
        for run_id in ("../escape", "D:/elsewhere", "", "."):
            with self.assertRaises(ValueError):
                locations({"run_id": run_id})
            with self.assertRaises(ValueError):
                locations_from_run_id(run_id)

    def test_empty_scene_acceptance_is_false_accept_not_nearest_truth(self):
        label, metrics = classify(accepted_line(), truth(present=False), POLICY)
        self.assertEqual(label, "false_accept_empty")
        self.assertIsNone(metrics)

    def test_rejection_is_not_scored_as_zero_error(self):
        label, metrics = classify(
            {"state": "rejected", "selected": None}, truth(), POLICY
        )
        self.assertEqual(label, "target_refused")
        self.assertIsNone(metrics)

    def test_line_distance_and_ambiguous_identity_semantics(self):
        metrics = line_metrics(
            accepted_line(.075)["selected"],
            np.array(truth()["target_endpoints_world"], dtype=float),
        )
        self.assertAlmostEqual(metrics["target_to_line_distance_p95"], .075)
        label, scored = classify(accepted_line(), truth(ambiguous=True), POLICY)
        self.assertEqual(label, "unsupported_identity_accept")
        self.assertIsNotNone(scored)


if __name__ == "__main__":
    unittest.main()
