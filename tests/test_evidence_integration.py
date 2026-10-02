"""Independent source-contract tests for surface/ridge integration, not scored cases."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import common_readout_evidence as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402

EMPTY = np.empty((0, 2, 3))
LINE = np.array([[[.013, .017, .019], [.013, .017, 1.019]]])


def line_points():
    return sample_segments(LINE, .005, 1000)


class EvidenceIntegrationTests(unittest.TestCase):
    def test_surface_only_early_return_obeys_total_output_budget(self):
        surface = np.r_[LINE, LINE + [.04, 0, 0]]

        def analyze(points, train_mask, config):
            return surface, np.ones(len(points), dtype=bool), {"state": "complete"}

        with patch.object(reader.section_evidence, "analyze", side_effect=analyze):
            result = reader.readout(line_points(), EMPTY, {"maximum_output_segments": 1})
        self.assertEqual(result["state"], "unmeasurable")
        self.assertIn("budget_exceeded", result["reason"])
        self.assertEqual(len(result["segments"]), 0)

    def test_combined_surface_and_ridge_obey_total_output_budget(self):
        def analyze(points, train_mask, config):
            return LINE + [.04, 0, 0], np.zeros(len(points), dtype=bool), {"state": "complete"}

        with patch.object(reader.section_evidence, "analyze", side_effect=analyze):
            result = reader.readout(line_points(), EMPTY, {"maximum_output_segments": 1})
        self.assertEqual(result["state"], "unmeasurable")
        self.assertIn("budget_exceeded", result["reason"])
        self.assertEqual(len(result["segments"]), 0)

    def test_unmeasurable_section_is_propagated_without_ridge_fallback(self):
        def analyze(points, train_mask, config):
            return EMPTY, np.zeros(len(points), dtype=bool), {
                "state": "unmeasurable", "reason": "section_group_budget_exceeded",
            }

        with patch.object(reader.section_evidence, "analyze", side_effect=analyze):
            with patch.object(reader, "_fit_ridge", side_effect=AssertionError("unexpected fallback")):
                result = reader.readout(line_points(), EMPTY, {})
        self.assertEqual(result["state"], "unmeasurable")
        self.assertEqual(result["reason"], "section_group_budget_exceeded")
        self.assertEqual(len(result["segments"]), 0)

    def test_accepted_and_unresolved_support_is_excluded_without_swallowing_other_ridge(self):
        cloud = np.r_[line_points(), line_points() + [.08, 0, 0]]
        original_fit = reader._fit_ridge
        original_runs = reader._ridge_runs

        def fit(train, *args):
            np.testing.assert_allclose(train[:, 0], .093, atol=1e-12)
            return original_fit(train, *args)

        def runs(points, *args):
            np.testing.assert_allclose(points[:, 0], .093, atol=1e-12)
            return original_runs(points, *args)

        for accepted in (True, False):
            with self.subTest(accepted=accepted):
                def analyze(points, train_mask, config):
                    consumed = points[:, 0] < .05
                    state = "accepted_surface" if accepted else "surface_unresolved"
                    return LINE if accepted else EMPTY, consumed, {"state": "complete", "groups": [{"state": state}]}

                with patch.object(reader.section_evidence, "analyze", side_effect=analyze):
                    with patch.object(reader, "_fit_ridge", side_effect=fit):
                        with patch.object(reader, "_ridge_runs", side_effect=runs):
                            result = reader.readout(cloud, EMPTY, {})
                self.assertEqual(result["components"], 2 if accepted else 1)
                self.assertEqual(result["resolution_state"], "resolved" if accepted else "partially_resolved")
                self.assertEqual(result["surface_consumed_points"], len(line_points()))
                self.assertTrue(any(np.allclose(segment[:, 0], .093) for segment in result["segments"]))

    def test_surface_only_gap_is_not_merged_or_filled_on_early_return(self):
        surface = np.array([[[.013, .017, .019], [.013, .017, .419]],
                            [[.013, .017, .719], [.013, .017, 1.019]]])

        def analyze(points, train_mask, config):
            return surface, np.ones(len(points), dtype=bool), {"state": "complete"}

        with patch.object(reader.section_evidence, "analyze", side_effect=analyze):
            result = reader.readout(line_points(), EMPTY, {})
        np.testing.assert_array_equal(result["segments"], surface)
        self.assertEqual(result["components"], 2)

    def test_real_surface_and_disjoint_ridge_both_survive_integration(self):
        theta, z = np.meshgrid(np.arange(32) * (2 * np.pi / 32), np.linspace(.019, 1.019, 201))
        shell = np.c_[.02 * np.cos(theta.ravel()), .02 * np.sin(theta.ravel()), z.ravel()]
        ridge = np.c_[np.full(201, .08), np.zeros(201), np.linspace(.019, 1.019, 201)]
        original = np.r_[shell, ridge]
        result = reader.readout(original, EMPTY, {})
        self.assertEqual(result["section_evidence"]["accepted_surface_groups"], 1)
        self.assertEqual(result["components"], 2, result["segments"])
        centers = np.sort(result["segments"].mean(axis=1)[:, 0])
        np.testing.assert_allclose(centers, [0., .08], atol=.001)
        self.assertEqual(result["surface_consumed_points"], len(shell))
        np.testing.assert_array_equal(original, np.r_[shell, ridge])


if __name__ == "__main__":
    unittest.main()
