"""Contracts and numerical checks; no classification thresholds tuned on the new suite."""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.section_model_controls import generate  # noqa: E402
from creator_eval.section_model_diagnostic import (  # noqa: E402
    DEFAULTS,
    diagnose,
    ellipse_distances,
    fit_model,
    summarize,
    two_circle_distances,
)


class SectionModelDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "configs/section_model_diagnostic_v1.json").read_text(encoding="utf-8"))
        cls.cases, cls.labels = generate(cls.protocol)

    def test_protocol_is_balanced_and_labels_stay_out_of_inputs(self):
        self.assertEqual(len(self.cases), 64)
        self.assertEqual(Counter(r["family"] for r in self.labels), {name:8 for name in self.protocol["generation"]["families"]})
        self.assertEqual(self.protocol["method"], DEFAULTS)
        for case in self.cases:
            self.assertEqual(set(case), {"case_id", "points", "voxel_size"})
        for label in self.labels:
            self.assertTrue(label["missing_evidence_is_not_free_space"])

    def test_phases_are_exactly_paired_translations_of_the_same_noise(self):
        g = self.protocol["generation"]
        delta = (np.array(g["phases_voxels"][1])-g["phases_voxels"][0])*g["voxel_m"]
        for first, second in zip(self.cases[::2], self.cases[1::2]):
            np.testing.assert_allclose(second["points"]-first["points"], np.broadcast_to(delta, first["points"].shape), atol=3e-16, rtol=0.)
        for label in self.labels:
            matched = [r for r in self.labels if r["family"]==label["family"]]
            self.assertEqual(len({r["paired_noise_seed"] for r in matched}), 1)

    def test_deterministic_generation_repeats_every_point(self):
        again, labels = generate(self.protocol)
        self.assertEqual(labels, self.labels)
        for first, second in zip(self.cases, again):
            np.testing.assert_array_equal(first["points"], second["points"])

    def test_ellipse_distance_matches_circle_and_known_surface(self):
        rng = np.random.default_rng(314)
        points = rng.uniform(-5, 5, (300, 2))
        parameters = np.array([.3, -.2, np.log(2.), np.log(2.), .7])
        np.testing.assert_allclose(ellipse_distances(points, parameters), np.linalg.norm(points-parameters[:2], axis=1)-2., atol=1e-10)
        t = np.linspace(0, 2*np.pi, 137, endpoint=False)
        surface = np.c_[3*np.cos(t), np.sin(t)]
        np.testing.assert_allclose(ellipse_distances(surface, [0, 0, np.log(3.), 0., 0.]), 0., atol=1e-10)

    def test_dual_circle_residual_and_line_fit_have_known_solutions(self):
        t = np.linspace(0, 2*np.pi, 100, endpoint=False)
        points = np.concatenate([np.c_[np.cos(t)+x, np.sin(t)] for x in (-2., 2.)])
        np.testing.assert_allclose(two_circle_distances(points, np.array([-2., 0., 0., 2., 0., 0.])), 0., atol=1e-12)
        line = np.c_[np.linspace(-4, 4, 50), np.full(50, .7)]
        fit = fit_model("line", line, np.ones(50), 10., DEFAULTS)
        self.assertEqual(fit["parameter_count"], 2)
        np.testing.assert_allclose((line-np.array(fit["parameters"][:2]))@np.array(fit["parameters"][2:]), 0., atol=1e-12)

    def test_weighted_summary_does_not_hide_large_holdout_residuals(self):
        score = summarize(np.array([0., 1., 3.]), np.array([1., 1., 2.]))
        self.assertAlmostEqual(score["rmse_voxels"], np.sqrt(19/4))
        self.assertEqual(score["p95_voxels"], 3.)

    def test_fold_results_are_explicit_and_never_emit_or_accept_a_model(self):
        # Small mathematical fixture, distinct from the 64 research conditions.
        t = np.linspace(0, 2*np.pi, 16, endpoint=False)
        cloud = np.concatenate([np.c_[.06*np.cos(t), .04*np.sin(t), np.full(16, z)] for z in np.linspace(0, 1., 24)])
        result = diagnose(cloud, .02, {"maximum_nfev": 2})
        self.assertFalse(result["emits_axis"])
        self.assertNotIn("segments", result)
        self.assertEqual(len(result["representations"]), 2)
        for representation in result["representations"]:
            self.assertEqual(len(representation["rows"]), 6)
            for row in representation["rows"]:
                self.assertFalse(set(row["train_slices"]) & set(row["test_slices"]))
                self.assertEqual(sorted(row["train_slices"]+row["test_slices"]), list(range(6)))
                self.assertEqual(row["fit"]["state"], "fitted")
                self.assertEqual(len(row["heldout"]), 3)
                json.dumps(row, allow_nan=False)

    def test_invalid_or_insufficient_point_input_is_not_silently_scored(self):
        for cloud in (np.empty((0, 3)), np.full((12, 3), np.nan)):
            with self.assertRaises(ValueError):
                diagnose(cloud, .02)


if __name__ == "__main__":
    unittest.main()
