"""Independent training-axis sampling scale contracts; no scored inputs or truth files."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import common_readout_context as frozen  # noqa: E402
from creator_eval.ridge_sampling_evidence import (  # noqa: E402
    DEFAULTS,
    policy,
    train_scale,
)

FALLBACK_GAP = frozen.policy({})["neighbor_radius_voxels"] * frozen.policy({})["voxel_size"]


class RidgeSamplingEvidenceTests(unittest.TestCase):
    def test_defaults_preserve_frozen_ridge_voxel_fallback(self):
        self.assertEqual(policy({})["voxel_size"], .01)
        self.assertEqual(policy({})["voxel_size"], frozen.policy({})["voxel_size"])
        self.assertEqual(policy({})["neighbor_radius_voxels"], frozen.policy({})["neighbor_radius_voxels"])
        for key in DEFAULTS:
            if key in ("voxel_size", "neighbor_radius_voxels"):
                continue
            self.assertTrue(key.startswith("ridge_sampling_"), key)
        self.assertEqual(DEFAULTS["ridge_sampling_layer_merge_voxels"], .05)
        self.assertEqual(DEFAULTS["ridge_sampling_minimum_layers"], 6)
        self.assertEqual(DEFAULTS["ridge_sampling_lower_quantile"], .25)
        self.assertEqual(DEFAULTS["ridge_sampling_cluster_relative_tolerance"], .2)
        self.assertEqual(DEFAULTS["ridge_sampling_minimum_repeats"], 3)
        self.assertEqual(DEFAULTS["ridge_sampling_minimum_repeat_fraction"], .2)
        self.assertEqual(DEFAULTS["ridge_sampling_minimum_lattice_fraction"], .8)
        self.assertEqual(DEFAULTS["ridge_sampling_lattice_relative_tolerance"], .15)
        self.assertEqual(DEFAULTS["ridge_sampling_maximum_pitch_voxels"], 8.)
        self.assertEqual(DEFAULTS["ridge_sampling_gap_factor"], 1.75)
        self.assertEqual(DEFAULTS["ridge_sampling_maximum_points"], 200000)
        self.assertEqual(DEFAULTS["ridge_sampling_maximum_layers"], 20000)
        p = policy({})
        self.assertEqual(p["voxel_size"], .01)
        override = policy({"voxel_size": .02})
        self.assertEqual(override["voxel_size"], .02)
        self.assertEqual(override["ridge_sampling_gap_factor"], DEFAULTS["ridge_sampling_gap_factor"])

    def test_all_legal_policy_keys_are_accepted(self):
        legal = dict(ridge_sampling_layer_merge_voxels=.05, ridge_sampling_minimum_layers=6,
            ridge_sampling_lower_quantile=.25, ridge_sampling_cluster_relative_tolerance=.2,
            ridge_sampling_minimum_repeats=3, ridge_sampling_minimum_repeat_fraction=.2,
            ridge_sampling_minimum_lattice_fraction=.8, ridge_sampling_lattice_relative_tolerance=.15,
            ridge_sampling_maximum_pitch_voxels=8., ridge_sampling_gap_factor=1.75,
            ridge_sampling_maximum_points=200000, ridge_sampling_maximum_layers=20000)
        p = policy(legal)
        for key, value in legal.items():
            self.assertEqual(p[key], value)

    def test_policy_rejects_unknown_nonpositive_nonfinite_boolean_and_bad_budgets(self):
        for config in ({"ridge_sampling_bad": 1}, {"ridge_sampling_maximum_points_typo": 100},
                {"voxel_size": 0.}, {"voxel_size": -.01}, {"voxel_size": float("nan")},
                {"voxel_size": True}, {"neighbor_radius_voxels": 0.}, {"neighbor_radius_voxels": False},
                {"ridge_sampling_layer_merge_voxels": 0.}, {"ridge_sampling_gap_factor": 0.},
                {"ridge_sampling_gap_factor": float("inf")}, {"ridge_sampling_maximum_pitch_voxels": 0.},
                {"ridge_sampling_maximum_pitch_voxels": float("nan")},
                {"ridge_sampling_minimum_layers": 0}, {"ridge_sampling_minimum_repeats": 0},
                {"ridge_sampling_lower_quantile": 1.5}, {"ridge_sampling_lower_quantile": -.1},
                {"ridge_sampling_cluster_relative_tolerance": 1.2},
                {"ridge_sampling_minimum_lattice_fraction": 2.},
                {"ridge_sampling_lattice_relative_tolerance": float("nan")},
                {"ridge_sampling_maximum_points": 200000.0}, {"ridge_sampling_maximum_points": 0},
                {"ridge_sampling_maximum_points": True}, {"ridge_sampling_maximum_layers": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                policy(config)

    def test_regular_single_line_and_missing_integer_multiples_learn_pitch(self):
        full = np.arange(34)*.013
        for projection in (full, np.delete(full, [3, 5])):
            with self.subTest(projection=projection):
                result = train_scale(projection, {})
                for key in ("state", "training_only", "training_layer_count", "trained_pitch_m",
                            "bin_width_m", "gap_limit_m", "learning_state", "reason"):
                    self.assertIn(key, result)
                self.assertIsInstance(result, dict)
                self.assertEqual(result["state"], "complete")
                self.assertIs(result["training_only"], True)
                self.assertEqual(result["training_layer_count"], len(projection))
                self.assertAlmostEqual(result["trained_pitch_m"], .013, places=9)
                self.assertAlmostEqual(result["bin_width_m"], .013, places=9)
                self.assertAlmostEqual(result["gap_limit_m"], 1.75*.013, places=9)
                self.assertGreater(result["bin_width_m"], policy({})["voxel_size"])
                self.assertIsNone(result["reason"])
                self.assertIsInstance(result["learning_state"], str)

    def test_dense_continuous_projection_keeps_original_voxel_scale(self):
        result = train_scale(np.linspace(0., 1., 20001), {})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["training_layer_count"], 1)
        self.assertTrue(result["trained_pitch_m"] is None
                        or result["trained_pitch_m"] <= policy({})["voxel_size"])
        self.assertEqual(result["bin_width_m"], policy({})["voxel_size"])
        self.assertEqual(result["gap_limit_m"], FALLBACK_GAP)
        self.assertEqual(result["reason"], "insufficient_training_layers")

    def test_small_regular_pitch_never_widens_beyond_voxel(self):
        result = train_scale(np.arange(30)*.004, {})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["training_layer_count"], 30)
        self.assertTrue(result["trained_pitch_m"] is None
                        or result["trained_pitch_m"] <= policy({})["voxel_size"])
        self.assertEqual(result["bin_width_m"], policy({})["voxel_size"])
        self.assertEqual(result["gap_limit_m"], FALLBACK_GAP)

    def test_fewer_than_six_layers_do_not_authorize_widening(self):
        for projection in (np.arange(3)*.1, np.arange(5)*.013):
            with self.subTest(projection=projection):
                result = train_scale(projection, {})
                self.assertEqual(result["state"], "complete")
                self.assertEqual(result["training_layer_count"], len(projection))
                self.assertEqual(result["bin_width_m"], policy({})["voxel_size"])
                self.assertEqual(result["gap_limit_m"], FALLBACK_GAP)

    def test_six_layers_is_the_minimum_for_learning(self):
        fallback = train_scale(np.arange(5)*.013, {})
        learned = train_scale(np.arange(6)*.013, {})
        self.assertEqual(fallback["bin_width_m"], policy({})["voxel_size"])
        self.assertGreater(learned["bin_width_m"], policy({})["voxel_size"])
        self.assertAlmostEqual(learned["bin_width_m"], .013, places=9)
        self.assertAlmostEqual(learned["trained_pitch_m"], .013, places=9)

    def test_order_and_duplicates_do_not_change_scale(self):
        training = np.arange(34)*.013
        first = train_scale(training, {})
        repeated = train_scale(np.repeat(training[::-1], 3), {})
        self.assertEqual(first["training_layer_count"], repeated["training_layer_count"])
        self.assertEqual(first["bin_width_m"], repeated["bin_width_m"])
        self.assertEqual(first["gap_limit_m"], repeated["gap_limit_m"])
        self.assertAlmostEqual(first["trained_pitch_m"], repeated["trained_pitch_m"], places=12)

    def test_translation_does_not_change_scale(self):
        first = train_scale(np.arange(34)*.013, {})
        shifted = train_scale(np.arange(34)*.013+.0037, {})
        self.assertEqual(first["training_layer_count"], shifted["training_layer_count"])
        np.testing.assert_allclose(first["bin_width_m"], shifted["bin_width_m"], atol=1e-12)
        np.testing.assert_allclose(first["gap_limit_m"], shifted["gap_limit_m"], atol=1e-12)
        np.testing.assert_allclose(first["trained_pitch_m"], shifted["trained_pitch_m"], atol=1e-12)

    def test_long_isolated_gap_does_not_inflate_pitch(self):
        block = np.arange(30)*.013
        result = train_scale(np.r_[block, 4.7+block], {})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["training_layer_count"], 60)
        self.assertAlmostEqual(result["trained_pitch_m"], .013, places=9)
        self.assertAlmostEqual(result["bin_width_m"], .013, places=9)
        self.assertLess(result["bin_width_m"], .05)

    def test_pitch_above_cap_falls_back_to_voxel_scale(self):
        for config, pitch in (({}, .12), ({"ridge_sampling_maximum_pitch_voxels": 1.0}, .013)):
            with self.subTest(config=config, pitch=pitch):
                result = train_scale(np.arange(20)*pitch, config)
                self.assertEqual(result["state"], "complete")
                self.assertEqual(result["bin_width_m"], policy({})["voxel_size"])
                self.assertEqual(result["gap_limit_m"], FALLBACK_GAP)

    def test_clearly_irregular_spacing_does_not_widen_scale(self):
        irregular = np.r_[0., np.cumsum(np.tile([.007, .011, .019], 40))]
        result = train_scale(irregular, {})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["training_layer_count"], len(irregular))
        self.assertEqual(result["bin_width_m"], policy({})["voxel_size"])
        self.assertEqual(result["gap_limit_m"], FALLBACK_GAP)

    def test_learned_gap_limit_uses_gap_factor(self):
        result = train_scale(np.arange(34)*.013, {"ridge_sampling_gap_factor": 2.0})
        self.assertAlmostEqual(result["bin_width_m"], .013, places=9)
        self.assertAlmostEqual(result["gap_limit_m"], 2.0*.013, places=9)

    def test_budget_exceeded_is_unmeasurable_not_empty_rejection(self):
        training = np.arange(10)*.013
        for config in ({"ridge_sampling_maximum_points": 3}, {"ridge_sampling_maximum_layers": 6}):
            with self.subTest(config=config):
                result = train_scale(training, config)
                self.assertEqual(result["state"], "unmeasurable")
                self.assertIs(result["training_only"], True)
                self.assertIsInstance(result["reason"], str)
                self.assertTrue(result["reason"])

    def test_train_scale_rejects_nonfinite_and_non_1d_projections(self):
        for projection in ([[0., .1], [.2, .3]], np.array([0., np.nan, .2]),
                np.array([0., np.inf, .2]), np.array([0., -np.inf, .2]), 5.):
            with self.subTest(projection=projection), self.assertRaises(ValueError):
                train_scale(projection, {})


if __name__ == "__main__":
    unittest.main()
