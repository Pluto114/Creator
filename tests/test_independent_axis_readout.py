"""Raw support preservation and coverage-ranked geometry-only readout contracts."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import independent_axis_readout as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_independent_axis_readout import IndependentAxisReadout  # noqa: E402
from creator_eval.rgb_single_axis_readout import SingleAxisReadout  # noqa: E402
from test_single_axis_readout import views  # noqa: E402


def line():
    return np.c_[np.linspace(0, 1, 101), np.zeros((101, 2))]


def yes(values):
    return np.ones(len(values), bool)


class IndependentAxisContracts(unittest.TestCase):
    def test_shared_noisy_axis_estimates_do_not_supply_independent_structure(self):
        x = np.linspace(-1, 1, 201)
        values = np.c_[x, .0025*np.sin(37*x), x*0]
        points, inverse, _ = reader.raw_pool(values, .01, np.zeros(3))
        first = dict(centroid=np.zeros(3), direction=np.array([1., 0, 0]))
        direction = np.array([1., .008, 0])
        direction /= np.linalg.norm(direction)
        second = dict(centroid=np.array([0, .004, 0]), direction=direction)
        result = reader.independent_support(points, inverse, first, second, .01)
        self.assertFalse(result["independent"])
        self.assertEqual(result["exclusive_coverage_fractions"][1], 0.)
        reverse = reader.independent_support(points, inverse, second, first, .01)
        self.assertEqual(reverse["exclusive_coverage_fractions"], result["exclusive_coverage_fractions"][::-1])

    def test_resolvable_shallow_cross_and_parallel_stay_independently_ambiguous(self):
        x = np.linspace(-.5, .5, 101)
        first = np.c_[x, x*0, x*0]
        for second in (first+[0, .03, 0], np.c_[x, .025*x, x*0]):
            result = reader.readout(np.r_[first, second], .01, [0, 0, 0], yes)
            self.assertEqual(result["reason"], "strong_competing_axis")
            self.assertTrue(result["competitor_independence"]["independent"])
            self.assertGreaterEqual(min(result["competitor_independence"]["exclusive_coverage_fractions"]), .5)

    def test_fixed_axis_raw_support_survives_off_axis_points_inside_same_voxels(self):
        original = line()+[.001, .001, .001]
        offsets = np.c_[np.zeros(20), np.linspace(.006, .008, 20), np.linspace(.001, .008, 20)]
        augmented = np.r_[original, (original[:, None]+offsets).reshape(-1, 3)]
        before, after = [], []
        for values, destination in ((original, before), (augmented, after)):
            points, inverse, _ = reader.raw_pool(values, .01, np.zeros(3))
            support = reader.axis_support(points, inverse, np.array([0, .001, .001]), np.array([1., 0, 0]), .01)
            destination.append(points[support["representatives"]])
        np.testing.assert_array_equal(before[0], after[0])

    def test_long_sparse_axis_outranks_dense_short_structure(self):
        x = np.linspace(0, 1, 61)
        sparse = np.c_[x, x*0, x*0]
        xx, zz = np.meshgrid(np.linspace(0, .25, 101), np.linspace(0, .08, 17))
        dense = np.c_[xx.ravel(), np.full(xx.size, .2), zz.ravel()]
        result = reader.readout(np.r_[sparse, dense], .01, [0, 0, 0], yes)
        self.assertEqual(result["components"], 1)
        np.testing.assert_allclose(result["segments"][0], [[0, 0, 0], [1, 0, 0]], atol=1e-12)
        self.assertAlmostEqual(result["model"]["support_coverage_m"], 1.)

    def test_resolvable_second_track_inside_same_voxels_preserves_ambiguity(self):
        original = line()+[.001, .001, .001]
        offsets = np.c_[np.zeros(10), np.linspace(.006, .008, 10), np.linspace(.001, .008, 10)]
        values = np.r_[original, (original[:, None]+offsets).reshape(-1, 3)]
        result = reader.readout(values, .01, [0, 0, 0], yes)
        # The far track is >v from the original despite sharing axis voxels.
        # Preserving raw geometry must not average away this real ambiguity.
        self.assertEqual(result["reason"], "strong_competing_axis")
        self.assertEqual(result["components"], 0)

    def test_local_proposals_find_sparse_axis_hidden_from_global_random_seeds(self):
        rod = np.c_[np.linspace(0, 1, 61), np.zeros((61, 2))]
        noise = np.random.default_rng(71).uniform([-.5, .03, -1], [1.5, 2, 1], (8000, 3))
        result = reader.readout(np.r_[rod, noise], .01, [0, 0, 0], yes)
        self.assertEqual(result["components"], 1)
        np.testing.assert_allclose(result["segments"][0], [[0, 0, 0], [1, 0, 0]], atol=1e-12)

    def test_candidate_cache_contains_no_per_point_support_arrays(self):
        points, inverse, _ = reader.raw_pool(line(), .01, np.zeros(3))
        result = reader.refine(points, inverse, np.zeros(3), np.array([1., 0, 0]), .01)
        self.assertIsInstance(result, dict)
        self.assertNotIn("indices", result)
        self.assertNotIn("axial", result)
        self.assertNotIn("representatives", result)

    def test_shallow_crossing_is_compared_on_same_pool_without_consumption(self):
        x = np.linspace(-.5, .5, 101)
        points = np.r_[np.c_[x, x*0, x*0], np.c_[x, .04*x, x*0]]
        result = reader.readout(points, .01, [0, 0, 0], yes)
        self.assertEqual(result["reason"], "strong_competing_axis")
        self.assertGreaterEqual(result["competitor_model"]["support_coverage_m"],
                                .8*result["model"]["support_coverage_m"])
        self.assertEqual(result["components"], 0)

    def test_off_axis_gap_points_cannot_fill_an_observed_axis_gap(self):
        ends = line()
        ends = ends[(ends[:, 0] < .3) | (ends[:, 0] > .7)]
        noise = np.c_[np.linspace(.3, .7, 21), np.full(21, .2), np.zeros(21)]
        result = reader.readout(np.r_[ends, noise], .01, [0, 0, 0], yes)
        self.assertEqual(result["components"], 2)
        for segment in result["segments"]:
            self.assertFalse(min(segment[:, 0]) <= .5 <= max(segment[:, 0]))

    def test_point_and_curve_inputs_are_identical_after_common_sampling(self):
        segment = np.array([[[-.8, 0, 2.], [.8, 0, 2.]]])
        sampled = sample_segments(segment, .01, 2000000)
        config = dict(voxel_size=.02, origin=[0, 0, 0])
        base = IndependentAxisReadout(sampled, views())(np.empty((0, 2, 3)), config)
        curve = IndependentAxisReadout(np.empty((0, 3)), views())(segment, config)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        np.testing.assert_allclose(base["segments"], segment, atol=1e-12)

    def test_raw_support_and_sampling_volume_unchanged_from_global_adapter(self):
        values = line()+[0, 0, 2]
        segment = np.array([[[0, 0, 2], [1, 0, 2]]])
        config = dict(voxel_size=.01, origin=[0, 0, 0])
        old, new = [adapter(values, views())(segment, config) for adapter in (SingleAxisReadout, IndependentAxisReadout)]
        for key in ("full_input_point_count", "supported_base_point_count", "full_input_segment_count",
                    "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram"):
            self.assertEqual(old[key], new[key], key)

    def test_invalid_empty_and_degenerate_are_distinguished(self):
        for values in (np.zeros((5, 2)), np.array([[np.nan, 0, 0]])):
            with self.assertRaises(ValueError):
                reader.readout(values, .01, [0, 0, 0], yes)
        for size in (0, -1, np.nan, True):
            with self.assertRaises(ValueError):
                reader.readout(line(), size, [0, 0, 0], yes)
        for values in (np.empty((0, 3)), np.zeros((100, 3))):
            result = reader.readout(values, .01, [0, 0, 0], yes)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["components"], 0)
            self.assertTrue(result["reason"])

    def test_budgets_are_unmeasurable_not_successful_empty_results(self):
        for key in ("maximum_unique_points", "maximum_voxels", "maximum_axis_samples", "maximum_point_hypothesis_checks"):
            with self.subTest(key=key), patch.dict(reader.DEFAULTS, {key: 5}):
                self.assertEqual(reader.readout(line(), .01, [0, 0, 0], yes)["state"], "unmeasurable")

    def test_invalid_callback_and_runtime_failure_remain_visible(self):
        with self.assertRaises(ValueError):
            reader.readout(line(), .01, [0, 0, 0], lambda p: np.ones(len(p)))
        with patch.object(reader, "readout", side_effect=RuntimeError("visible failure")):
            result = IndependentAxisReadout(line()+[0, 0, 2], views())(
                np.empty((0, 2, 3)), dict(voxel_size=.01, origin=[0, 0, 0]))
        self.assertEqual(result["state"], "error")
        self.assertIn("visible failure", result["reason"])

    def test_input_and_policy_are_not_mutated_or_method_conditioned(self):
        values, origin = line(), np.zeros(3)
        before = values.copy()
        result = reader.readout(values, .01, origin, yes)
        np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(origin, np.zeros(3))
        self.assertTrue(result["fixed_axis_support_set_monotone"])
        self.assertFalse(result["exhaustive_axis_search"])
        with self.assertRaises(ValueError):
            IndependentAxisReadout(values, views(), dict(minimum_views=2, horizontal_padding_px=1))
        with self.assertRaises(ValueError):
            IndependentAxisReadout(values, views())(np.empty((0, 2, 3)),
                dict(voxel_size=.01, origin=[0, 0, 0], method="candidate"))


if __name__ == "__main__":
    unittest.main()
