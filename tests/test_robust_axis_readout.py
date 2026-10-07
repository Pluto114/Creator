"""Geometry-only robust reader mechanisms; no fixture scores or model labels."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import robust_axis_readout as reader  # noqa: E402
from creator_eval import single_axis_readout as previous  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_robust_axis_readout import RobustAxisReadout  # noqa: E402
from creator_eval.rgb_single_axis_readout import SingleAxisReadout  # noqa: E402
from test_single_axis_readout import views  # noqa: E402


def line():
    return np.c_[np.linspace(0, 1, 101), np.zeros((101, 2))]


def supported(points):
    return np.ones(len(points), bool)


class RobustAxisContracts(unittest.TestCase):
    def test_fixed_consensus_does_not_average_a_line_with_off_axis_noise(self):
        rng = np.random.default_rng(5)
        noise = rng.uniform([0, .06, -.12], [1, .22, .12], (250, 3))
        values = np.r_[line(), noise]
        old = previous.readout(values, .01, [0, 0, 0], supported)
        new = reader.readout(values, .01, [0, 0, 0], supported)
        self.assertGreater(abs(old["model"]["centroid"][1]), .04)
        self.assertEqual(len(new["segments"]), 1)
        np.testing.assert_allclose(new["segments"][0], [[0, 0, 0], [1, 0, 0]], atol=1e-12)
        self.assertGreater(new["excluded_voxels"], 200)

    def test_axis_offline_geometry_cannot_bridge_an_observed_axis_gap(self):
        ends = line()
        ends = ends[(ends[:, 0] < .3) | (ends[:, 0] > .7)]
        # Distant geometric occupancies exist at every axial gap position.
        # They must not support the winning line merely through projection.
        gap_points = np.c_[np.linspace(.3, .7, 21), np.full(21, .2), np.zeros(21)]
        result = reader.readout(np.r_[ends, gap_points], .01, [0, 0, 0], supported)
        self.assertEqual(len(result["segments"]), 2)
        for segment in result["segments"]:
            self.assertFalse(min(segment[:, 0]) <= .5 <= max(segment[:, 0]))
        self.assertLess(result["geometry_supported_samples"], result["axis_sample_count"])

    def test_equal_parallel_axes_and_cross_cannot_pick_one_by_seed(self):
        first = line()
        for second in (first+[0, .2, 0], first[:, [1, 0, 2]]):
            with self.subTest(second=second[-1]):
                result = reader.readout(np.r_[first, second], .01, [0, 0, 0], supported)
                self.assertEqual(result["reason"], "strong_competing_axis")
                self.assertEqual(len(result["segments"]), 0)

    def test_rotated_translated_line_preserves_observed_extent(self):
        direction = np.array([1., 2., 3.])/np.sqrt(14)
        values = np.array([.43, -.21, 2.7])+np.linspace(0, 1, 101)[:, None]*direction
        result = reader.readout(values, .01, [.1, -.2, 0], supported)
        np.testing.assert_allclose(result["segments"][0], values[[0, -1]], atol=1e-12)

    def test_shallow_crossing_shared_support_is_not_consumed_by_first_axis(self):
        x = np.linspace(-.5, .5, 101)
        values = np.r_[np.c_[x, x*0, x*0], np.c_[x, .04*x, x*0]]
        result = reader.readout(values, .01, [0, 0, 0], supported)
        self.assertEqual(result["reason"], "strong_competing_axis")
        self.assertEqual(len(result["segments"]), 0)
        first, second = result["model"], result["competitor_model"]
        self.assertLess(second["inlier_count"], .8*first["inlier_count"])
        self.assertGreaterEqual(second["full_pool_comparison_support"], .8*first["full_pool_comparison_support"])

    def test_replication_and_input_order_do_not_change_consensus(self):
        value = line()
        reference = reader.readout(value, .01, [0, 0, 0], supported)
        duplicated = np.r_[value, value[::-1], value[::3]]
        np.random.default_rng(21).shuffle(duplicated)
        actual = reader.readout(duplicated, .01, [0, 0, 0], supported)
        np.testing.assert_array_equal(reference["segments"], actual["segments"])
        np.testing.assert_array_equal(reference["model"]["inlier_mask"], actual["model"]["inlier_mask"])

    def test_identical_point_and_curve_samples_have_identical_output(self):
        segment = np.array([[[-.8, 0, 2.], [.8, 0, 2.]]])
        samples = sample_segments(segment, .01, 2000000)
        call = dict(voxel_size=.02, origin=[0., 0., 0.])
        points = RobustAxisReadout(samples, views())(np.empty((0, 2, 3)), call)
        curve = RobustAxisReadout(np.empty((0, 3)), views())(segment, call)
        np.testing.assert_array_equal(points["segments"], curve["segments"])
        np.testing.assert_allclose(curve["segments"], segment, atol=1e-12)

    def test_same_input_volume_and_sampling_counts_as_global_tls_adapter(self):
        values = line()+[0, 0, 2]
        segments = np.array([[[0, 0, 2.], [1, 0, 2.]]])
        call = dict(voxel_size=.01, origin=[0., 0., 0.])
        old, new = [adapter(values, views())(segments, call) for adapter in (SingleAxisReadout, RobustAxisReadout)]
        for key in ("full_input_point_count", "supported_base_point_count", "full_input_segment_count",
                    "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram"):
            self.assertEqual(old[key], new[key], key)

    def test_rgb_support_still_splits_continuous_consensus(self):
        result = reader.readout(line(), .01, [0, 0, 0], lambda p: (p[:, 0] < .4) | (p[:, 0] > .6))
        self.assertEqual(len(result["segments"]), 2)
        self.assertLess(result["jointly_supported_samples"], result["geometry_supported_samples"])

    def test_invalid_and_empty_inputs_differ(self):
        for value in (np.zeros((4, 2)), np.array([[np.nan, 0, 0]])):
            with self.assertRaises(ValueError):
                reader.readout(value, .01, [0, 0, 0], supported)
        for size in (0, -1, True, np.nan):
            with self.assertRaises(ValueError):
                reader.readout(line(), size, [0, 0, 0], supported)
        for value in (np.empty((0, 3)), np.zeros((20, 3))):
            result = reader.readout(value, .01, [0, 0, 0], supported)
            self.assertEqual(result["state"], "complete")
            self.assertTrue(result["reason"])
            self.assertEqual(len(result["segments"]), 0)

    def test_budget_failures_are_not_successful_empty_results(self):
        for key in ("maximum_voxels", "maximum_axis_samples"):
            with self.subTest(key=key), patch.dict(reader.DEFAULTS, {key: 5}):
                result = reader.readout(line(), .01, [0, 0, 0], supported)
                self.assertEqual(result["state"], "unmeasurable")

    def test_callback_contract_errors_and_adapter_failures_are_visible(self):
        with self.assertRaises(ValueError):
            reader.readout(line(), .01, [0, 0, 0], lambda p: np.ones(len(p)))
        with patch.object(reader, "readout", side_effect=RuntimeError("visible failure")):
            result = RobustAxisReadout(line()+[0, 0, 2], views())(
                np.empty((0, 2, 3)), dict(voxel_size=.01, origin=[0, 0, 0]))
        self.assertEqual(result["state"], "error")
        self.assertIn("visible failure", result["reason"])

    def test_no_method_identity_or_policy_override_and_no_input_mutation(self):
        values, origin = line(), np.zeros(3)
        before = values.copy()
        result = reader.readout(values, .01, origin, supported)
        np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(origin, np.zeros(3))
        self.assertTrue(result["consensus_view_ids_are_geometric_pool_not_source_views"])
        self.assertFalse(result["within_voxel_sampling_invariant"])
        with self.assertRaises(ValueError):
            RobustAxisReadout(values, views(), dict(minimum_views=2, horizontal_padding_px=1.))
        with self.assertRaises(ValueError):
            RobustAxisReadout(values, views())(np.empty((0, 2, 3)), dict(voxel_size=.01, origin=origin, method="candidate"))


if __name__ == "__main__":
    unittest.main()
