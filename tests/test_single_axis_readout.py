"""Mechanisms for the declared local single-axis shared representation."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import single_axis_readout as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_single_axis_readout import SingleAxisReadout  # noqa: E402


def points():
    return np.c_[np.linspace(-.8, .8, 161), np.zeros(161), np.full(161, 2.)]


def views():
    return [dict(view_id=f"v{i}", K_index=np.array([[50., 0, 50], [0, 50, 50], [0, 0, 1]]),
        world_to_camera_cv=np.c_[np.eye(3), np.zeros(3)], left=np.zeros((100, 1)),
        right=np.full((100, 1), 99.), size_wh=[100, 100], row_states={}) for i in range(3)]


class SingleAxisContracts(unittest.TestCase):
    def test_clean_line_and_exact_point_curve_sampling_equivalence(self):
        segment = np.array([[[-.8, 0, 2], [.8, 0, 2]]])
        sampled = sample_segments(segment, .01, 2000000)
        base = SingleAxisReadout(sampled, views())(np.empty((0, 2, 3)), dict(voxel_size=.02, origin=[0., 0., 0.]))
        curve = SingleAxisReadout(np.empty((0, 3)), views())(segment, dict(voxel_size=.02, origin=[0., 0., 0.]))
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        np.testing.assert_allclose(base["segments"], segment, atol=1e-12)

    def test_repeated_and_permuted_samples_do_not_increase_voxel_weight(self):
        value = points()
        one = reader.readout(value, .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
        repeated = np.concatenate((value, value[::3], value[::-1]))
        two = reader.readout(repeated, .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
        np.testing.assert_array_equal(one["segments"], two["segments"])
        self.assertEqual(one["occupied_voxels"], two["occupied_voxels"])

    def test_duplicate_curves_and_base_curve_overlap_do_not_double_length(self):
        segment = np.array([[[-.8, 0, 2], [.8, 0, 2]]])
        adapter = SingleAxisReadout(points(), views())
        config = dict(voxel_size=.02, origin=[0, 0, 0])
        one, two = adapter(segment, config), adapter(np.repeat(segment, 7, axis=0), config)
        np.testing.assert_array_equal(one["segments"], two["segments"])
        self.assertEqual(one["unique_count"], two["unique_count"])

    def test_geometry_gap_is_not_filled_by_unanimous_rgb(self):
        value = points()
        value = value[(value[:, 0] < -.2) | (value[:, 0] > .2)]
        result = reader.readout(value, .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
        self.assertEqual(len(result["segments"]), 2)
        self.assertTrue(all(not (min(s[:, 0]) <= 0 <= max(s[:, 0])) for s in result["segments"]))

    def test_rgb_unknown_splits_continuous_geometry_without_interpolation(self):
        result = reader.readout(points(), .02, [0, 0, 0], lambda p: np.abs(p[:, 0]) > .2)
        self.assertEqual(len(result["segments"]), 2)
        self.assertLess(result["jointly_supported_samples"], result["geometry_supported_samples"])

    def test_midline_between_ambiguous_intervals_is_not_supported(self):
        v = views()
        for view in v:
            view["left"] = np.tile([10., 80.], (100, 1))
            view["right"] = np.tile([20., 90.], (100, 1))
        value = np.c_[np.zeros(161), np.linspace(-.8, .8, 161), np.full(161, 2.)]
        result = SingleAxisReadout(value, v)(np.empty((0, 2, 3)), dict(voxel_size=.02, origin=[0, 0, 0]))
        self.assertEqual(len(result["segments"]), 0)
        self.assertEqual(result["supported_base_point_count"], 0)

    def test_symmetric_axes_do_not_pick_a_convenient_axis(self):
        axis = np.linspace(-1, 1, 101)
        value = np.r_[np.c_[axis, axis*0, axis*0], np.c_[axis*0, axis, axis*0]]
        result = reader.readout(value, .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
        self.assertEqual(len(result["segments"]), 0)
        self.assertTrue(result["reason"])

    def test_invalid_and_degenerate_inputs_are_distinguished(self):
        for value in (np.zeros((4, 2)), np.array([[np.nan, 0, 0]])):
            with self.assertRaises(ValueError):
                reader.readout(value, .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
        for size in (0, -1, np.nan, True):
            with self.assertRaises(ValueError):
                reader.readout(points(), size, [0, 0, 0], lambda p: np.ones(len(p), bool))
        for value in (np.empty((0, 3)), np.zeros((20, 3))):
            result = reader.readout(value, .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
            self.assertEqual(result["state"], "complete")
            self.assertTrue(result["reason"])
            self.assertEqual(len(result["segments"]), 0)

    def test_support_callback_contract_and_exceptions_remain_visible(self):
        with self.assertRaises(ValueError):
            reader.readout(points(), .02, [0, 0, 0], lambda p: np.ones(len(p)))
        adapter = SingleAxisReadout(points(), views())
        with patch.object(reader, "readout", side_effect=RuntimeError("visible failure")):
            result = adapter(np.empty((0, 2, 3)), dict(voxel_size=.02, origin=[0, 0, 0]))
        self.assertEqual(result["state"], "error")
        self.assertIn("visible failure", result["reason"])

    def test_no_policy_or_candidate_label_override(self):
        with self.assertRaises(ValueError):
            SingleAxisReadout(points(), views(), dict(minimum_views=2, horizontal_padding_px=1.))
        adapter = SingleAxisReadout(points(), views())
        with self.assertRaises(ValueError):
            adapter(np.empty((0, 2, 3)), dict(voxel_size=.02, origin=[0, 0, 0], method="candidate"))

    def test_budget_exhaustion_is_unmeasurable_not_success(self):
        with patch.dict(reader.DEFAULTS, maximum_voxels=5):
            result = reader.readout(points(), .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
        self.assertEqual(result["state"], "unmeasurable")
        with patch.dict(reader.DEFAULTS, maximum_axis_samples=5):
            result = reader.readout(points(), .02, [0, 0, 0], lambda p: np.ones(len(p), bool))
        self.assertEqual(result["state"], "unmeasurable")

    def test_no_input_mutation_and_scope_explicit(self):
        value, origin = points(), np.array([0., 0., 0.])
        original = value.copy()
        result = reader.readout(value, .02, origin, lambda p: np.ones(len(p), bool))
        np.testing.assert_array_equal(value, original)
        np.testing.assert_array_equal(origin, np.zeros(3))
        self.assertIn("predeclared_local_single_axis", result["scope"])
        self.assertFalse(result["within_voxel_sampling_invariant"])


if __name__ == "__main__":
    unittest.main()
