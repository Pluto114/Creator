"""Shared single-axis readout contract checks; no model weights or Blender required.

Implements the six approved local tests for creator_eval.single_axis_readout:
equal-weight TLS over deduplicated per-voxel means, geometric + callback support,
gap preservation, rejection with a non-empty reason, and input immutability.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.single_axis_readout import readout

VOXEL = 0.01
ORIGIN = np.zeros(3, dtype=float)


def all_supported(points):
    """Callback that accepts every sampled point."""
    return np.ones(len(points), dtype=bool)


def x_axis_points(xs):
    """Build a finite Nx3 float array on the clean x axis."""
    xs = np.asarray(xs, dtype=float)
    return np.c_[xs, np.zeros(len(xs)), np.zeros(len(xs))]


def canonical_segments(segments):
    """Order endpoints along x and segments by their lower endpoint."""
    ordered = [np.asarray(seg)[np.argsort(seg[:, 0])] for seg in segments]
    ordered.sort(key=lambda seg: tuple(np.asarray(seg)[0]))
    return np.asarray(ordered)


class SingleAxisReadoutContractTests(unittest.TestCase):
    def assert_contract_result(self, result):
        self.assertIsInstance(result, dict)
        for key in ("state", "reason", "segments", "components"):
            self.assertIn(key, result)

    def test_clean_x_axis_gives_one_segment_over_the_observed_extent(self):
        values = x_axis_points(np.linspace(0.0, 1.0, 101))
        result = readout(values, VOXEL, ORIGIN.copy(), all_supported)
        self.assert_contract_result(result)
        self.assertEqual(result["state"], "complete")
        self.assertFalse(result["reason"])
        self.assertEqual(len(result["segments"]), 1)
        segment = np.asarray(result["segments"][0])
        self.assertEqual(segment.shape, (2, 3))
        np.testing.assert_allclose(
            segment[np.argsort(segment[:, 0])],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            atol=1e-6,
        )

    def test_shuffled_duplicates_reproduce_the_same_segments(self):
        values = x_axis_points(np.linspace(0.0, 1.0, 101))
        baseline = readout(values, VOXEL, ORIGIN.copy(), all_supported)
        rng = np.random.default_rng(7)
        jumbled = np.repeat(values, rng.integers(1, 4, size=len(values)), axis=0)
        rng.shuffle(jumbled)
        result = readout(jumbled, VOXEL, ORIGIN.copy(), all_supported)
        self.assertEqual(result["state"], baseline["state"])
        np.testing.assert_array_equal(
            canonical_segments(result["segments"]),
            canonical_segments(baseline["segments"]),
        )

    def test_two_separated_clusters_give_two_segments_with_a_central_gap(self):
        xs = np.concatenate([np.linspace(0.0, 0.3, 31), np.linspace(0.7, 1.0, 31)])
        values = x_axis_points(xs)
        result = readout(values, VOXEL, ORIGIN.copy(), all_supported)
        self.assert_contract_result(result)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 2)
        for segment in result["segments"]:
            xs = np.asarray(segment)[:, 0]
            self.assertFalse(xs.min() < 0.5 < xs.max())

    def test_callback_gap_splits_the_axis_without_bridging(self):
        values = x_axis_points(np.linspace(0.0, 1.0, 101))

        def allowed(points):
            xs = np.asarray(points)[:, 0]
            return (xs < 0.4) | (xs > 0.6)

        result = readout(values, VOXEL, ORIGIN.copy(), allowed)
        self.assert_contract_result(result)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 2)
        for segment in result["segments"]:
            xs = np.asarray(segment)[:, 0]
            self.assertTrue((xs < 0.5).all() or (xs > 0.5).all())

    def test_equal_strength_cross_is_rejected_with_nonempty_reason(self):
        arm = np.linspace(-0.5, 0.5, 101)
        values = np.vstack([x_axis_points(arm), np.c_[np.zeros(101), arm, np.zeros(101)]])
        result = readout(values, VOXEL, ORIGIN.copy(), all_supported)
        self.assert_contract_result(result)
        self.assertEqual(len(result["segments"]), 0)
        self.assertTrue(isinstance(result["reason"], str) and result["reason"])

    def test_readout_does_not_modify_values_or_origin(self):
        values = x_axis_points(np.linspace(0.0, 1.0, 101))
        origin = ORIGIN.copy()

        def allowed(points):
            xs = np.asarray(points)[:, 0]
            return (xs < 0.4) | (xs > 0.6)

        saved_values = values.copy()
        saved_origin = origin.copy()
        readout(values, VOXEL, origin, all_supported)
        np.testing.assert_array_equal(values, saved_values)
        np.testing.assert_array_equal(origin, saved_origin)

        readout(values, VOXEL, origin, allowed)
        np.testing.assert_array_equal(values, saved_values)
        np.testing.assert_array_equal(origin, saved_origin)


if __name__ == "__main__":
    unittest.main()
