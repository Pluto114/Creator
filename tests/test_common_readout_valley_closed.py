"""Focused development checks for a valley gate; not held-out efficacy evidence."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_split import principal_axes, split_component  # noqa: E402
from creator_eval.common_readout_valley import readout as open_band_readout  # noqa: E402
from creator_eval.common_readout_valley_closed import (  # noqa: E402
    DEFAULTS,
    guarded_split_component,
    readout,
)
from creator_eval.line_controls import curve_metrics  # noqa: E402
from creator_eval.readout_background_controls import member, panel  # noqa: E402

EMPTY = np.empty((0, 2, 3))


class ClosedBandValleyTests(unittest.TestCase):
    def test_exact_binary_grid_boundary_points_cannot_disappear(self):
        voxel = 1 / 32
        x, z = np.meshgrid((np.arange(8) - 3.5) * voxel, (np.arange(40) + .5) * voxel, indexing="ij")
        cloud = np.c_[x.ravel(), np.full(x.size, voxel / 2), z.ravel()]
        previous = open_band_readout(cloud, EMPTY, {"voxel_size": voxel})
        self.assertEqual(previous["components"], 2)
        corrected = readout(cloud, EMPTY, {"voxel_size": voxel})
        self.assertEqual(corrected["components"], 0)
        self.assertEqual(corrected["accepted_splits"], 0)
        self.assertTrue(all(row["gap_points"] > 0 for check in corrected["split_valley_checks"] for row in check["axial_slices"]))

    def test_uniform_planar_strip_is_not_evidence_for_two_modes(self):
        result = readout(panel(1.23, .24), EMPTY, {"voxel_size": .02})
        self.assertEqual(result["accepted_splits"], 0)
        self.assertEqual(result["components"], 0)
        self.assertGreater(len(result["split_valley_checks"]), 0)
        self.assertTrue(all(check["supported_slices"] < check["required_slices"] for check in result["split_valley_checks"]))

    def test_resolved_parallel_members_keep_both_axes(self):
        truth = np.array([[[.007, .013, .011], [.007, .013, 1.241]], [[.187, .013, .011], [.187, .013, 1.241]]])
        points = np.concatenate([member(segment, .032, noise=.001, seed=500 + i) for i, segment in enumerate(truth)])
        result = readout(points, EMPTY, {"voxel_size": .02})
        score = curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005)
        self.assertEqual(result["components"], 2, result)
        self.assertGreater(score["recovery_fraction"], .95)
        self.assertGreater(score["precision_fraction"], .95)

    def test_circle_and_half_circle_are_not_split_into_two_members(self):
        truth = np.array([[[.007, .013, .011], [.007, .013, 1.241]]])
        for angular_range in ((0., 2 * np.pi), (-np.pi / 2, np.pi / 2)):
            with self.subTest(angular_range=angular_range):
                points = member(truth[0], .05, noise=.0015, seed=502, angular_range=angular_range)
                result = readout(points, EMPTY, {"voxel_size": .018})
                score = curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005)
                self.assertEqual(result["accepted_splits"], 0)
                self.assertEqual(result["components"], 1)
                self.assertGreater(score["recovery_fraction"], .95)
                self.assertGreater(score["precision_fraction"], .95)

    def test_a_gap_in_one_axial_slice_does_not_authorize_a_full_split(self):
        x, z = np.meshgrid(np.arange(-.09, .10, .02), np.arange(.01, 1.22, .02), indexing="ij")
        cloud = np.c_[x.ravel(), np.full(x.size, .01), z.ravel()]
        cloud = cloud[(cloud[:, 2] >= .21) | (np.abs(cloud[:, 0]) > .08)]
        _, eigen, vectors, axis = principal_axes(cloud)
        policy = {**DEFAULTS, "voxel_size": .02}
        children, _ = split_component(cloud, eigen, vectors, axis, policy)
        self.assertIsNotNone(children)
        gated, reason, detail = guarded_split_component(cloud, eigen, vectors, axis, policy)
        self.assertIsNone(gated)
        self.assertEqual(reason, "transverse_valley_not_persistent")
        self.assertLess(detail["supported_slices"], detail["required_slices"])

    def test_order_and_multiplicity_do_not_change_the_gate_or_geometry(self):
        points = panel(1.23, .24)
        first = readout(points, EMPTY, {"voxel_size": .02})
        second = readout(np.repeat(points[::-1], 3, axis=0), EMPTY, {"voxel_size": .02})
        np.testing.assert_array_equal(first["segments"], second["segments"])
        self.assertEqual(first["split_valley_checks"], second["split_valley_checks"])

    def test_invalid_valley_policies_are_rejected(self):
        for config in ({"split_valley_axial_slices": 1000}, {"split_valley_maximum_density_ratio": 1.}, {"split_valley_minimum_fraction": 0.}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                readout(np.empty((0, 3)), EMPTY, config)


if __name__ == "__main__":
    unittest.main()
