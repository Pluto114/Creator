"""Regression cases motivating v3; these are development, never new blind evidence."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from check_common_readout_ceiling import tube  # noqa: E402
from creator_eval.common_readout_split import axial_runs, readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402

EMPTY = np.empty((0, 2, 3))


class SplitReadoutTests(unittest.TestCase):
    def test_perfect_line_does_not_fragment_at_floating_bin_boundaries(self):
        truth = [[[.013, .017, .011], [.013, .017, 1.011]]]
        result = readout(tube(truth[0], 0., 0., 1), EMPTY, {"voxel_size": .02})
        self.assertEqual(result["components"], 1)
        self.assertGreater(curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005)["recovery_fraction"], .98)

    def test_real_distance_gap_survives_numerical_tolerance(self):
        projection = np.r_[np.arange(0, .41, .02), np.arange(.6, 1.01, .02)]
        self.assertEqual(len(axial_runs(projection, .02, 1.75)), 2)
        truth = np.array([[[.013, .017, .011], [.013, .017, .411]], [[.013, .017, .611], [.013, .017, 1.011]]])
        result = readout(np.concatenate([tube(s, 0., 0., 1) for s in truth]), EMPTY, {"voxel_size": .02})
        self.assertEqual(result["components"], 2)
        self.assertEqual(gap_coverage(result["segments"], [truth[0, 1], truth[1, 0]], tolerance=.025, spacing=.005)["guarded_interior"]["covered_fraction"], 0)

    def test_both_previously_false_middle_axes_split_into_supported_members(self):
        for origin, length, radius, noise, voxel, seed in (([.013, .017, .011], 1., .05, .0025, .04, 22092026),
                                                          ([.028, .033, -.027], 1.25, .042, .004, .036, 22092027)):
            with self.subTest(voxel=voxel):
                a = np.array(origin)
                truth = np.array([[a, a + [0, 0, length]], [a + [.16, 0, 0], a + [.16, 0, length]]])
                points = np.concatenate([tube(s, radius, noise, seed + i) for i, s in enumerate(truth)])
                result = readout(points, EMPTY, {"voxel_size": voxel})
                self.assertEqual(result["components"], 2, result["split_decisions"])
                score = curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005)
                self.assertGreater(score["recovery_fraction"], .95)
                self.assertGreater(score["precision_fraction"], .95)

    def test_order_and_multiplicity_do_not_change_geometry(self):
        points = tube([[.013, .017, .011], [.013, .017, 1.011]], .02, .0025, 1)
        first = readout(points, EMPTY, {"voxel_size": .02})
        second = readout(np.repeat(points[::-1], 3, axis=0), EMPTY, {"voxel_size": .02})
        np.testing.assert_array_equal(first["segments"], second["segments"])

    def test_branch_is_not_a_second_parallel_rod(self):
        points = np.concatenate([tube(np.asarray(s, float), 0., 0., 1) for s in ([[0, 0, 0], [0, 0, 1]], [[0, 0, .5], [.4, 0, .5]])])
        result = readout(points, EMPTY, {"voxel_size": .02})
        self.assertEqual(result["components"], 0)

    def test_invalid_inputs_and_resource_failures_are_explicit(self):
        for config in ({"voxel_size": -1}, {"maximum_split_depth": 100}, {"split_minimum_fraction": .9}, {"gt_axis": [0, 0, 1]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                readout(np.empty((0, 3)), EMPTY, config)
        points = np.c_[np.zeros(10), np.zeros(10), np.arange(10)]
        self.assertEqual(readout(points, EMPTY, {"maximum_voxels": 2})["state"], "unmeasurable")


if __name__ == "__main__":
    unittest.main()
