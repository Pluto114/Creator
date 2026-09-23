"""Development controls for robust sections; separate from blind efficacy evidence."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from check_common_readout_ceiling import tube  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.common_readout_sections import readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402

EMPTY = np.empty((0, 2, 3))


class SectionReadoutTests(unittest.TestCase):
    def score(self, points, truth, voxel=.017):
        result = readout(points, EMPTY, {"voxel_size": voxel})
        score = curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005)
        return result, score

    def test_thick_clean_and_noisy_surfaces_survive_finer_grid(self):
        truth = np.array([[[.021, -.013, .007], [.301, .157, 1.207]]])
        for noise in (0., .003):
            with self.subTest(noise=noise):
                result, score = self.score(tube(truth[0], .06, noise, 91), truth)
                self.assertGreater(score["recovery_fraction"], .95, result)
                self.assertGreater(score["precision_fraction"], .95, result)

    def test_half_surface_uses_circle_center_not_visible_surface_mean(self):
        truth = np.array([[[.021, -.013, .007], [.021, -.013, 1.207]]])
        points = tube(truth[0], .05, .002, 92)
        points = points[points[:, 0] > truth[0, 0, 0]]
        result, score = self.score(points, truth)
        self.assertGreater(score["recovery_fraction"], .95, result)
        self.assertGreater(score["precision_fraction"], .95, result)
        self.assertLess(score["truth_to_prediction"]["distance_p95"], .015)
        self.assertEqual(result["sections"][0]["model"], "circle")

    def test_local_unstructured_contamination_does_not_delete_entire_rod(self):
        truth = np.array([[[.021, -.013, .007], [.021, -.013, 1.207]]])
        points = tube(truth[0], .05, .003, 93)
        clutter = np.random.default_rng(94).uniform([-.1, -.13, .3], [.14, .11, .6], size=(1500, 3))
        result, score = self.score(np.r_[points, clutter], truth)
        self.assertGreater(score["recovery_fraction"], .9, result)
        self.assertGreater(score["precision_fraction"], .9, result)

    def test_coherent_side_arm_is_not_silently_trimmed(self):
        main = tube(np.array([[.021, -.013, .007], [.021, -.013, 1.207]]), .05, .003, 95)
        branch = tube(np.array([[.021, -.013, .55], [.371, -.013, .55]]), 0., 0., 96)
        result = readout(np.r_[main, branch], EMPTY, {"voxel_size": .017})
        self.assertEqual(result["components"], 0, result)

    def test_planar_negative_does_not_become_circle_axis(self):
        x, y = np.meshgrid(np.arange(31) * .017, np.arange(31) * .017)
        result = readout(np.c_[x.ravel(), y.ravel(), np.zeros(x.size)], EMPTY, {"voxel_size": .017})
        self.assertEqual(result["components"], 0)

    def test_real_missing_interval_stays_missing(self):
        truth = np.array([[[.021, -.013, .007], [.021, -.013, .407]], [[.021, -.013, .607], [.021, -.013, 1.007]]])
        result, _ = self.score(np.concatenate([tube(s, .04, .002, 97 + i) for i, s in enumerate(truth)]), truth)
        self.assertEqual(result["components"], 2, result)
        # This checks missing support beyond the voxel diagonal and scoring tolerance.
        # It does NOT replace the original experiment's separately reported gap metric.
        guard = np.sqrt(3) * .017 + .025
        inner_start, inner_end = truth[0, 1].copy(), truth[1, 0].copy()
        inner_start[2] += guard
        inner_end[2] -= guard
        inner = gap_coverage(result["segments"], [inner_start, inner_end], tolerance=.025, spacing=.005)
        self.assertEqual(inner["guarded_interior"]["covered_fraction"], 0)

    def test_points_and_input_curve_receive_identical_occupancy_path(self):
        segment = np.array([[[.021, -.013, .007], [.021, -.013, 1.207]]])
        points = sample_segments(segment, .017 / 2, 2000000)
        from_points = readout(points, EMPTY, {"voxel_size": .017})
        from_curve = readout(np.empty((0, 3)), segment, {"voxel_size": .017})
        np.testing.assert_array_equal(from_points["segments"], from_curve["segments"])
        self.assertEqual(from_points["occupied_voxels"], from_curve["occupied_voxels"])

    def test_order_and_duplicate_samples_do_not_change_result(self):
        points = tube(np.array([[.021, -.013, .007], [.021, -.013, 1.207]]), .05, .003, 98)
        first = readout(points, EMPTY, {"voxel_size": .017})
        second = readout(np.repeat(points[::-1], 2, axis=0), EMPTY, {"voxel_size": .017})
        np.testing.assert_array_equal(first["segments"], second["segments"])

    def test_budgets_and_unknown_policy_rejected(self):
        for config in ({"circle_trials": 0}, {"circle_trials": 100000}, {"gt_radius": .05}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                readout(np.empty((0, 3)), EMPTY, config)
        result = readout(np.c_[np.zeros(8), np.zeros(8), np.arange(8)], EMPTY, {"maximum_voxels": 2})
        self.assertEqual(result["state"], "unmeasurable")


if __name__ == "__main__":
    unittest.main()
