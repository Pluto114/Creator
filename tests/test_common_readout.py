"""Analytic readout controls: all methods enter the same occupancy/PCA graph."""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.common_readout import readout, sample_segments
from creator_eval.line_controls import curve_metrics, gap_coverage

EMPTY = np.empty((0, 2, 3))
CONFIG = {"voxel_size": .02}


class CommonReadoutTests(unittest.TestCase):
    def test_sampled_curve_and_identical_point_cloud_read_identically(self):
        line = np.array([[[.013, .013, .013], [.013, .013, 1.013]]])
        points = sample_segments(line, .01, 1000)
        cloud = readout(points, EMPTY, CONFIG)
        curve = readout(np.empty((0, 3)), line, CONFIG)
        self.assertGreater(cloud["components"], 0)
        np.testing.assert_array_equal(cloud["segments"], curve["segments"])
        self.assertGreater(curve_metrics(cloud["segments"], line, tolerance=.03, spacing=.01)["recovery_fraction"], .9)

    def test_density_and_input_order_do_not_change_the_common_result(self):
        points = np.c_[np.full(100, .01), np.full(100, .01), np.linspace(.01, 1.01, 100)]
        first = readout(points, EMPTY, CONFIG)
        second = readout(np.repeat(points[::-1], 10, axis=0), EMPTY, CONFIG)
        np.testing.assert_array_equal(first["segments"], second["segments"])

    def test_true_gap_is_not_joined(self):
        truth = np.array([[[.01, .01, .01], [.01, .01, .4]], [[.01, .01, .6], [.01, .01, 1.01]]])
        result = readout(np.empty((0, 3)), truth, CONFIG)
        self.assertEqual(result["components"], 2)
        gap = [[.01, .01, .4], [.01, .01, .6]]
        score = gap_coverage(result["segments"], gap, tolerance=.02, spacing=.005)
        self.assertEqual(score["guarded_interior"]["covered_fraction"], 0)

    def test_plane_is_not_automatically_exported_as_a_new_rod(self):
        x, y = np.meshgrid(np.arange(30) * .02, np.arange(30) * .02)
        points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        result = readout(points, EMPTY, CONFIG)
        self.assertEqual(result["components"], 0)

    def test_complete_candidate_keeps_unrelated_base_curves(self):
        old = np.array([[[.01, .01, .01], [.01, .01, 1.01]]])
        new = old + [1., 0., 0.]
        base = sample_segments(old, .01, 1000)
        complete = readout(base, new, CONFIG)
        self.assertEqual(complete["components"], 2)
        self.assertGreater(curve_metrics(complete["segments"], np.concatenate((old, new)), tolerance=.03, spacing=.01)["recovery_fraction"], .9)

    def test_budget_failure_is_unmeasurable_not_successful_empty_geometry(self):
        points = np.c_[np.zeros(20), np.zeros(20), np.arange(20)]
        r = readout(points, EMPTY, {**CONFIG, "maximum_voxels": 4})
        self.assertEqual(r["state"], "unmeasurable")
        r = readout(np.empty((0, 3)), np.array([[[0, 0, 0], [0, 0, 2]]]), {**CONFIG, "maximum_curve_samples": 10})
        self.assertEqual(r["state"], "unmeasurable")

    def test_fixed_region_is_applied_equally_to_cloud_and_curve_samples(self):
        line = np.array([[[0, -.5, 3], [0, .5, 3]]], float)
        outside = line + [2, 0, 0]
        region = {"minimum_views": 1, "views": [{"view_id": "v0", "K_index": [[100, 0, 100], [0, 100, 100], [0, 0, 1]],
                    "world_to_camera_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                    "guides": [{"xyxy": [[100, 50], [100, 150]], "half_width_px": 10}]}]}
        cloud = readout(sample_segments(np.concatenate((line, outside)), .01, 1000), EMPTY, CONFIG, region)
        mixed = readout(sample_segments(line, .01, 1000), outside, CONFIG, region)
        np.testing.assert_array_equal(cloud["segments"], mixed["segments"])
        self.assertEqual(mixed["region_curve_sample_count"], 0)
        self.assertEqual(mixed["components"], 1)

    def test_replay_cli_starts_as_a_fresh_process(self):
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "scripts/run_g1_point_patch.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_and_bad_inputs_have_explicit_semantics(self):
        self.assertEqual(readout(np.empty((0, 3)), EMPTY, CONFIG)["reason"], "empty_input")
        for points, config in (([[np.nan, 0, 0]], CONFIG), ([[0, 0, 0]], {"voxel_size": -1}), ([[0, 0, 0]], {"from_ground_truth": True})):
            with self.subTest(config=config), self.assertRaises(ValueError):
                readout(points, EMPTY, config)


if __name__ == "__main__":
    unittest.main()
