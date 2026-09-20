"""Analytic geometry tests for the optional edge-ray radius diagnostic."""
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_radius_consistency import axis_ray_distances, diagnose_edge_radii, summary


class EdgeRadiusTests(unittest.TestCase):
    def test_diagnostic_cli_starts_in_its_own_interpreter(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_rod_radius_diagnostic.py"
        run = subprocess.run([sys.executable, "-B", str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_analytic_cylinder_tangents_have_constant_distance_at_every_image_row(self):
        radius, depth = .35, 5.0
        x = radius / np.sqrt(depth * depth - radius * radius)
        rays = np.array([[sign * x, y, 1] for sign in (-1, 1) for y in (-.8, 0, .7)])
        distances, valid = axis_ray_distances([0, 0, depth], [0, 1, 0], [0, 0, 0], rays)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(distances, radius, atol=1e-12, rtol=0)

    def test_rigid_transform_axis_anchor_shift_and_ray_scale_preserve_distances(self):
        rays = np.array([[-.1, -.4, 1], [.1, .8, 1]])
        a, v, c = np.array([0, 0, 5.]), np.array([0, 1., 0]), np.zeros(3)
        expected, _ = axis_ray_distances(a, v, c, rays)
        theta = .73
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        shift = np.array([2., -3., 4.])
        got, valid = axis_ray_distances(rotation @ (a + 17 * v) + shift, rotation @ (-3 * v), shift, rays @ rotation.T * 7)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)

    def test_pixel_camera_pipeline_recovers_analytic_tangent_distances(self):
        radius, anchor = .1, np.array([.1, 0, 5.])
        views, observations = [], []
        for index, cx in enumerate((-1., -.5, .5, 1.)):
            angle = .1 * (index - 1)
            rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
            center = np.array([cx, 0, 0])
            q = anchor - center
            beta, alpha = math.atan2(q[0], q[2]), math.asin(radius / np.linalg.norm(q))
            xs = []
            for theta in (beta - alpha, beta + alpha):
                ray = rotation @ np.array([math.sin(theta), 0, math.cos(theta)])
                xs.append(160 + 300 * ray[0] / ray[2])
            left, right = sorted(xs)
            raw = {"rows": [{"y": y, "candidates": [{"left_edge": {"x": left}, "right_edge": {"x": right}}]} for y in (80, 120, 160)]}
            views.append({"view_id": str(index), "K_index": [[300, 0, 160], [0, 300, 120], [0, 0, 1]],
                          "world_to_camera_cv": np.c_[rotation, -rotation @ center], "y_range": [80, 160],
                          "candidates": [{"line": [1, 0, -(left + right) / 2], "row_matches": [[i, 0, 0] for i in range(3)]}]})
            observations.append(raw)
        hypothesis = {"model": {"anchor": anchor, "direction": [0, 1, 0]}, "supporting_views": list(range(4)),
                      "matches": [{"candidate_index": 0} for _ in range(4)]}
        result = diagnose_edge_radii(hypothesis, views, observations)
        self.assertEqual(result["all_supported_edge_rays"]["count"], 24)
        self.assertAlmostEqual(result["all_supported_edge_rays"]["median"], radius, places=12)
        self.assertLess(result["all_supported_edge_rays"]["relative_p90_p10_spread"], 1e-12)

    def test_parallel_and_backwards_rays_are_invalid_not_zero_radius(self):
        values, valid = axis_ray_distances([0, 0, 5], [0, 1, 0], [0, 0, 0], [[0, 1, 0], [.1, 0, -1]])
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(values).all())
        self.assertIsNone(summary(values)["median"])

    def test_an_offset_axis_breaks_equal_left_right_tangent_distance(self):
        x = .35 / np.sqrt(25 - .35 ** 2)
        values, valid = axis_ray_distances([.2, 0, 5], [0, 1, 0], [0, 0, 0], [[-x, 0, 1], [x, 0, 1]])
        self.assertTrue(valid.all())
        self.assertGreater(abs(values[0] - values[1]), .39)
        self.assertGreater(summary(values)["relative_p90_p10_spread"], .8)


if __name__ == "__main__":
    unittest.main()
