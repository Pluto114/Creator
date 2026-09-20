"""Test cylinder-screen geometry against an independently derived quadratic."""
import copy
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from creator_eval.line_controls import LineFitDegenerate  # noqa: E402
from creator_eval.rod_cylinder_gate import (  # noqa: E402
    recheck_finite_cylinder,
    screen_cylinder,
    select_cylinder_hypothesis,
    silhouette_x,
)
from run_rod_cylinder_controls import analytic_case  # noqa: E402

EXTENT = {"minimum_second_to_first_plane_ratio": 1e-5, "minimum_direction_gap_ratio": 1e-8}


class CylinderScreenTests(unittest.TestCase):
    def test_exact_tangents_across_scales_views_and_tilt(self):
        for radius in (.004, .015, .08, .25):
            for tilt in (0, .18):
                with self.subTest(radius=radius, tilt=tilt):
                    h, views, obs = analytic_case(radius, tilt)
                    result = screen_cylinder(h, views, obs)
                    self.assertTrue(result["passed"])
                    self.assertAlmostEqual(result["radius"], radius, places=10)
                    for view, raw in zip(views, obs):
                        got = silhouette_x(h["model"], view, radius, [r["y"] for r in raw["rows"]])
                        expected = [[r["candidates"][0][s + "_edge"]["x"] for r in raw["rows"]] for s in ("left", "right")]
                        np.testing.assert_allclose(got, expected, atol=1e-8, rtol=0)

    def test_half_pixel_noise_and_view_balancing(self):
        h, views, obs = analytic_case(noise=.5, counts=[600, 60, 60, 60])
        self.assertTrue(screen_cylinder(h, views, obs)["passed"])
        for row in obs[-1]["rows"]:
            row["candidates"][0]["right_edge"]["x"] += 3
        result = screen_cylinder(h, views, obs)
        self.assertFalse(result["passed"])
        self.assertFalse(result["views"][-1]["passed"])

    def test_ninety_percent_boundary_counts_rows_per_side(self):
        h, views, obs = analytic_case(counts=[100] * 4)
        for row in obs[0]["rows"][:10]:
            row["candidates"][0]["right_edge"]["x"] += 3
        result = screen_cylinder(h, views, obs)
        self.assertEqual(result["views"][0]["right"]["within_tolerance_fraction"], .9)
        self.assertTrue(result["passed"])
        obs[0]["rows"][10]["candidates"][0]["right_edge"]["x"] += 3
        self.assertFalse(screen_cylinder(h, views, obs)["passed"])

    def test_wrong_axis_and_other_real_object_have_different_semantics(self):
        h, views, obs = analytic_case()
        self.assertTrue(screen_cylinder(h, views, obs)["passed"])  # Geometry alone cannot know whose rod this is.
        h["model"]["anchor"][0] += .06
        self.assertFalse(screen_cylinder(h, views, obs)["passed"])

    def test_transform_scale_axis_reversal_and_view_order_invariance(self):
        h, views, obs = analytic_case(tilt=.18)
        old = screen_cylinder(h, views, obs)
        theta = .6
        r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        shift, scale = np.array([.3, -.1, .6]), 7.
        h["model"]["anchor"] = scale * r @ (h["model"]["anchor"] + 12 * h["model"]["direction"]) + shift
        h["model"]["direction"] = -r @ h["model"]["direction"]
        for view in views:
            ext = view["world_to_camera_cv"]
            rotation = ext[:, :3] @ r.T
            view["world_to_camera_cv"] = np.c_[rotation, scale * ext[:, 3] - rotation @ shift]
        result = screen_cylinder(h, views[::-1], obs[::-1])
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["radius"], scale * old["radius"], places=10)

    def test_degenerate_cameras_bad_rows_and_invalid_policies_fail(self):
        h, views, obs = analytic_case()
        for radius in (0, -1, 100):
            with self.assertRaises(LineFitDegenerate):
                silhouette_x(h["model"], views[0], radius, [120])
        for policy in ({"edge_tolerance_px": float("nan")}, {"minimum_side_fraction": 0}, {"minimum_views": True}):
            with self.assertRaises(ValueError):
                screen_cylinder(h, views, obs, policy)
        views[0]["candidates"][0]["row_matches"].append([0, 0, 0])
        with self.assertRaises(ValueError):
            screen_cylinder(h, views, obs)

    def test_incomplete_search_does_not_become_unique_and_input_is_unchanged(self):
        h, views, obs = analytic_case()
        original = copy.deepcopy(h)
        association = {"state": "ambiguous", "selected": h, "alternatives": [], "search_complete": False}
        result = select_cylinder_hypothesis(association, views, obs, EXTENT)
        self.assertEqual(result["state"], "ambiguous")
        self.assertEqual(result["reason"], "search_budget_exhausted")
        np.testing.assert_array_equal(h["model"]["anchor"], original["model"]["anchor"])
        self.assertEqual(association["state"], "ambiguous")

    def test_same_assignment_is_refitted_once_and_final_axis_is_rechecked(self):
        h, views, obs = analytic_case()
        association = {"state": "ambiguous", "selected": h, "alternatives": [copy.deepcopy(h)], "search_complete": True}
        result = select_cylinder_hypothesis(association, views, obs, EXTENT)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["cylinder_screen"]["distinct_assignment_count"], 1)
        wrong = copy.deepcopy(h["model"])
        wrong["anchor"][0] += .06
        finite = {"state": "accepted", "line": wrong, "segments": [[[0, 0, 5], [0, 1, 5]]],
                  "shadow_segments": [[[0, 0, 5], [0, 1, 5]]], "rejection_reasons": []}
        checked = recheck_finite_cylinder(finite, h, views, obs)
        self.assertEqual(checked["state"], "rejected")
        self.assertEqual(len(checked["segments"]), 0)
        self.assertEqual(len(checked["shadow_segments"]), 1)
        self.assertEqual(finite["state"], "accepted")

    def test_control_cli_is_standalone(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_rod_cylinder_controls.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
