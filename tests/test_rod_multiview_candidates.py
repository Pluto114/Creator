"""Tests for multi-hypothesis image lines and calibrated cross-view association."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_multiview_candidates import (
    associate_multiview_lines,
    enumerate_image_lines,
)


def camera(x):
    return {
        "K_index": np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]),
        "world_to_camera_cv": np.c_[np.eye(3), [-x, 0, 0]],
    }


def projected_line(x, view):
    points = np.array([[x, -1, 6, 1], [x, 1, 6, 1]])
    pixels = points @ (view["K_index"] @ view["world_to_camera_cv"]).T
    return np.cross(pixels[0], pixels[1])


class MultiViewCandidateTests(unittest.TestCase):
    def test_image_enumerator_retains_two_supported_lines(self):
        rows = []
        for y in range(20, 181, 2):
            rows.append({"y": y, "candidates": [{"center_x": 60.0}, {"center_x": 75.0}]})
        models = enumerate_image_lines(
            {"rows": rows},
            {"minimum_rows": 50, "minimum_y_span": 100, "inlier_distance_px": 0.5,
             "trials": 400, "seed": 9, "deduplicate_separation_px": 0.75, "maximum_models": 4},
        )
        self.assertEqual(len(models), 2)
        self.assertEqual(sorted(round(model["intercept"]) for model in models), [60, 75])

    def test_unique_consistent_line_is_accepted(self):
        views = []
        for index, x in enumerate([-1, -.5, 0, .5, 1]):
            view = {"view_id": str(index), **camera(x), "y_range": [60, 180]}
            view["candidates"] = [{"line": projected_line(0.1, view)}]
            views.append(view)
        result = associate_multiview_lines(views, self.config(4))
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["selected"]["support_view_count"], 5)
        np.testing.assert_allclose(result["selected"]["model"]["anchor"][[0, 2]], [.1, 6], atol=1e-10)

    def test_two_consistent_lines_are_ambiguous(self):
        views = []
        for index, x in enumerate([-1, -.5, 0, .5, 1]):
            view = {"view_id": str(index), **camera(x), "y_range": [60, 180]}
            view["candidates"] = [{"line": projected_line(value, view)} for value in (-.15, .2)]
            views.append(view)
        result = associate_multiview_lines(views, self.config(4))
        self.assertEqual(result["state"], "ambiguous")
        self.assertGreaterEqual(len(result["alternatives"]), 1)

    def test_inconsistent_lines_do_not_reach_four_view_support(self):
        views = []
        for index, (camera_x, offset) in enumerate(zip([-1, -.5, 0, .5, 1], [-12, 7, -9, 13, -4])):
            view = {"view_id": str(index), **camera(camera_x), "y_range": [60, 180]}
            base = projected_line(0, view)
            base = base / base[0]
            base[2] -= offset
            view["candidates"] = [{"line": base}]
            views.append(view)
        result = associate_multiview_lines(views, self.config(4))
        self.assertEqual(result["state"], "rejected")

    @staticmethod
    def config(minimum):
        return {"fit_view_count": 3, "minimum_support_views": minimum,
                "reprojection_threshold_px": 1, "deduplicate_separation_px": .6,
                "ambiguity_support_fraction": .9, "ambiguity_separation_px": 2.5,
                "maximum_hypotheses": 256}


if __name__ == "__main__":
    unittest.main()
