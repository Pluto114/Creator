"""Tests for multi-hypothesis image lines and calibrated cross-view association."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_multiview_candidates import (
    apply_guide_identity_guard,
    apply_nested_band_guard,
    associate_multiview_lines,
    enumerate_image_lines,
    image_line_from_endpoints,
)


def camera(x, *, homogeneous=False):
    extrinsic = np.c_[np.eye(3), [-x, 0, 0]]
    if homogeneous:
        extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
    return {
        "K_index": np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]),
        "world_to_camera_cv": extrinsic,
    }


def projected_line(x, view):
    points = np.array([[x, -1, 6, 1], [x, 1, 6, 1]])
    extrinsic = np.asarray(view["world_to_camera_cv"])
    pixels = points @ (view["K_index"] @ extrinsic[:3, :]).T
    return np.cross(pixels[0], pixels[1])


class MultiViewCandidateTests(unittest.TestCase):
    def test_image_enumerator_retains_two_supported_lines(self):
        rows = []
        for y in range(20, 181, 2):
            rows.append(
                {"y": y, "candidates": [
                    {"center_x": 60.0, "width": 4.0,
                     "left_edge": {"x": 58.0}, "right_edge": {"x": 62.0}},
                    {"center_x": 75.0, "width": 4.0,
                     "left_edge": {"x": 73.0}, "right_edge": {"x": 77.0}},
                ]}
            )
        models = enumerate_image_lines(
            {"rows": rows},
            {"minimum_rows": 50, "minimum_y_span": 100, "inlier_distance_px": 0.5,
             "trials": 400, "seed": 9, "deduplicate_separation_px": 0.75, "maximum_models": 4},
        )
        self.assertEqual(len(models), 2)
        self.assertEqual(sorted(round(model["intercept"]) for model in models), [60, 75])

    def test_nested_band_is_warning_not_widest_candidate_selection(self):
        rows = []
        for y in range(20, 181, 2):
            narrow = {"center_x": 60.0, "width": 4.0,
                      "left_edge": {"x": 58.0}, "right_edge": {"x": 62.0}}
            wide = {"center_x": 61.0, "width": 12.0,
                    "left_edge": {"x": 55.0}, "right_edge": {"x": 67.0}}
            rows.append({"y": y, "candidates": [narrow, wide]})
        models = enumerate_image_lines(
            {"rows": rows},
            {"minimum_rows": 50, "minimum_y_span": 100, "inlier_distance_px": 0.5,
             "trials": 400, "seed": 9, "deduplicate_separation_px": 0.75, "maximum_models": 4},
        )
        narrow = next(model for model in models if round(model["intercept"]) == 60)
        self.assertEqual(narrow["enclosing_wider_row_fraction"], 1)
        views = [{"candidates": [narrow]} for _ in range(3)]
        accepted = {"state": "accepted", "reason": "unique", "selected": {
            "supporting_views": [0, 1, 2],
            "matches": [{"candidate_index": 0}] * 3,
        }}
        guarded = apply_nested_band_guard(
            accepted, views, {"minimum_nested_views": 2, "minimum_nested_row_fraction": .25}
        )
        self.assertEqual(guarded["state"], "ambiguous")
        self.assertEqual(accepted["state"], "accepted")

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

    def test_homogeneous_camera_reprojects_after_fitting(self):
        views = []
        for index, x in enumerate([-1, -.5, 0, .5, 1]):
            view = {
                "view_id": str(index), **camera(x, homogeneous=True), "y_range": [60, 180]
            }
            view["candidates"] = [{"line": projected_line(0.1, view)}]
            views.append(view)
        result = associate_multiview_lines(views, self.config(4))
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["selected"]["support_view_count"], 5)

    def test_guide_identity_guard_rejects_a_different_physical_line(self):
        views = []
        for index, x in enumerate([-1, -.5, 0, .5, 1]):
            view = {"view_id": str(index), **camera(x), "y_range": [60, 180]}
            guide = projected_line(0.1, view)
            view["guide_line"] = guide / np.linalg.norm(guide[:2])
            views.append(view)
        accepted = {
            "state": "accepted",
            "reason": "unique_multiview_line",
            "selected": {
                "model": {"anchor": [0.5, 0, 6], "direction": [0, 1, 0]},
            },
        }
        guarded = apply_guide_identity_guard(
            accepted, views,
            {"maximum_median_residual_px": 7.5, "minimum_views": 3,
             "minimum_fraction": .75},
        )
        self.assertEqual(guarded["state"], "rejected")
        self.assertEqual(accepted["state"], "accepted")
        accepted["selected"]["model"]["anchor"][0] = 0.1
        guarded = apply_guide_identity_guard(
            accepted, views,
            {"maximum_median_residual_px": 7.5, "minimum_views": 3,
             "minimum_fraction": .75},
        )
        self.assertEqual(guarded["state"], "accepted")

    def test_image_line_from_endpoints_validates_orientation(self):
        line = image_line_from_endpoints([[60, 20], [62, 180]])
        np.testing.assert_allclose(line @ [60, 20, 1], 0, atol=1e-12)
        with self.assertRaises(ValueError):
            image_line_from_endpoints([[20, 60], [180, 60]])

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
