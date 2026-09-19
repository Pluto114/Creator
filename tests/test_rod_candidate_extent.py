"""No-truth checks for binding a chosen image candidate to finite RGB support."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_candidate_extent import bound_selected_candidate


def fixture(*, missing_middle=False, raw_middle_state="unknown", support_views=5):
    intrinsic = np.array([[600.0, 0, 400], [0, 600.0, 300], [0, 0, 1]])
    centers = [[-2, 0, 0], [-1, .3, 0], [0, -.2, 0], [1, .4, 0], [2, -.3, 0]]
    views, extracted = [], []
    for index, center in enumerate(centers):
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = -np.asarray(center)
        projection = intrinsic @ extrinsic[:3]
        endpoints = np.array([[.25, -1, 6, 1], [.25, 1, 6, 1]]) @ projection.T
        line = np.cross(endpoints[0], endpoints[1])
        rows, matches = [], []
        for row_index, position in enumerate(np.linspace(-1, 1, 401)):
            projected = np.array([.25, position, 6, 1]) @ projection.T
            x, y = projected[:2] / projected[2]
            missing = missing_middle and abs(position) < .2
            rows.append({
                "guide_x": float(x + 20), "y": float(y),
                "status": raw_middle_state if missing else "ambiguous",
                "candidates": [
                    {"center_x": float(x + 30), "width": 1.0},
                    {"center_x": float(x), "width": 1.0},
                ],
            })
            if not missing:
                matches.append([row_index, 1, 0.0])
        views.append({
            "view_id": str(index), "K_index": intrinsic.tolist(),
            "world_to_camera_cv": extrinsic.tolist(), "size_wh": [800, 600],
            "y_range": [float(rows[0]["y"]), float(rows[-1]["y"])],
            "candidates": [{"line": [1, 0, 0], "row_matches": []},
                           {"line": line.tolist(), "row_matches": matches}],
        })
        extracted.append({"rows": rows})
    association = {
        "state": "accepted",
        "selected": {
            "supporting_views": list(range(support_views)),
            "matches": [{"candidate_index": 1} for _ in views],
        },
    }
    return association, views, extracted


def crosses_middle(result):
    return any(start[1] < 0 < end[1] for start, end in result["segments"])


class RodCandidateExtentTests(unittest.TestCase):
    def test_correct_selected_rows_preserve_finite_endpoints_and_inputs(self):
        association, views, observations = fixture()
        before = copy.deepcopy((association, views, observations))
        result = bound_selected_candidate(association, views, observations)
        self.assertEqual((association, views, observations), before)
        self.assertEqual(result["state"], "accepted", result["rejection_reasons"])
        self.assertEqual(result["segments"].shape, (1, 2, 3))
        np.testing.assert_allclose(result["line"]["anchor"], [.25, 0, 6], atol=1e-10)
        endpoints = result["segments"][0, :, 1]
        self.assertTrue(np.all(np.abs(endpoints) <= 1 + 1e-10))
        self.assertTrue(np.all(np.abs(endpoints) >= .99))
        self.assertTrue(result["geometry"]["passed"])
        self.assertEqual(len(result["geometry"]["leave_one_out"]), 5)
        self.assertTrue(np.all(result["evidence"]["positive_view_counts"] == 5))
        for selection in result["candidate_selection"]:
            self.assertEqual(selection["candidate_index"], 1)
            self.assertEqual(selection["row_matches"], [[i, 1] for i in range(401)])

    def test_true_gap_and_occlusion_are_not_filled_from_an_infinite_line(self):
        for state in ("absent", "occluded", "unknown", "ambiguous"):
            with self.subTest(state=state):
                args = fixture(missing_middle=True, raw_middle_state=state)
                result = bound_selected_candidate(*args)
                self.assertEqual(result["state"], "accepted", result["rejection_reasons"])
                self.assertEqual(len(result["segments"]), 2)
                self.assertFalse(crosses_middle(result))
                self.assertLess(result["segments"][0, 1, 1], -.19)
                self.assertGreater(result["segments"][1, 0, 1], .19)
                self.assertFalse(result["evidence"]["absent_view_counts"].any())
                self.assertFalse(result["ablations"]["negative_veto"])
                # Raw rows even contain a plausible center here. Without a selected
                # row match it stays unknown; seeing candidates is not support.
                middle = np.abs(result["evidence"]["parameters"]) < .18
                self.assertTrue(np.all(result["evidence"]["positive_view_counts"][middle] == 0))

    def test_unselected_view_cannot_supply_positive_rows(self):
        association, views, observations = fixture(support_views=4)
        result = bound_selected_candidate(association, views, observations)
        self.assertEqual(result["state"], "accepted", result["rejection_reasons"])
        self.assertTrue(np.all(result["evidence"]["positive_view_counts"] == 4))
        self.assertIsNone(result["candidate_selection"][-1]["candidate_index"])
        self.assertEqual(result["candidate_selection"][-1]["row_matches"], [])

    def test_three_view_identity_is_insufficient_for_finite_leave_one_out(self):
        result = bound_selected_candidate(*fixture(support_views=3))
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(len(result["segments"]), 0)
        self.assertIn("too_few_fit_views_for_leave_one_out", result["rejection_reasons"])

    def test_rejected_and_ambiguous_associations_never_emit_even_shadow_segments(self):
        for state in ("rejected", "ambiguous"):
            result = bound_selected_candidate({"state": state}, [], [])
            self.assertEqual(result["state"], "rejected")
            self.assertEqual(result["segments"].shape, (0, 2, 3))
            self.assertEqual(result["shadow_segments"].shape, (0, 2, 3))
            self.assertEqual(result["rejection_reasons"], ["association_not_accepted"])

    def test_duplicate_and_invalid_matches_do_not_silently_change_identity(self):
        association, views, observations = fixture()
        views[0]["candidates"][1]["row_matches"].append([0, 0, 0.0])
        with self.assertRaisesRegex(ValueError, "each raw row only once"):
            bound_selected_candidate(association, views, observations)
        views[0]["candidates"][1]["row_matches"][-1] = [9999, 0, 0.0]
        with self.assertRaisesRegex(ValueError, "outside its source collection"):
            bound_selected_candidate(association, views, observations)
        with self.assertRaisesRegex(ValueError, "view order and count"):
            bound_selected_candidate(association, views, observations[:-1])


if __name__ == "__main__":
    unittest.main()
