"""Analytic identity controls: clicks describe object membership, not 3D matches."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_foreground_identity import select_foreground_identity


def fixture(two_rods=False):
    views, raw = [], []
    centers = [-1., -.3, .4, 1.]
    xs = [0., .6] if two_rods else [0.]
    for i, cx in enumerate(centers):
        candidates = [{"line": [1, 0, -(100 + 20 * (x - cx))]} for x in xs]
        views.append({"view_id": str(i), "K_index": [[100, 0, 100], [0, 100, 100], [0, 0, 1]],
                      "world_to_camera_cv": [[1, 0, 0, -cx], [0, 1, 0, 0], [0, 0, 1, 0]],
                      "size_wh": [200, 200], "y_range": [50, 150], "candidates": candidates,
                      "rgb_sha256": str(i) * 64})
        raw.append({"rows": [{"y": y, "candidates": [{"left_edge": {"x": 96 + 20 * (x - cx)},
                                                       "right_edge": {"x": 104 + 20 * (x - cx)}} for x in xs]} for y in range(70, 131)]})
    proposals = []
    for j, x in enumerate(xs):
        selection = [{"view_id": str(i), "row_matches": [[ri, j] for ri in range(61)]} for i in range(4)]
        proposals.append({"ordinal": j, "finite": {"state": "accepted", "segments": [[[x, -1.5, 5], [x, 1.5, 5]]],
                                                   "candidate_selection": selection}})
    anchors = [{"view_id": str(i), "xy": [100 - 20 * centers[i], 100], "uncertainty_xy_px": [.5, .5], "rgb_sha256": str(i) * 64} for i in (0, 3)]
    return {"search_complete": True, "proposals": proposals}, views, raw, anchors


class ForegroundIdentityTests(unittest.TestCase):
    def test_unique_supported_candidate_does_not_move_geometry(self):
        bundle, views, raw, anchors = fixture()
        before = copy.deepcopy(bundle)
        result = select_foreground_identity(bundle, views, raw, anchors)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["segments"], bundle["proposals"][0]["finite"]["segments"])
        self.assertEqual(bundle, before)
        self.assertFalse(result["geometry_changed"])

    def test_anchors_on_different_parts_are_not_triangulated_as_one_point(self):
        bundle, views, raw, anchors = fixture()
        anchors[0]["xy"][1] = 80
        anchors[1]["xy"][1] = 120
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "accepted")

    def test_same_rgb_different_valid_picks_select_different_objects(self):
        bundle, views, raw, anchors = fixture(True)
        first = select_foreground_identity(bundle, views, raw, anchors)
        for anchor in anchors:
            anchor["xy"][0] += 12
        second = select_foreground_identity(bundle, views, raw, anchors)
        self.assertEqual((first["selected_ordinal"], second["selected_ordinal"]), (0, 1))
        # If the user consistently points at the wrong object, geometry cannot read their mind.
        self.assertEqual(second["state"], "accepted")

    def test_mixed_object_picks_and_background_clicks_fail(self):
        bundle, views, raw, anchors = fixture(True)
        anchors[1]["xy"][0] += 12
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "rejected")
        for anchor in anchors:
            anchor["xy"][0] += 30
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "rejected")

    def test_missing_clicks_never_upgrade_geometry_to_identity(self):
        bundle, views, raw, anchors = fixture()
        for count in (0, 1):
            self.assertEqual(select_foreground_identity(bundle, views, raw, anchors[:count])["state"], "unresolved")

    def test_uncertainty_crossing_boundary_is_not_snapped_inside(self):
        bundle, views, raw, anchors = fixture()
        anchors[0]["xy"][0] += 3.8
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "unresolved")
        anchors[0]["xy"][0] += 2
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "rejected")

    def test_missing_row_is_unknown_and_an_unknown_rival_blocks_uniqueness(self):
        bundle, views, raw, anchors = fixture()
        rival = copy.deepcopy(bundle["proposals"][0])
        rival["ordinal"] = 1
        rival["finite"]["candidate_selection"][0]["row_matches"] = [[ri, 0] for ri in range(61) if ri != 30]
        bundle["proposals"].append(rival)
        result = select_foreground_identity(bundle, views, raw, anchors)
        self.assertEqual(result["state"], "unresolved")
        self.assertEqual([r["state"] for r in result["proposal_audit"]], ["supported", "unresolved"])

    def test_dropped_view_does_not_silently_reenter_identity_support(self):
        bundle, views, raw, anchors = fixture()
        bundle["proposals"][0]["finite"]["candidate_selection"][0]["row_matches"] = []
        result = select_foreground_identity(bundle, views, raw, anchors)
        self.assertEqual(result["state"], "unresolved")
        self.assertEqual(result["proposal_audit"][0]["anchors"][0]["rows"][0]["reason"], "no_selected_measured_row")
        # A genuinely different supplied view can resolve it; the selector never invents that click.
        anchors[0] = {"view_id": "1", "xy": [106, 100], "uncertainty_xy_px": [.5, .5], "rgb_sha256": "1" * 64}
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "accepted")

    def test_real_gap_and_endpoint_are_not_filled_for_clicks(self):
        bundle, views, raw, anchors = fixture()
        bundle["proposals"][0]["finite"]["segments"] = [[[0, -1.5, 5], [0, -.3, 5]], [[0, .3, 5], [0, 1.5, 5]]]
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "rejected")
        for a in anchors:
            a["xy"][1] = 80
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "accepted")
        anchors[0]["xy"][1] = 70
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "unresolved")

    def test_vertical_uncertainty_requires_all_pixel_rows(self):
        bundle, views, raw, anchors = fixture()
        anchors[0]["uncertainty_xy_px"][1] = 1.0
        bundle["proposals"][0]["finite"]["candidate_selection"][0]["row_matches"] = [[ri, 0] for ri in range(61) if ri != 29]
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "unresolved")

    def test_duplicate_views_cameras_and_detached_images_are_rejected(self):
        for problem in ("view", "camera", "hash", "bounds", "nan"):
            bundle, views, raw, anchors = fixture()
            if problem == "view":
                anchors[1] = copy.deepcopy(anchors[0])
            if problem == "camera":
                views[3]["world_to_camera_cv"] = views[0]["world_to_camera_cv"]
            if problem == "hash":
                anchors[0]["rgb_sha256"] = 'f' * 64
            if problem == "bounds":
                anchors[0]["xy"] = [-1, 100]
            if problem == "nan":
                anchors[0]["uncertainty_xy_px"][0] = float('nan')
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                select_foreground_identity(bundle, views, raw, anchors)

    def test_incomplete_search_or_failed_geometry_cannot_publish(self):
        bundle, views, raw, anchors = fixture()
        bundle["search_complete"] = False
        self.assertEqual(select_foreground_identity(bundle, views, raw, anchors)["state"], "unresolved")
        bundle["search_complete"] = True
        bundle["proposals"][0]["finite"]["state"] = "rejected"
        result = select_foreground_identity(bundle, views, raw, anchors)
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(np.asarray(result["segments"]).shape, (0, 2, 3))


if __name__ == "__main__":
    unittest.main()
