"""Analytic fixed-camera checks for conservative RGB rod evidence; no scene GT or models."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_evidence import DEFAULT_CONFIG, STATE_CODES, build_rod_candidate


def fixture(gap_states=None, centers=None):
    centers = centers if centers is not None else [[-2, 0, 0], [-1, 0.3, 0], [0, -0.2, 0], [1, 0.4, 0], [2, -0.3, 0]]
    intrinsic = np.array([[600.0, 0, 400], [0, 600.0, 300], [0, 0, 1]])
    endpoints = np.array([[0.25, -1, 6, 1], [0.25, 1, 6, 1]])
    result = []
    for index, center in enumerate(centers):
        ext = np.c_[np.eye(3), -np.asarray(center)]
        projected = endpoints @ (intrinsic @ ext).T
        line = np.cross(projected[0], projected[1])
        observations = []
        for value in np.linspace(-1, 1, 401):
            point = np.array([0.25, value, 6, 1]) @ (intrinsic @ ext).T
            xy = point[:2] / point[2]
            state = gap_states[index] if gap_states is not None and abs(value) < 0.2 else "accepted"
            if state == "missing":
                continue
            observation = {"xy": xy.tolist(), "width_px": 1.0, "state": state}
            if state == "absent":
                observation["negative_window_x"] = [float(xy[0] - 8), float(xy[0] + 8)]
            observations.append(observation)
        result.append({"view_id": f"view_{index}", "K_index": intrinsic.copy(),
                       "world_to_camera_cv": ext.copy(), "size_wh": [800, 600],
                       "image_line": line, "observations": observations})
    return result


def segments_cross_center(result, field="segments"):
    return any(first[1] < 0 < last[1] for first, last in result[field])


class RodEvidenceTests(unittest.TestCase):
    def test_exact_known_line_and_finite_endpoints(self):
        result = build_rod_candidate(fixture())
        self.assertEqual(result["state"], "accepted", result["rejection_reasons"])
        self.assertEqual(len(result["segments"]), 1)
        np.testing.assert_allclose(result["line"]["anchor"], [0.25, 0, 6], atol=1e-10)
        np.testing.assert_allclose(result["line"]["direction"], [0, 1, 0], atol=1e-10)
        endpoints = result["segments"][0]
        self.assertGreaterEqual(endpoints[0, 1], -1 - 1e-10)
        self.assertLessEqual(endpoints[1, 1], 1 + 1e-10)
        self.assertLess(endpoints[0, 1], -0.99)
        self.assertGreater(endpoints[1, 1], 0.99)
        self.assertTrue(result["geometry"]["passed"])
        self.assertEqual(len(result["geometry"]["leave_one_out"]), 5)

    def test_known_gap_kept_without_dilation_or_bridge(self):
        result = build_rod_candidate(fixture(["absent"] * 5))
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(len(result["segments"]), 2)
        self.assertFalse(segments_cross_center(result))
        # The analytic gap is 0.4 world units. Sampling does not extrapolate into it.
        self.assertLess(result["segments"][0, 1, 1], -0.19)
        self.assertGreater(result["segments"][1, 0, 1], 0.19)
        self.assertGreater(np.max(result["evidence"]["absent_view_counts"]), 0)

    def test_two_explicit_background_hypotheses_veto_three_positive_views(self):
        views = fixture(["accepted", "accepted", "accepted", "absent", "absent"])
        full = build_rod_candidate(views)
        ablated = build_rod_candidate(views, negative_veto=False)
        self.assertEqual(full["state"], "accepted")
        self.assertFalse(segments_cross_center(full))
        self.assertTrue(segments_cross_center(ablated))
        self.assertGreater(full["evidence"]["vetoed_despite_positive_sample_count"], 0)
        np.testing.assert_array_equal(full["line"]["anchor"], ablated["line"]["anchor"])

    def test_missing_unknown_and_occluded_are_not_negative_votes(self):
        for status in ("missing", "unknown", "occluded", "ambiguous"):
            result = build_rod_candidate(fixture(["accepted"] * 3 + [status] * 2))
            self.assertEqual(result["state"], "accepted")
            self.assertTrue(segments_cross_center(result))
            self.assertFalse(result["evidence"]["absent_view_counts"].any())
        missing = build_rod_candidate(fixture(["missing"] * 5))
        self.assertFalse(segments_cross_center(missing))
        self.assertFalse(missing["evidence"]["absent_view_counts"].any())

    def test_negative_window_must_cover_projected_point(self):
        views = fixture(["accepted"] * 3 + ["absent"] * 2)
        for view in views[-2:]:
            for item in view["observations"]:
                if item["state"] == "absent":
                    item["negative_window_x"] = [item["xy"][0] + 30, item["xy"][0] + 40]
        result = build_rod_candidate(views)
        self.assertTrue(segments_cross_center(result))
        self.assertFalse(result["evidence"]["absent_view_counts"].any())
        # A measured strip can legitimately be wider than the fallback center tolerance.
        for view in views[-2:]:
            for item in view["observations"]:
                if item["state"] == "absent":
                    x = item["xy"][0]
                    item["negative_window_x"] = [x - 8, x + 8]
                    item["xy"][0] += 7
        result = build_rod_candidate(views)
        self.assertFalse(segments_cross_center(result))

    def test_negative_without_window_uses_only_local_center_tolerance(self):
        views = fixture(["accepted"] * 3 + ["absent"] * 2)
        for view in views[-2:]:
            for item in view["observations"]:
                if item["state"] == "absent":
                    item.pop("negative_window_x")
                    item["xy"][0] += 4
        result = build_rod_candidate(views)
        self.assertTrue(segments_cross_center(result))
        self.assertFalse(result["evidence"]["absent_view_counts"].any())

    def test_duplicating_records_does_not_create_view_support(self):
        views = fixture(["accepted"] * 2 + ["unknown"] * 3)
        ordinary = build_rod_candidate(views)
        for view in views[:2]:
            view["observations"] = view["observations"] * 8
        duplicate = build_rod_candidate(views)
        np.testing.assert_array_equal(ordinary["evidence"]["positive_view_counts"], duplicate["evidence"]["positive_view_counts"])
        np.testing.assert_allclose(ordinary["segments"], duplicate["segments"])
        self.assertFalse(segments_cross_center(duplicate))

    def test_duplicate_view_names_or_camera_centers_rejected(self):
        views = fixture()
        views[-1]["view_id"] = views[0]["view_id"]
        result = build_rod_candidate(views)
        self.assertIn("duplicate_view_id", result["rejection_reasons"])
        views = fixture()
        views[-1] = copy.deepcopy(views[0])
        views[-1]["view_id"] = "different_name_same_camera"
        result = build_rod_candidate(views)
        self.assertIn("duplicate_camera_center_not_independent_evidence", result["rejection_reasons"])

    def test_parallel_backprojection_planes_and_three_fit_views_rejected(self):
        parallel = fixture(centers=[[0, y, 0] for y in (-2, -1, 0, 1, 2)])
        result = build_rod_candidate(parallel)
        self.assertEqual(result["state"], "rejected")
        self.assertIn("backprojection_planes", result["rejection_reasons"][0])
        few = build_rod_candidate(fixture()[:3])
        self.assertIn("too_few_fit_views_for_leave_one_out", few["rejection_reasons"])

    def test_wrongly_shifted_local_observations_cannot_support_line(self):
        views = fixture()
        for view in views:
            for item in view["observations"]:
                item["xy"][0] += 30
        for gate in (True, False):
            result = build_rod_candidate(views, geometric_gate=gate)
            self.assertEqual(result["state"], "rejected")
            self.assertEqual(len(result["segments"]), 0)
            self.assertTrue(result["ablations"]["positive_pixel_distance_gate"])

    def test_camera_perturbation_rejects_but_retains_shadow_and_loo_reason(self):
        views = fixture()
        views[-1]["world_to_camera_cv"][0, 3] += 0.06
        gated = build_rod_candidate(views)
        ablated = build_rod_candidate(views, geometric_gate=False)
        self.assertEqual(gated["state"], "rejected", gated["rejection_reasons"])
        self.assertFalse(gated["geometry"]["passed"])
        self.assertTrue(any("leave_one_out" in item for item in gated["rejection_reasons"]))
        self.assertEqual(len(gated["segments"]), 0)
        self.assertGreater(len(gated["shadow_segments"]), 0)
        self.assertEqual(ablated["state"], "accepted")
        np.testing.assert_allclose(gated["shadow_segments"], ablated["segments"])
        np.testing.assert_array_equal(gated["evidence"]["view_states"], ablated["evidence"]["view_states"])

    def test_conflicting_positive_negative_same_view_is_ambiguous(self):
        views = fixture()
        for view in views[2:]:
            additions = []
            for item in view["observations"]:
                if abs(item["xy"][1] - (300 - 100 * (-view["world_to_camera_cv"][1, 3]))) < 10:
                    additions.append({**copy.deepcopy(item), "state": "absent"})
            view["observations"].extend(additions)
        result = build_rod_candidate(views)
        self.assertFalse(segments_cross_center(result))
        self.assertTrue(np.any(result["evidence"]["view_states"] == STATE_CODES["ambiguous"]))

    def test_inputs_and_cameras_unchanged(self):
        views = fixture(["absent"] * 5)
        original = copy.deepcopy(views)
        first = build_rod_candidate(views)
        second = build_rod_candidate(views)
        for actual, prior in zip(views, original):
            np.testing.assert_array_equal(actual["K_index"], prior["K_index"])
            np.testing.assert_array_equal(actual["world_to_camera_cv"], prior["world_to_camera_cv"])
            np.testing.assert_array_equal(actual["image_line"], prior["image_line"])
            self.assertEqual(actual["observations"], prior["observations"])
        np.testing.assert_array_equal(first["segments"], second["segments"])
        self.assertEqual(DEFAULT_CONFIG["minimum_positive_views"], 3)

    def test_width_cap_cannot_turn_far_away_pixels_into_support(self):
        views = fixture()
        for view in views:
            for item in view["observations"]:
                item["xy"][0] += 10
                item["width_px"] = 1000
        result = build_rod_candidate(views, geometric_gate=False)
        self.assertEqual(result["state"], "rejected")
        self.assertIn("insufficient_distinct_view_accepted_extent", result["rejection_reasons"])

    def test_world_unit_and_rigid_frame_change_preserves_known_image_line(self):
        views = fixture()
        original = build_rod_candidate(views)
        angle = np.radians(27)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        scale, translation = 3.0, np.array([8.0, -4, 2])
        for view in views:
            ext = view["world_to_camera_cv"]
            new_rotation = ext[:, :3] @ rotation.T
            view["world_to_camera_cv"] = np.c_[new_rotation, scale * ext[:, 3] - new_rotation @ translation]
        transformed = build_rod_candidate(views)
        self.assertEqual(transformed["state"], "accepted", transformed["rejection_reasons"])
        self.assertAlmostEqual(transformed["camera_span"], scale * original["camera_span"])
        self.assertAlmostEqual(transformed["evidence"]["step"], scale * original["evidence"]["step"])
        # Transform back only in the analytic test, never inside the implementation.
        restored = ((transformed["segments"] - translation) @ rotation) / scale
        np.testing.assert_allclose(restored[0, :, 0], 0.25, atol=1e-10)
        np.testing.assert_allclose(restored[0, :, 2], 6, atol=1e-10)
        self.assertGreaterEqual(restored[0, 0, 1], -1 - 1e-10)
        self.assertLessEqual(restored[0, 1, 1], 1 + 1e-10)
        self.assertTrue(np.all(transformed["evidence"]["positive_view_counts"] == 5))

    def test_policy_and_malformed_observation_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_rod_candidate(fixture(), {"step_camera_span_fraction": 0})
        with self.assertRaises(ValueError):
            build_rod_candidate(fixture(), {"invented_parameter": 3})
        views = fixture()
        views[0]["observations"][0]["negative_window_x"] = [8, -8]
        with self.assertRaisesRegex(ValueError, "negative_window_x"):
            build_rod_candidate(views)


if __name__ == "__main__":
    unittest.main()
