"""Pixel evidence checks, including coherent wrong data that must still pass."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_validation_evidence import check_endpoint_reprojection, check_target_anchors


def fixture():
    frames = []
    for i, x in enumerate([-.8, -.4, 0., .4, .8]):
        frames.append(dict(view_id=f"v{i}", size_wh=[640, 480],
            source_sha256=hashlib.sha256(f"measurement-{i}".encode()).hexdigest(),
            source_kind="synthetic_pixel_measurements",
            K_index=[[100., 0., 320.], [0., 100., 240.], [0., 0., 1.]],
            world_to_camera_cv=[[1., 0., 0., -x], [0., 1., 0., 0.], [0., 0., 1., 0.]]))
    segments = [[[0., -1., 4.], [0., 1., 4.]]]
    endpoints = [dict(endpoint_id=f"end-{i}", observations=observations(frames, point))
                 for i, point in enumerate(segments[0])]
    anchors = [dict(o, uncertainty_xy_px=[.5, .5]) for o in observations(frames[:2], [0., 0., 4.])]
    return frames, segments, endpoints, anchors


def observations(frames, point):
    result = []
    for frame in frames:
        xyz = np.asarray(frame["world_to_camera_cv"]) @ np.r_[point, 1.]
        q = np.asarray(frame["K_index"]) @ xyz
        result.append(dict(view_id=frame["view_id"], source_sha256=frame["source_sha256"],
                           xy=(q[:2] / q[2]).tolist()))
    return result


class EndpointEvidenceTests(unittest.TestCase):
    def test_clean_endpoints_leave_every_observation_out_and_do_not_mutate(self):
        frames, _, tracks, _ = fixture()
        saved = copy.deepcopy((frames, tracks))
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "supported")
        self.assertLess(result["maximum_residual_px"], 1e-10)
        for endpoint in result["per_endpoint"]:
            self.assertEqual(endpoint["scored_count"], 5)
            for row in endpoint["held_out"]:
                self.assertNotIn(row["view_id"], row["fit_view_ids"])
                self.assertEqual(len(row["fit_view_ids"]), 4)
        self.assertEqual((frames, tracks), saved)
        self.assertFalse(result["rod_geometry_modified"])
        json.dumps(result, allow_nan=False)

    def test_four_views_are_enough_but_three_preserve_unresolved_rows(self):
        frames, _, tracks, _ = fixture()
        for track in tracks:
            track["observations"] = track["observations"][:4]
        self.assertEqual(check_endpoint_reprojection(frames, tracks)["state"], "supported")
        tracks[0]["observations"].pop()
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "unresolved")
        self.assertEqual(len(result["per_endpoint"][0]["held_out"]), 3)
        self.assertIsNone(result["per_endpoint"][0]["maximum_residual_px"])

    def test_corrupt_heldout_pixel_does_not_enter_its_triangulation(self):
        frames, _, tracks, _ = fixture()
        original = check_endpoint_reprojection(frames, tracks)
        tracks[0]["observations"][0]["xy"][1] += 20.
        result = check_endpoint_reprojection(frames, tracks)
        row = result["per_endpoint"][0]["held_out"][0]
        self.assertEqual(result["state"], "contradicted")
        self.assertAlmostEqual(row["residual_px"], 20.)
        np.testing.assert_allclose(row["predicted_xy"], original["per_endpoint"][0]["held_out"][0]["predicted_xy"])

    def test_coherent_wrong_endpoint_can_pass_so_this_is_not_truth(self):
        frames, _, _, _ = fixture()
        tracks = [dict(endpoint_id="coherent-other-point", observations=observations(frames, [1.2, .8, 8.]))]
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "supported")
        self.assertIn("not 3D accuracy", result["scope"])

    def test_duplicate_endpoint_or_view_is_unresolved(self):
        frames, _, tracks, _ = fixture()
        tracks[1]["endpoint_id"] = tracks[0]["endpoint_id"]
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "unresolved")
        self.assertIn("duplicate_endpoint_id", result["reasons"])
        _, _, tracks, _ = fixture()
        tracks[0]["observations"][1] = copy.deepcopy(tracks[0]["observations"][0])
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "unresolved")
        self.assertIn("duplicate_endpoint_view", result["per_endpoint"][0]["reasons"])

    def test_nonfinite_or_missing_measurement_remains_null(self):
        for change in (lambda o: o.update(xy=[float("nan"), 20.]), lambda o: o.pop("xy"),
                       lambda o: o.update(source_sha256="0" * 64), lambda o: o.update(xy=[-2., 10.])):
            with self.subTest(change=change):
                frames, _, tracks, _ = fixture()
                change(tracks[0]["observations"][0])
                result = check_endpoint_reprojection(frames, tracks)
                self.assertEqual(result["state"], "unresolved")
                self.assertIsNone(result["per_endpoint"][0]["held_out"][0]["residual_px"])
                json.dumps(result, allow_nan=False)

    def test_missing_endpoint_tracks_are_not_zero_error(self):
        frames, _, _, _ = fixture()
        result = check_endpoint_reprojection(frames, [])
        self.assertEqual(result["state"], "unresolved")
        self.assertIsNone(result["maximum_residual_px"])

    def test_behind_camera_and_parallel_rays_are_unresolved(self):
        frames, _, _, _ = fixture()
        tracks = [dict(endpoint_id="behind", observations=observations(frames, [0., 0., -4.]))]
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "unresolved")
        self.assertIsNone(result["maximum_residual_px"])
        self.assertIn("point_behind_camera", result["per_endpoint"][0]["held_out"][0]["reasons"])
        for observation in tracks[0]["observations"]:
            observation["xy"] = [320., 240.]
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "unresolved")
        self.assertIsNone(result["maximum_residual_px"])

    def test_distinct_centers_on_one_optical_ray_do_not_constrain_depth(self):
        frames, _, _, _ = fixture()
        for i, frame in enumerate(frames):
            frame["world_to_camera_cv"][0][3] = 0.
            frame["world_to_camera_cv"][2][3] = float(i + 1)
        tracks = [dict(endpoint_id="axial", observations=observations(frames, [0., 0., 4.]))]
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "unresolved")
        self.assertIsNone(result["maximum_residual_px"])
        self.assertIn("degenerate_triangulation", result["per_endpoint"][0]["held_out"][0]["reasons"])

    def test_invalid_identifiers_do_not_leak_nonfinite_json(self):
        frames, segments, tracks, anchors = fixture()
        tracks[0]["endpoint_id"] = float("nan")
        tracks[0]["observations"][0]["view_id"] = float("nan")
        anchors[0]["view_id"] = float("nan")
        a = check_endpoint_reprojection(frames, tracks)
        b = check_target_anchors(frames, segments, anchors)
        self.assertEqual(a["state"], "unresolved")
        self.assertEqual(b["state"], "unresolved")
        json.dumps([a, b], allow_nan=False)

    def test_repeated_centers_do_not_count_as_new_views(self):
        frames, _, tracks, _ = fixture()
        frames[1]["world_to_camera_cv"] = copy.deepcopy(frames[0]["world_to_camera_cv"])
        result = check_endpoint_reprojection(frames, tracks)
        self.assertEqual(result["state"], "unresolved")
        self.assertIn("repeated_camera_center", result["per_endpoint"][0]["reasons"])


class TargetEvidenceTests(unittest.TestCase):
    def test_clean_fixed_geometry_supported_and_inputs_unchanged(self):
        frames, segments, _, anchors = fixture()
        saved = copy.deepcopy((frames, segments, anchors))
        result = check_target_anchors(frames, segments, anchors)
        self.assertEqual(result["state"], "supported")
        self.assertEqual((frames, segments, anchors), saved)
        self.assertFalse(result["geometry_modified"])
        self.assertFalse(result["target_selected"])
        self.assertAlmostEqual(result["per_anchor"][0]["uncertainty_radius_px"], np.sqrt(.5))
        json.dumps(result, allow_nan=False)

    def test_uncertainty_disk_supported_crossing_and_contradicted(self):
        for offset, expected in ((1., "supported"), (1.8, "unresolved"), (3., "contradicted")):
            with self.subTest(offset=offset):
                frames, segments, _, anchors = fixture()
                anchors[0]["xy"][0] += offset
                result = check_target_anchors(frames, segments, anchors)
                self.assertEqual(result["state"], expected)
                self.assertAlmostEqual(result["per_anchor"][0]["distance_px"], offset)

    def test_exact_threshold_rules(self):
        frames, segments, _, anchors = fixture()
        anchors[0]["xy"][0] += 2.
        anchors[0]["uncertainty_xy_px"] = [0., 0.]
        self.assertEqual(check_target_anchors(frames, segments, anchors)["state"], "supported")
        anchors[0]["xy"][0] += .5
        anchors[0]["uncertainty_xy_px"] = [.5, 0.]
        self.assertEqual(check_target_anchors(frames, segments, anchors)["state"], "unresolved")

    def test_finite_endpoints_and_internal_gap_are_not_bridged(self):
        frames, segments, _, anchors = fixture()
        anchors = [dict(o, uncertainty_xy_px=[.5, .5]) for o in observations(frames[:2], [0., 1.2, 4.])]
        self.assertEqual(check_target_anchors(frames, segments, anchors)["state"], "contradicted")
        frames, _, _, anchors = fixture()
        gapped = [[[0., -1., 4.], [0., -.2, 4.]], [[0., .2, 4.], [0., 1., 4.]]]
        result = check_target_anchors(frames, gapped, anchors)
        self.assertEqual(result["state"], "contradicted")
        self.assertAlmostEqual(result["per_anchor"][0]["distance_px"], 5.)

    def test_one_anchor_cannot_support_but_can_contradict(self):
        frames, segments, _, anchors = fixture()
        self.assertEqual(check_target_anchors(frames, segments, anchors[:1])["state"], "unresolved")
        anchors[0]["xy"][0] += 5.
        result = check_target_anchors(frames, segments, anchors[:1])
        self.assertEqual(result["state"], "contradicted")
        self.assertIn("insufficient_anchor_views", result["reasons"])

    def test_missing_anchor_does_not_override_real_contradiction(self):
        frames, segments, _, anchors = fixture()
        anchors[0]["xy"][0] += 5.
        anchors[1].pop("source_sha256")
        result = check_target_anchors(frames, segments, anchors)
        self.assertEqual(result["state"], "contradicted")
        self.assertIsNone(result["per_anchor"][1]["distance_px"])

    def test_same_view_or_center_cannot_support_twice(self):
        frames, segments, _, anchors = fixture()
        duplicate = [anchors[0], copy.deepcopy(anchors[0])]
        self.assertEqual(check_target_anchors(frames, segments, duplicate)["state"], "unresolved")
        frames[1]["world_to_camera_cv"] = copy.deepcopy(frames[0]["world_to_camera_cv"])
        anchors[1]["xy"] = copy.deepcopy(anchors[0]["xy"])
        result = check_target_anchors(frames, segments, anchors)
        self.assertEqual(result["state"], "unresolved")
        self.assertIn("repeated_camera_center", result["reasons"])

    def test_missing_nonfinite_zero_length_and_behind_segments_remain_null(self):
        frames, _, _, anchors = fixture()
        for segments in ([], [[[0., 0., 4.], [0., 0., 4.]]], [[[0., 0., 4.], [float("nan"), 1., 4.]]],
                         [[[0., -1., -4.], [0., 1., -4.]]]):
            with self.subTest(segments=segments):
                result = check_target_anchors(frames, segments, anchors)
                self.assertEqual(result["state"], "unresolved")
                self.assertIsNone(result["maximum_distance_px"])
                self.assertTrue(all(row["distance_px"] is None for row in result["per_anchor"]))
                json.dumps(result, allow_nan=False)

    def test_anchors_and_full_uncertainty_rectangle_must_stay_in_image(self):
        for xy, uncertainty in (([640., 240.], [.5, .5]), ([639., 240.], [1., .5]),
                                ([float("nan"), 240.], [.5, .5]), ([320., 240.], [-.5, .5])):
            with self.subTest(xy=xy, uncertainty=uncertainty):
                frames, segments, _, anchors = fixture()
                anchors[0].update(xy=xy, uncertainty_xy_px=uncertainty)
                result = check_target_anchors(frames, segments, anchors)
                self.assertEqual(result["state"], "unresolved")
                self.assertIsNone(result["per_anchor"][0]["distance_px"])
                json.dumps(result, allow_nan=False)

    def test_camera_and_provenance_errors_remain_unresolved(self):
        mutations = [lambda f: f.update(source_kind="pretend_rgb"), lambda f: f.update(source_sha256="not-a-hash"),
                     lambda f: f.update(K_index=None), lambda f: f.update(size_wh=[0, 480]),
                     lambda f: f["K_index"][0].__setitem__(0, -100.),
                     lambda f: f["world_to_camera_cv"][0].__setitem__(0, -1.),
                     lambda f: f["world_to_camera_cv"][0].__setitem__(3, float("nan"))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                frames, segments, tracks, anchors = fixture()
                mutation(frames[0])
                a = check_target_anchors(frames, segments, anchors)
                b = check_endpoint_reprojection(frames, tracks)
                self.assertEqual(a["state"], "unresolved")
                self.assertEqual(b["state"], "unresolved")
                json.dumps([a, b], allow_nan=False)

    def test_duplicate_frame_id_and_missing_frame_are_explicit(self):
        frames, segments, _, anchors = fixture()
        frames.append(copy.deepcopy(frames[0]))
        result = check_target_anchors(frames, segments, anchors)
        self.assertEqual(result["state"], "unresolved")
        self.assertIn("duplicate_view_id", result["reasons"])
        frames, segments, _, anchors = fixture()
        anchors[1]["view_id"] = "missing"
        result = check_target_anchors(frames, segments, anchors)
        self.assertEqual(result["state"], "unresolved")
        self.assertIn("missing_frame", result["per_anchor"][1]["reasons"])

    def test_rgb_receipts_and_homogeneous_cameras_use_same_geometry(self):
        frames, segments, tracks, anchors = fixture()
        for frame in frames:
            frame["source_kind"] = "rgb"
            frame["world_to_camera_cv"].append([0., 0., 0., 1.])
        self.assertEqual(check_target_anchors(frames, segments, anchors)["state"], "supported")
        self.assertEqual(check_endpoint_reprojection(frames, tracks)["state"], "supported")

    def test_threshold_is_frozen_not_a_posthoc_fit_knob(self):
        frames, segments, tracks, anchors = fixture()
        for value in (1., 3., float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_target_anchors(frames, segments, anchors, threshold_px=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_endpoint_reprojection(frames, tracks, threshold_px=value)


if __name__ == "__main__":
    unittest.main()
