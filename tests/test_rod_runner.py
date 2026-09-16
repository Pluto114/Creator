"""Runner adapter contracts; run locally with the existing DA3 environment.

Importing the actual runner currently needs Pillow and OpenCV via its I/O helpers.
Keep these tests out of the NumPy-only CI job unless those imports are separated.
No model, dataset, or ground-truth file is accessed by these analytic checks.
"""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rod_evidence import observation_view, occupancy_candidate, original_intrinsics


class OriginalIntrinsicTests(unittest.TestCase):
    def test_estimated_K_preserves_rays_under_exact_half_pixel_resize(self):
        native = np.array([
            [[705.0, 0, 253.2], [0, 693.0, 139.1], [0, 0, 1]],
            [[699.0, 0, 250.5], [0, 688.0, 140.7], [0, 0, 1]],
        ])
        prior = native.copy()
        original_pixels = np.array([[0, 0, 1], [1919, 1079, 1], [1037.25, 586.7, 1]])
        for native_wh in ([504, 280], [756, 420]):
            scale = np.asarray(native_wh) / [1920, 1080]
            resized_pixels = original_pixels.copy()
            resized_pixels[:, :2] = (original_pixels[:, :2] + 0.5) * scale - 0.5
            lifted = original_intrinsics(native, [1920, 1080], native_wh)
            for index in range(2):
                np.testing.assert_allclose(
                    original_pixels @ np.linalg.inv(lifted[index]).T,
                    resized_pixels @ np.linalg.inv(native[index]).T,
                    atol=1e-12,
                )
        np.testing.assert_array_equal(native, prior)

    def test_oracle_edge_K_lifts_to_correct_original_index_K(self):
        original_edge = np.array([
            [[2666.0, 0, 960.0], [0, 2670.0, 540.0], [0, 0, 1]],
            [[2600.0, 0, 954.3], [0, 2640.0, 536.7], [0, 0, 1]],
        ])
        expected_index = original_edge.copy()
        expected_index[:, :2, 2] -= 0.5
        for native_wh in ([504, 280], [756, 420]):
            scale = np.asarray(native_wh) / [1920, 1080]
            native_edge = np.diag([*scale, 1]) @ original_edge
            original_native = native_edge.copy()
            lifted = original_intrinsics(native_edge, [1920, 1080], native_wh, oracle=True)
            np.testing.assert_allclose(lifted, expected_index, atol=1e-10)
            np.testing.assert_array_equal(native_edge, original_native)

    def test_two_half_pixel_resize_stages_equal_combined_lift(self):
        original_index = np.array([[[2600.0, 0, 959.5], [0, 2650.0, 539.5], [0, 0, 1]]])
        def resize_transform(source, target):
            sx, sy = np.asarray(target) / source
            return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        for intermediate, native_wh in (([504, 284], [504, 280]), ([756, 425], [756, 420])):
            first = resize_transform(np.array([1920, 1080]), intermediate)
            second = resize_transform(np.array(intermediate), native_wh)
            native = second @ first @ original_index
            restored = original_intrinsics(native, [1920, 1080], native_wh)
            np.testing.assert_allclose(restored, original_index, atol=1e-10)


class ObservationAdapterTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rows = [
            {"y": 10, "guide_x": 20, "status": "ambiguous", "candidates": [{"center_x": 16.0, "width": 3}, {"center_x": 21.5, "width": 4}]},
            {"y": 14, "guide_x": 21, "status": "absent", "candidates": [], "absence_window_xyxy": [12, 12, 30, 17]},
            {"y": 18, "guide_x": 22, "status": "unknown", "candidates": []},
            {"y": 22, "guide_x": 23, "status": "observed", "candidates": [{"center_x": 37.0, "width": 3}]},
            {"y": 26, "guide_x": 24, "status": "ambiguous", "candidates": [{"center_x": 15.0, "width": 3}, {"center_x": 33.0, "width": 3}]},
        ]
        fitted = {"usable": True, "slope": 0.25, "intercept": 19.0, "row_matches": [{"row_index": 0, "candidate_index": 1}]}
        frame = {"frame_id": "view_test", "size_wh": [80, 60]}
        intrinsic = np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]])
        camera = np.c_[np.eye(3), np.zeros(3)]
        return {"rows": rows}, fitted, frame, intrinsic, camera

    def test_only_matched_candidates_become_positive_no_unknown_gap_fill(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        view = observation_view(*args)
        states = [item["state"] for item in view["observations"]]
        self.assertEqual(states, ["accepted", "absent", "unknown", "unknown", "ambiguous"])
        self.assertEqual(view["observations"][0]["xy"], [21.5, 10])
        self.assertEqual(view["observations"][0]["width_px"], 4)
        self.assertEqual(len(view["observations"]), len(args[0]["rows"]))
        self.assertEqual(view["image_line"], [1, -0.25, -19.0])
        self.assertEqual(args[:3], before[:3])
        np.testing.assert_array_equal(args[3], before[3])
        np.testing.assert_array_equal(args[4], before[4])

    def test_exclusive_scan_window_becomes_inclusive_negative_pixel_window(self):
        view = observation_view(*self.fixture())
        absent = view["observations"][1]
        self.assertEqual(absent["xy"], [21, 14])
        self.assertEqual(absent["negative_window_x"], [12, 29])
        self.assertNotIn("negative_window_x", view["observations"][2])

    def test_unusable_fit_does_not_reuse_its_provisional_matches(self):
        args = list(self.fixture())
        args[1]["usable"] = False
        view = observation_view(*args)
        self.assertIsNone(view["image_line"])
        self.assertEqual([item["state"] for item in view["observations"]], ["ambiguous", "absent", "unknown", "ambiguous", "ambiguous"])

    def test_empty_observations_and_occupancy_return_explicit_rejection(self):
        args = list(self.fixture())
        args[0] = {"rows": []}
        args[1] = {"usable": False}
        view = observation_view(*args)
        self.assertEqual(view["observations"], [])
        self.assertIsNone(view["image_line"])
        result = occupancy_candidate([view], {})
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["rejection_reasons"], ["fewer_than_three_image_lines"])

    def test_usable_lines_with_no_positive_rows_cannot_publish_a_segment(self):
        intrinsic = np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]])
        endpoints = np.array([[0, -1, 5, 1], [0, 1, 5, 1]])
        views = []
        for index, center in enumerate(([-1, 0, 0], [0, 1, 0], [1, -1, 0])):
            camera = np.c_[np.eye(3), -np.array(center)]
            xy = endpoints @ (intrinsic @ camera).T
            views.append({"view_id": str(index), "K_index": intrinsic, "world_to_camera_cv": camera,
                          "image_line": np.cross(xy[0], xy[1]), "observations": []})
        result = occupancy_candidate(views, {"bin_width_camera_span_fraction": 0.002, "minimum_views": 3, "dilation_bins": 1, "minimum_run_bins": 3})
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(np.asarray(result["segments"]).shape, (0, 2, 3))
        self.assertEqual(result["rejection_reasons"], ["no_occupied_interval"])


if __name__ == "__main__":
    unittest.main()
