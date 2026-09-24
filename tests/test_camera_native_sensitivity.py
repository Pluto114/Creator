"""Native comparison must expose drift, abstention, and lost finite extent."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.camera_bundle import CameraGauge
from creator_eval.camera_native_sensitivity import (
    camera_difference,
    check_native_gauge,
    segment_difference,
)


def cameras():
    k = np.tile([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]], (3, 1, 1))
    locations = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    e = np.concatenate([np.tile(np.eye(3), (3, 1, 1)), -locations[:, :, None]], axis=2)
    return k, e


def accepted(segments):
    return dict(state="accepted", segments=segments)


class CameraNativeSensitivityTests(unittest.TestCase):
    def test_same_cameras_compare_without_mutation_or_alignment(self):
        k, e = cameras()
        saved_k, saved_e = k.copy(), e.copy()
        result = camera_difference(k, e, k, e, e)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["compared_pose_view_count"], 2)
        self.assertEqual(result["center_delta_over_baseline_max"], 0.)
        self.assertEqual(result["rotation_degrees_max"], 0.)
        self.assertEqual(result["absolute_log_focal_ratio_max"], 0.)
        self.assertFalse(result["alignment_performed"])
        np.testing.assert_array_equal(k, saved_k)
        np.testing.assert_array_equal(e, saved_e)

    def test_center_rotation_and_focal_changes_are_visible(self):
        k, e = cameras()
        moved_k, moved_e = k.copy(), e.copy()
        theta = np.radians(12.)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                             [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        moved_e[1, :, :3] = rotation
        moved_e[1, :, 3] = -rotation @ np.array([1., .2, 0.])
        moved_k[:, [0, 1], [0, 1]] *= 1.1
        result = camera_difference(k, e, moved_k, moved_e, e)
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["center_delta_over_baseline_max"], .1)
        self.assertAlmostEqual(result["center_delta_over_baseline_median"], .05)
        self.assertAlmostEqual(result["rotation_degrees_max"], 12.)
        self.assertAlmostEqual(result["rotation_degrees_median"], 6.)
        self.assertAlmostEqual(result["absolute_log_focal_ratio_max"], np.log(1.1))

    def test_changed_first_pose_or_scale_is_not_realigned_away(self):
        k, e = cameras()
        translated = e.copy()
        translated[:, 0, 3] -= .5
        resized = e.copy()
        resized[:, :, 3] *= 1.1
        for changed, reason in ((translated, "first_camera_pose_changed"),
                                (resized, "first_last_baseline_changed")):
            with self.subTest(reason=reason):
                audit = check_native_gauge(e, changed)
                self.assertFalse(audit["comparable"])
                self.assertIn(reason, audit["reason"])
                result = camera_difference(k, e, k, changed, e)
                self.assertFalse(result["comparable"])
                self.assertIsNone(result["center_delta_over_baseline_max"])
                self.assertFalse(result["alignment_performed"])
        # Shared drift of both results still violates their original gauge.
        self.assertFalse(camera_difference(k, translated, k, translated, e)["comparable"])

    def test_float32_rotations_survive_two_zero_parameter_gauge_decodes(self):
        k, _ = cameras()
        rotations = []
        for angle in (.37, .83, -.29):
            rotations.append([[np.cos(angle), -np.sin(angle), 0.],
                              [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        rotations = np.asarray(rotations, dtype=np.float32).astype(float)
        locations = np.array([[.3, -.2, .1], [1.3, .4, .1], [2.3, -.2, .1]])
        translations = -np.einsum("nij,nj->ni", rotations, locations)
        initial = np.concatenate((rotations, translations[:, :, None]), axis=2).astype(np.float32).astype(float)
        current_k, current_e = k, initial
        for _ in range(2):
            gauge = CameraGauge(current_k, current_e, True)
            current_k, current_e = gauge.decode(np.zeros(gauge.parameter_count), native=True)
        result = check_native_gauge(initial, current_e)
        self.assertGreater(result["baseline_relative_delta"], 1e-8)
        self.assertLess(result["baseline_relative_delta"], 1e-6)
        self.assertTrue(result["comparable"])
        self.assertTrue(camera_difference(k, initial, current_k, current_e, initial)["comparable"])
        # Tolerance must not erase an actual baseline change.
        tampered = current_e.copy()
        tampered[-1, :, 3] *= 1.0001
        rejected = check_native_gauge(initial, tampered)
        self.assertFalse(rejected["comparable"])
        self.assertIn("first_last_baseline_changed", rejected["reason"])

    def test_missing_camera_is_missing_and_malformed_rotation_is_error(self):
        k, e = cameras()
        result = camera_difference(k, e, None, None, e)
        self.assertFalse(result["comparable"])
        self.assertIsNone(result["rotation_degrees_max"])
        self.assertIn("missing_candidate_camera", result["reason"])
        malformed = e.copy()
        malformed[1, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "proper"):
            camera_difference(k, e, k, malformed, e)


class FiniteSegmentSensitivityTests(unittest.TestCase):
    def test_parallel_offset_and_native_scale_normalization(self):
        reference = accepted([[[0., 0., 0.], [1., 0., 0.]]])
        candidate = accepted([[[0., .2, 0.], [1., .2, 0.]]])
        saved = copy.deepcopy(reference)
        result = segment_difference(reference, candidate, 2.)
        self.assertTrue(result["comparable"])
        self.assertAlmostEqual(result["symmetric_p95_over_baseline"], .1)
        self.assertAlmostEqual(result["length_ratio"], 1.)
        self.assertEqual(result["segment_count_delta"], 0)
        self.assertEqual(reference, saved)
        scaled = segment_difference(accepted(np.array(reference["segments"]) * 7),
                                    accepted(np.array(candidate["segments"]) * 7), 14.)
        self.assertAlmostEqual(scaled["symmetric_p95_over_baseline"],
                               result["symmetric_p95_over_baseline"])
        self.assertEqual(result["normalized_tolerance"], .001)

    def test_collinear_gap_is_visible_despite_identical_infinite_axis(self):
        reference = accepted([[[0., 0., 0.], [2., 0., 0.]]])
        candidate = accepted([[[0., 0., 0.], [.5, 0., 0.]],
                              [[1.5, 0., 0.], [2., 0., 0.]]])
        result = segment_difference(reference, candidate, 2.)
        self.assertEqual(result["segment_count_delta"], 1)
        self.assertAlmostEqual(result["length_ratio"], .5)
        self.assertGreater(result["reference_to_candidate_p95_over_baseline"], .2)
        self.assertLess(result["candidate_to_reference_p95_over_baseline"], 1e-12)

    def test_empty_or_abstained_geometry_never_becomes_zero_error(self):
        nonempty = accepted([[[0., 0., 0.], [1., 0., 0.]]])
        empty = accepted([])
        rejected = dict(state="rejected", segments=[])
        for reference, candidate in ((empty, empty), (nonempty, empty), (empty, nonempty),
                                     (rejected, rejected), (nonempty, rejected), (None, nonempty)):
            with self.subTest(reference=reference, candidate=candidate):
                result = segment_difference(reference, candidate, 2.)
                self.assertFalse(result["comparable"])
                self.assertIsNone(result["symmetric_p95_over_baseline"])
                self.assertIsNone(result["length_ratio"])
                self.assertIsNotNone(result["reason"])

    def test_invalid_scale_and_overlapping_duplicates_are_errors(self):
        segment = [[[0., 0., 0.], [1., 0., 0.]]]
        for baseline in (0., -1., np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "baseline"):
                segment_difference(accepted(segment), accepted(segment), baseline)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            segment_difference(accepted(segment), accepted(segment + segment), 2.)


if __name__ == "__main__":
    unittest.main()
