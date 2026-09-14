"""Known-answer checks for the baseline diagnostics; no model weights needed."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.native_diagnostics import (
    AlignmentDegenerate,
    align_cameras,
    apply_similarity,
    confidence_masks,
    fit_sim3,
    region_report,
    unproject,
)


class NativeDiagnosticsTests(unittest.TestCase):
    def test_camera_alignment_and_world_points_together(self):
        angles = np.linspace(-0.5, 0.5, 5)
        centers = np.c_[np.sin(angles), np.cos(angles), np.zeros(5)]
        rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = np.array([4.0, -2, 7])
        predicted, truth = [], []
        for center in centers:
            camera_to_world = np.eye(4)
            camera_to_world[:3, 3] = center
            predicted.append(np.linalg.inv(camera_to_world))
            true_camera_to_world = np.eye(4)
            true_camera_to_world[:3, :3] = rotation
            true_camera_to_world[:3, 3] = apply_similarity(center, 3, rotation, translation)
            truth.append(np.linalg.inv(true_camera_to_world))
        config = {
            "minimum_second_to_first_singular_ratio": 0.001,
            "poor_fit_rmse_over_camera_span": 0.1,
            "leave_one_camera_out_diagnostic": True,
        }
        scale, fitted_rotation, fitted_translation, report = align_cameras(predicted, truth, config)
        k = np.array([[2.0, 0, 0.5], [0, 2, 0.5], [0, 0, 1]])
        source_points = unproject(np.full((2, 2), 4.0), k, predicted[0])
        true_points = unproject(np.full((2, 2), 12.0), k, truth[0])
        np.testing.assert_allclose(
            apply_similarity(source_points, scale, fitted_rotation, fitted_translation),
            true_points,
            atol=1e-12,
        )
        self.assertLess(max(report["orientation_errors_degrees"]), 1e-5)
        self.assertLess(
            max(v["held_out_camera_error_m"] for v in report["leave_one_camera_out"]), 1e-12
        )

    def test_known_planar_similarity(self):
        angle = np.linspace(-0.5, 0.5, 5)
        source = np.c_[np.sin(angle), np.cos(angle), np.zeros(5)]
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        target = apply_similarity(source, 3.0, r, np.array([4.0, -2.0, 7.0]))
        scale, rotation, translation, report = fit_sim3(source, target)
        self.assertAlmostEqual(scale, 3.0)
        np.testing.assert_allclose(rotation, r, atol=1e-12)
        np.testing.assert_allclose(translation, [4, -2, 7], atol=1e-12)
        self.assertLess(report["camera_rmse_m"], 1e-12)

    def test_collinear_centers_rejected(self):
        source = np.c_[np.arange(5), np.zeros((5, 2))]
        with self.assertRaises(AlignmentDegenerate):
            fit_sim3(source, source)

    def test_mirror_is_not_a_valid_rotation(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        target = source * np.array([-1, 1, 1])
        _, r, _, report = fit_sim3(source, target)
        self.assertAlmostEqual(np.linalg.det(r), 1.0)
        self.assertGreater(report["camera_rmse_m"], 0.1)

    def test_unprojection_half_pixel_conversion(self):
        # K_edge的主点是1；像素(0,0)的物理中心在(0.5,0.5)。
        k_index = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1.0]])
        actual = unproject(np.full((2, 2), 4.0), k_index, np.eye(4))
        np.testing.assert_allclose(actual[0, 0], [-1, -1, 4])
        np.testing.assert_allclose(actual[1, 1], [1, 1, 4])

    def test_percentile_ties_are_not_forced_to_sixty_percent(self):
        d = np.ones((2, 5))
        c = np.full((2, 5), 2.0)
        masks, threshold = confidence_masks(
            d,
            c,
            {"default_percentile": 40, "default_upper_percentile": 90, "fixed_confidence": 1.05},
        )
        self.assertEqual(threshold, 2.0)
        self.assertTrue(masks["default_p40"].all())

    def test_empty_rod_region_is_not_zero_error(self):
        result = region_report(np.zeros(4, bool), np.ones(4, bool), np.zeros(4), np.zeros(4))
        self.assertIsNone(result["retained_fraction"])
        self.assertIsNone(result["scaled_camera_z_abs_error_m"]["median"])

    def test_filtering_reports_lost_support_alongside_error(self):
        region = np.ones(4, bool)
        kept = np.array([1, 1, 0, 0], bool)
        result = region_report(region, kept, np.array([0, 0, 5, 5]))
        self.assertEqual(result["retained_fraction"], 0.5)
        self.assertEqual(result["scaled_camera_z_abs_error_m"]["median"], 0.0)
        self.assertEqual(result["available_pixels"], 4)


if __name__ == "__main__":
    unittest.main()
