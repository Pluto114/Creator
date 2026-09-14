"""Known-answer checks for line controls; no model weights or Blender required."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.line_controls import (
    LineFitDegenerate,
    closest_ray_line_parameters,
    curve_metrics,
    deterministic_ransac_line,
    fit_line_tls,
    fit_multiview_line,
    gap_coverage,
    point_to_segments_distance,
    sample_segments_midpoint,
    support_intervals,
)


def line(start=0.0, end=1.0, offset=0.0):
    return np.array([[start, offset, 0.0], [end, offset, 0.0]])


class LineControlTests(unittest.TestCase):
    def test_tls_recovers_axis_and_full_observed_extent(self):
        points = np.c_[np.linspace(-2, 3, 11), np.full(11, 4.0), np.full(11, -1.0)]
        fitted = fit_line_tls(points)
        np.testing.assert_allclose(fitted["segment"], [[-2, 4, -1], [3, 4, -1]])
        self.assertAlmostEqual(fitted["extent"], 5)
        self.assertAlmostEqual(fitted["perpendicular_rmse"], 0)
        np.testing.assert_allclose(fit_line_tls(points[::-1])["segment"], fitted["segment"])

    def test_tls_rejects_missing_or_ambiguous_support(self):
        for points in (np.ones((2, 3)), np.ones((5, 3)), np.vstack([np.eye(3), -np.eye(3)])):
            with self.assertRaises(LineFitDegenerate):
                fit_line_tls(points)
        with self.assertRaises(LineFitDegenerate):
            fit_line_tls([[0, 0, 0], [1, 0, 0], [np.nan, 0, 0]])

    def test_midpoint_arc_weights_and_split_density(self):
        points, weights = sample_segments_midpoint([line(0, 0.5), line(0.5, 2)], 0.25)
        np.testing.assert_allclose(points[:, 0], np.arange(0.125, 2, 0.25))
        self.assertAlmostEqual(weights.sum(), 2)
        # 切成很多段不会按段数加分。这里20%弧长正确，剩下80%偏离。
        truth = [line(0, 2)]
        prediction = [line(0, 0.4), line(0.4, 2, offset=1)]
        dense_prediction = [line(0, 0.2), line(0.2, 0.4), line(0.4, 2, offset=1)]
        for value in (prediction, dense_prediction):
            result = curve_metrics(value, truth, tolerance=0.0, spacing=0.01)
            self.assertAlmostEqual(result["precision_fraction"], 0.2)
            self.assertAlmostEqual(result["prediction_to_truth"]["distance_median"], 1)
            self.assertAlmostEqual(result["false_predicted_length"], 1.6)

    def test_duplicate_or_overlapping_segments_are_rejected(self):
        for prediction in ([line(), line()], [line(), line(0.5, 1.5)], [line(), line()[::-1]]):
            with self.assertRaisesRegex(ValueError, "overlapping_segments"):
                curve_metrics(prediction, [line()], tolerance=0.01, spacing=0.01)
        # 仅相接的端点没有正长度重叠。
        curve_metrics([line(0, 0.5), line(0.5, 1)], [line()], tolerance=0.01, spacing=0.01)

    def test_identical_offset_and_finite_endpoints(self):
        exact = curve_metrics([line()], [line()], tolerance=0.01, spacing=0.01)
        self.assertAlmostEqual(exact["recovery_fraction"], 1)
        self.assertAlmostEqual(exact["precision_fraction"], 1)
        self.assertAlmostEqual(exact["truth_to_prediction"]["distance_p95"], 0)
        shifted = curve_metrics([line(offset=0.2)], [line()], tolerance=0.1, spacing=0.01)
        self.assertAlmostEqual(shifted["recovery_fraction"], 0)
        self.assertAlmostEqual(shifted["precision_fraction"], 0)
        self.assertAlmostEqual(shifted["prediction_to_truth"]["distance_median"], 0.2)
        np.testing.assert_allclose(
            point_to_segments_distance(np.array([[-1, 0, 0], [2, 0, 0]]), [line()]), [1, 1]
        )

    def test_broken_prediction_and_false_bridge_have_different_accounts(self):
        broken = [line(0, 0.4), line(0.6, 1)]
        missing = curve_metrics(broken, [line()], tolerance=1e-10, spacing=0.001)
        self.assertAlmostEqual(missing["recovery_fraction"], 0.8)
        self.assertAlmostEqual(missing["precision_fraction"], 1)
        bridged = curve_metrics([line()], broken, tolerance=1e-10, spacing=0.001)
        self.assertAlmostEqual(bridged["recovery_fraction"], 1)
        self.assertAlmostEqual(bridged["precision_fraction"], 0.8)
        self.assertAlmostEqual(bridged["false_predicted_length"], 0.2)

    def test_gap_endpoint_tolerance_is_not_a_bridge(self):
        broken = [line(0, 0.4), line(0.6, 1)]
        gap = line(0.4, 0.6)
        correct = gap_coverage(broken, gap, tolerance=0.01, spacing=0.001)
        self.assertGreater(correct["whole_gap"]["covered_fraction"], 0)
        self.assertAlmostEqual(correct["guarded_interior"]["covered_fraction"], 0)
        bridge = gap_coverage([line()], gap, tolerance=0.01, spacing=0.001)
        self.assertAlmostEqual(bridge["guarded_interior"]["covered_fraction"], 1)
        short = gap_coverage([line()], line(0, 0.01), tolerance=0.01, spacing=0.001)
        self.assertIsNone(short["guarded_interior"]["covered_fraction"])

    def test_empty_prediction_is_not_perfect_precision(self):
        empty = curve_metrics([], [line()], tolerance=0.01, spacing=0.01)
        self.assertAlmostEqual(empty["recovery_fraction"], 0)
        self.assertIsNone(empty["precision_fraction"])
        self.assertIsNone(empty["truth_to_prediction"]["distance_median"])
        self.assertEqual(empty["prediction_to_truth"]["reason"], "empty_prediction")
        no_truth = curve_metrics([line()], [], tolerance=0.01, spacing=0.01)
        self.assertIsNone(no_truth["recovery_fraction"])
        self.assertAlmostEqual(no_truth["precision_fraction"], 0)
        self.assertAlmostEqual(no_truth["false_predicted_length"], 1)

    @staticmethod
    def synthetic_image_lines(centers):
        intrinsic = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
        endpoints = np.array([[0.5, -1, 5, 1], [0.5, 1, 5, 1]])
        cameras, lines = [], []
        for center in centers:
            camera = np.c_[np.eye(3), -np.asarray(center)]
            pixels = endpoints @ (intrinsic @ camera).T
            cameras.append(camera)
            lines.append(np.cross(pixels[0], pixels[1]))
        return np.array(lines), intrinsic, np.array(cameras)

    def test_multiview_planes_recover_known_world_line(self):
        lines, intrinsic, cameras = self.synthetic_image_lines([[-2, 0, 0], [2, 1, 0], [0, -2, 0]])
        fitted = fit_multiview_line(lines, intrinsic, cameras)
        np.testing.assert_allclose(fitted["direction"], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(fitted["anchor"], [0.5, 0, 5], atol=1e-12)
        self.assertLess(fitted["plane_rmse"], 1e-12)
        self.assertLess(abs(fitted["anchor"] @ fitted["direction"]), 1e-12)
        scaled = fit_multiview_line(lines * np.array([2, -4, 0.5])[:, None], intrinsic, cameras)
        np.testing.assert_allclose(scaled["anchor"], fitted["anchor"], atol=1e-12)
        np.testing.assert_allclose(scaled["direction"], fitted["direction"], atol=1e-12)

    def test_multiview_coincident_planes_and_too_few_views_rejected(self):
        lines, intrinsic, cameras = self.synthetic_image_lines([[0, 0, 0], [0, 1, 0], [0, -2, 0]])
        with self.assertRaisesRegex(LineFitDegenerate, "backprojection_planes"):
            fit_multiview_line(lines, intrinsic, cameras)
        with self.assertRaisesRegex(LineFitDegenerate, "too_few_views"):
            fit_multiview_line(lines[:2], intrinsic, cameras[:2])


class RaySupportTests(unittest.TestCase):
    def test_closest_ray_line_parameters_have_known_world_answers(self):
        origins = np.array([[0, 0, 0], [0, 0, 0], [0, 2, 0]], float)
        rays = np.array([[1, 0, 1], [2, 0, 1], [1, 0, 1]], float)
        result = closest_ray_line_parameters([0, 0, 5], [1, 0, 0], origins, rays)
        np.testing.assert_allclose(result["line_t"], [5, 10, 5])
        np.testing.assert_allclose(result["ray_s"], [5, 5, 5])
        np.testing.assert_allclose(result["closest_separation"], [0, 0, 2])
        self.assertTrue(result["valid"].all())
        scaled = closest_ray_line_parameters([0, 0, 5], [2, 0, 0], origins, rays * 2)
        np.testing.assert_allclose(scaled["line_t"], result["line_t"] / 2)
        np.testing.assert_allclose(scaled["ray_s"], result["ray_s"] / 2)

    def test_parallel_invalid_and_backwards_rays_do_not_support_a_line(self):
        rays = np.array([[1, 0, 0], [1, 0, 1e-8], [0, 0, 0], [1, 0, -1]], float)
        result = closest_ray_line_parameters([0, 0, 5], [1, 0, 0], np.zeros((4, 3)), rays)
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["line_t"][:3]).all())
        self.assertLess(result["ray_s"][-1], 0)

    def test_support_intervals_keep_a_gap_and_explicit_dilation_extent(self):
        values = np.r_[np.arange(0.05, 0.4, 0.1), np.arange(0.85, 1.2, 0.1)]
        result = support_intervals([values, values, values], 0.1, dilation_bins=1)
        np.testing.assert_allclose(result, [[-0.1, 0.5], [0.7, 1.3]], atol=1e-12)
        undilated = support_intervals([values, values, values], 0.1, dilation_bins=0)
        np.testing.assert_allclose(undilated, [[0, 0.4], [0.8, 1.2]], atol=1e-12)

    def test_single_view_duplicates_cannot_vote_as_multiple_views(self):
        samples = np.array([0.05, 0.15, 0.25])
        one_view = support_intervals([np.tile(samples, 1000), [], []], 0.1)
        self.assertEqual(one_view.shape, (0, 2))
        normal = support_intervals([samples, samples, samples], 0.1)
        duplicated = support_intervals([np.tile(samples, 1000), samples, samples], 0.1)
        np.testing.assert_allclose(normal, duplicated)
        self.assertEqual(support_intervals([], 0.1).shape, (0, 2))

    def test_support_grid_anchor_and_short_runs(self):
        values = np.array([-0.26, -0.16, -0.06])
        result = support_intervals([values] * 3, 0.1, dilation_bins=0)
        np.testing.assert_allclose(result, [[-0.3, 0]], atol=1e-12)
        short = support_intervals([[0.05]] * 3, 0.1, dilation_bins=0, minimum_run_bins=3)
        self.assertEqual(short.shape, (0, 2))

class RansacLineTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rng = np.random.default_rng(19)
        inliers = np.c_[np.linspace(-2, 3, 90), np.full(90, 4.0), np.full(90, -1.0)]
        inliers[:, 1:] += rng.normal(0, 0.003, size=(90, 2))
        outliers = rng.uniform([-2, -1, 2], [3, 2, 5], size=(60, 3))
        points = np.vstack([inliers, outliers])
        views = np.arange(len(points)) % 3
        return points, views

    def test_ransac_finds_line_among_outliers(self):
        points, views = self.fixture()
        fitted = deterministic_ransac_line(points, views, 0.03, 0.5)
        self.assertEqual(fitted["inlier_count"], 90)
        self.assertTrue(fitted["inlier_mask"][:90].all())
        self.assertFalse(fitted["inlier_mask"][90:].any())
        np.testing.assert_allclose(fitted["segment"], [[-2, 4, -1], [3, 4, -1]], atol=0.002)
        self.assertEqual(fitted["view_support_counts"], {"0": 30, "1": 30, "2": 30})
        self.assertLess(fitted["perpendicular_rmse"], 0.01)

    def test_ransac_rejects_single_view_support(self):
        points, _ = self.fixture()
        with self.assertRaisesRegex(LineFitDegenerate, "distinct_views"):
            deterministic_ransac_line(points, np.zeros(len(points), int), 0.03, 0.5)
        # 虽然整个输入有三个视图，真正那条线只在第一张图里有，仍然不能通过。
        views = np.zeros(len(points), int)
        views[90:] = np.arange(60) % 2 + 1
        with self.assertRaisesRegex(LineFitDegenerate, "required_view_support"):
            deterministic_ransac_line(points, views, 0.01, 0.5, min_inliers=30)

    def test_ransac_repeats_bitwise_with_same_seed(self):
        points, views = self.fixture()
        first = deterministic_ransac_line(points, views, 0.03, 0.5, seed=7)
        second = deterministic_ransac_line(points, views, 0.03, 0.5, seed=7)
        np.testing.assert_array_equal(first["inlier_mask"], second["inlier_mask"])
        np.testing.assert_array_equal(first["segment"], second["segment"])
        self.assertEqual(first["winning_trial"], second["winning_trial"])

if __name__ == "__main__":
    unittest.main()
