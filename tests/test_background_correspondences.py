"""Known-answer geometry and isolation tests; no project RGB or GT is loaded."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.background_correspondences import (
    exclusion_mask,
    fit_rgb_geometry,
    homography_distances,
    mesh_direction,
    mutual_ratio_matches,
    residual_summary,
    spatial_split,
    world_from_depth,
)
from creator_eval.camera_diagnostics import fundamental_from_cameras, sampson_distances

SPLIT = {"cell_size_px": 128, "seed": 0, "validation_modulus": 3, "validation_residue": 0}
GEOMETRY = {"minimum_matches": 30, "minimum_train": 16, "minimum_validation": 8,
            "seed": 113, "ransac_threshold_px": 2., "confidence": .99, "maximum_iterations": 5000,
            "validation_fraction_within_2px": .6, "planar_fraction_margin": .1}


def camera(x=0):
    transform = np.eye(4)
    transform[0, 3] = -x
    return {"K_index": np.array([[100., 0, 49.5], [0, 100., 49.5], [0, 0, 1]]),
            "K_edge": np.array([[100., 0, 50.], [0, 100., 50.], [0, 0, 1]]),
            "world_to_camera_cv": transform, "size_wh": [100, 100], "clip_start": .1, "clip_end": 100.}


def plane_cast(cam, edge_uv):
    rays = np.c_[edge_uv, np.ones(len(edge_uv))] @ np.linalg.inv(cam["K_edge"]).T
    z = 10 / rays[:, 2]
    return z, np.ones(len(z), dtype=np.uint32), z * np.linalg.norm(rays, axis=1)


class BackgroundMathTests(unittest.TestCase):
    def test_spatial_assignment_survives_input_permutation(self):
        points = np.array([[x * 128 + 1, y * 128 + 4] for x in range(6) for y in range(5)], float)
        labels, report = spatial_split(points, points, SPLIT)
        perm = np.random.default_rng(2).permutation(len(points))
        permuted, _ = spatial_split(points[perm], points[perm], SPLIT)
        np.testing.assert_array_equal(labels[perm], permuted)
        self.assertEqual(report["target_cells_shared_across_split"], 0)
        self.assertGreater(report["train_count"], 0)
        self.assertGreater(report["validation_count"], 0)

    def test_duplicate_orientation_keypoints_stay_on_same_side(self):
        points = np.array([[15, 20], [15, 20], [120, 100]], float)
        labels, _ = spatial_split(points, points, SPLIT)
        self.assertTrue(np.all(labels == labels[0]))

    def test_empty_split_remains_empty(self):
        mask, result = spatial_split(np.empty((0, 2)), np.empty((0, 2)), SPLIT)
        self.assertEqual(mask.shape, (0,))
        self.assertEqual(result["train_count"], 0)

    def test_invalid_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            spatial_split([[np.nan, 0]], [[0, 0]], SPLIT)

    def test_invalid_residual_does_not_disappear_from_fraction(self):
        result = residual_summary([0, 1, np.nan, 10])
        self.assertEqual(result["fraction_within_2px"], .5)
        self.assertEqual(result["invalid_count"], 1)
        self.assertIsNone(residual_summary([])["median_px"])

    def test_homography_known_translation_both_directions(self):
        points = np.array([[1., 2], [30., 50], [100., 200]])
        h = np.array([[1., 0, 3], [0, 1., -5], [0, 0, 1]])
        np.testing.assert_allclose(homography_distances(points, points + [3, -5], h), 0)
        np.testing.assert_allclose(homography_distances(points, points + [4, -5], h), 1)

    def test_singular_homography_has_no_valid_residual(self):
        self.assertTrue(np.isnan(homography_distances([[1, 2]], [[1, 2]], np.zeros((3, 3)))).all())

    def test_mask_excludes_centers_only_and_preserves_outside_rows(self):
        mask = exclusion_mask([100, 100], {"target": [50, 60]}, [20, 80], 3)
        self.assertEqual(mask[20, 50], 0)
        self.assertEqual(mask[80, 60], 0)
        self.assertEqual(mask[19, 50], 255)
        self.assertEqual(mask[20, 40], 255)

    def test_index_edge_boundary_and_metric_reprojection(self):
        a, b = np.array([[50., 50]]), np.array([[40., 50]])
        report = mesh_direction(a, b, camera(), camera(1), plane_cast, .002)
        np.testing.assert_allclose(report["source_world"], [[.05, .05, 10]])
        np.testing.assert_allclose(report["expected_target_xy"], b)
        np.testing.assert_allclose(report["transfer_error_px"], 0)
        self.assertTrue(report["source_point_visible_in_target"][0])

    def test_wrong_physical_match_can_still_be_epipolar(self):
        report = mesh_direction([[50, 50]], [[70, 50]], camera(), camera(1), plane_cast, .002)
        self.assertAlmostEqual(report["transfer_error_px"][0], 30)
        first, second = camera(), camera(1)
        fundamental = fundamental_from_cameras(first["K_index"], first["world_to_camera_cv"],
                                               second["K_index"], second["world_to_camera_cv"])
        self.assertAlmostEqual(sampson_distances([[50, 50]], [[70, 50]], fundamental)[0], 0)

    def test_occlusion_remains_unscored(self):
        def cast(cam, uv):
            z, ids, ranges = plane_cast(cam, uv)
            if cam["world_to_camera_cv"][0, 3] == -1:
                z[:] = 5
            return z, ids, ranges

        report = mesh_direction([[50, 50]], [[40, 50]], camera(), camera(1), cast, .002)
        self.assertTrue(report["target_in_frame"][0])
        self.assertFalse(report["source_point_visible_in_target"][0])
        self.assertTrue(report["occluded"][0])
        self.assertTrue(np.isnan(report["transfer_error_px"][0]))

    def test_outside_target_stays_distinct_from_occlusion(self):
        report = mesh_direction([[50, 50]], [[40, 50]], camera(), camera(100), plane_cast, .002)
        self.assertTrue(report["source_hit"][0])
        self.assertFalse(report["target_in_frame"][0])
        self.assertFalse(report["source_point_visible_in_target"][0])

    def test_no_hit_is_unknown(self):
        def cast(cam, uv):
            return np.full(len(uv), np.nan), np.zeros(len(uv), np.uint32), np.full(len(uv), np.nan)

        report = mesh_direction([[50, 50]], [[40, 50]], camera(), camera(1), cast, .002)
        self.assertFalse(report["source_hit"][0])
        self.assertFalse(report["source_point_visible_in_target"][0])

    def test_depth_shape_validation(self):
        with self.assertRaises(ValueError):
            world_from_depth([[1, 2]], [1, 2], camera())


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV integration tests use the existing DA3 environment")
class BackgroundOpenCVTests(unittest.TestCase):
    def setUp(self):
        import cv2
        cv2.setNumThreads(1)

    def test_insufficient_features_never_get_an_estimated_model(self):
        result = fit_rgb_geometry([[0, 0]], [[0, 0]], [False], GEOMETRY)
        self.assertEqual(result["state"], "inconclusive")
        self.assertEqual(result["models"], {})

    def test_exact_planar_motion_is_flagged_as_ambiguous_pose_evidence(self):
        a = np.random.default_rng(3).uniform(0, 500, (90, 2))
        b = a + [23, 11]
        validation = np.arange(len(a)) % 3 == 0
        result = fit_rgb_geometry(a, b, validation, GEOMETRY)
        self.assertEqual(result["models"]["H"]["validation"]["fraction_within_2px"], 1)
        self.assertTrue(result["planar_or_repetitive_explanation_possible"])

    def test_validation_observations_cannot_change_fitted_matrices(self):
        rng = np.random.default_rng(4)
        world = rng.uniform([-3, -2, 4], [3, 2, 10], (120, 3))
        a = 300 * world[:, :2] / world[:, 2, None] + [320, 240]
        second = world - [1, 0, 0]
        b = 300 * second[:, :2] / second[:, 2, None] + [320, 240]
        validation = np.arange(len(a)) % 3 == 0
        first = fit_rgb_geometry(a, b, validation, GEOMETRY)
        altered = b.copy()
        altered[validation] += [80, 70]
        second = fit_rgb_geometry(a, altered, validation, GEOMETRY)
        for kind in ("F", "H"):
            np.testing.assert_array_equal(first["models"][kind]["matrix"], second["models"][kind]["matrix"])
            np.testing.assert_array_equal(first["models"][kind]["train_inlier_mask"], second["models"][kind]["train_inlier_mask"])
        self.assertEqual(first["models"]["F"]["validation"]["fraction_within_2px"], 1)
        self.assertEqual(second["models"]["F"]["validation"]["fraction_within_2px"], 0)

    def test_mutual_ratio_rejects_ambiguous_descriptor_ties(self):
        first = {"descriptors": np.array([[0., 0], [10, 10]], np.float32)}
        second = {"descriptors": np.array([[0., 0], [0, 0], [10, 10]], np.float32)}
        ids, _ = mutual_ratio_matches(first, second, "SIFT", .75)
        np.testing.assert_array_equal(ids, [[1, 2]])


if __name__ == "__main__":
    unittest.main()
