"""Known-answer contracts for creator_eval.foreground_support; no Blender or weights needed."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.foreground_support import (  # noqa: E402
    fixed_seed_expansion,
    novel_support_mask,
)


class ForegroundSupportContractTests(unittest.TestCase):
    """Eight contracts; expectations are hand-derived from simple geometry."""

    def test_known_2v_tube_expansion_and_untouched_values(self):
        voxel_size, origin = 1.0, np.zeros(3)
        seeds = np.arange(7)
        values = np.c_[np.arange(7, dtype=float), np.zeros(7), np.zeros(7)]
        values = np.vstack([
            values,
            [0.5, 0.8, 0.8],      # 7: inside the 2v tube
            [10.0, -0.9, 0.4],    # 8: inside, beyond seed extent (infinite axis)
            [4.0, 1.3, 0.0],      # 9: inside
            [-1.5, -1.0, 1.2],    # 10: inside
            [3.0, 0.2, 0.2],      # 11: inside
            [0.5, 0.0, 3.0],      # 12: outside the 2v tube
            [7.0, 2.7, 1.5],      # 13: outside
            [-2.0, -2.2, -1.6],   # 14: outside
        ])
        original = values.copy()
        result = fixed_seed_expansion(values, seeds, voxel_size, origin)
        self.assertEqual(result["state"], "complete")
        self.assertIsNone(result["reason"])
        self.assertTrue(result["original_seed_preserved"])
        self.assertFalse(result["iterative_expansion"])
        self.assertEqual(result["added_point_count"], 5)
        np.testing.assert_array_equal(result["indices"], np.arange(12))
        np.testing.assert_array_equal(values, original)

    def test_expansion_uses_only_original_seed_axis_without_growth(self):
        voxel_size, origin = 1.0, np.zeros(3)
        seeds = np.arange(7)
        values = np.c_[np.arange(7, dtype=float), np.zeros(7), np.zeros(7)]
        support = np.c_[np.arange(-10.0, 10.0), np.full(20, 1.5), np.zeros(20)]
        beyond = np.c_[np.arange(-5.0, 35.0, 2.0), np.full(20, 3.0), np.zeros(20)]
        values = np.vstack([values, support, beyond])
        result = fixed_seed_expansion(values, seeds, voxel_size, origin)
        self.assertEqual(result["state"], "complete")
        self.assertFalse(result["iterative_expansion"])
        self.assertEqual(result["added_point_count"], 20)
        np.testing.assert_array_equal(result["indices"], np.arange(27))

    def test_all_original_seeds_preserved_including_far_outlier(self):
        voxel_size, origin = 1.0, np.zeros(3)
        values = np.c_[np.arange(16, dtype=float), np.zeros(16), np.zeros(16)]
        values = np.vstack([values, [7.5, 5.0, 0.0]])  # index 16: far off the axis
        seeds = np.arange(17)
        result = fixed_seed_expansion(values, seeds, voxel_size, origin)
        self.assertEqual(result["state"], "complete")
        self.assertIsNone(result["reason"])
        self.assertIsNotNone(result["seed_model"])
        self.assertTrue(result["original_seed_preserved"])
        self.assertEqual(result["seed_point_count"], 17)
        self.assertEqual(result["added_point_count"], 0)
        np.testing.assert_array_equal(result["indices"], np.arange(17))
        self.assertIn(16, result["indices"].tolist())

    def test_translation_keeps_indices_and_moves_centroid_consistently(self):
        voxel_size, origin = 1.0, np.zeros(3)
        seeds = np.arange(6)
        values = np.c_[np.arange(6, dtype=float), np.zeros(6), np.zeros(6)]
        values = np.vstack([
            values,
            [0.5, 1.3, 0.0],     # 6: inside
            [3.0, 0.0, 3.0],     # 7: outside
            [-2.0, -0.9, 0.4],   # 8: inside
            [8.0, 0.8, 0.8],     # 9: inside
            [5.5, 2.7, -1.5],    # 10: outside
        ])
        delta = np.array([2.5, -1.0, 4.0])
        result = fixed_seed_expansion(values, seeds, voxel_size, origin)
        shifted = fixed_seed_expansion(
            values + delta, seeds, voxel_size, origin + delta
        )
        self.assertEqual(result["state"], "complete")
        np.testing.assert_array_equal(
            result["indices"], [0, 1, 2, 3, 4, 5, 6, 8, 9]
        )
        np.testing.assert_array_equal(shifted["indices"], result["indices"])
        self.assertEqual(shifted["added_point_count"], result["added_point_count"])
        np.testing.assert_allclose(
            shifted["seed_model"]["centroid"],
            result["seed_model"]["centroid"] + delta,
            atol=1e-9,
        )
        x_axis = np.array([1.0, 0.0, 0.0])
        first_dir = result["seed_model"]["direction"]
        second_dir = shifted["seed_model"]["direction"]
        for direction in (first_dir, second_dir):
            alignment = abs(float(direction @ x_axis))
            self.assertAlmostEqual(alignment, 1.0, places=9)
        self.assertAlmostEqual(abs(float(first_dir @ second_dir)), 1.0, places=9)

    def test_invalid_parameters_are_rejected(self):
        values = np.c_[np.arange(7, dtype=float), np.zeros(7), np.zeros(7)]
        seeds = np.arange(7)
        origin = np.zeros(3)
        values_nan = values.copy()
        values_nan[2, 1] = np.nan
        values_inf = values.copy()
        values_inf[0, 2] = np.inf
        cases = [
            ("values_nan", values_nan, seeds, 1.0, origin),
            ("values_inf", values_inf, seeds, 1.0, origin),
            ("values_1d", np.zeros(6), seeds, 1.0, origin),
            ("values_nx2", np.zeros((6, 2)), seeds, 1.0, origin),
            ("voxel_zero", values, seeds, 0.0, origin),
            ("voxel_negative", values, seeds, -1.0, origin),
            ("voxel_nan", values, seeds, np.nan, origin),
            ("voxel_inf", values, seeds, np.inf, origin),
            ("voxel_bool", values, seeds, True, origin),
            ("voxel_numpy_bool", values, seeds, np.bool_(True), origin),
            ("origin_2d", values, seeds, 1.0, np.array([0.0, 0.0])),
            ("origin_nan", values, seeds, 1.0, np.array([0.0, 0.0, np.nan])),
            ("origin_inf", values, seeds, 1.0, np.array([0.0, np.inf, 0.0])),
            ("seed_duplicate", values, np.array([0, 0, 1, 2, 3, 4]), 1.0, origin),
            (
                "seed_float",
                values,
                np.array([0, 1, 2, 3, 4, 5], dtype=float),
                1.0,
                origin,
            ),
            (
                "seed_negative",
                values,
                np.array([-1, 0, 1, 2, 3, 4]),
                1.0,
                origin,
            ),
            (
                "seed_out_of_range",
                values,
                np.array([0, 1, 2, 3, 4, 7]),
                1.0,
                origin,
            ),
            ("seed_2d", values, np.array([[0, 1], [2, 3]]), 1.0, origin),
        ]
        for label, points, indices, voxel_size, origin_value in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    fixed_seed_expansion(points, indices, voxel_size, origin_value)
        unsigned = fixed_seed_expansion(
            values, np.arange(7, dtype=np.uint32), 1.0, np.zeros(3)
        )
        self.assertEqual(unsigned["state"], "complete")

    def test_underconstrained_and_empty_seeds_never_bootstrap(self):
        voxel_size, origin = 1.0, np.zeros(3)
        values = np.c_[np.arange(5, dtype=float), np.zeros(5), np.zeros(5)]
        pool = np.c_[np.arange(0.25, 5.0, 0.25), np.full(19, 1.0), np.zeros(19)]
        values = np.vstack([values, pool])
        seeds = np.arange(5)
        result = fixed_seed_expansion(values, seeds, voxel_size, origin)
        self.assertEqual(result["state"], "complete")
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"])
        self.assertIsNone(result["seed_model"])
        self.assertTrue(result["original_seed_preserved"])
        np.testing.assert_array_equal(result["indices"], np.arange(5))

        empty = fixed_seed_expansion(
            values, np.array([], dtype=int), voxel_size, origin
        )
        self.assertEqual(empty["state"], "complete")
        self.assertIsInstance(empty["reason"], str)
        self.assertTrue(empty["reason"])
        self.assertIsNone(empty["seed_model"])
        np.testing.assert_array_equal(empty["indices"], np.array([], dtype=int))

    def test_mask_known_endpoint_boundary_and_gap_answers(self):
        segments = np.array([
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ])
        points = np.array([
            [1.0, 0.0, 0.0],    # on segment -> 0
            [0.0, 0.0, 0.0],    # first endpoint -> 0
            [-1.0, 0.0, 0.0],   # exactly radius from endpoint -> explained
            [-1.5, 0.0, 0.0],   # 1.5 beyond endpoint -> novel (finite segment)
            [8.0, 0.0, 0.0],    # second endpoint -> 0
            [10.0, 0.0, 0.0],   # 2 beyond endpoint -> novel
            [4.0, 0.0, 0.0],    # gap center, distance 2 -> novel
            [4.0, 0.5, 0.0],    # sqrt(4.25) -> novel
            [1.0, 1.0, 0.0],    # perpendicular exactly radius -> explained
            [1.0, 1.5, 0.0],    # 1.5 above segment -> novel
            [4.0, 2.0, 0.0],    # sqrt(8) in the gap -> novel
            [3.0, 0.0, 0.0],    # exactly radius from endpoint 2 -> explained
            [5.0, 0.0, 0.0],    # exactly radius from endpoint 6 -> explained
            [1.0, 0.0, 2.0],    # 2 above segment -> novel
        ])
        expected = np.array([
            False, False, False, True, False, True, True,
            True, False, True, True, False, False, True,
        ])
        mask = novel_support_mask(points, segments, 1.0)
        np.testing.assert_array_equal(mask, expected)

    def test_empty_segments_all_true_and_invalid_inputs_rejected(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 1.0],
            [5.0, 0.0, 0.0],
            [-1.0, 2.0, 0.5],
        ])
        segments = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        nan_segments = segments.copy()
        nan_segments[0, 1, 2] = np.nan
        inf_segments = segments.copy()
        inf_segments[0, 0, 0] = np.inf
        nan_points = points.copy()
        nan_points[1, 0] = np.nan
        inf_points = points.copy()
        inf_points[2, 1] = np.inf

        mask = novel_support_mask(points, np.empty((0, 2, 3)), 1.0)
        np.testing.assert_array_equal(mask, np.ones(4, dtype=bool))
        no_points = novel_support_mask(np.empty((0, 3)), np.empty((0, 2, 3)), 1.0)
        self.assertEqual(no_points.shape, (0,))
        self.assertEqual(no_points.dtype, np.bool_)

        cases = [
            ("radius_neg", dict(points=points, segments=segments, radius=-0.5)),
            ("radius_nan", dict(points=points, segments=segments, radius=np.nan)),
            ("radius_inf", dict(points=points, segments=segments, radius=np.inf)),
            ("radius_bool", dict(points=points, segments=segments, radius=True)),
            (
                "radius_np_bool",
                dict(points=points, segments=segments, radius=np.bool_(True)),
            ),
            ("seg_nan", dict(points=points, segments=nan_segments, radius=1.0)),
            ("seg_inf", dict(points=points, segments=inf_segments, radius=1.0)),
            ("seg_2d", dict(points=points, segments=np.zeros((2, 3)), radius=1.0)),
            (
                "seg_shape",
                dict(points=points, segments=np.zeros((1, 3, 3)), radius=1.0),
            ),
            ("points_nan", dict(points=nan_points, segments=segments, radius=1.0)),
            ("points_inf", dict(points=inf_points, segments=segments, radius=1.0)),
            (
                "points_nx2",
                dict(points=np.zeros((4, 2)), segments=segments, radius=1.0),
            ),
        ]
        for label, kwargs in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    novel_support_mask(**kwargs)


if __name__ == "__main__":
    unittest.main()
