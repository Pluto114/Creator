"""Known-answer contract checks for merge_collinear_segments; no model weights or Blender required."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.chain_segment_union import merge_collinear_segments  # noqa: E402


def segments(*endpoint_pairs):
    return np.array(endpoint_pairs, dtype=float)


def canonical(merged):
    """Sort the two endpoints inside every segment and the segments themselves.

    The contract promises a deterministic output order but does not pin the
    exact sorting rule, so known-answer checks compare normalized endpoint
    sets.  Exact equality is intended: output endpoints must come verbatim
    from the original input.
    """
    array = np.asarray(merged, dtype=float)
    if array.size == 0:
        return array.reshape(0, 2, 3)
    array = array.reshape(-1, 2, 3)
    array = np.sort(array, axis=1)
    order = np.lexsort(
        (
            array[:, 1, 2], array[:, 1, 1], array[:, 1, 0],
            array[:, 0, 2], array[:, 0, 1], array[:, 0, 0],
        )
    )
    return array[order]


class ChainSegmentUnionContractTests(unittest.TestCase):
    def test_x_axis_nested_overlap_and_touch_merge_into_true_range(self):
        nested = [[0, 0, 0], [4, 0, 0]]
        inside = [[1, 0, 0], [3, 0, 0]]
        overlap = [[2, 0, 0], [5, 0, 0]]
        touch = [[5, 0, 0], [7, 0, 0]]
        merged = merge_collinear_segments(segments(nested, inside, overlap, touch))
        self.assertEqual(merged.shape, (1, 2, 3))
        np.testing.assert_array_equal(canonical(merged), canonical([[[0, 0, 0], [7, 0, 0]]]))
        original_endpoints = segments(nested, inside, overlap, touch).reshape(-1, 3)
        for endpoint in merged[0]:
            self.assertTrue(
                any(np.array_equal(endpoint, known) for known in original_endpoints),
                f"endpoint {endpoint} is not taken from the original input",
            )

    def test_reorder_reversal_and_duplicates_are_idempotent(self):
        first = [[0, 0, 0], [4, 0, 0]]
        second = [[2, 0, 0], [6, 0, 0]]
        third = [[6, 0, 0], [8, 0, 0]]
        baseline = merge_collinear_segments(segments(first, second, third))
        self.assertEqual(baseline.shape, (1, 2, 3))
        np.testing.assert_array_equal(canonical(baseline), canonical([[[0, 0, 0], [8, 0, 0]]]))
        reordered = merge_collinear_segments(segments(third, first, second))
        np.testing.assert_array_equal(reordered, baseline)
        reversed_segments = segments(first[::-1], second[::-1], third[::-1])
        np.testing.assert_array_equal(merge_collinear_segments(reversed_segments), baseline)
        duplicated = merge_collinear_segments(segments(first, second, third, second, first))
        np.testing.assert_array_equal(duplicated, baseline)
        np.testing.assert_array_equal(merge_collinear_segments(baseline), baseline)

    def test_positive_gap_between_collinear_segments_is_not_bridged(self):
        left = [[0, 0, 0], [2, 0, 0]]
        right = [[3, 0, 0], [5, 0, 0]]
        merged = merge_collinear_segments(segments(left, right))
        self.assertEqual(merged.shape, (2, 2, 3))
        np.testing.assert_array_equal(canonical(merged), canonical(segments(left, right)))

    def test_distinct_parallel_axes_stay_separate_without_averaging(self):
        low = [[0, 0, 0], [2, 0, 0]]
        high = [[0, 1, 0], [2, 1, 0]]
        merged = merge_collinear_segments(segments(low, high))
        self.assertEqual(merged.shape, (2, 2, 3))
        np.testing.assert_array_equal(canonical(merged), canonical(segments(low, high)))

    def test_crossing_nonparallel_segments_stay_separate(self):
        horizontal = [[-1, 0, 0], [1, 0, 0]]
        vertical = [[0, -1, 0], [0, 1, 0]]
        merged = merge_collinear_segments(segments(horizontal, vertical))
        self.assertEqual(merged.shape, (2, 2, 3))
        np.testing.assert_array_equal(canonical(merged), canonical(segments(horizontal, vertical)))
        with self.subTest(scenario="touching_perpendicular"):
            x_axis = [[0, 0, 0], [2, 0, 0]]
            y_axis = [[0, 0, 0], [0, 2, 0]]
            touched = merge_collinear_segments(segments(x_axis, y_axis))
            self.assertEqual(touched.shape, (2, 2, 3))
            np.testing.assert_array_equal(canonical(touched), canonical(segments(x_axis, y_axis)))

    def test_empty_input_returns_empty_float_and_invalid_inputs_raise(self):
        empty = merge_collinear_segments(np.zeros((0, 2, 3)))
        self.assertEqual(empty.shape, (0, 2, 3))
        self.assertTrue(np.issubdtype(empty.dtype, np.floating))
        invalid_cases = [
            ("nan", [[[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]]),
            ("inf", [[[0.0, 0.0, 0.0], [np.inf, 1.0, 0.0]]]),
            ("zero_length", [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
        ]
        for name, endpoint_pairs in invalid_cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    merge_collinear_segments(segments(*endpoint_pairs))


if __name__ == "__main__":
    unittest.main()

