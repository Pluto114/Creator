"""Robust single-axis shared-readout contract tests for creator_eval.robust_axis_readout.

No model weights, GPU or Blender required. The implementation under test is
maintained by the coordinating agent; these checks pin the shared-readout
contract only, following the approved task specification.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.robust_axis_readout import readout


def x_line(count=101):
    return np.c_[np.linspace(0.0, 1.0, count), np.zeros(count), np.zeros(count)]


def always_true(count):
    """Support callback tolerant of being called with no args or with the point set."""
    fallback = np.ones(count, dtype=bool)

    def support(*args, **_kwargs):
        if args and np.asarray(args[0]).ndim == 2:
            return np.ones(len(args[0]), dtype=bool)
        return fallback

    return support


class RobustAxisReadoutTests(unittest.TestCase):
    def test_single_x_axis_reads_one_complete_segment(self):
        # 规格1：x轴101点，全true，1段端点0..1。
        values = x_line(101)
        result = readout(values, 0.01, [0.0, 0.0, 0.0], always_true(len(values)))
        for key in ("state", "reason", "segments", "components"):
            self.assertIn(key, result)
        self.assertEqual(result["state"], "complete")
        segments = np.asarray(result["segments"])
        self.assertEqual(segments.shape, (1, 2, 3))
        endpoints = segments[0]
        np.testing.assert_allclose(np.sort(endpoints[:, 0]), [0.0, 1.0], atol=0.01)
        np.testing.assert_allclose(endpoints[:, 1:], np.zeros((2, 2)), atol=0.01)

    def test_duplicated_and_shuffled_points_keep_identical_segments(self):
        # 规格2：相同坐标shuffle+重复，segments与原结果全同。
        values = x_line(101)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(values))
        jumbled = np.vstack([values[order], values[order[:17]]])
        reference = readout(values, 0.01, [0.0, 0.0, 0.0], always_true(len(values)))
        result = readout(jumbled, 0.01, [0.0, 0.0, 0.0], always_true(len(jumbled)))
        self.assertEqual(result["state"], "complete")
        np.testing.assert_array_equal(
            np.asarray(result["segments"]), np.asarray(reference["segments"])
        )

    def test_far_offline_points_do_not_derail_the_axis(self):
        # 规格3：x线101点加五个离线远点，全true，仍1段贴原x轴（atol=.01）。
        values = np.vstack(
            [
                x_line(101),
                np.array(
                    [
                        [0.0, 0.3, 0.0],
                        [0.2, -0.4, 0.1],
                        [0.4, 0.3, 0.2],
                        [0.6, -0.3, 0.3],
                        [0.8, 0.4, -0.3],
                    ]
                ),
            ]
        )
        result = readout(values, 0.01, [0.0, 0.0, 0.0], always_true(len(values)))
        self.assertEqual(result["state"], "complete")
        segments = np.asarray(result["segments"])
        self.assertEqual(segments.shape[0], 1)
        endpoints = segments[0]
        np.testing.assert_allclose(np.sort(endpoints[:, 0]), [0.0, 1.0], atol=0.01)
        np.testing.assert_allclose(endpoints[:, 1:], np.zeros((2, 2)), atol=0.01)

    def test_two_equal_parallel_lines_are_rejected(self):
        # 规格4：两条等强平行线，全true，拒绝、空segments且reason非空。
        lower = np.c_[np.linspace(0.0, 1.0, 101), np.full(101, -0.1), np.zeros(101)]
        upper = np.c_[np.linspace(0.0, 1.0, 101), np.full(101, 0.1), np.zeros(101)]
        values = np.vstack([lower, upper])
        result = readout(values, 0.01, [0.0, 0.0, 0.0], always_true(len(values)))
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(np.asarray(result["segments"])), 0)
        self.assertTrue(result["reason"])

    def test_two_separated_clusters_read_two_segments_without_bridging(self):
        # 规格5：x簇0..0.3与0.7..1各31点，全true，恰2段且无段覆盖x=.5。
        left = np.c_[np.linspace(0.0, 0.3, 31), np.zeros(31), np.zeros(31)]
        right = np.c_[np.linspace(0.7, 1.0, 31), np.zeros(31), np.zeros(31)]
        values = np.vstack([left, right])
        result = readout(values, 0.01, [0.0, 0.0, 0.0], always_true(len(values)))
        segments = np.asarray(result["segments"])
        self.assertEqual(segments.shape[0], 2)
        for segment in segments:
            low, high = np.sort(segment[:, 0])
            self.assertFalse(low <= 0.5 <= high)

    def test_support_gap_splits_one_line_into_two_segments(self):
        # 规格6：连续x轴101点，callback允许x<.4或x>.6，恰2段不跨中间。
        values = x_line(101)

        def support(*args, **_kwargs):
            points = np.asarray(args[0]) if args else values
            xs = points[:, 0]
            return (xs < 0.4) | (xs > 0.6)

        result = readout(values, 0.01, [0.0, 0.0, 0.0], support)
        segments = np.asarray(result["segments"])
        self.assertEqual(segments.shape[0], 2)
        for segment in segments:
            low, high = np.sort(segment[:, 0])
            self.assertFalse(low <= 0.5 <= high)


if __name__ == "__main__":
    unittest.main()
