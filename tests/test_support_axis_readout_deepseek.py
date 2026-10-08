"""Known-answer contracts for support axis readout; no model weights or Blender required."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.support_axis_readout import readout  # noqa: E402


def _all_supported(points):
    return np.ones(len(points), dtype=bool)


class SupportAxisReadoutContractTests(unittest.TestCase):
    """6项原始点支持保真轴读取契约；验收测试由Codex在真实实现上运行。"""

    @staticmethod
    def _canonical(segments):
        """按x方向规范端点次序并按起点排序，保持空数组形状为(N,2,3)。"""
        segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
        swapped = segments[:, 0, 0] > segments[:, 1, 0]
        segments[swapped] = segments[swapped][:, ::-1]
        return segments[np.argsort(segments[:, 0, 0])]

    def test_clean_x_axis_reads_one_segment_with_observed_endpoints(self):
        values = np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)]
        result = readout(values, 0.01, [0.0, 0.0, 0.0], _all_supported)
        for key in ("state", "reason", "segments", "components"):
            self.assertIn(key, result)
        self.assertEqual(result["state"], "complete")
        segments = self._canonical(result["segments"])
        self.assertEqual(segments.shape, (1, 2, 3))
        np.testing.assert_allclose(segments[0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-9)

    def test_shuffle_and_duplicates_keep_exact_segments(self):
        base = np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)]
        rng = np.random.default_rng(7)
        duplicated = np.vstack([base, base[::2], base[10:40]])
        rng.shuffle(duplicated)
        clean = readout(base, 0.01, [0.0, 0.0, 0.0], _all_supported)
        shuffled = readout(duplicated, 0.01, [0.0, 0.0, 0.0], _all_supported)
        self.assertEqual(clean["state"], "complete")
        self.assertEqual(shuffled["state"], "complete")
        clean_segments = self._canonical(clean["segments"])
        shuffled_segments = self._canonical(shuffled["segments"])
        self.assertEqual(shuffled_segments.shape, (1, 2, 3))
        np.testing.assert_array_equal(shuffled_segments, clean_segments)

    def test_gap_splits_x_axis_into_two_segments_not_crossing_midpoint(self):
        values = np.vstack([
            np.c_[np.linspace(0.0, 0.3, 31), np.zeros(31), np.zeros(31)],
            np.c_[np.linspace(0.7, 1.0, 31), np.zeros(31), np.zeros(31)],
        ])
        result = readout(values, 0.01, [0.0, 0.0, 0.0], _all_supported)
        self.assertEqual(result["state"], "complete")
        segments = self._canonical(result["segments"])
        self.assertEqual(segments.shape, (2, 2, 3))
        for segment in segments:
            self.assertTrue((segment[:, 0] < 0.5).all() or (segment[:, 0] > 0.5).all())
        np.testing.assert_allclose(segments[0], [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], atol=0.010000001)
        np.testing.assert_allclose(segments[1], [[0.7, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=0.010000001)

    def test_support_callback_cuts_middle_and_keeps_two_segments(self):
        values = np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)]

        def support(points):
            return (points[:, 0] < 0.4) | (points[:, 0] > 0.6)

        result = readout(values, 0.01, [0.0, 0.0, 0.0], support)
        self.assertEqual(result["state"], "complete")
        segments = self._canonical(result["segments"])
        self.assertEqual(segments.shape, (2, 2, 3))
        for segment in segments:
            self.assertTrue((segment[:, 0] < 0.5).all() or (segment[:, 0] > 0.5).all())
        left, right = segments[0], segments[1]
        np.testing.assert_allclose(left[0, 0], 0.0, atol=1e-9)
        self.assertLess(left[1, 0], 0.5)
        self.assertGreater(right[0, 0], 0.5)
        np.testing.assert_allclose(right[1, 0], 1.0, atol=1e-9)

    def test_equal_parallel_lines_reject_with_strong_competing_axis(self):
        values = np.vstack([
            np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)],
            np.c_[np.linspace(0.0, 1.0, 101), np.full(101, 0.2), np.zeros(101)],
        ])
        result = readout(values, 0.01, [0.0, 0.0, 0.0], _all_supported)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(np.asarray(result["segments"]).size, 0)
        self.assertEqual(result["reason"], "strong_competing_axis")

    def test_translated_axis_keeps_its_own_offset(self):
        values = np.c_[np.linspace(0.0, 1.0, 101), np.full(101, 0.2), np.full(101, 0.3)]
        result = readout(values, 0.01, [0.0, 0.0, 0.0], _all_supported)
        self.assertEqual(result["state"], "complete")
        segments = self._canonical(result["segments"])
        self.assertEqual(segments.shape, (1, 2, 3))
        np.testing.assert_allclose(segments[0], [[0.0, 0.2, 0.3], [1.0, 0.2, 0.3]], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
