"""Contract checks for guide-direction block readout; no model weights or Blender required."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.guided_block_axis_readout import readout  # noqa: E402


class GuidedBlockAxisReadoutContractTests(unittest.TestCase):
    """Exactly six contract checks for creator_eval.guided_block_axis_readout.readout."""

    GUIDE = np.array([1.0, 0.0, 0.0])

    def _readout(self, values, support_fn):
        return readout(values, 0.01, [0, 0, 0], support_fn, self.GUIDE)

    def _assert_complete_result(self, result):
        self.assertIsInstance(result, dict)
        for key in ("state", "reason", "segments", "components"):
            self.assertIn(key, result)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["segments"].ndim, 3)
        self.assertEqual(result["segments"].shape[1:], (2, 3))

    def _ordered_segments(self, segments):
        ordered = np.asarray([seg[np.argsort(seg @ self.GUIDE)] for seg in segments])
        return ordered[np.argsort([seg[0] @ self.GUIDE for seg in ordered])]

    def test_01_continuous_axis_yields_one_segment_with_exact_endpoints(self):
        values = np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)]
        result = self._readout(values, lambda pts: np.ones(len(pts), dtype=bool))
        self._assert_complete_result(result)
        self.assertEqual(result["segments"].shape[0], 1)
        np.testing.assert_allclose(
            self._ordered_segments(result["segments"]),
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            atol=1e-9,
        )

    def test_02_shuffled_and_duplicated_points_repeat_segments_exactly(self):
        base = np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)]
        rng = np.random.default_rng(11)
        shuffled = base[rng.permutation(len(base))]
        duplicated = np.vstack([shuffled, base[rng.integers(0, len(base), 41)]])
        def support(pts):
            return np.ones(len(pts), dtype=bool)
        first = self._readout(base, support)
        second = self._readout(shuffled, support)
        third = self._readout(duplicated, support)
        np.testing.assert_array_equal(first["segments"], second["segments"])
        np.testing.assert_array_equal(first["segments"], third["segments"])
        np.testing.assert_array_equal(second["segments"], third["segments"])

    def test_03_two_blocks_never_bridge_the_unsupported_gap(self):
        left = np.c_[np.linspace(0.0, 0.3, 31), np.zeros(31), np.zeros(31)]
        right = np.c_[np.linspace(0.7, 1.0, 31), np.zeros(31), np.zeros(31)]
        result = self._readout(np.vstack([left, right]), lambda pts: np.ones(len(pts), dtype=bool))
        self._assert_complete_result(result)
        self.assertEqual(result["segments"].shape[0], 2)
        ordered = self._ordered_segments(result["segments"])
        np.testing.assert_allclose(
            ordered,
            [[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], [[0.7, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            atol=0.010000001,
        )
        for seg in ordered:
            xs = seg[:, 0]
            self.assertFalse(np.min(xs) < 0.5 < np.max(xs))

    def test_04_support_callback_gap_splits_continuous_axis(self):
        values = np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)]

        def support(points):
            x = points[:, 0]
            return (x < 0.4) | (x > 0.6)

        result = self._readout(values, support)
        self._assert_complete_result(result)
        self.assertEqual(result["segments"].shape[0], 2)
        ordered = self._ordered_segments(result["segments"])
        np.testing.assert_allclose(
            ordered,
            [[[0.0, 0.0, 0.0], [0.39, 0.0, 0.0]], [[0.61, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            atol=0.010000001,
        )
        for seg in ordered:
            xs = seg[:, 0]
            self.assertFalse(np.min(xs) < 0.5 < np.max(xs))

    def test_05_split_double_track_is_rejected_with_nonempty_reason(self):
        x = np.linspace(0.0, 1.0, 101)
        lower = np.c_[x, np.full(101, -0.02), np.zeros(101)]
        upper = np.c_[x, np.full(101, 0.02), np.zeros(101)]
        result = self._readout(np.vstack([lower, upper]), lambda pts: np.ones(len(pts), dtype=bool))
        self._assert_complete_result(result)
        self.assertEqual(result["segments"].shape[0], 0)
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"].strip())

    def test_06_translated_line_keeps_offset_endpoints(self):
        offset = np.array([0.0, 0.2, 0.3])
        values = np.c_[np.linspace(0.0, 1.0, 101), np.zeros(101), np.zeros(101)] + offset
        result = self._readout(values, lambda pts: np.ones(len(pts), dtype=bool))
        self._assert_complete_result(result)
        self.assertEqual(result["segments"].shape[0], 1)
        np.testing.assert_allclose(
            self._ordered_segments(result["segments"]),
            [[[0.0, 0.2, 0.3], [1.0, 0.2, 0.3]]],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
