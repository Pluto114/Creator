"""Evaluation-side interval and finite-axis checks; uses the existing image I/O environment."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_rod_edges import projected_axis_at_rows, visible_intervals


class EdgeAuditTests(unittest.TestCase):
    def test_pixel_footprints_keep_real_gaps(self):
        self.assertEqual(visible_intervals(np.array([0, 1, 1, 0, 1, 0], bool)), [[.5, 2.5], [3.5, 4.5]])
        self.assertEqual(visible_intervals(np.zeros(6, bool)), [])

    def test_projected_axis_is_finite_and_does_not_fill_gap(self):
        camera = {"K_index": [[10, 0, 0], [0, 10, 0], [0, 0, 1]], "world_to_camera_cv": np.eye(4), "clip_start": .1}
        segments = [[[2, 1, 10], [4, 3, 10]], [[5, 4, 10], [7, 6, 10]]]
        result = projected_axis_at_rows(segments, camera, np.array([0, 1, 2, 3.5, 5, 7]))
        np.testing.assert_allclose(result, [np.nan, 2, 3, np.nan, 6, np.nan], equal_nan=True)

    def test_overlapping_projected_truth_requires_explicit_handling(self):
        camera = {"K_index": np.eye(3), "world_to_camera_cv": np.eye(4), "clip_start": .1}
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            projected_axis_at_rows([[[0, 0, 1], [0, 2, 1]], [[1, 0, 1], [1, 2, 1]]], camera, np.array([1]))


if __name__ == "__main__":
    unittest.main()
