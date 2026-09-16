"""Known sampling failure and its predeclared dense-row control; no data or GT IO."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_evidence import build_rod_candidate


def perfect_rod_views(row_stride):
    """A perfect vertical rod at z=16, viewed by five parallel lateral cameras.

    Camera span 16 gives the frozen 0.016 world sampling step. With fy/z=125,
    each step moves the projection by exactly two original-image pixels.
    """
    views = []
    intrinsic = np.array([[2000.0, 0, 1100], [0, 2000.0, 1000], [0, 0, 1]])
    for index, center_x in enumerate([-8.0, -4.0, 0.0, 4.0, 8.0]):
        extrinsic = np.eye(4)
        extrinsic[0, 3] = -center_x
        image_x = 1100.0 - 125 * center_x
        views.append(
            {
                "view_id": str(index),
                "K_index": intrinsic,
                "world_to_camera_cv": extrinsic,
                "size_wh": [2201, 2001],
                "image_line": [1, 0, -image_x],
                "observations": [
                    {"xy": [image_x, y], "state": "accepted", "width_px": 4}
                    for y in range(960, 1041, row_stride)
                ],
            }
        )
    return views


class RodEvidenceSamplingTests(unittest.TestCase):
    def test_sparse_perfect_observations_can_be_rejected_by_grid_phase(self):
        result = build_rod_candidate(perfect_rod_views(row_stride=4))
        self.assertTrue(result["geometry"]["passed"])
        self.assertAlmostEqual(result["evidence"]["step"], 0.016)
        np.testing.assert_array_equal(
            result["evidence"]["positive_view_counts"], [5, 0] * 20 + [5]
        )
        self.assertEqual(result["state"], "rejected")
        self.assertIn("no_consecutive_supported_segment", result["rejection_reasons"])
        self.assertEqual(result["segments"].shape, (0, 2, 3))
        # 这不是几何恢复失败：所有已测量行都完美。每隔一行没有观测，
        # 恰好把每条支持段切成单点；不能把它写成模型没看见杆。

    def test_dense_rows_remove_this_aliasing_without_changing_geometry(self):
        sparse = build_rod_candidate(perfect_rod_views(row_stride=4))
        dense = build_rod_candidate(perfect_rod_views(row_stride=1))
        self.assertTrue(dense["geometry"]["passed"])
        self.assertEqual(sparse["config"], dense["config"])
        np.testing.assert_allclose(sparse["line"]["anchor"], dense["line"]["anchor"])
        np.testing.assert_allclose(sparse["line"]["direction"], dense["line"]["direction"])
        np.testing.assert_array_equal(
            sparse["evidence"]["parameters"], dense["evidence"]["parameters"]
        )
        self.assertEqual(dense["state"], "accepted")
        self.assertEqual(dense["segments"].shape, (1, 2, 3))
        np.testing.assert_allclose(dense["segments"][0], [[0, -0.32, 16], [0, 0.32, 16]], atol=1e-10)
        self.assertTrue(np.all(dense["evidence"]["positive_view_counts"] == 5))
        # 逐行观测仅消除了这个已知采样失配；此解析结果不证明真实图片也会恢复。


if __name__ == "__main__":
    unittest.main()
