"""Fixed fold-consensus re-check: 6 mechanism tests for agreed_segments."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import ridge_fold_consensus  # noqa: E402


class RidgeFoldConsensusTests(unittest.TestCase):
    """0.2 m x-axis segments, 1 mm perturbations, 5 mm match threshold."""

    def setUp(self):
        self.config = {"voxel_size": .01, "merge_distance_voxels": .5}
        self.empty = np.empty((0, 2, 3))
        self.segment = np.array([[[0., 0., 0.], [0.2, 0., 0.]]])

    def test_reversed_endpoints_of_same_segment_are_confirmed(self):
        first = self.segment
        second = first[:, ::-1, :]  # 反向端点
        result = ridge_fold_consensus.agreed_segments(first, second, self.empty, self.config)
        self.assertEqual(result.shape, (1, 2, 3))
        np.testing.assert_array_equal(result, first)  # 保留first实际几何

    def test_empty_second_fold_yields_no_output(self):
        result = ridge_fold_consensus.agreed_segments(
            self.segment, self.empty, self.empty, self.config)
        self.assertEqual(result.shape, (0, 2, 3))
        self.assertEqual(result.size, 0)

    def test_partial_overlap_and_parallel_separation_do_not_match(self):
        first = np.array([
            [[0., 0., 0.], [0.2, 0., 0.]],
            [[0., 0., 0.], [0.2, 0., 0.]],
        ])
        partial = np.array([[[0.05, 0., 0.], [0.25, 0., 0.]]])    # 中间重叠、端点差50mm
        parallel = np.array([[[0.25, 0., 0.], [0.45, 0., 0.]]])   # 平行分离50mm
        second = np.concatenate([partial, parallel], axis=0)
        result = ridge_fold_consensus.agreed_segments(first, second, self.empty, self.config)
        self.assertEqual(result.shape, (0, 2, 3))

    def test_segments_already_in_existing_are_not_repeated(self):
        known = self.segment
        fresh = np.array([[[0.4, 0., 0.], [0.6, 0., 0.]]])
        first = np.concatenate([known, fresh], axis=0)
        existing = known.copy()
        result = ridge_fold_consensus.agreed_segments(first, first, existing, self.config)
        self.assertEqual(result.shape, (1, 2, 3))
        np.testing.assert_array_equal(result, fresh)

    def test_approximate_match_keeps_first_coordinates_not_average(self):
        first = self.segment
        perturbed = np.array([[[0.001, 0.001, 0.001], [0.199, 0.001, 0.001]]])  # 约1mm扰动
        result = ridge_fold_consensus.agreed_segments(
            first, perturbed, self.empty, self.config)
        self.assertEqual(result.shape, (1, 2, 3))
        np.testing.assert_array_equal(result, first)  # 不平均、不延长

    def test_input_arrays_are_not_modified(self):
        first = self.segment
        second = first[:, ::-1, :].copy()
        existing = np.array([[[0.4, 0., 0.], [0.6, 0., 0.]]])
        first_before, second_before, existing_before = \
            first.copy(), second.copy(), existing.copy()
        ridge_fold_consensus.agreed_segments(first, second, existing, self.config)
        np.testing.assert_array_equal(first, first_before)
        np.testing.assert_array_equal(second, second_before)
        np.testing.assert_array_equal(existing, existing_before)


if __name__ == "__main__":
    unittest.main()


