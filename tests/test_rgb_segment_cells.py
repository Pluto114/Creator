"""Projective-cell clipping contracts using explicit synthetic membership tables."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_segment_cells as cells  # noqa: E402
from creator_eval.rgb_chain_support import chain_support_mask  # noqa: E402


def context(height=8, unknown_rows=()):
    """A hand-authored downstream support-consumer fixture, not chain discovery."""
    views, memberships = [], []
    for index in range(4):
        left, right = np.full((height, 1), 4.), np.full((height, 1), 6.)
        members = np.ones((height, 1, 1), bool)
        for row in unknown_rows:
            left[row] = right[row] = np.nan
            members[row] = False
        views.append(dict(view_id=str(index), K_index=np.eye(3),
            world_to_camera_cv=np.c_[np.eye(3), np.zeros(3)], left=left, right=right))
        memberships.append(members)
    return dict(views=views, row_members=memberships,
        association=dict(fallback_to_raw_union=False, assignments=[[[0]] * 4]))


def dense(segments, count=1001):
    result = [a + np.linspace(0., 1., count)[:, None] * (b-a) for a, b in segments]
    return np.concatenate(result) if result else np.empty((0, 3))


class ProjectiveCellContracts(unittest.TestCase):
    def test_unknown_row_missed_by_sparse_uniform_samples_is_clipped(self):
        support = context(unknown_rows=(3,))
        segment = np.array([[[5., .2, 1.], [5., 6.2, 1.]]])
        self.assertTrue(chain_support_mask(dense(segment, 4), support).all())
        result = cells.clip_segments(segment, support, 0.)
        self.assertEqual(len(result["segments"]), 2)
        self.assertTrue(chain_support_mask(dense(result["segments"]), support).all())
        self.assertLess(result["segments"][0, 1, 1], 2.5)
        self.assertGreaterEqual(result["segments"][1, 0, 1], 3.5)
        self.assertGreater(result["trimmed_open_endpoints"], 0)

    def test_perspective_nonconstant_depth_uses_rational_row_event(self):
        support = context(unknown_rows=(3,))
        segment = np.array([[[5., .2, 1.], [10., 12.4, 2.]]])
        result = cells.clip_segments(segment, support, 0.)
        self.assertEqual(len(result["segments"]), 2)
        self.assertTrue(chain_support_mask(dense(result["segments"]), support).all())
        a, b = segment[0]
        expected_t = (2.5*a[2]-a[1]) / ((b[1]-a[1])-2.5*(b[2]-a[2]))
        actual_t = (result["segments"][0, 1, 2]-a[2]) / (b[2]-a[2])
        self.assertAlmostEqual(actual_t, expected_t, places=8)
        endpoints = result["segments"].reshape(-1, 3)
        self.assertLess(np.linalg.norm(np.cross(endpoints-a, b-a), axis=1).max(), 1e-10)

    def test_horizontal_band_boundaries_clip_without_a_row_change(self):
        support = context()
        segment = np.array([[[1., 2., 1.], [9., 2., 1.]]])
        result = cells.clip_segments(segment, support, 0.)
        np.testing.assert_allclose(result["segments"], [[[3., 2., 1.], [7., 2., 1.]]], atol=1e-12)
        self.assertTrue(chain_support_mask(dense(result["segments"]), support).all())

    def test_row_dependent_horizontal_band_cannot_be_interpolated(self):
        support = context()
        for view in support["views"]:
            view["left"][3, 0], view["right"][3, 0] = 8., 10.
        segment = np.array([[[5., .2, 1.], [5., 6.2, 1.]]])
        result = cells.clip_segments(segment, support, 0.)
        self.assertEqual(len(result["segments"]), 2)
        self.assertTrue(chain_support_mask(dense(result["segments"]), support).all())

    def test_three_views_cannot_be_assembled_from_different_chains(self):
        support = context()
        support["association"]["assignments"] = [[[0]]*4, [[1]]*4]
        for index in range(4):
            members = np.zeros((8, 1, 2), bool)
            members[:, :, 0 if index < 2 else 1] = True
            support["row_members"][index] = members
        result = cells.clip_segments(np.array([[[5., .2, 1.], [5., 6.2, 1.]]]), support, 0.)
        self.assertEqual(result["segments"].shape, (0, 2, 3))

    def test_positive_depth_boundary_is_clipped_as_an_open_endpoint(self):
        support = context()
        segment = np.array([[[2.5, 1., .5], [-2.5, -1., -.5]]])
        result = cells.clip_segments(segment, support, 0.)
        self.assertEqual(len(result["segments"]), 1)
        self.assertTrue(np.all(result["segments"][..., 2] > 1e-10))
        self.assertTrue(chain_support_mask(dense(result["segments"]), support).all())

    def test_minimum_length_applies_after_actual_finite_clipping(self):
        support = context()
        segment = np.array([[[1., 2., 1.], [9., 2., 1.]]])
        result = cells.clip_segments(segment, support, 4.01)
        self.assertEqual(result["segments"].shape, (0, 2, 3))
        self.assertEqual(result["discarded_short_runs"], 1)

    def test_each_budget_raises_without_a_partial_success_return(self):
        support = context()
        segment = np.array([[[5., .2, 1.], [5., 6.2, 1.]]])
        for field in ("maximum_cells", "maximum_membership_checks"):
            with self.subTest(field=field), patch.dict(cells.DEFAULTS, {field: 0}):
                with self.assertRaisesRegex(OverflowError, "rgb_cell_budget_exceeded"):
                    cells.clip_segments(segment, support, 0.)

    def test_no_input_geometry_or_context_mutation(self):
        support = context(unknown_rows=(3,))
        original = copy.deepcopy(support)
        segment = np.array([[[5., .2, 1.], [5., 6.2, 1.]]])
        before = segment.copy()
        cells.clip_segments(segment, support, 0.)
        np.testing.assert_array_equal(segment, before)
        for a, b in zip(support["views"], original["views"]):
            np.testing.assert_array_equal(a["left"], b["left"])
            np.testing.assert_array_equal(a["right"], b["right"])


if __name__ == "__main__":
    unittest.main()
