"""Analytic image-support controls independent of the scored fixture objects."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_supported_readout import (  # noqa: E402
    SupportedReadout,
    support_mask,
    support_views,
)


def fixture():
    frames, cameras = [], []
    for i, x in enumerate([-.5, 0, .5, 1.]):
        cx = 50-20*x
        rows = [dict(y=y, status="observed", candidates=[dict(left_edge=dict(x=cx-.5), right_edge=dict(x=cx+.5))]) for y in range(10, 91)]
        frames.append(dict(view_id=str(i), size_wh=[101, 101], observations=dict(rows=rows)))
        cameras.append(dict(view_id=str(i), state="validated", K_index=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
                            world_to_camera_cv=np.c_[np.eye(3), [-x, 0, 0]].tolist()))
    return frames, cameras


class FixtureRGBReadoutTests(unittest.TestCase):
    def test_multiview_volume_accepts_supported_line_rejects_background_and_behind_camera(self):
        frames, cameras = fixture()
        points = np.array([[0, 0, 5], [0, .7, 5], [.2, 0, 5], [0, 0, 10], [0, 0, -5]])
        np.testing.assert_array_equal(support_mask(points, support_views(frames, cameras)), [1, 1, 0, 0, 0])

    def test_gap_unknown_ambiguity_never_supply_positive_votes(self):
        for state in ("unknown", "absent", "ambiguous"):
            frames, cameras = fixture()
            for frame in frames[:2]:
                frame["observations"]["rows"][40]["status"] = state
            np.testing.assert_array_equal(support_mask([[0, 0, 5], [0, .05, 5]], support_views(frames, cameras)), [0, 1])

    def test_row_cell_does_not_dilate_missing_row(self):
        frames, cameras = fixture()
        for frame in frames:
            frame["observations"]["rows"][40]["status"] = "absent"
        # y=49.51..50.49 stays in absent row 50, although adjacent rows are observed.
        points = [[0, y, 5] for y in [-.026, -.024, .024, .026]]
        np.testing.assert_array_equal(support_mask(points, support_views(frames, cameras)), [1, 0, 0, 1])

    def test_duplicate_views_and_invalid_observed_pair_fail(self):
        frames, cameras = fixture()
        views = support_views(frames, cameras)
        with self.assertRaises(ValueError):
            support_mask([[0, 0, 5]], [views[0]]*3)
        frames[0]["observations"]["rows"][0]["candidates"] *= 2
        with self.assertRaises(ValueError):
            support_views(frames, cameras)

    def test_point_and_curve_representation_use_same_mask_and_voxels(self):
        frames, cameras = fixture()
        views = support_views(frames, cameras)
        config = dict(voxel_size=.02)
        segment = np.array([[[0, -1., 5], [0, 1., 5]]])
        points = sample_segments(segment, .01, 2000000)
        line_result = SupportedReadout(np.empty((0, 3)), views)(segment, config)
        point_result = SupportedReadout(points, views)(np.empty((0, 2, 3)), config)
        self.assertEqual(line_result["occupied_voxels"], point_result["occupied_voxels"])
        np.testing.assert_array_equal(line_result["segments"], point_result["segments"])
        self.assertGreater(len(line_result["segments"]), 0)

    def test_filter_is_not_object_identity_and_does_not_mutate_raw_evidence(self):
        frames, cameras = fixture()
        original = copy.deepcopy(frames)
        views = support_views(frames, cameras)
        self.assertTrue(support_mask([[0, 0, 5]], views)[0])
        # The same pixels could be a painted line: this interface has no semantic identity claim.
        self.assertEqual(frames, original)


if __name__ == "__main__":
    unittest.main()
