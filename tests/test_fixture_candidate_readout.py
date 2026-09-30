"""Analytic candidate-union controls; no fixture data or physical-score inputs."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_candidate_readout import (  # noqa: E402
    CandidateSupportedReadout,
    support_mask,
    support_views,
    support_votes,
)

EMPTY_POINTS = np.empty((0, 3))
EMPTY_SEGMENTS = np.empty((0, 2, 3))


def fixture(offsets=(0.,)):
    """Four translated pinholes, with candidate rows generated analytically."""
    frames, cameras = [], []
    for i, x in enumerate([-.5, 0., .5, 1.]):
        center = 50 - 20 * x
        rows = [dict(
            y=y, status="observed" if len(offsets) == 1 else "ambiguous",
            candidates=[dict(left_edge=dict(x=center+offset-.5),
                             right_edge=dict(x=center+offset+.5)) for offset in offsets],
        ) for y in range(10, 91)]
        frames.append(dict(view_id=str(i), size_wh=[101, 101], observations=dict(rows=rows)))
        cameras.append(dict(
            view_id=str(i), state="validated",
            K_index=[[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
            world_to_camera_cv=np.c_[np.eye(3), [-x, 0., 0.]].tolist(),
        ))
    return frames, cameras


class FixtureCandidateReadoutTests(unittest.TestCase):
    def test_unambiguous_support_still_rejects_wrong_depth_and_background(self):
        frames, cameras = fixture()
        points = [[0., 0., 5.], [0., .7, 5.], [.2, 0., 5.], [0., 0., 10.], [0., 0., -5.]]
        np.testing.assert_array_equal(support_mask(points, support_views(frames, cameras)), [1, 1, 0, 0, 0])

    def test_ambiguous_rows_preserve_both_intervals_but_not_their_envelope(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        points = [[-.25, 0., 5.], [.25, 0., 5.], [0., 0., 5.], [.5, 0., 5.]]
        views = support_views(frames, cameras)
        np.testing.assert_array_equal(support_votes(points, views), [4, 4, 0, 0])
        np.testing.assert_array_equal(support_mask(points, views), [1, 1, 0, 0])

    def test_each_view_casts_at_most_one_vote_despite_duplicate_and_overlapping_pairs(self):
        frames, cameras = fixture(offsets=(-.25, .25))
        for frame in frames[:2]:
            for row in frame["observations"]["rows"]:
                row["candidates"] *= 10
        for frame in frames[2:]:
            for row in frame["observations"]["rows"]:
                row.update(status="unknown", candidates=[])
        views = support_views(frames, cameras)
        np.testing.assert_array_equal(support_votes([[0., 0., 5.]], views), [2])
        self.assertFalse(support_mask([[0., 0., 5.]], views)[0])

    def test_unknown_absent_and_unrecorded_rows_supply_no_positive_support(self):
        for state in ("unknown", "absent", "unrecorded"):
            with self.subTest(state=state):
                frames, cameras = fixture(offsets=(-5., 5.))
                for frame in frames[:2]:
                    rows = frame["observations"]["rows"]
                    if state == "unrecorded":
                        rows.pop(40)
                    else:
                        rows[40].update(status=state, candidates=[])
                points = [[-.25, 0., 5.], [-.25, .05, 5.]]
                np.testing.assert_array_equal(support_mask(points, support_views(frames, cameras)), [0, 1])

    def test_nearest_row_cells_do_not_dilate_support_into_a_missing_row(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        for frame in frames:
            frame["observations"]["rows"][40].update(status="absent", candidates=[])
        # Projection y=49.51..50.49 remains in absent row 50.
        points = [[-.25, y, 5.] for y in [-.026, -.024, .024, .026]]
        np.testing.assert_array_equal(support_mask(points, support_views(frames, cameras)), [1, 0, 0, 1])

    def test_candidate_duplicates_order_and_synchronized_view_order_do_not_change_votes(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        points = np.array([[-.25, 0., 5.], [.25, 0., 5.], [0., 0., 5.]])
        expected = support_votes(points, support_views(frames, cameras))
        for frame in frames:
            frame["observations"]["rows"].reverse()
            for row in frame["observations"]["rows"]:
                row["candidates"] = row["candidates"][::-1] * 3
        np.testing.assert_array_equal(support_votes(points, support_views(frames[::-1], cameras[::-1])), expected)
        np.testing.assert_array_equal(support_mask(points[::-1], support_views(frames, cameras)), (expected >= 3)[::-1])

    def test_invalid_edge_pairs_row_coordinates_and_view_identity_are_rejected(self):
        def bad_pair(frames):
            frames[0]["observations"]["rows"][0]["candidates"][1]["right_edge"]["x"] = float("nan")

        def reversed_pair(frames):
            pair = frames[0]["observations"]["rows"][0]["candidates"][1]
            pair["left_edge"]["x"], pair["right_edge"]["x"] = pair["right_edge"]["x"], pair["left_edge"]["x"]

        def outside_pair(frames):
            frames[0]["observations"]["rows"][0]["candidates"][1]["right_edge"]["x"] = 101.

        def fractional_row(frames):
            frames[0]["observations"]["rows"][0]["y"] = 10.5

        def duplicate_row(frames):
            frames[0]["observations"]["rows"].append(copy.deepcopy(frames[0]["observations"]["rows"][0]))

        for change in (bad_pair, reversed_pair, outside_pair, fractional_row, duplicate_row):
            with self.subTest(change=change.__name__):
                frames, cameras = fixture(offsets=(-5., 5.))
                change(frames)
                with self.assertRaises(ValueError):
                    support_views(frames, cameras)
        frames, cameras = fixture()
        with self.assertRaises(ValueError):
            support_views(frames, cameras[::-1])
        with self.assertRaises(ValueError):
            support_views([frames[0]]*3, [cameras[0]]*3)
        views = support_views(frames, cameras)
        with self.assertRaises(ValueError):
            support_votes([[0., 0., 5.]], [views[0]]*3)

    def test_point_and_segment_representations_have_identical_occupancy_and_curves(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        views = support_views(frames, cameras)
        segments = np.array([[[-.25, -1., 5.], [-.25, 1., 5.]], [[.25, -1., 5.], [.25, 1., 5.]]])
        for voxel in (.02, .04):
            with self.subTest(voxel=voxel):
                points = sample_segments(segments, voxel/2, 2000000)
                config = dict(voxel_size=voxel)
                line_result = CandidateSupportedReadout(EMPTY_POINTS, views)(segments, config)
                point_result = CandidateSupportedReadout(points, views)(EMPTY_SEGMENTS, config)
                repeated_result = CandidateSupportedReadout(np.repeat(points[::-1], 3, axis=0), views)(EMPTY_SEGMENTS, config)
                self.assertEqual(line_result["occupied_voxels"], point_result["occupied_voxels"])
                self.assertEqual(line_result["occupied_voxels"], repeated_result["occupied_voxels"])
                np.testing.assert_array_equal(line_result["segments"], point_result["segments"])
                np.testing.assert_array_equal(line_result["segments"], repeated_result["segments"])
                self.assertEqual(line_result["components"], 2)
                # Resolved parallel lines must not become an unsupported middle axis.
                self.assertTrue(np.all(np.abs(line_result["segments"][..., 0]) > .15))

    def test_a_resolvable_axial_gap_survives_the_full_readout(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        for frame in frames:
            for row in frame["observations"]["rows"]:
                if 46 <= row["y"] <= 54:
                    row.update(status="unknown", candidates=[])
        segments = np.array([[[-.25, -1., 5.], [-.25, 1., 5.]], [[.25, -1., 5.], [.25, 1., 5.]]])
        views = support_views(frames, cameras)
        for voxel in (.02, .04):
            with self.subTest(voxel=voxel):
                result = CandidateSupportedReadout(EMPTY_POINTS, views)(segments, dict(voxel_size=voxel))
                self.assertEqual(result["components"], 4)
                self.assertTrue(all(np.max(segment[:, 1]) < -.15 or np.min(segment[:, 1]) > .15 for segment in result["segments"]))

    def test_known_limitation_coarse_voxels_can_emit_an_unsupported_middle_axis(self):
        # Frozen counterexample, NOT an acceptance criterion: the shared reader's
        # coarse connectivity/PCA can create a line outside the accepted RGB union.
        # A future common-reader fix must deliberately retire/update this test.
        frames, cameras = fixture(offsets=(-2., 2.))
        views = support_views(frames, cameras)
        segments = np.array([[[-.1, -1., 5.], [-.1, 1., 5.]], [[.1, -1., 5.], [.1, 1., 5.]]])
        self.assertTrue(support_mask(segments.reshape(-1, 3), views).all())
        result = CandidateSupportedReadout(EMPTY_POINTS, views)(segments, dict(voxel_size=.2))
        self.assertEqual(result["components"], 1)
        self.assertEqual(result["resolution_state"], "resolved")
        midpoint = result["segments"].mean(axis=1)
        np.testing.assert_allclose(midpoint[:, 0], 0., atol=1e-12)
        self.assertFalse(support_mask(midpoint, views).any())

    def test_support_preserves_inputs_and_does_not_claim_foreground_identity(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        original_frames, original_cameras = copy.deepcopy(frames), copy.deepcopy(cameras)
        points = np.array([[-.25, 0., 5.], [.25, 0., 5.], [0., 0., 5.]])
        original_points = points.copy()
        views = support_views(frames, cameras)
        reader = CandidateSupportedReadout(points, views)
        result = reader(EMPTY_SEGMENTS, dict(voxel_size=.02))
        self.assertEqual(frames, original_frames)
        self.assertEqual(cameras, original_cameras)
        np.testing.assert_array_equal(points, original_points)
        self.assertEqual(result["supported_base_point_count"], 2)
        # A painted line with identical pixels gets identical votes: no identity is inferred.
        np.testing.assert_array_equal(support_votes(points, views), [4, 4, 0])


if __name__ == "__main__":
    unittest.main()
