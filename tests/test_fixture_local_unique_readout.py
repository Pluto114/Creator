"""Pre-freeze point-local ambiguity controls, independent of fixture scoring."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "tests"))
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_candidate_readout import (  # noqa: E402
    CandidateSupportedReadout,
)
from creator_eval.rgb_candidate_readout import support_mask as union_mask  # noqa: E402
from creator_eval.rgb_candidate_readout import support_views as union_views  # noqa: E402
from creator_eval.rgb_local_unique_readout import (  # noqa: E402
    LocalUniqueReadout,
    support_mask,
    support_views,
    support_votes,
)
from creator_eval.rgb_supported_readout import support_mask as row_unique_mask  # noqa: E402
from creator_eval.rgb_supported_readout import support_views as row_unique_views  # noqa: E402
from test_fixture_candidate_readout import (  # noqa: E402
    EMPTY_POINTS,
    EMPTY_SEGMENTS,
    fixture,
)


class FixtureLocalUniqueReadoutTests(unittest.TestCase):
    def test_disjoint_candidates_each_support_their_own_projection_not_the_middle(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        points = [[-.25, 0., 5.], [.25, 0., 5.], [0., 0., 5.]]
        np.testing.assert_array_equal(support_votes(points, support_views(frames, cameras)), [4, 4, 0])
        np.testing.assert_array_equal(support_mask(points, support_views(frames, cameras)), [1, 1, 0])
        # This is intentionally different from rejecting an entire ambiguous row.
        self.assertFalse(row_unique_mask(points, row_unique_views(frames, cameras)).any())

    def test_overlap_abstains_but_the_unique_sides_vote_including_padding_boundaries(self):
        frames, cameras = fixture(offsets=(-1., 1.))
        # Raw pairs [-1.5,-.5], [.5,1.5]; padding gives overlap [-.5,.5].
        points = [[x, 0., 5.] for x in [-.1, -.026, -.025, 0., .025, .026, .1]]
        np.testing.assert_array_equal(support_votes(points, support_views(frames, cameras)), [4, 4, 0, 0, 0, 4, 4])
        self.assertTrue(union_mask(points, union_views(frames, cameras)).all())
        # Even an exactly shared padded endpoint belongs to both closed intervals.
        frames, cameras = fixture(offsets=(-1.5, 1.5))
        np.testing.assert_array_equal(
            support_votes([[-.001, 0., 5.], [0., 0., 5.], [.001, 0., 5.]], support_views(frames, cameras)),
            [4, 0, 4],
        )

    def test_exact_duplicates_do_not_create_ambiguity_but_distinct_overlapping_pairs_do(self):
        frames, cameras = fixture(offsets=(0., 0., 0.))
        np.testing.assert_array_equal(support_votes([[0., 0., 5.]], support_views(frames, cameras)), [4])
        frames, cameras = fixture(offsets=(0., .01))
        np.testing.assert_array_equal(support_votes([[0., 0., 5.]], support_views(frames, cameras)), [0])

    def test_three_distinct_views_are_still_required(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        for frame in frames[3:]:
            for row in frame["observations"]["rows"]:
                row.update(status="unknown", candidates=[])
        point = [[-.25, 0., 5.]]
        np.testing.assert_array_equal(support_votes(point, support_views(frames, cameras)), [3])
        self.assertTrue(support_mask(point, support_views(frames, cameras))[0])
        for row in frames[2]["observations"]["rows"]:
            row.update(status="absent", candidates=[])
        views = support_views(frames, cameras)
        np.testing.assert_array_equal(support_votes(point, views), [2])
        self.assertFalse(support_mask(point, views)[0])
        with self.assertRaises(ValueError):
            support_votes(point, [views[0]]*3)
        with self.assertRaises(ValueError):
            support_views([frames[0]]*3, [cameras[0]]*3)

    def test_unknown_absent_and_unrecorded_rows_remain_holes_without_vertical_dilation(self):
        for state in ("unknown", "absent", "unrecorded"):
            with self.subTest(state=state):
                frames, cameras = fixture(offsets=(-5., 5.))
                for frame in frames:
                    rows = frame["observations"]["rows"]
                    if state == "unrecorded":
                        rows.pop(40)
                    else:
                        rows[40].update(status=state, candidates=[])
                points = [[-.25, y, 5.] for y in [-.026, -.024, .024, .026]]
                np.testing.assert_array_equal(support_mask(points, support_views(frames, cameras)), [1, 0, 0, 1])

    def test_all_unique_fixture_matches_union_readout_exactly(self):
        frames, cameras = fixture()
        segments = np.array([[[0., -1., 5.], [0., 1., 5.]]])
        points = np.array([[0., .03, 5.], [.5, 0., 5.], [0., 0., 10.]])
        for voxel in (.02, .04):
            with self.subTest(voxel=voxel):
                config = dict(voxel_size=voxel)
                local = LocalUniqueReadout(points, support_views(frames, cameras))(segments, config)
                union = CandidateSupportedReadout(points, union_views(frames, cameras))(segments, config)
                for key in ("occupied_voxels", "supported_base_point_count", "supported_curve_sample_count", "components"):
                    self.assertEqual(local[key], union[key])
                np.testing.assert_array_equal(local["segments"], union["segments"])

    def test_point_segment_order_and_multiplicity_invariance_preserves_a_resolvable_gap(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        for frame in frames:
            for row in frame["observations"]["rows"]:
                if 46 <= row["y"] <= 54:
                    row.update(status="unknown", candidates=[])
        segments = np.array([[[-.25, -1., 5.], [-.25, 1., 5.]], [[.25, -1., 5.], [.25, 1., 5.]]])
        views = support_views(frames, cameras)
        for voxel in (.02, .04):
            with self.subTest(voxel=voxel):
                config = dict(voxel_size=voxel)
                points = sample_segments(segments, voxel/2, 2000000)
                line = LocalUniqueReadout(EMPTY_POINTS, views)(segments, config)
                cloud = LocalUniqueReadout(points, views)(EMPTY_SEGMENTS, config)
                repeated = LocalUniqueReadout(np.repeat(points[::-1], 3, axis=0), views)(EMPTY_SEGMENTS, config)
                self.assertEqual(line["components"], 4)
                self.assertEqual(line["occupied_voxels"], cloud["occupied_voxels"])
                self.assertEqual(line["occupied_voxels"], repeated["occupied_voxels"])
                np.testing.assert_array_equal(line["segments"], cloud["segments"])
                np.testing.assert_array_equal(line["segments"], repeated["segments"])
                self.assertTrue(all(np.max(segment[:, 1]) < -.15 or np.min(segment[:, 1]) > .15 for segment in line["segments"]))

    def test_candidate_row_and_synchronized_view_order_do_not_change_votes(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        points = np.array([[-.25, 0., 5.], [.25, 0., 5.], [0., 0., 5.]])
        expected = support_votes(points, support_views(frames, cameras))
        for frame in frames:
            frame["observations"]["rows"].reverse()
            for row in frame["observations"]["rows"]:
                row["candidates"] = row["candidates"][::-1] * 3
        np.testing.assert_array_equal(support_votes(points, support_views(frames[::-1], cameras[::-1])), expected)
        np.testing.assert_array_equal(support_votes(points[::-1], support_views(frames, cameras)), expected[::-1])

    def test_row_unique_local_unique_and_union_form_nested_support_sets(self):
        frames, cameras = fixture(offsets=(-5., -.5, .5, 5.))
        for frame in frames:
            for row in frame["observations"]["rows"]:
                if row["y"] < 48:
                    row.update(status="observed", candidates=row["candidates"][:1])
        x, y, z = np.meshgrid(np.linspace(-.5, .5, 41), [-.3, 0., .3], [-5., 4., 5., 6., 10.], indexing="ij")
        points = np.c_[x.ravel(), y.ravel(), z.ravel()]
        strict = row_unique_mask(points, row_unique_views(frames, cameras))
        local = support_mask(points, support_views(frames, cameras))
        union = union_mask(points, union_views(frames, cameras))
        self.assertTrue(strict.any())
        self.assertTrue((local & ~strict).any())
        self.assertTrue((union & ~local).any())
        self.assertFalse((strict & ~local).any())
        self.assertFalse((local & ~union).any())

    def test_known_limitation_unique_pixels_can_correspond_to_different_lines_between_views(self):
        # Counterexample, not qualification: three physical analytic lines at z=5
        # can uniquely support a ghost at z=10 through different line identities.
        frames, cameras = fixture(offsets=(-5., 0., 5.))
        frames, cameras = frames[:3], cameras[:3]
        points = [[-.25, 0., 5.], [0., 0., 5.], [.25, 0., 5.], [0., 0., 10.]]
        views = support_views(frames, cameras)
        np.testing.assert_array_equal(support_votes(points, views), [3, 3, 3, 3])
        self.assertTrue(support_mask(points, views).all())
        self.assertFalse(row_unique_mask(points, row_unique_views(frames, cameras)).any())

    def test_raw_evidence_cameras_points_and_segments_are_not_mutated(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        original_frames, original_cameras = copy.deepcopy(frames), copy.deepcopy(cameras)
        points = np.array([[-.25, 0., 5.], [.25, 0., 5.], [0., 0., 5.]])
        segments = np.array([[[-.25, -1., 5.], [-.25, 1., 5.]]])
        original_points, original_segments = points.copy(), segments.copy()
        result = LocalUniqueReadout(points, support_views(frames, cameras))(segments, dict(voxel_size=.02))
        self.assertEqual(result["supported_base_point_count"], 2)
        self.assertEqual(frames, original_frames)
        self.assertEqual(cameras, original_cameras)
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(segments, original_segments)


if __name__ == "__main__":
    unittest.main()
