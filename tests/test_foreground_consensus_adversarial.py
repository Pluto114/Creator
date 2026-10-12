"""Independent normal-only adversarial geometry contracts.

These hand-authored RGB/camera contexts use explicitly synthetic required-view
metadata. They are not foreground annotations or experiment GT/score inputs.
Actual point coordinates and RGB support are retained through the real fitter.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import foreground_consensus_readout as consensus  # noqa: E402
from creator_eval import rgb_chain_support as chain  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402


def geometry(slope=0., intercept=0., ys=None):
    ys = np.linspace(-1., 1., 401) if ys is None else np.asarray(ys, float)
    return np.column_stack((intercept + slope * ys, ys, np.full(len(ys), 3.)))


def mechanism_context(lines=((0., 0.),), missing=None):
    """Exact near-vertical image families x=slope*y+intercept, z=3."""
    frames, cameras = [], []
    k = np.array([[100., 0., 100.], [0., 100., 100.], [0., 0., 1.]])
    for index, camera_x in enumerate((-.9, -.3, .3, .9)):
        image_lines = [[1., -slope, -(100. + (intercept - camera_x) * 100. / 3. - 100. * slope)]
                       for slope, intercept in lines]
        rows = []
        for row in range(60, 141):
            absent = missing is not None and missing(index, row)
            pixels = [-(line[1] * row + line[2]) for line in image_lines]
            rows.append(dict(y=row,
                status="unknown" if absent else ("observed" if len(lines) == 1 else "ambiguous"),
                candidates=[] if absent else [dict(center_x=x, left_edge=dict(x=x-1.),
                    right_edge=dict(x=x+1.), width=2.) for x in pixels]))
        observations = dict(rows=rows)
        pool = dict(candidates=[dict(line=line) for line in image_lines])
        frames.append(dict(view_id=str(index), size_wh=[201, 201],
            guide_xyxy=[[100, 60], [100, 140]], rgb_sha256="a" * 64,
            observations=observations, pool=pool,
            observation_sha256=canonical_hash(observations), pool_sha256=canonical_hash(pool)))
        cameras.append(dict(view_id=str(index), state="validated", K_index=k.tolist(),
            world_to_camera_cv=np.c_[np.eye(3), [-camera_x, 0., 0.]].tolist()))
    context = chain.build_context(frames, cameras)
    claims = [dict(view_id=view, xy=[100., 100.], uncertainty_xy_px=[.5, .5], rgb_sha256="a" * 64)
              for view in ("0", "3")]
    context["association"].update(anchors=claims, anchors_sha256=canonical_hash(claims),
        anchor_view_count=2, anchor_point_count=2,
        synthetic_test_metadata_only=True, anchor_selection_performed=False)
    context["sha256"] = canonical_hash(context["association"])
    return context


class ForegroundConsensusAdversarialContracts(unittest.TestCase):
    def test_parallel_axes_more_than_four_voxels_apart_both_survive(self):
        context = mechanism_context(((0., -.15), (0., .15)))
        points = np.r_[geometry(intercept=-.15), geometry(intercept=.15)]
        graph = consensus.readout(points, .025, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "complete")
        self.assertEqual(graph["components"], 2)
        np.testing.assert_allclose(np.sort(graph["segments"][:, 0, 0]), [-.15, .15], atol=1e-10)
        np.testing.assert_allclose(graph["segments"][:, 0, 0], graph["segments"][:, 1, 0], atol=1e-10)
        self.assertFalse(graph["target_identity_confirmed"])

    def test_crossing_axes_keep_both_arms_outside_resolution_exclusion(self):
        context = mechanism_context(((-.4, 0.), (.4, 0.)))
        points = np.r_[geometry(slope=-.4), geometry(slope=.4)]
        graph = consensus.readout(points, .02, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "complete")
        endpoints = np.asarray(graph["segments"]).reshape(-1, 3)
        for slope in (-.4, .4):
            own = endpoints[np.abs(endpoints[:, 0] - slope * endpoints[:, 1]) < .025]
            self.assertGreater(len(own), 1)
            self.assertLess(float(own[:, 1].min()), -.8)
            self.assertGreater(float(own[:, 1].max()), .8)
        self.assertFalse(graph["target_identity_confirmed"])

    def test_sparse_required_seed_can_restore_actual_same_chain_stroke_extent(self):
        context = mechanism_context(missing=lambda view, row: view == 0 and abs(row - 100) > 3)
        points = geometry()
        graph = consensus.readout(points, .01, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "complete")
        self.assertGreater(graph["components"], 0)
        endpoints = np.asarray(graph["segments"]).reshape(-1, 3)
        self.assertLess(float(endpoints[:, 1].min()), -.9)
        self.assertGreater(float(endpoints[:, 1].max()), .9)
        np.testing.assert_allclose(endpoints[:, 0], 0., atol=1e-10)

    def test_short_tilted_seed_does_not_iteratively_walk_to_far_points(self):
        context = mechanism_context(missing=lambda view, row: view == 0 and abs(row - 100) > 1)
        points = geometry()
        central = np.abs(points[:, 1]) <= .05
        points[central, 0] = .05 * points[central, 1]
        graph = consensus.readout(points, .0025, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "complete")
        self.assertGreater(graph["components"], 0)
        endpoints = np.asarray(graph["segments"]).reshape(-1, 3)
        self.assertLess(float(np.max(np.abs(endpoints[:, 1]))), .3)
        self.assertFalse(graph["target_identity_confirmed"])

    def test_actual_geometric_gap_is_not_filled_by_growth(self):
        context = mechanism_context()
        points = geometry(ys=np.r_[np.linspace(-1., -.3, 141), np.linspace(.3, 1., 141)])
        graph = consensus.readout(points, .02, [0., 0., 0.], context)
        self.assertEqual(graph["components"], 2)
        for segment in graph["segments"]:
            self.assertTrue(segment[:, 1].max() < -.25 or segment[:, 1].min() > .25)

    def test_all_view_unknown_gap_is_not_filled_by_actual_points(self):
        context = mechanism_context(missing=lambda view, row: 95 <= row <= 105)
        graph = consensus.readout(geometry(), .02, [0., 0., 0.], context)
        self.assertEqual(graph["components"], 2)
        for segment in graph["segments"]:
            self.assertTrue(segment[:, 1].max() < 0. or segment[:, 1].min() > 0.)

    def test_nearby_support_aliases_do_not_become_independent_rods(self):
        context = mechanism_context(((0., -.006), (0., .006)))
        points = np.r_[geometry(intercept=-.006), geometry(intercept=.006)]
        graph = consensus.readout(points, .02, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "complete")
        self.assertEqual(graph["components"], 1)
        self.assertFalse(graph["target_identity_confirmed"])
        self.assertLess(float(np.max(np.abs(np.asarray(graph["segments"])[..., 0]))), .007)

    def test_duplicate_and_ordered_actual_points_do_not_change_resolution(self):
        context = mechanism_context(((0., -.15), (0., .15)))
        points = np.r_[geometry(intercept=-.15), geometry(intercept=.15)]
        before = consensus.readout(points, .025, [0., 0., 0.], context)
        after = consensus.readout(np.r_[points[::-1], points, points], .025, [0., 0., 0.], context)
        np.testing.assert_array_equal(before["segments"], after["segments"])
        self.assertEqual(before["components"], after["components"])


if __name__ == "__main__":
    unittest.main()
