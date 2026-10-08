"""Exact pair geometry, exhaustive member matching and shared support contracts."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_chain_support as chain  # noqa: E402
from creator_eval.rgb_candidate_readout import support_votes  # noqa: E402
from creator_eval.rgb_chain_axis_readout import ChainAxisReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402


def fixture(xs=(0.,)):
    frames, cameras = [], []
    k = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
    for index, camera_x in enumerate((-.9, -.3, .3, .9)):
        pixels = [100+100*(x-camera_x)/3 for x in xs]
        rows = [dict(y=y, status="observed" if len(xs) == 1 else "ambiguous",
            candidates=[dict(center_x=x, left_edge=dict(x=x-1), right_edge=dict(x=x+1), width=2.)
                        for x in pixels]) for y in range(60, 141)]
        observations = dict(rows=rows)
        pool = dict(candidates=[dict(line=[1., 0., -x]) for x in pixels])
        frames.append(dict(view_id=str(index), size_wh=[201, 201], guide_xyxy=[[100, 60], [100, 140]],
            rgb_sha256="a"*64, observations=observations, pool=pool,
            observation_sha256=canonical_hash(observations), pool_sha256=canonical_hash(pool)))
        cameras.append(dict(view_id=str(index), state="validated", K_index=k.tolist(),
            world_to_camera_cv=np.c_[np.eye(3), [-camera_x, 0., 0.]].tolist()))
    return frames, cameras


class ChainSupportContracts(unittest.TestCase):
    def test_packed_votes_equal_naive_for_ragged_sets_and_bit_boundaries(self):
        rng = np.random.default_rng(801)
        for count in (0, 1, 7, 8, 9, 65, 4099):
            arrays = [rng.random((count, width)) < .2 for width in (0, 3, 5, 2, 8)]
            assignments = [[tuple(np.flatnonzero(rng.random(a.shape[1]) < .4)) for a in arrays] for _ in range(25)]
            for minimum in range(1, 6):
                expected = np.zeros(count, bool)
                for assignment in assignments:
                    votes = np.zeros(count, int)
                    for a, ids in zip(arrays, assignment):
                        if ids:
                            votes += a[:, ids].any(axis=1)
                    expected |= votes >= minimum
                np.testing.assert_array_equal(chain.consistent_chain_mask(arrays, assignments, minimum), expected)

    def test_exact_pair_proposals_recover_visible_line(self):
        frames, cameras = fixture()
        context = chain.build_context(frames, cameras)
        association = context["association"]
        self.assertTrue(association["search_complete"])
        self.assertEqual(association["attempted_pairs"], 6)
        self.assertEqual(association["chain_count"], 1)
        self.assertFalse(association["target_identity_confirmed"])
        points = [[0., 0., 3.], [0., .5, 3.], [1., 0., 3.], [0., 0., -3.]]
        np.testing.assert_array_equal(chain.chain_support_mask(points, context), [True, True, False, False])

    def test_world_translation_preserves_mask_and_assignments(self):
        frames, cameras = fixture((-.3, .3))
        before = chain.build_context(frames, cameras)
        shift = np.array([17., -11., 23.])
        moved = copy.deepcopy(cameras)
        for camera in moved:
            e = np.asarray(camera["world_to_camera_cv"])
            e[:, 3] -= e[:, :3] @ shift
            camera["world_to_camera_cv"] = e.tolist()
        after = chain.build_context(frames, moved)
        points = np.array([[-.3, .2, 3.], [.3, -.5, 3.], [0., 0., 3.], [0., 0., -3.]])
        np.testing.assert_array_equal(chain.chain_support_mask(points, before), chain.chain_support_mask(points+shift, after))
        self.assertEqual(before["association"]["assignments"], after["association"]["assignments"])

    def test_pair_budget_falls_back_for_every_point(self):
        frames, cameras = fixture()
        context = chain.build_context(frames, cameras, {**chain.DEFAULTS, "maximum_pairs": 5})
        self.assertFalse(context["association"]["search_complete"])
        self.assertEqual(context["association"]["reason"], "pair_budget_exceeded")
        points = np.array([[0., 0., 3.], [0., 0., 4.], [.2, 0., 3.]])
        np.testing.assert_array_equal(chain.chain_support_mask(points, context), support_votes(points, context["views"]) >= 3)

    def test_chain_budget_never_uses_partial_exclusion(self):
        frames, cameras = fixture((-.3, .3))
        context = chain.build_context(frames, cameras, {**chain.DEFAULTS, "maximum_chains": 1})
        self.assertFalse(context["association"]["search_complete"])
        self.assertEqual(context["association"]["reason"], "chain_budget_exceeded")
        self.assertEqual(context["association"]["assignments"], [])
        np.testing.assert_array_equal(chain.chain_support_mask([[-.3, 0., 3.], [.3, 0., 3.]], context), [True, True])

    def test_two_branches_and_pool_order_preserved(self):
        frames, cameras = fixture((-.3, .3))
        first = chain.build_context(frames, cameras)
        for frame in frames:
            frame["pool"]["candidates"].reverse()
            frame["pool_sha256"] = canonical_hash(frame["pool"])
        second = chain.build_context(frames, cameras)
        points = [[-.3, 0., 3.], [.3, 0., 3.]]
        np.testing.assert_array_equal(chain.chain_support_mask(points, first), [True, True])
        np.testing.assert_array_equal(chain.chain_support_mask(points, second), [True, True])

    def test_guide_horizontal_position_is_not_identity(self):
        frames, cameras = fixture()
        first = chain.build_context(frames, cameras)
        for frame in frames:
            frame["guide_xyxy"] = [[15., 60], [195., 140]]
        second = chain.build_context(frames, cameras)
        self.assertEqual(first["association"]["assignments"], second["association"]["assignments"])

    def test_nested_measured_pairs_not_discarded_by_nearest_match(self):
        frames, cameras = fixture((0., 0.))
        for frame in frames:
            for row in frame["observations"]["rows"]:
                pair = row["candidates"][1]
                pair["left_edge"]["x"] -= 3
                pair["right_edge"]["x"] += 3
            frame["observation_sha256"] = canonical_hash(frame["observations"])
        context = chain.build_context(frames, cameras)
        for members in context["row_members"]:
            self.assertTrue(members[100, :, :].all())
            self.assertFalse(members[10].any())

    def test_midpoint_prefilter_exact_against_full_search(self):
        rng = np.random.default_rng(513)
        ys = np.linspace(20., 459., 9)
        projected = np.c_[np.ones(83), rng.normal(0, .01, 83), -rng.uniform(280., 340., 83)]
        candidates = np.r_[projected.copy(), np.c_[np.ones(129), rng.normal(0, .01, 129), -rng.uniform(280., 340., 129)]]
        candidates = np.r_[candidates, projected[:3]+[0, 0, 1.5], projected[:3]+[0, 0, -1.5]]
        actual = chain._matching_candidates(projected, candidates, ys, 1.5)
        for index, seed in enumerate(projected):
            px = -(seed[1]*ys+seed[2])/seed[0]
            cx = -(candidates[:, 1, None]*ys+candidates[:, 2, None])/candidates[:, 0, None]
            expected = np.flatnonzero(np.median(np.abs(px-cx), axis=1) <= 1.5)
            self.assertEqual(tuple(expected), actual[index])

    def test_empty_pool_is_complete_zero_support_not_identity(self):
        frames, cameras = fixture()
        for frame in frames:
            frame["pool"]["candidates"] = []
            frame["pool_sha256"] = canonical_hash(frame["pool"])
        context = chain.build_context(frames, cameras)
        self.assertTrue(context["association"]["search_complete"])
        self.assertFalse(chain.chain_support_mask([[0., 0., 3.]], context)[0])

    def test_coincident_cameras_do_not_prove_a_chain(self):
        frames, cameras = fixture()
        for camera in cameras:
            camera["world_to_camera_cv"] = copy.deepcopy(cameras[0]["world_to_camera_cv"])
        # Equal cameras and equal image lines imply identical planes, not depth.
        for frame in frames:
            frame["pool"] = copy.deepcopy(frames[0]["pool"])
            frame["pool_sha256"] = canonical_hash(frame["pool"])
        context = chain.build_context(frames, cameras)
        self.assertEqual(context["association"]["nondegenerate_pairs"], 0)
        self.assertEqual(context["association"]["chain_count"], 0)

    def test_exact_raw_hashes_required(self):
        frames, cameras = fixture()
        frames[0]["pool"]["candidates"][0]["line"][2] += 1
        with self.assertRaises(ValueError):
            chain.build_context(frames, cameras)

    def test_batching_does_not_change_point_decisions(self):
        frames, cameras = fixture((-.3, .3))
        context = chain.build_context(frames, cameras)
        points = np.tile([[-.3, 0., 3.], [.3, 0., 3.], [0., 0., 3.]], (11, 1))
        expected = chain.chain_support_mask(points, context)
        with patch.object(chain, "POINT_BATCH_SIZE", 2):
            actual = chain.chain_support_mask(points, context)
        np.testing.assert_array_equal(actual, expected)

    def test_adapter_fits_real_points_not_pair_line(self):
        frames, cameras = fixture()
        context = chain.build_context(frames, cameras)
        points = np.c_[np.full(101, .015), np.linspace(-1., 1., 101), np.full(101, 3.)]
        adapter = ChainAxisReadout(points, context)
        result = adapter(np.empty((0, 2, 3)), dict(voxel_size=.02, origin=[0., 0., 0.]))
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 1)
        np.testing.assert_allclose(result["segments"][..., 0], .015, atol=1e-12)
        np.testing.assert_allclose(result["segments"][..., 2], 3., atol=1e-12)
        self.assertTrue(result["base_chain_mask_subset_raw"])
        self.assertFalse(result["target_identity_confirmed"])
        self.assertEqual(result["chain_context_sha256"], canonical_hash(context["association"]))

    def test_adapter_base_and_curve_have_same_membership(self):
        frames, cameras = fixture()
        context = chain.build_context(frames, cameras)
        values = np.array([[0., 0., 3.], [.3, 0., 3.]])
        adapter = ChainAxisReadout(values, context)
        segments = np.stack((values, values+[0., .005, 0.]), axis=1)
        result = adapter(segments, dict(voxel_size=.02, origin=[0., 0., 0.]))
        self.assertEqual(result["supported_base_point_count"], 1)
        self.assertEqual(result["supported_curve_sample_count"], result["full_curve_sample_count"]//2)

    def test_invalid_memberships_and_duplicate_view_inputs_rejected(self):
        with self.assertRaises(ValueError):
            chain.consistent_chain_mask([np.ones((1, 1))]*3, [])
        with self.assertRaises(ValueError):
            chain.consistent_chain_mask([np.ones((1, 1), bool)]*3, [[[1], [0], [0]]])
        frames, cameras = fixture()
        frames[1]["view_id"] = frames[0]["view_id"]
        with self.assertRaises(ValueError):
            chain.build_context(frames, cameras)


if __name__ == "__main__":
    unittest.main()
