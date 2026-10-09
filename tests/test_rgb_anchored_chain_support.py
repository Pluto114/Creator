"""Explicit anchor claims and support filtering; no truth-based selection."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_anchored_chain_support as anchored  # noqa: E402
from creator_eval import rgb_chain_support as chain  # noqa: E402
from creator_eval.rgb_anchored_chain_axis_readout import AnchoredChainAxisReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402


def anchors(frames, candidate=0):
    return [dict(view_id=f["view_id"], rgb_sha256=f["rgb_sha256"],
        xy=[-f["pool"]["candidates"][candidate]["line"][2], 100.], uncertainty_xy_px=[.5, .5])
        for f in (frames[0], frames[3])]


class AnchoredChainContracts(unittest.TestCase):
    def test_two_explicit_views_select_one_physical_branch(self):
        frames, cameras = fixture((-.3, .3))
        before = chain.build_context(frames, cameras)
        after = anchored.condition_context(before, frames, cameras, anchors(frames))
        points = [[-.3, 0., 3.], [.3, 0., 3.]]
        np.testing.assert_array_equal(chain.chain_support_mask(points, before), [True, True])
        np.testing.assert_array_equal(chain.chain_support_mask(points, after), [True, False])
        self.assertFalse(after["association"]["target_identity_confirmed"])
        self.assertEqual(after["association"]["positive_support_state"], "supported")

    def test_missing_anchors_are_unknown_without_positive_support(self):
        frames, cameras = fixture()
        for values in ([], anchors(frames)[:1]):
            context = anchored.build_context(frames, cameras, values)
            self.assertEqual(context["association"]["positive_support_state"], "unresolved")
            self.assertFalse(chain.chain_support_mask([[0., 0., 3.]], context)[0])

    def test_hash_and_camera_center_validation_reused(self):
        frames, cameras = fixture()
        selected = anchors(frames)
        selected[0]["rgb_sha256"] = "f"*64
        with self.assertRaises(ValueError):
            anchored.build_context(frames, cameras, selected)
        cameras[3]["world_to_camera_cv"] = copy.deepcopy(cameras[0]["world_to_camera_cv"])
        with self.assertRaises(ValueError):
            anchored.build_context(frames, cameras, anchors(frames))

    def test_context_is_immutable_and_bound_to_full_normal_inputs(self):
        frames, cameras = fixture()
        before = chain.build_context(frames, cameras)
        digest = canonical_hash(before["association"])
        after = anchored.condition_context(before, frames, cameras, anchors(frames))
        self.assertEqual(canonical_hash(before["association"]), digest)
        self.assertIs(before["row_members"], after["row_members"])
        changed = copy.deepcopy(frames)
        changed[0]["rgb_sha256"] = "b"*64
        with self.assertRaises(ValueError):
            anchored.condition_context(before, changed, cameras, anchors(changed))

    def test_uncertainty_cannot_snap_to_measured_band(self):
        frames, cameras = fixture()
        selected = anchors(frames)
        selected[0]["uncertainty_xy_px"] = [1.5, .5]
        context = anchored.build_context(frames, cameras, selected)
        self.assertEqual(context["association"]["anchor_audit"][0]["raw_measurement_state"], "unresolved")
        self.assertFalse(chain.chain_support_mask([[0., 0., 3.]], context)[0])
        self.assertEqual(context["association"]["chain_state_counts"], {"unresolved": 1})

    def test_vertical_box_requires_every_observed_row(self):
        frames, cameras = fixture()
        frames[0]["observations"]["rows"][41].update(status="unknown", candidates=[])
        frames[0]["observation_sha256"] = canonical_hash(frames[0]["observations"])
        selected = anchors(frames)
        selected[0]["uncertainty_xy_px"] = [.5, 1.]
        context = anchored.build_context(frames, cameras, selected)
        self.assertEqual(context["association"]["anchor_audit"][0]["covered_rows"], [99, 100, 101])
        self.assertEqual(context["association"]["chain_state_counts"], {"unresolved": 1})

    def test_incomplete_search_falls_back_explicitly_not_as_identity(self):
        frames, cameras = fixture()
        context = anchored.build_context(frames, cameras, anchors(frames), {**chain.DEFAULTS, "maximum_pairs": 1})
        self.assertTrue(context["association"]["fallback_to_raw_union"])
        self.assertFalse(context["association"]["positive_chain_selection_complete"])
        self.assertFalse(context["association"]["target_identity_confirmed"])

    def test_conflicting_view_claims_do_not_become_a_new_triangulated_rod(self):
        frames, cameras = fixture((-.3, .3))
        selected = anchors(frames)
        selected[1] = anchors(frames, 1)[1]
        context = anchored.build_context(frames, cameras, selected)
        self.assertEqual(context["association"]["chain_count"], 0)
        self.assertFalse(context["association"]["geometry_changed"])
        self.assertFalse(context["association"]["anchors_used_as_3d_correspondences"])

    def test_anchor_order_does_not_change_mask(self):
        frames, cameras = fixture((-.3, .3))
        a = anchored.build_context(frames, cameras, anchors(frames))
        b = anchored.build_context(frames, cameras, list(reversed(anchors(frames))))
        self.assertEqual(a["association"]["assignments"], b["association"]["assignments"])

    def test_actual_geometry_not_projected_to_anchor_hypothesis(self):
        frames, cameras = fixture()
        context = anchored.build_context(frames, cameras, anchors(frames))
        points = np.c_[np.full(101, .015), np.linspace(-1., 1., 101), np.full(101, 3.)]
        result = AnchoredChainAxisReadout(points, context)(np.empty((0, 2, 3)), dict(voxel_size=.02, origin=[0., 0., 0.]))
        self.assertEqual(len(result["segments"]), 1)
        np.testing.assert_allclose(result["segments"][..., 0], .015, atol=1e-12)
        self.assertEqual(result["support_domain_kind"], "anchor_conditioned_positive_only_not_all_possible")
        self.assertIn("Positive support conditional", result["measurement_scope"])

    def test_multiple_supported_assignments_remain_unconfirmed(self):
        frames, cameras = fixture((0., 0.))
        context = chain.build_context(frames, cameras)
        context["association"]["assignments"] = [[(0,)]*4, [(1,)]*4]
        context["association"]["chain_count"] = 2
        context["association"]["assignment_sha256"] = canonical_hash(context["association"]["assignments"])
        context["sha256"] = canonical_hash(context["association"])
        after = anchored.condition_context(context, frames, cameras, anchors(frames))
        self.assertEqual(after["association"]["chain_count"], 2)
        self.assertFalse(after["association"]["target_identity_confirmed"])
        self.assertEqual(after["association"]["identity_state"], "unresolved")

    def test_empty_and_disjoint_rows_keep_three_states_distinct(self):
        self.assertEqual(anchored.classify_anchor_rows([], 0, 1), "unresolved")
        self.assertEqual(anchored.classify_anchor_rows([[], [(2, 3)]], 0, 1), "unresolved")
        self.assertEqual(anchored.classify_anchor_rows([[(-1, 2)], [(2, 3)]], 0, 1), "unresolved")
        self.assertEqual(anchored.classify_anchor_rows([[(1, 3)]], 0, 1), "unresolved")
        with self.assertRaises(ValueError):
            anchored.classify_anchor_rows([[(2, 1)]], 0, 1)


if __name__ == "__main__":
    unittest.main()
