"""Sparse stroke conjunction, without triangulating clicks or filling gaps."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_chain_support as chain  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_anchored_chain_support import anchors  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402


def marks(frames):
    result = anchors(frames)
    extra = copy.deepcopy(result)
    for anchor in extra:
        anchor["xy"][1] = 120.
    return result+extra


class StrokeContracts(unittest.TestCase):
    def test_same_candidate_must_support_every_mark_in_view(self):
        frames, cameras = fixture((-.3, .3))
        selected = marks(frames)
        selected[-1]["xy"][0] = anchors(frames, 1)[1]["xy"][0]
        context = stroke.build_context(frames, cameras, selected)
        self.assertEqual(context["association"]["chain_count"], 0)
        self.assertEqual(context["association"]["anchor_view_count"], 2)
        self.assertEqual(context["association"]["anchor_point_count"], 4)

    def test_positive_marks_keep_selected_branch_without_changing_arrays(self):
        frames, cameras = fixture((-.3, .3))
        old = chain.build_context(frames, cameras)
        old_sha = old["sha256"]
        new = stroke.condition_stroke_context(old, frames, cameras, marks(frames))
        self.assertEqual(old["sha256"], old_sha)
        self.assertIs(new["row_members"], old["row_members"])
        np.testing.assert_array_equal(chain.chain_support_mask([[-.3, 0., 3.], [.3, 0., 3.]], new), [True, False])
        self.assertFalse(new["association"]["stroke_interpolation_performed"])
        self.assertFalse(new["association"]["target_identity_confirmed"])

    def test_missing_middle_row_not_interpolated_by_stroke(self):
        frames, cameras = fixture()
        for frame in frames:
            frame["observations"]["rows"][50].update(status="unknown", candidates=[])
            frame["observation_sha256"] = canonical_hash(frame["observations"])
        context = stroke.build_context(frames, cameras, marks(frames))
        # y_world=.3 projects to row 110, strictly between marks 100 and 120.
        self.assertFalse(chain.chain_support_mask([[0., .3, 3.]], context)[0])
        self.assertTrue(chain.chain_support_mask([[0., 0., 3.]], context)[0])

    def test_stroke_order_and_duplicate_marks_preserve_mask(self):
        frames, cameras = fixture()
        selected = marks(frames)
        a = stroke.build_context(frames, cameras, selected)
        b = stroke.build_context(frames, cameras, list(reversed(selected))+selected[:1])
        self.assertEqual(a["association"]["assignments"], b["association"]["assignments"])

    def test_unknown_mark_not_a_negative_claim(self):
        frames, cameras = fixture()
        selected = marks(frames)
        selected[-1]["uncertainty_xy_px"] = [2., .5]
        context = stroke.build_context(frames, cameras, selected)
        self.assertEqual(context["association"]["chain_count"], 0)
        self.assertEqual(context["association"]["conditioning_layers"][-1]["chain_state_counts"], {"unresolved": 1})

    def test_independent_views_and_distinct_rows_required(self):
        frames, cameras = fixture()
        for selected in (anchors(frames), marks(frames)[:1], anchors(frames)*2):
            with self.assertRaises(ValueError):
                stroke.build_context(frames, cameras, selected)

    def test_layer_source_and_final_assignment_hashes_are_bound(self):
        frames, cameras = fixture()
        context = stroke.build_context(frames, cameras, marks(frames))
        association = context["association"]
        self.assertEqual(association["assignment_sha256"], canonical_hash(association["assignments"]))
        self.assertEqual(context["sha256"], canonical_hash(association))
        self.assertEqual(len(association["conditioning_layers"]), 2)


if __name__ == "__main__":
    unittest.main()
