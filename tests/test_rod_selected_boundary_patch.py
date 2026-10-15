"""Seven contracts for native-selected RGB outer bounds, without reader changes."""

import copy
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rod_selected_boundary_patch as patch  # noqa: E402
from creator_eval.anchored_boundary_readout import measured_boundary_context  # noqa: E402
from creator_eval.consensus_boundary_readout import boundary_support_mask  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_anchored_boundary_readout import line  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def native_inputs(*, xs=(0.,), supporting=(0, 1, 2, 3), select=None):
    """Attach exact original pool matches and the finite adapter's recorded copy."""
    frames, cameras = fixture(xs)
    selections, matches = [], []
    for view_index, frame in enumerate(frames):
        chosen_rows = []
        for row_index, row in enumerate(frame["observations"]["rows"]):
            raw_choice = 0 if select is None else select(view_index, row_index, row)
            if raw_choice is not None:
                chosen_rows.append([row_index, raw_choice, 0.])
        for candidate_index, candidate in enumerate(frame["pool"]["candidates"]):
            candidate["row_matches"] = chosen_rows if candidate_index == 0 else []
        frame["pool_sha256"] = canonical_hash(frame["pool"])
        active = view_index in supporting
        matches.append(dict(candidate_index=0 if active else None))
        selections.append(dict(view_id=frame["view_id"], candidate_index=0 if active else None,
            row_matches=[[item[0], item[1]] for item in chosen_rows] if active else []))
    baseline = dict(method="baseline", state="accepted",
        proposal=dict(state="accepted", selected=dict(supporting_views=list(supporting), matches=matches)),
        finite=dict(state="accepted", candidate_selection=selections))
    return frames, cameras, baseline


def tail_inputs():
    return native_inputs(select=lambda view, index, row: None if view == 0 and row["y"] >= 122 else 0)


class SelectedBoundaryContracts(unittest.TestCase):
    def test_exact_raw_row_and_edge_pair_choices_not_nearest_alternative(self):
        frames, cameras, baseline = native_inputs(xs=(0., 0.),
            select=lambda view, index, row: index % 2)
        context = patch.build_selected_context(frames, cameras, baseline)
        self.assertEqual(context["association"]["assignments"], [[(0,), (0,), (0,), (0,)]])
        for frame, mask in zip(frames, context["row_members"]):
            self.assertEqual(mask.shape, (201, 2, 1))
            self.assertEqual(int(mask.sum()), 81)
            for index, row in enumerate(frame["observations"]["rows"]):
                self.assertTrue(mask[row["y"], index % 2, 0])
                self.assertFalse(mask[row["y"], 1 - index % 2, 0])

    def test_unselected_foreign_pair_cannot_lend_a_boundary_vote(self):
        context = patch.build_selected_context(*native_inputs(xs=(0., .09)))
        strict = measured_boundary_context(context)
        np.testing.assert_array_equal(boundary_support_mask([[0., 0., 3.], [.09, 0., 3.]], strict),
                                      [True, False])
        for view in context["views"]:
            self.assertTrue(np.isnan(view["left"][:, 1]).all())
            self.assertTrue(np.isnan(view["right"][:, 1]).all())

    def test_unmatched_measured_rows_stay_unknown_without_interpolation(self):
        inputs = native_inputs(select=lambda view, index, row: None if row["y"] == 110 else 0)
        context = patch.build_selected_context(*inputs)
        for mask in context["row_members"]:
            self.assertFalse(mask[110].any())
        strict = measured_boundary_context(context)
        np.testing.assert_array_equal(boundary_support_mask([[0., .27, 3.], [0., .3, 3.],
                                                            [0., .33, 3.]], strict),
                                      [True, False, True])
        # A missing internal row cannot split or fill existing native geometry.
        source = np.concatenate((line(-1., .1), line(.2, 1.)))
        result = patch.refine_boundary(source, measured_context(), context)
        np.testing.assert_array_equal(result["segments"], source)

    def test_three_distinct_views_reject_and_duplicates_cannot_make_four(self):
        context = patch.build_selected_context(*native_inputs(supporting=(0, 1, 2)))
        self.assertEqual(context["association"]["assignments"][0][-1], ())
        self.assertFalse(context["row_members"][-1].any())
        result = patch.refine_boundary(line(), measured_context(), context)
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(result["segments"].shape, (0, 2, 3))
        self.assertEqual(result["rejection_reasons"], ["no_four_view_selected_native_boundary_support"])
        frames, cameras, baseline = native_inputs()
        frames[-1]["view_id"] = cameras[-1]["view_id"] = "0"
        baseline["finite"]["candidate_selection"][-1]["view_id"] = "0"
        with self.assertRaisesRegex(ValueError, "Duplicate views"):
            patch.build_selected_context(frames, cameras, baseline)

    def test_outer_intersection_preserves_axis_internal_gap_and_no_extension(self):
        source = np.concatenate((line(-1., -.2, .015), line(.2, 1., .015)))
        context = patch.build_selected_context(*tail_inputs())
        result = patch.refine_boundary(source, measured_context(), context)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["segments"].shape, source.shape)
        np.testing.assert_array_equal(result["segments"][0], source[0])
        np.testing.assert_array_equal(result["segments"][1, 0], source[1, 0])
        np.testing.assert_array_equal(result["segments"][..., (0, 2)], source[..., (0, 2)])
        self.assertAlmostEqual(result["segments"][1, 1, 1], .645, places=8)
        self.assertGreater(result["selected_boundary_removed_length_m"], .35)
        self.assertEqual(result["previous_boundary_removed_length_m"], 0.)
        self.assertTrue(np.all(result["segments"][..., 1] >= source[..., 1].min()))
        self.assertTrue(np.all(result["segments"][..., 1] <= source[..., 1].max()))
        # Old admission padding is not physical selected edge support.
        rejected = patch.refine_boundary(line(x=.045), measured_context(), context)
        self.assertEqual(rejected["state"], "rejected")

    def test_native_inputs_shared_context_and_selected_context_remain_unchanged(self):
        inputs, shared, source = tail_inputs(), measured_context(), line(x=.015)
        before_inputs, before_shared, before_source = pickle.dumps(inputs), pickle.dumps(shared), source.copy()
        context = patch.build_selected_context(*inputs)
        before_selected = pickle.dumps(context)
        patch.refine_boundary(source, shared, context)
        self.assertEqual(pickle.dumps(inputs), before_inputs)
        self.assertEqual(pickle.dumps(shared), before_shared)
        self.assertEqual(pickle.dumps(context), before_selected)
        np.testing.assert_array_equal(source, before_source)

    def test_metadata_binds_one_native_selection_and_rejects_tampered_finite_copy(self):
        inputs, shared, source = tail_inputs(), measured_context(), line(x=.015)
        context = patch.build_selected_context(*inputs)
        result = patch.refine_boundary(source, shared, context)
        self.assertEqual(result["method"], "baseline_selected_boundary")
        self.assertEqual(result["policy"]["version"], "selected-boundary-patch-1")
        self.assertEqual(result["policy"]["minimum_boundary_views"], 4)
        self.assertEqual(result["policy"]["horizontal_boundary_padding_px"], 0.)
        self.assertEqual(result["method_sha256"], canonical_hash(result["policy"]))
        self.assertEqual(result["parent_native_sha256"], canonical_hash(source))
        self.assertEqual(result["context_sha256"], shared["sha256"])
        self.assertEqual(result["selected_context_sha256"], context["sha256"])
        self.assertEqual(result["native_selected_rows"][0]["selected_row_count"], 62)
        self.assertFalse(result["gt_read"])
        self.assertEqual(result["extra_annotations"], 0)
        self.assertFalse(result["native_axis_refitted"])
        self.assertFalse(result["common_reader_changed"])
        self.assertTrue(result["original_baseline_preserved_as_control"])
        changed = copy.deepcopy(inputs)
        changed[2]["finite"]["candidate_selection"][0]["row_matches"].pop()
        with self.assertRaisesRegex(ValueError, "Finite row assignments differ"):
            patch.build_selected_context(*changed)


if __name__ == "__main__":
    unittest.main()
