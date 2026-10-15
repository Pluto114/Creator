"""Five contracts separating native axis eligibility from finite width votes."""

import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rod_width_boundary_patch as four_view  # noqa: E402
from creator_eval import rod_width_finite_patch as patch  # noqa: E402
from creator_eval.rod_evidence import DEFAULT_CONFIG  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_anchored_boundary_readout import changed_context, line  # noqa: E402
from test_rod_selected_boundary_patch import native_inputs  # noqa: E402
from test_rod_width_boundary_patch import shared_context, width_inputs  # noqa: E402


class WidthFiniteContracts(unittest.TestCase):
    def test_three_width_qualified_views_retain_tail_rejected_by_four_view_width_rule(self):
        inputs = width_inputs(lambda view, y: 5. if view == 0 and y >= 122 else 2.)
        source, shared = inputs[2]["segments"], shared_context(inputs)
        context = patch.build_width_context(*inputs)
        self.assertEqual(sum(bool(mask[130, 0, 0]) for mask in context["row_members"]), 3)
        result = patch.refine_boundary(source, shared, context)
        old = four_view.refine_boundary(source, shared, four_view.build_width_context(*inputs))
        self.assertEqual(result["state"], "accepted")
        np.testing.assert_array_equal(result["segments"], source)
        self.assertAlmostEqual(old["segments"][0, 1, 1], .645, places=8)
        self.assertEqual(DEFAULT_CONFIG["minimum_positive_views"], 3)

    def test_only_two_width_qualified_views_cannot_keep_terminal_support(self):
        inputs = width_inputs(lambda view, y: 5. if view < 2 and y >= 122 else 2.)
        context = patch.build_width_context(*inputs)
        self.assertEqual(sum(bool(mask[130, 0, 0]) for mask in context["row_members"]), 2)
        source = inputs[2]["segments"]
        result = patch.refine_boundary(source, shared_context(inputs), context)
        self.assertEqual(result["state"], "accepted")
        self.assertAlmostEqual(result["segments"][0, 1, 1], .645, places=8)
        np.testing.assert_array_equal(result["segments"][0, 0], source[0, 0])

    def test_three_original_native_fit_views_are_not_promoted_to_four_view_axis(self):
        inputs = native_inputs(supporting=(0, 1, 2))
        inputs[2]["segments"] = line()
        with self.assertRaisesRegex(ValueError, "At least four original native fit views"):
            patch.build_width_context(*inputs)

    def test_raw_four_view_parent_cap_and_original_gap_remain_hard_limits(self):
        source = np.concatenate((line(-1., -.2, .015), line(.2, 1., .015)))
        inputs = width_inputs(lambda view, y: 5. if view == 0 and y >= 122 else 2., source=source)
        context = patch.build_width_context(*inputs)

        def missing_parent_tail(frame, row):
            if frame["view_id"] == "0" and row["y"] >= 122:
                row.update(status="unknown", candidates=[])

        shared = changed_context(missing_parent_tail)
        result = patch.refine_boundary(source, shared, context)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["segments"].shape, source.shape)
        np.testing.assert_array_equal(result["segments"][0], source[0])
        np.testing.assert_array_equal(result["segments"][1, 0], source[1, 0])
        np.testing.assert_array_equal(result["segments"][..., (0, 2)], source[..., (0, 2)])
        self.assertAlmostEqual(result["segments"][1, 1, 1], .645, places=8)
        self.assertTrue(np.all(result["segments"][..., 1] >= source[..., 1].min()))
        self.assertTrue(np.all(result["segments"][..., 1] <= source[..., 1].max()))

    def test_input_immutability_and_fixed_staged_policy_provenance(self):
        inputs = width_inputs(lambda view, y: 5. if view == 0 and y >= 122 else 2.)
        shared, source = shared_context(inputs), inputs[2]["segments"].copy()
        before_inputs, before_shared, original = pickle.dumps(inputs), pickle.dumps(shared), source.copy()
        context = patch.build_width_context(*inputs)
        before_context = pickle.dumps(context)
        result = patch.refine_boundary(source, shared, context)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(pickle.dumps(inputs), before_inputs)
        self.assertEqual(pickle.dumps(shared), before_shared)
        self.assertEqual(pickle.dumps(context), before_context)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(result["method"], "baseline_width_finite")
        self.assertEqual(result["policy"], patch.POLICY)
        self.assertEqual(result["policy"]["minimum_native_fit_views"], 4)
        self.assertEqual(result["policy"]["minimum_raw_boundary_views"], 4)
        self.assertEqual(result["policy"]["minimum_boundary_views"], DEFAULT_CONFIG["minimum_positive_views"])
        self.assertEqual(result["original_native_fit_view_count"], 4)
        self.assertEqual(result["method_sha256"], canonical_hash(patch.POLICY))
        self.assertEqual(result["parent_native_sha256"], canonical_hash(source))
        self.assertEqual(result["context_sha256"], shared["sha256"])
        self.assertEqual(result["width_context_sha256"], context["sha256"])
        self.assertFalse(result["native_axis_refitted"])
        self.assertFalse(result["gt_read"])
        self.assertEqual(result["extra_annotations"], 0)
        self.assertFalse(result["common_reader_changed"])
        self.assertFalse(result["physical_radius_estimated"])
        self.assertTrue(result["original_baseline_preserved_as_control"])


if __name__ == "__main__":
    unittest.main()
