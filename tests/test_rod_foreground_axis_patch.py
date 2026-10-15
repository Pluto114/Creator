"""Six direct contracts for an RGB-band-fitted native axis and finite patch."""

import copy
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rod_foreground_axis_patch as patch  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_anchored_boundary_readout import line  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def update_signature(context):
    context["sha256"] = canonical_hash(context["association"])
    return context


class ForegroundAxisPatchContracts(unittest.TestCase):
    def test_two_real_annotated_views_correct_native_axis_from_measured_rgb(self):
        source, context = line(-1., 1., .015), measured_context()
        self.assertEqual(len({a["view_id"] for a in context["association"]["anchors"]}), 2)
        result = patch.refine_boundary(source, context)
        self.assertEqual(result["state"], "accepted")
        np.testing.assert_allclose(result["segments"][..., 0], 0., atol=1e-12)
        np.testing.assert_allclose(result["segments"][..., 2], 3., atol=1e-12)
        np.testing.assert_allclose(result["segments"][..., 1], source[..., 1], atol=1e-12)
        self.assertTrue(result["native_axis_refitted"])

    def test_shifted_clicks_within_same_measured_bands_do_not_move_axis(self):
        context, source = measured_context(), line(x=.015)
        shifted = copy.deepcopy(context)
        for anchor in shifted["association"]["anchors"]:
            anchor["xy"][0] -= .6
        update_signature(shifted)
        first = patch.refine_boundary(source, context)
        second = patch.refine_boundary(source, shifted)
        self.assertEqual(first["state"], "accepted")
        self.assertEqual(second["state"], "accepted")
        np.testing.assert_array_equal(first["segments"], second["segments"])

    def test_projection_preserves_component_order_orientation_and_inner_gap(self):
        source = np.concatenate((line(.8, .2, .015), line(-.2, -.8, .015)))
        result = patch.refine_boundary(source, measured_context())
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["segments"].shape, source.shape)
        np.testing.assert_allclose(result["segments"][..., 1], source[..., 1], atol=1e-12)
        np.testing.assert_allclose(result["segments"][..., 0], 0., atol=1e-12)
        self.assertGreater(result["segments"][0, 1, 1], .19)
        self.assertLess(result["segments"][1, 0, 1], -.19)

    def test_absent_too_few_and_one_row_marks_reject_without_native_fallback(self):
        original = measured_context()
        subsets = ([], original["association"]["anchors"][:3],
                   [original["association"]["anchors"][0], original["association"]["anchors"][3]])
        for anchors in subsets:
            with self.subTest(mark_count=len(anchors)):
                context = copy.deepcopy(original)
                context["association"]["anchors"] = copy.deepcopy(anchors)
                result = patch.refine_boundary(line(x=.015), update_signature(context))
                self.assertEqual(result["state"], "rejected")
                self.assertEqual(result["segments"].shape, (0, 2, 3))
                self.assertTrue(result["rejection_reasons"])

    def test_coincident_marked_camera_planes_reject_without_native_fallback(self):
        context = measured_context()
        first, last = context["views"][0], context["views"][3]
        for key in ("K_index", "world_to_camera_cv", "left", "right"):
            last[key] = first[key].copy()
        for anchor in context["association"]["anchors"]:
            if anchor["view_id"] == "3":
                anchor["xy"][0] += 60.
        result = patch.refine_boundary(line(x=.015), update_signature(context))
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(result["segments"].shape, (0, 2, 3))
        self.assertTrue(result["rejection_reasons"])

    def test_inputs_unchanged_and_metadata_binds_new_axis_policy_without_gt(self):
        context, source = measured_context(), line(x=.015)
        before, original = pickle.dumps(context), source.copy()
        result = patch.refine_boundary(source, context)
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(pickle.dumps(context), before)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(result["method"], "baseline_foreground_axis")
        self.assertEqual(result["method_sha256"], canonical_hash(result["policy"]))
        self.assertEqual(result["parent_native_sha256"], canonical_hash(source))
        self.assertEqual(result["context_sha256"], context["sha256"])
        self.assertFalse(result["gt_read"])
        self.assertEqual(result["extra_annotations"], 0)
        self.assertFalse(result["common_reader_changed"])
        self.assertTrue(result["native_axis_refitted"])
        self.assertTrue(result["original_baseline_preserved_as_control"])


if __name__ == "__main__":
    unittest.main()
