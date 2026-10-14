"""Five contracts for a new native boundary patch, not a common reader change."""

import copy
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rod_boundary_patch as patch  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_anchored_boundary_readout import changed_context, line  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def tail_context():
    def change(frame, row):
        if frame["view_id"] == "0" and row["y"] >= 122:
            row.update(status="unknown", candidates=[])
    return changed_context(change)


class RodBoundaryPatchContracts(unittest.TestCase):
    def test_native_axis_and_interior_gap_preserved_without_extension(self):
        source = np.concatenate((line(-1., -.2, .015), line(.2, 1., .015)))
        result = patch.refine_boundary(source, tail_context())
        self.assertEqual(result["state"], "accepted")
        self.assertEqual(result["segments"].shape, source.shape)
        np.testing.assert_array_equal(result["segments"][0], source[0])
        np.testing.assert_array_equal(result["segments"][1, 0], source[1, 0])
        np.testing.assert_array_equal(result["segments"][..., 0], source[..., 0])
        np.testing.assert_array_equal(result["segments"][..., 2], source[..., 2])
        self.assertAlmostEqual(result["segments"][1, 1, 1], .645, places=8)
        self.assertTrue(np.all(result["segments"][..., 1] >= source[..., 1].min()))
        self.assertTrue(np.all(result["segments"][..., 1] <= source[..., 1].max()))
        self.assertFalse(result["native_axis_refitted"])

    def test_native_input_and_shared_context_are_not_mutated(self):
        source, context = line(x=.015), tail_context()
        original, before = source.copy(), pickle.dumps(context)
        result = patch.refine_boundary(source, context)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(pickle.dumps(context), before)
        self.assertEqual(result["parent_native_sha256"], canonical_hash(original))
        self.assertTrue(result["original_baseline_preserved_as_control"])

    def test_empty_or_unobserved_native_output_is_rejected_not_fallback(self):
        context = measured_context()
        for source in (np.empty((0, 2, 3)), line(x=.045)):
            with self.subTest(source_points=len(source)):
                result = patch.refine_boundary(source, context)
                self.assertEqual(result["state"], "rejected")
                self.assertEqual(result["segments"].shape, (0, 2, 3))
                self.assertEqual(result["rejection_reasons"], ["no_four_view_native_boundary_support"])

    def test_incomplete_context_is_rejected(self):
        context = copy.deepcopy(measured_context())
        context["association"]["fallback_to_raw_union"] = True
        with self.assertRaisesRegex(ValueError, "Complete original foreground chain context"):
            patch.refine_boundary(line(), context)

    def test_different_native_inputs_share_one_rule_and_no_gt_metadata(self):
        context = tail_context()
        sources = (line(-1., 1., .015), line(-.8, .8, .015))
        originals = [source.copy() for source in sources]
        results = [patch.refine_boundary(source, context) for source in sources]
        self.assertEqual(results[0]["policy"], results[1]["policy"])
        self.assertEqual(results[0]["method_sha256"], results[1]["method_sha256"])
        self.assertNotEqual(results[0]["parent_native_sha256"], results[1]["parent_native_sha256"])
        for result, source, original in zip(results, sources, originals):
            self.assertEqual(result["method"], "baseline_boundary")
            self.assertEqual(result["context_sha256"], context["sha256"])
            self.assertEqual(result["policy"]["minimum_boundary_views"], 4)
            self.assertEqual(result["policy"]["horizontal_boundary_padding_px"], 0.)
            self.assertFalse(result["gt_read"])
            self.assertEqual(result["extra_annotations"], 0)
            self.assertFalse(result["common_reader_changed"])
            self.assertFalse(result["policy"]["base_point_suppression"])
            self.assertAlmostEqual(result["segments"][0, 1, 1], .645, places=8)
            np.testing.assert_array_equal(source, original)


if __name__ == "__main__":
    unittest.main()
