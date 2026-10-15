"""Six contracts for selected-pair width contamination, not physical radius."""

import copy
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval import rod_width_boundary_patch as patch  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_anchored_boundary_readout import line  # noqa: E402
from test_rod_selected_boundary_patch import native_inputs  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def width_inputs(width=None, *, select=None, source=None):
    frames, cameras, baseline = native_inputs(select=select)
    baseline["segments"] = line() if source is None else source.copy()
    for index, frame in enumerate(frames):
        for row in frame["observations"]["rows"]:
            value = 2. if width is None else width(index, row["y"])
            candidate = row["candidates"][0]
            candidate["width"] = value
            candidate["left_edge"]["x"] = candidate["center_x"] - value / 2
            candidate["right_edge"]["x"] = candidate["center_x"] + value / 2
        frame["observation_sha256"] = canonical_hash(frame["observations"])
    return frames, cameras, baseline


def shared_context(inputs):
    return stroke.build_context(inputs[0], inputs[1], measured_context()["association"]["anchors"])


class WidthBoundaryContracts(unittest.TestCase):
    def test_wide_tail_loses_votes_but_exact_two_pixel_allowance_remains(self):
        inputs = width_inputs(lambda view, y: 5. if y >= 122 else 4. if y == 120 else 2.)
        context = patch.build_width_context(*inputs)
        for mask, view in zip(context["row_members"], context["views"]):
            self.assertTrue(mask[120, 0, 0])
            self.assertFalse(mask[122, 0, 0])
            self.assertTrue(np.isnan(view["left"][122, 0]))
            self.assertTrue(np.isnan(view["right"][122, 0]))
        result = patch.refine_boundary(inputs[2]["segments"], shared_context(inputs), context)
        self.assertEqual(result["state"], "accepted")
        self.assertAlmostEqual(result["segments"][0, 1, 1], .645, places=8)
        np.testing.assert_array_equal(result["segments"][0, 0], inputs[2]["segments"][0, 0])
        self.assertGreater(result["width_boundary_removed_length_m"], .35)
        for item in context["association"]["width_diagnostics"]:
            self.assertEqual(item["reference_state"], "measured")
            self.assertEqual(item["reference_row_count"], 33)
            self.assertEqual(item["reference_width_px"], 2.)
            self.assertEqual(item["maximum_width_px"], 4.)
            self.assertEqual(item["retained_row_count"], 62)
            self.assertEqual([row["row"] for row in item["removed_rows"]], list(range(122, 141)))

    def test_each_view_uses_its_own_central_width_reference(self):
        def measured_width(view, y):
            reference = 2. + 2. * view
            return reference + (2. if y == 130 else 2.01 if y == 131 else 0.)
        context = patch.build_width_context(*width_inputs(measured_width))
        for index, mask in enumerate(context["row_members"]):
            self.assertTrue(mask[130, 0, 0])
            self.assertFalse(mask[131, 0, 0])
            item = context["association"]["width_diagnostics"][index]
            self.assertEqual(item["reference_width_px"], 2. + 2. * index)
            self.assertEqual(item["maximum_width_px"], 4. + 2. * index)

    def test_missing_central_rows_add_no_filter_and_fabricate_no_support(self):
        inputs = width_inputs(lambda view, y: 8. if y == 130 else 2.,
            select=lambda view, index, row: None if 84 <= row["y"] <= 116 and row["y"] not in (100, 101) else 0)
        context = patch.build_width_context(*inputs)
        for mask in context["row_members"]:
            self.assertTrue(mask[130, 0, 0])
            self.assertFalse(mask[110, 0, 0])
            self.assertTrue(mask[100, 0, 0])
            self.assertTrue(mask[101, 0, 0])
        for item in context["association"]["width_diagnostics"]:
            self.assertEqual(item["reference_state"], "unavailable")
            self.assertEqual(item["reference_row_count"], 2)
            self.assertIsNone(item["reference_width_px"])
            self.assertIsNone(item["maximum_width_px"])
            self.assertEqual(item["retained_row_count"], item["selected_row_count"])
            self.assertEqual(item["removed_rows"], [])
        result = patch.refine_boundary(inputs[2]["segments"], shared_context(inputs), context)
        self.assertEqual(result["state"], "accepted")
        np.testing.assert_array_equal(result["segments"], inputs[2]["segments"])

    def test_internal_wide_unknown_neither_splits_geometry_nor_fills_native_gap(self):
        source = np.concatenate((line(-1., .1), line(.2, 1.)))
        inputs = width_inputs(lambda view, y: 8. if y == 110 else 2., source=source)
        context = patch.build_width_context(*inputs)
        self.assertTrue(all(not mask[110, 0, 0] for mask in context["row_members"]))
        result = patch.refine_boundary(source, shared_context(inputs), context)
        self.assertEqual(result["state"], "accepted")
        np.testing.assert_array_equal(result["segments"], source)

    def test_inputs_context_native_axis_and_internal_endpoint_are_unchanged(self):
        source = np.concatenate((line(-1., -.2, .015), line(.2, 1., .015)))
        inputs = width_inputs(lambda view, y: 5. if y >= 122 else 2., source=source)
        shared = shared_context(inputs)
        before_inputs, before_shared, before_source = pickle.dumps(inputs), pickle.dumps(shared), source.copy()
        context = patch.build_width_context(*inputs)
        before_context = pickle.dumps(context)
        result = patch.refine_boundary(source, shared, context)
        self.assertEqual(pickle.dumps(inputs), before_inputs)
        self.assertEqual(pickle.dumps(shared), before_shared)
        self.assertEqual(pickle.dumps(context), before_context)
        np.testing.assert_array_equal(source, before_source)
        np.testing.assert_array_equal(result["segments"][..., (0, 2)], source[..., (0, 2)])
        np.testing.assert_array_equal(result["segments"][0], source[0])
        np.testing.assert_array_equal(result["segments"][1, 0], source[1, 0])
        self.assertFalse(result["native_axis_refitted"])

    def test_policy_and_original_row_provenance_are_bound_without_gt(self):
        inputs = width_inputs(lambda view, y: 5. if y >= 122 else 2.)
        context, shared = patch.build_width_context(*inputs), shared_context(inputs)
        result = patch.refine_boundary(inputs[2]["segments"], shared, context)
        self.assertEqual(result["method"], "baseline_width_boundary")
        self.assertEqual(result["policy"], patch.POLICY)
        self.assertEqual(result["method_sha256"], canonical_hash(patch.POLICY))
        self.assertEqual(result["parent_native_sha256"], canonical_hash(inputs[2]["segments"]))
        self.assertEqual(result["context_sha256"], shared["sha256"])
        self.assertEqual(result["width_context_sha256"], context["sha256"])
        self.assertEqual(result["width_diagnostics"], context["association"]["width_diagnostics"])
        self.assertFalse(result["physical_radius_estimated"])
        self.assertFalse(result["gt_read"])
        self.assertEqual(result["extra_annotations"], 0)
        self.assertFalse(result["common_reader_changed"])
        self.assertTrue(result["original_baseline_preserved_as_control"])
        changed = copy.deepcopy(inputs)
        changed[2]["finite"]["candidate_selection"][0]["row_matches"].pop()
        with self.assertRaisesRegex(ValueError, "Finite row assignments differ"):
            patch.build_width_context(*changed)


if __name__ == "__main__":
    unittest.main()
