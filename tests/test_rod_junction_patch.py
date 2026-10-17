"""Seven focused contracts for explicit transverse terminal-connector evidence."""

import copy
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rod_junction_patch as patch  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_anchored_boundary_readout import line  # noqa: E402
from test_rod_selected_boundary_patch import native_inputs  # noqa: E402
from test_rod_width_boundary_patch import shared_context  # noqa: E402


def junction_inputs(*, tops=(72, 72, 72, 72), bottoms=(128, 128, 128, 128), source=None):
    frames, cameras, baseline = native_inputs()
    baseline["segments"] = line() if source is None else source.copy()
    images = []
    for index, frame in enumerate(frames):
        x = int(frame["observations"]["rows"][0]["candidates"][0]["center_x"])
        image = np.full((201, 201, 3), 20, dtype=np.uint8)
        top, bottom = tops[index], bottoms[index]
        image[72:129, x-1:x+2] = 200
        for y in (top, bottom):
            if y is not None:
                image[y-3:y+4, x-16:x+17] = 200
        images.append(image)
    return frames, cameras, baseline, images


class JunctionContracts(unittest.TestCase):
    def test_bounded_member_on_both_flanks_uses_center_not_first_visible_rod_edge(self):
        inputs = junction_inputs()
        context = patch.build_junction_context(*inputs)
        self.assertEqual([len(side) for side in context["measurements"]], [4, 4])
        for diagnostic in context["diagnostics"]:
            self.assertEqual(diagnostic["state"], "measured")
            for side, expected_y in enumerate((72., 128.)):
                detection = diagnostic["endpoints"][side]
                self.assertEqual(len(detection["flanks"]), 6)
                self.assertAlmostEqual(detection["pixel_xy"][1], expected_y, places=8)
                # The crossbar occupies seven rows; its inward visibility edge
                # is 3.5 pixels from the connector center and is not the result.
                inner_edge = detection["flanks"][0]["bounds_px"][1-side]
                self.assertAlmostEqual(abs(inner_edge-detection["center_px"]), 3.5, places=8)
        result = patch.refine_boundary(inputs[2]["segments"], shared_context(inputs[:3]), context)
        np.testing.assert_allclose(result["segments"][0, :, 1], [-.84, .84], atol=1e-10)
        self.assertFalse(result["physical_endpoint_observed"])
        self.assertTrue(result["structural_connection_prior_used"])

    def test_missing_one_sided_unbounded_and_continuing_X_profiles_stay_unknown(self):
        inputs = junction_inputs()
        context = patch.build_junction_context(*inputs)
        view = context["views"][0]
        endpoints = inputs[2]["segments"][0]
        x = 130
        for case in ("missing", "one_sided", "unbounded", "continuing_X"):
            with self.subTest(case=case):
                image = np.full((201, 201), 20./255.)
                image[72:129, x-1:x+2] = 200./255.
                if case == "one_sided":
                    image[69:76, x:x+17] = 200./255.
                elif case == "unbounded":
                    image[:76, x-16:x+17] = 200./255.
                elif case == "continuing_X":
                    image[69:76, x-16:x+17] = 200./255.
                    image[45:129, x-1:x+2] = 200./255.
                detected = patch.detect_view_junctions(image, view, endpoints, 2.)
                self.assertIsNone(detected["endpoints"][0])

    def test_two_distinct_consistent_views_required_and_duplicates_cannot_vote(self):
        for count in (1, 2):
            with self.subTest(count=count):
                tops = tuple(72 if index < count else None for index in range(4))
                inputs = junction_inputs(tops=tops, bottoms=(None,)*4)
                context = patch.build_junction_context(*inputs)
                self.assertEqual(len(context["measurements"][0]), count)
                result = patch.refine_boundary(inputs[2]["segments"], shared_context(inputs[:3]), context)
                self.assertAlmostEqual(result["segments"][0, 0, 1], -.84 if count == 2 else -1.)
                self.assertEqual(result["junction_decisions"][0]["state"], "applied" if count == 2 else "unchanged")
        repeated = copy.deepcopy(context)
        repeated["measurements"][0] = [context["measurements"][0][0]]*2
        with self.assertRaisesRegex(ValueError, "Duplicate junction cameras"):
            patch.refine_boundary(inputs[2]["segments"], shared_context(inputs[:3]), repeated)

    def test_conflicting_detected_views_cannot_select_a_favorable_subset(self):
        inputs = junction_inputs(tops=(72, 72, 78, 78), bottoms=(None,)*4)
        # End the rod at each local junction, retaining a valid T rather than X.
        for index in (2, 3):
            image = inputs[3][index]
            x = int(inputs[0][index]["observations"]["rows"][0]["candidates"][0]["center_x"])
            image[:75, x-1:x+2] = 20
        context = patch.build_junction_context(*inputs)
        self.assertEqual(len(context["measurements"][0]), 4)
        result = patch.refine_boundary(inputs[2]["segments"], shared_context(inputs[:3]), context)
        np.testing.assert_array_equal(result["segments"], inputs[2]["segments"])
        self.assertEqual(result["junction_decisions"][0]["reason"], "inconsistent_junction_views")

    def test_no_extension_and_native_axis_and_existing_internal_gap_preserved(self):
        source = np.concatenate((line(-1., -.2), line(.2, 1.)))
        inputs = junction_inputs(source=source)
        result = patch.refine_boundary(source, shared_context(inputs[:3]), patch.build_junction_context(*inputs))
        self.assertEqual(result["segments"].shape, source.shape)
        np.testing.assert_array_equal(result["segments"][..., (0, 2)], source[..., (0, 2)])
        np.testing.assert_array_equal(result["segments"][[0, 1], [1, 0]], source[[0, 1], [1, 0]])
        np.testing.assert_allclose(result["segments"][[0, 1], [0, 1], 1], [-.84, .84], atol=1e-10)
        context = patch.build_junction_context(*inputs)
        for measurement, view in zip(context["measurements"][0], context["views"]):
            measurement["parameter_m"] = -.1
            measurement["pixel_xy"] = patch.project(context["anchor"]-.1*context["direction"], view)
        result = patch.refine_boundary(source, shared_context(inputs[:3]), context)
        np.testing.assert_array_equal(result["segments"][0, 0], source[0, 0])
        self.assertEqual(result["junction_decisions"][0]["reason"], "junction_outside_stable_extent_no_extension")

    def test_inputs_unchanged_and_hashes_bind_original_images_and_fixed_policy(self):
        inputs = junction_inputs()
        shared = shared_context(inputs[:3])
        before_inputs, before_shared = pickle.dumps(inputs), pickle.dumps(shared)
        context = patch.build_junction_context(*inputs)
        before_context = pickle.dumps(context)
        result = patch.refine_boundary(inputs[2]["segments"], shared, context)
        self.assertEqual(pickle.dumps(inputs), before_inputs)
        self.assertEqual(pickle.dumps(shared), before_shared)
        self.assertEqual(pickle.dumps(context), before_context)
        self.assertEqual(result["method"], "baseline_junction")
        self.assertEqual(result["method_sha256"], canonical_hash(patch.POLICY))
        self.assertEqual(result["junction_context_sha256"], context["sha256"])
        self.assertEqual(result["parent_native_sha256"], canonical_hash(inputs[2]["segments"]))
        self.assertFalse(result["gt_read"])
        self.assertFalse(result["native_axis_refitted"])
        self.assertFalse(result["common_reader_changed"])
        self.assertEqual(result["extra_annotations"], 0)
        changed = copy.deepcopy(inputs)
        changed[3][0][0, 0, 0] += 1
        self.assertNotEqual(patch.build_junction_context(*changed)["sha256"], context["sha256"])

    def test_native_four_view_axis_and_original_uint8_RGB_are_required(self):
        inputs = junction_inputs()
        changed = copy.deepcopy(inputs)
        changed[2]["proposal"]["selected"]["supporting_views"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            patch.build_junction_context(*changed)
        changed = copy.deepcopy(inputs)
        changed[3][0] = changed[3][0].astype(float)/255.
        with self.assertRaisesRegex(ValueError, "Unmodified uint8 RGB"):
            patch.build_junction_context(*changed)


if __name__ == "__main__":
    unittest.main()
