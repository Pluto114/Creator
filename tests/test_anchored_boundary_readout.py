"""Six mechanisms for strict measured outer limits on unchanged actual axes."""

import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import anchored_boundary_readout as reader  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_anchored_boundary_readout import AnchoredBoundaryReadout  # noqa: E402
from creator_eval.rgb_bracketed_occupancy_readout import BracketedOccupancyReadout  # noqa: E402
from creator_eval.rgb_chain_support import chain_support_mask  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def changed_context(change):
    frames, cameras = fixture()
    for frame in frames:
        for row in frame["observations"]["rows"]:
            change(frame, row)
        frame["observation_sha256"] = canonical_hash(frame["observations"])
    return stroke.build_context(frames, cameras, measured_context()["association"]["anchors"])


def line(left=-1., right=1., x=0.):
    return np.array([[[x, left, 3.], [x, right, 3.]]])


class AnchoredBoundaryMechanisms(unittest.TestCase):
    def test_outer_tail_missing_one_marked_view_is_trimmed_without_extension(self):
        def change(frame, row):
            if frame["view_id"] == "0" and row["y"] >= 122:
                row.update(status="unknown", candidates=[])
        context = changed_context(change)
        self.assertTrue(chain_support_mask([[0., .9, 3.]], context)[0])
        source = line()
        result = reader.trim_outer_envelope(source, context)
        self.assertEqual(len(result["segments"]), 1)
        np.testing.assert_array_equal(result["segments"][0, 0], source[0, 0])
        self.assertAlmostEqual(result["segments"][0, 1, 1], .645, places=8)
        self.assertGreater(result["boundary_removed_length_m"], .35)
        self.assertLessEqual(result["segments"][0, 1, 1], source[0, 1, 1])

    def test_admission_padding_does_not_define_measured_boundary(self):
        context = measured_context()
        source = np.array([[[0., -1., 3.], [.045, 1., 3.]]])
        self.assertTrue(chain_support_mask(source[0], context).all())
        result = reader.trim_outer_envelope(source, context)
        self.assertEqual(len(result["segments"]), 1)
        np.testing.assert_array_equal(result["segments"][0, 0], source[0, 0])
        # fx=100/depth=3, measured half-width=1px: x <= .03m, not padded .06m.
        self.assertAlmostEqual(result["segments"][0, 1, 0], .03, places=10)
        self.assertAlmostEqual(result["segments"][0, 1, 1], 1./3, places=9)

    def test_internal_unknown_adds_no_cut_and_existing_actual_gap_is_preserved(self):
        def change(frame, row):
            if row["y"] == 110:
                row.update(status="unknown", candidates=[])
        context = changed_context(change)
        source = np.concatenate((line(-1., .1), line(.2, 1.)))
        self.assertFalse(chain_support_mask([[0., .3, 3.]], context)[0])
        result = reader.trim_outer_envelope(source, context)
        np.testing.assert_array_equal(result["segments"], source)
        self.assertEqual(result["boundary_removed_length_m"], 0.)

    def test_all_strong_support_preserves_endpoints_inputs_and_context(self):
        context, source = measured_context(), line(-.8, .9, .015)
        before, points = pickle.dumps(context), source.copy()
        result = reader.trim_outer_envelope(source, context)
        np.testing.assert_array_equal(result["segments"], source)
        np.testing.assert_array_equal(source, points)
        self.assertEqual(pickle.dumps(context), before)
        self.assertEqual(result["boundary_removed_length_m"], 0.)

    def test_no_strong_positive_support_returns_empty_without_fallback(self):
        context, source = measured_context(), line(x=.045)
        self.assertTrue(chain_support_mask(source[0], context).all())
        result = reader.trim_outer_envelope(source, context)
        self.assertEqual(result["segments"].shape, (0, 2, 3))
        self.assertEqual(result["boundary_positive_cells"], 0)
        self.assertAlmostEqual(result["boundary_removed_length_m"], 2.)

    def test_fixture_base_curve_equal_and_bracketed_model_selection_unchanged(self):
        context, source = measured_context(), line(x=.015)
        points = sample_segments(source, .01, 1000)
        policy, empty = dict(voxel_size=.02, origin=[0., 0., 0.]), np.empty((0, 2, 3))
        old = BracketedOccupancyReadout(points, context)(empty, policy)
        base = AnchoredBoundaryReadout(points, context)(empty, policy)
        curve = AnchoredBoundaryReadout(np.empty((0, 3)), context)(source, policy)
        self.assertGreater(base["components"], 0)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["selected_candidate"], old["selected_candidate"])
        self.assertEqual(base["selected_candidate"], curve["selected_candidate"])
        chosen = base["selected_candidate"]
        self.assertEqual(canonical_hash(base["readouts"][chosen]["graph"]["model"]),
                         canonical_hash(old["readouts"][chosen]["graph"]["model"]))
        self.assertEqual(canonical_hash(base["readouts"]), canonical_hash(curve["readouts"]))
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])
        self.assertTrue(base["boundary_requires_all_marked_views"])
        self.assertEqual(base["boundary_horizontal_padding_px"], 0.)


if __name__ == "__main__":
    unittest.main()
