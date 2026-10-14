"""Six mechanisms for internally bracketed RGB holes without edge extension."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import bracketed_occupancy_readout as reader  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_bracketed_occupancy_readout import BracketedOccupancyReadout  # noqa: E402
from creator_eval.rgb_occupied_axis_readout import OccupiedAxisReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def segments(intervals):
    values = np.asarray(intervals, dtype=float)
    output = np.zeros((*values.shape, 3))
    output[..., 1] = values
    output[..., 2] = 3.
    return output


class BracketedOccupancyMechanisms(unittest.TestCase):
    def test_short_hole_inside_same_actual_interval_is_bridged(self):
        result = reader.bridge_internal_holes(segments([[0., .4], [.45, 1.]]),
                                               segments([[0., 1.]]), .1)
        np.testing.assert_array_equal(result["segments"], segments([[0., 1.]]))
        self.assertEqual(result["bridged_gap_count"], 1)
        self.assertAlmostEqual(result["bridged_length_m"], .05)

    def test_original_outer_endpoints_are_not_extended(self):
        output = segments([[.1, .4], [.45, .9]])
        result = reader.bridge_internal_holes(output, segments([[0., 1.]]), .1)
        np.testing.assert_array_equal(result["segments"], segments([[.1, .9]]))
        np.testing.assert_array_equal(result["segments"][0, 0], output[0, 0])
        np.testing.assert_array_equal(result["segments"][0, 1], output[-1, 1])

    def test_small_actual_occupancy_gap_is_never_bridged(self):
        occupied = segments([[0., .4], [.45, 1.]])
        result = reader.bridge_internal_holes(occupied.copy(), occupied, .1)
        np.testing.assert_array_equal(result["segments"], occupied)
        self.assertEqual(result["bridged_gap_count"], 0)
        self.assertEqual(result["bridged_length_m"], 0.)

    def test_rgb_hole_larger_than_one_voxel_is_not_bridged(self):
        output = segments([[0., .3], [.41, 1.]])
        result = reader.bridge_internal_holes(output, segments([[0., 1.]]), .1)
        np.testing.assert_array_equal(result["segments"], output)
        self.assertEqual(result["bridged_gap_count"], 0)

    def test_multiple_holes_are_deterministic_without_cumulative_threshold_drift(self):
        output = segments([[0., .2], [.28, .4], [.49, .6], [.72, 1.]])
        before = output.copy()
        occupied = segments([[0., 1.]])
        first = reader.bridge_internal_holes(output, occupied, .1)
        second = reader.bridge_internal_holes(output, occupied, .1)
        np.testing.assert_array_equal(first["segments"], segments([[0., .6], [.72, 1.]]))
        np.testing.assert_array_equal(first["segments"], second["segments"])
        np.testing.assert_array_equal(output, before)
        self.assertEqual(first["bridged_gap_count"], 2)
        self.assertAlmostEqual(first["bridged_length_m"], .17)

    def test_actual_fixture_base_curve_equal_and_selected_tls_unchanged(self):
        frames, cameras = fixture()
        for frame in frames:
            row = next(row for row in frame["observations"]["rows"] if row["y"] == 110)
            row.update(status="unknown", candidates=[])
            frame["observation_sha256"] = canonical_hash(frame["observations"])
        context = stroke.build_context(frames, cameras,
                                       measured_context()["association"]["anchors"])
        line = segments([[-1., 1.]])
        points = sample_segments(line, .02, 1000)
        policy = dict(voxel_size=.04, origin=[0., 0., 0.])
        empty = np.empty((0, 2, 3))
        old = OccupiedAxisReadout(points, context)(empty, policy)
        base = BracketedOccupancyReadout(points, context)(empty, policy)
        curve = BracketedOccupancyReadout(np.empty((0, 3)), context)(line, policy)
        self.assertEqual(old["components"], 2)
        self.assertEqual(base["components"], 1)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["selected_candidate"], old["selected_candidate"])
        self.assertEqual(base["selected_candidate"], curve["selected_candidate"])
        chosen = base["selected_candidate"]
        self.assertEqual(canonical_hash(base["readouts"][chosen]["graph"]["model"]),
                         canonical_hash(old["readouts"][chosen]["graph"]["model"]))
        self.assertEqual(canonical_hash(base["readouts"]), canonical_hash(curve["readouts"]))
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])
        np.testing.assert_array_equal(base["segments"][0, 0], old["segments"][0, 0])
        np.testing.assert_array_equal(base["segments"][-1, 1], old["segments"][-1, 1])


if __name__ == "__main__":
    unittest.main()
