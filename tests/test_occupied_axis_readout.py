"""Eight mechanisms for exact occupancy on one fixed, stroke-selected axis."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import occupied_axis_readout as reader  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval import single_axis_readout as axis  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_occupied_axis_readout import OccupiedAxisReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_stroke_selected_axis_readout import measured_context, vertical_points  # noqa: E402


class OccupiedAxisMechanisms(unittest.TestCase):
    def test_exact_one_voxel_union_and_actual_outer_endpoints(self):
        intervals = reader.occupied_intervals([0., 1., 4., 5., 6., 8.], 1.)
        np.testing.assert_array_equal(intervals, [[0., 2.], [3., 8.]])
        points = vertical_points(.015, np.linspace(-1., 1., 201))
        graph = reader.actual_axis(points, .02, np.zeros(3))
        self.assertEqual(graph["components"], 1)
        np.testing.assert_allclose(graph["segments"][0, :, 1], [-1., 1.], atol=1e-12)
        self.assertEqual(graph["projection_support_radius_voxels"], 1.)

    def test_narrow_actual_gap_over_two_voxels_is_not_sampled_away(self):
        ys = np.array([-6., -5., -4., -3., -2., 0., 2.01, 4., 5., 6., 7., 8., 9., 10.])
        # A half-voxel sample grid sees support everywhere, missing (1, 1.01).
        old_samples = np.linspace(-6., 10., 33)
        self.assertTrue(np.all(np.abs(old_samples[:, None] - ys).min(axis=1) <= 1.))
        self.assertGreater(2.01 - 0., 2.)
        graph = reader.actual_axis(vertical_points(0., ys), 1., np.zeros(3))
        self.assertEqual(graph["components"], 2)
        np.testing.assert_allclose(graph["segments"][..., 1], [[-6., 1.], [1.01, 10.]], atol=1e-12)
        self.assertGreater(graph["segments"][1, 0, 1] - graph["segments"][0, 1, 1], .009)

    def test_same_voxel_mean_does_not_erase_actual_point_occupancy(self):
        ys = np.array([-5., -4., -3., -2., -1., .01, .99, 3., 4., 5., 6., 7., 8.])
        points = vertical_points(0., ys)
        _, representatives = axis.voxel_representatives(points, 1., np.zeros(3))
        self.assertLess(np.abs(ys - 1.75).min(), 1.)
        self.assertGreater(np.abs(representatives[:, 1] - 1.75).min(), 1.)
        graph = reader.actual_axis(points, 1., np.zeros(3))
        self.assertEqual(graph["components"], 2)
        np.testing.assert_allclose(graph["segments"][..., 1], [[-5., 1.99], [2., 8.]], atol=1e-12)

    def test_rgb_ranking_preserves_fixed_actual_tls_model(self):
        context = measured_context()
        points = vertical_points(.015, np.linspace(-1., 1., 201))
        actual = reader.actual_axis(points, .02, np.zeros(3))
        result = reader.readout(points, .02, np.zeros(3), context)
        chosen = result["readouts"][result["selected_candidate"]]["graph"]
        self.assertEqual(canonical_hash(chosen["model"]), canonical_hash(actual["model"]))
        np.testing.assert_allclose(result["segments"][..., 0], .015, atol=1e-12)
        np.testing.assert_allclose(result["segments"][..., 2], 3., atol=1e-12)
        self.assertFalse(result["image_constructed_geometry"])

    def test_unknown_rgb_row_still_cuts_continuous_actual_occupancy(self):
        frames, cameras = fixture()
        claims = measured_context()["association"]["anchors"]
        for frame in frames:
            row = next(row for row in frame["observations"]["rows"] if row["y"] == 110)
            row.update(status="unknown", candidates=[])
            frame["observation_sha256"] = canonical_hash(frame["observations"])
        context = stroke.build_context(frames, cameras, claims)
        points = vertical_points(0., np.linspace(-1., 1., 201))
        result = reader.readout(points, .02, np.zeros(3), context)
        chosen = result["readouts"][result["selected_candidate"]]["graph"]
        self.assertEqual(chosen["components"], 1)
        self.assertEqual(result["components"], 2)
        # Rounded image row 110 occupies world y in [.285, .315].
        np.testing.assert_allclose(result["segments"][..., 1], [[-1., .285], [.315, 1.]], atol=1e-9)
        self.assertLess(result["segments"][0, 1, 1], .3)
        self.assertGreater(result["segments"][1, 0, 1], .3)

    def test_duplicate_and_reordered_points_keep_exact_output(self):
        context = measured_context()
        points = vertical_points(.015, np.linspace(-1., 1., 121))
        first = reader.readout(points, .02, np.zeros(3), context)
        second = reader.readout(np.concatenate((points[::-1], points[::3], points)),
                                .02, np.zeros(3), context)
        self.assertGreater(first["components"], 0)
        np.testing.assert_array_equal(first["segments"], second["segments"])
        self.assertEqual(first["selected_candidate"], second["selected_candidate"])
        self.assertEqual(canonical_hash(first["readouts"]), canonical_hash(second["readouts"]))

    def test_equal_base_and_curve_coordinates_have_identical_readout(self):
        context = measured_context()
        segments = np.array([[[.015, -1., 3.], [.015, 1., 3.]]])
        samples = sample_segments(segments, .01, 1000)
        policy = dict(voxel_size=.02, origin=[0., 0., 0.])
        base = OccupiedAxisReadout(samples, context)(np.empty((0, 2, 3)), policy)
        curve = OccupiedAxisReadout(np.empty((0, 3)), context)(segments, policy)
        self.assertGreater(base["components"], 0)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["selected_candidate"], curve["selected_candidate"])
        self.assertEqual(canonical_hash(base["readouts"]), canonical_hash(curve["readouts"]))
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])

    def test_four_voxel_gate_qualifies_axis_without_pruning_supported_fragments(self):
        diagonal = np.array([.1, .4, .7]) + np.linspace(0., 3.5, 101)[:, None] / np.sqrt(3.)
        rejected = reader.actual_axis(diagonal, 1., np.zeros(3))
        self.assertGreaterEqual(rejected["occupied_voxels"], 6)
        self.assertIsNotNone(rejected["model"])
        self.assertEqual(rejected["reason"], "below_minimum_axis_length")
        self.assertEqual(rejected["components"], 0)
        # The full 13v axis qualifies; both [0, 3] and [9, 13] stay supported.
        points = vertical_points(0., [0., 1., 2., 10., 11., 12., 13.])
        graph = reader.actual_axis(points, 1., np.zeros(3))
        self.assertEqual(graph["minimum_axis_extent_voxels"], 4.)
        self.assertEqual(graph["geometry_interval_count"], 2)
        self.assertEqual(graph["discarded_short_runs"], 0)
        self.assertEqual(graph["retained_short_fragments"], 1)
        self.assertEqual(graph["retained_short_length_m"], 3.)
        np.testing.assert_array_equal(graph["segments"][..., 1], [[0., 3.], [9., 13.]])
        # The real 3v fragment must also survive final RGB cell clipping.
        supported = vertical_points(0., np.array([0., 1., 2., 10., 11., 12., 13.]) * .02 - .2)
        result = reader.readout(supported, .02, np.zeros(3), measured_context())
        self.assertEqual(result["components"], 2)
        np.testing.assert_allclose(result["segments"][..., 1], [[-.2, -.14], [-.02, .06]], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
