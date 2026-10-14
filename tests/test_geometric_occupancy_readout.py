"""Six mechanisms for geometric occupancy after unchanged RGB axis selection."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import geometric_occupancy_readout as reader  # noqa: E402
from creator_eval import occupied_axis_readout as previous  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_chain_support import chain_support_mask  # noqa: E402
from creator_eval.rgb_geometric_occupancy_readout import GeometricOccupancyReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_stroke_selected_axis_readout import measured_context, vertical_points  # noqa: E402


class GeometricOccupancyMechanisms(unittest.TestCase):
    def test_selected_tls_model_and_candidate_ranking_are_unchanged(self):
        context = measured_context()
        points = vertical_points(.015, np.linspace(-1., 1., 201))
        old = previous.readout(points, .02, np.zeros(3), context)
        new = reader.readout(points, .02, np.zeros(3), context)
        self.assertIsNotNone(new["selected_candidate"])
        self.assertEqual(new["selected_candidate"], old["selected_candidate"])
        self.assertEqual(new["selected_source"], old["selected_source"])
        self.assertEqual(new["selected_band_score"], old["selected_band_score"])
        chosen = new["selected_candidate"]
        self.assertEqual(canonical_hash(new["readouts"][chosen]["graph"]["model"]),
                         canonical_hash(old["readouts"][chosen]["graph"]["model"]))
        self.assertFalse(new["image_constructed_geometry"])

    def test_output_equals_selected_actual_occupancy_without_endpoint_extension(self):
        context = measured_context()
        points = vertical_points(.015, np.linspace(-.9, .8, 171))
        new = reader.readout(points, .02, np.zeros(3), context)
        selected = new["readouts"][new["selected_candidate"]]["graph"]
        np.testing.assert_array_equal(new["segments"], selected["segments"])
        np.testing.assert_allclose(new["segments"][..., 1].ravel(), [-.9, .8], atol=1e-12)
        self.assertEqual(new["components"], len(selected["segments"]))

    def test_short_unknown_rgb_row_does_not_cut_supported_3d_occupancy(self):
        frames, cameras = fixture()
        claims = measured_context()["association"]["anchors"]
        for frame in frames:
            row = next(row for row in frame["observations"]["rows"] if row["y"] == 110)
            row.update(status="unknown", candidates=[])
            frame["observation_sha256"] = canonical_hash(frame["observations"])
        context = stroke.build_context(frames, cameras, claims)
        raw = vertical_points(0., np.linspace(-1., 1., 401))
        mask = chain_support_mask(raw, context)
        self.assertLess(int(mask.sum()), len(raw))
        points = raw[mask]
        old = previous.readout(points, .03, np.zeros(3), context)
        new = reader.readout(points, .03, np.zeros(3), context)
        self.assertEqual(old["components"], 2)
        self.assertEqual(new["components"], 1)
        np.testing.assert_allclose(new["segments"][..., 1], [[-1., 1.]], atol=1e-12)
        self.assertEqual(new["output_support_policy"],
                         "selected_actual_3d_occupancy_without_second_rgb_clip")
        self.assertFalse(new["continuous_rgb_support_required"])
        self.assertFalse(new["image_only_gaps_preserved"])
        self.assertEqual(new["minimum_resolvable_geometric_gap_voxels"], 2.)
        self.assertFalse(new["axis_refitted"])

    def test_actual_gap_larger_than_two_voxels_survives_positive_rgb(self):
        context = measured_context()
        self.assertTrue(chain_support_mask([[0., 0., 3.]], context)[0])
        points = vertical_points(0., np.r_[np.linspace(-1., -.3, 71), np.linspace(.3, 1., 71)])
        result = reader.readout(points, .02, np.zeros(3), context)
        self.assertEqual(result["components"], 2)
        np.testing.assert_allclose(result["segments"][..., 1], [[-1., -.28], [.28, 1.]], atol=1e-12)

    def test_duplicate_and_reordered_points_preserve_exact_output(self):
        context = measured_context()
        points = vertical_points(.015, np.linspace(-1., 1., 121))
        first = reader.readout(points, .02, np.zeros(3), context)
        second = reader.readout(np.concatenate((points[::-1], points[::3], points)),
                                .02, np.zeros(3), context)
        self.assertGreater(first["components"], 0)
        np.testing.assert_array_equal(first["segments"], second["segments"])
        self.assertEqual(first["selected_candidate"], second["selected_candidate"])
        self.assertEqual(canonical_hash(first["readouts"]), canonical_hash(second["readouts"]))

    def test_equal_base_and_curve_coordinates_keep_identical_readout(self):
        context = measured_context()
        segments = np.array([[[.015, -1., 3.], [.015, 1., 3.]]])
        samples = sample_segments(segments, .01, 1000)
        policy = dict(voxel_size=.02, origin=[0., 0., 0.])
        base = GeometricOccupancyReadout(samples, context)(np.empty((0, 2, 3)), policy)
        curve = GeometricOccupancyReadout(np.empty((0, 3)), context)(segments, policy)
        self.assertGreater(base["components"], 0)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["selected_candidate"], curve["selected_candidate"])
        self.assertEqual(canonical_hash(base["readouts"]), canonical_hash(curve["readouts"]))
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])


if __name__ == "__main__":
    unittest.main()
