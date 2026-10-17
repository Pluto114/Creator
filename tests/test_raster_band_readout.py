"""Fixed one-pixel equivalence ranks existing, clipped actual support only."""

import pickle
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import raster_band_readout as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_raster_band_readout import RasterBandReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_bracketed_occupancy_readout import segments  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def candidate(intervals, excess, *, clipped=None, signature=None):
    row = dict(source="original_actual_TLS", support_set_sha256=signature,
        graph=dict(segments=segments(intervals), model=dict(perpendicular_rmse=.001,
            centroid=np.array([0., 0., 3.]), direction=np.array([0., 1., 0.]))),
        band_score=dict(maximum_band_excess_px=excess, band_excess_rms_px=excess / 2))
    if clipped is not None:
        row["rgb_cell_clip"] = dict(segments=segments(clipped))
    return row


def old_result(rows):
    selected = rows[0].get("rgb_cell_clip", rows[0]["graph"])["segments"].copy()
    return dict(state="complete", selected_candidate=0, readouts=rows,
                segments=selected, components=len(selected), reason=None)


class RasterBandMechanisms(unittest.TestCase):
    def test_longer_existing_candidate_wins_within_inclusive_one_pixel(self):
        rows = [candidate([[0., .4]], 0., clipped=[[0., .4]]),
                candidate([[0., 1.]], 1., clipped=[[0., 1.]])]
        geometry_before = [canonical_hash(row["graph"]) for row in rows]
        result = reader.select_existing(old_result(rows), np.empty((0, 3)), .1, {})
        self.assertEqual(result["selected_candidate"], 1)
        self.assertEqual(result["previous_selected_candidate"], 0)
        self.assertTrue(result["selection_changed"])
        np.testing.assert_array_equal(result["segments"], rows[1]["graph"]["segments"])
        self.assertEqual([canonical_hash(row["graph"]) for row in rows], geometry_before)
        self.assertEqual(result["raster_admissible_count"], 2)
        self.assertEqual(result["raster_band_policy_sha256"], canonical_hash(reader.POLICY))
        self.assertEqual(result["raster_band_policy"]["axis_policy"],
                         "original_actual_point_TLS_hypotheses_no_refit")

    def test_longer_outside_one_pixel_is_not_eligible(self):
        rows = [candidate([[0., .4]], 0., clipped=[[0., .4]]),
                candidate([[0., 20.]], np.nextafter(1., np.inf), clipped=[[0., 20.]])]
        result = reader.select_existing(old_result(rows), np.empty((0, 3)), .1, {})
        self.assertEqual(result["selected_candidate"], 0)
        self.assertEqual(result["raster_admissible_count"], 1)
        self.assertNotIn("raster_final_length_m", rows[1])

    def test_no_admissible_candidate_preserves_old_output_exactly(self):
        rows = [candidate([[0., 1.]], 1.01, clipped=[[.1, .8]]),
                candidate([[0., 20.]], 2., clipped=[[0., 20.]])]
        source = old_result(rows)
        before = source["segments"].copy()
        with patch.object(reader.cells, "clip_segments", side_effect=AssertionError("unexpected clip")):
            result = reader.select_existing(source, np.empty((0, 3)), .1, {})
        np.testing.assert_array_equal(result["segments"], before)
        self.assertEqual(result["selected_candidate"], 0)
        self.assertFalse(result["selection_changed"])
        self.assertEqual(result["raster_selection_state"], "fallback_unchanged")
        self.assertEqual(result["raster_evaluated_count"], 0)

    def test_rank_uses_clipped_and_bridged_length_not_raw_length(self):
        rows = [candidate([[0., 4.]], 0., clipped=[[0., .3]]),
                candidate([[0., .85]], .5, clipped=[[0., .85]]),
                candidate([[0., .9]], .8, clipped=[[0., .4], [.5, .9]])]
        result = reader.select_existing(old_result(rows), np.empty((0, 3)), .1, {})
        self.assertEqual(result["selected_candidate"], 2)
        self.assertAlmostEqual(rows[0]["raster_final_length_m"], .3)
        self.assertAlmostEqual(rows[2]["raster_final_length_m"], .9)
        self.assertEqual(result["bridged_gap_count"], 1)
        np.testing.assert_array_equal(result["segments"], segments([[0., .9]]))

    def test_unclipped_candidate_uses_its_own_original_assignment(self):
        rows = [candidate([[0., .4]], 0., clipped=[[0., .4]]),
                candidate([[0., 1.]], .5, signature="own-group")]
        context = dict(association=dict(assignments=["parent"], chain_count=1), sha256="original")
        before = canonical_hash(context)
        values = np.array([[0., .4, 3.], [0., 0., 3.], [0., .4, 3.]])
        values_before = values.copy()
        groups = [dict(support_set_sha256="own-group", assignments=["own-original-chain"])]
        seen = []

        def clip(occupied, local, floor):
            seen.append(local["association"]["assignments"])
            np.testing.assert_array_equal(occupied, rows[1]["graph"]["segments"])
            self.assertGreater(floor, 0.)
            return dict(segments=segments([[.1, .9]]))

        with patch.object(reader, "support_groups", return_value=groups) as recover:
            with patch.object(reader.cells, "clip_segments", side_effect=clip):
                result = reader.select_existing(old_result(rows), values, .1, context)
        self.assertEqual(seen, [["own-original-chain"]])
        np.testing.assert_array_equal(recover.call_args.args[0], np.unique(values, axis=0))
        np.testing.assert_array_equal(values, values_before)
        self.assertEqual(canonical_hash(context), before)
        self.assertEqual(result["selected_candidate"], 1)

    def test_true_occupancy_gap_and_rgb_outer_endpoints_are_preserved(self):
        rows = [candidate([[0., .4], [.45, 1.]], .5,
                          clipped=[[.1, .4], [.45, .9]])]
        result = reader.select_existing(old_result(rows), np.empty((0, 3)), .1, {})
        np.testing.assert_array_equal(result["segments"], segments([[.1, .4], [.45, .9]]))
        self.assertEqual(result["components"], 2)
        self.assertEqual(result["bridged_gap_count"], 0)
        self.assertLess(reader.length(result["segments"]), reader.length(rows[0]["graph"]["segments"]))

    def test_actual_adapter_base_curve_equal_inputs_and_existing_tls_unchanged(self):
        context = measured_context()
        context_before = pickle.dumps(context, protocol=5)
        line = np.array([[[.015, -1., 3.], [.015, 1., 3.]]])
        before_line = line.copy()
        policy = dict(voxel_size=.02, origin=[0., 0., 0.])
        points = sample_segments(line, .01, 1000)
        before_points = points.copy()
        empty = np.empty((0, 2, 3))
        base = RasterBandReadout(points, context)(empty, policy)
        curve = RasterBandReadout(np.empty((0, 3)), context)(line, policy)
        self.assertEqual(base["state"], "complete")
        self.assertGreater(base["components"], 0)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["selected_candidate"], curve["selected_candidate"])
        self.assertEqual(canonical_hash(base["readouts"]), canonical_hash(curve["readouts"]))
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])
        self.assertTrue(base["base_curve_sampling_and_ranking_identical"])
        self.assertFalse(base["axis_refitted"])
        self.assertFalse(base["image_constructed_geometry"])
        model = base["readouts"][base["selected_candidate"]]["graph"]["model"]
        axis = np.asarray(model["direction"])
        residual = np.cross(base["segments"] - model["centroid"], axis)
        np.testing.assert_allclose(residual, 0., atol=1e-12)
        np.testing.assert_array_equal(points, before_points)
        np.testing.assert_array_equal(line, before_line)
        self.assertEqual(pickle.dumps(context, protocol=5), context_before)
        self.assertEqual(base["raster_band_policy_sha256"], canonical_hash(reader.POLICY))
        self.assertEqual(base["extra_annotations"], 0)


if __name__ == "__main__":
    unittest.main()
