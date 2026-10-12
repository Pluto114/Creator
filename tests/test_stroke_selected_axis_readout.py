"""Six mechanisms for one stroke-selected local axis, not general topology."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval import stroke_selected_axis_readout as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.line_controls import fit_line_tls  # noqa: E402
from creator_eval.rgb_chain_support import chain_support_mask  # noqa: E402
from creator_eval.rgb_stroke_selected_axis_readout import StrokeSelectedAxisReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402


def measured_context(*, nested=False):
    """Six genuine in-band synthetic claims: three measured rows, two views."""
    frames, cameras = fixture((0., 0.) if nested else (0.,))
    if nested:
        for frame in frames:
            for row in frame["observations"]["rows"]:
                wide = row["candidates"][0]
                wide["left_edge"]["x"] = wide["center_x"] - 4
                wide["right_edge"]["x"] = wide["center_x"] + 4
                wide["width"] = 8.
            frame["observation_sha256"] = canonical_hash(frame["observations"])
    claims = []
    for frame in (frames[0], frames[3]):
        for y in (80, 100, 120):
            row = next(row for row in frame["observations"]["rows"] if row["y"] == y)
            measured = row["candidates"][-1]
            x = (measured["left_edge"]["x"] + measured["right_edge"]["x"]) / 2 + .4
            claims.append(dict(view_id=frame["view_id"], rgb_sha256=frame["rgb_sha256"],
                               xy=[x, float(y)], uncertainty_xy_px=[.2, .2]))
    return stroke.build_context(frames, cameras, claims)


def vertical_points(x, ys):
    ys = np.asarray(ys, float)
    return np.c_[np.full(len(ys), x), ys, np.full(len(ys), 3.)]


class StrokeSelectedAxisMechanisms(unittest.TestCase):
    def test_narrowest_measured_band_not_click_center(self):
        context = measured_context(nested=True)
        bands = reader.foreground_bands(context)
        self.assertEqual(len(bands), 6)
        self.assertEqual({band["row"] for band in bands}, {80, 100, 120})
        clicks = {(a["view_id"], int(a["xy"][1])): a["xy"][0]
                  for a in context["association"]["anchors"]}
        for band in bands:
            center = 130. if band["view_id"] == "0" else 70.
            self.assertEqual((band["left"], band["right"]), (center - 1, center + 1))
            self.assertAlmostEqual(clicks[band["view_id"], band["row"]], center + .4)
            self.assertNotEqual((band["left"] + band["right"]) / 2,
                                clicks[band["view_id"], band["row"]])

    def test_image_ranking_does_not_move_actual_tls_geometry(self):
        context = measured_context()
        points = vertical_points(.015, np.linspace(-1., 1., 201))
        before = points.copy()
        result = reader.readout(points, .02, [0., 0., 0.], context)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["components"], 1)
        np.testing.assert_allclose(result["segments"][..., 0], .015, atol=1e-12)
        np.testing.assert_allclose(result["segments"][..., 2], 3., atol=1e-12)
        model = result["readouts"][result["selected_candidate"]]["graph"]["model"]
        model_before = canonical_hash(model)
        reader.projected_band_score(model, reader.foreground_bands(context))
        self.assertEqual(canonical_hash(model), model_before)
        np.testing.assert_array_equal(points, before)
        self.assertFalse(result["image_constructed_geometry"])
        self.assertFalse(result["preserves_arbitrary_multiple_axes"])

    def test_low_support_foreground_scores_above_dense_wrong_axis(self):
        bands = reader.foreground_bands(measured_context())
        foreground = vertical_points(0., np.linspace(-1., 1., 17))
        wrong = vertical_points(.045, np.linspace(-1., 1., 161))
        self.assertGreater(len(wrong), 9 * len(foreground))
        good_score = reader.projected_band_score(fit_line_tls(foreground), bands)
        bad_score = reader.projected_band_score(fit_line_tls(wrong), bands)
        # fx=100, depth=3: the actual wrong axis is 1.5px from a 1px half-band.
        self.assertAlmostEqual(good_score["maximum_band_excess_px"], 0.)
        self.assertAlmostEqual(good_score["worst_view_rms"], 0.)
        self.assertAlmostEqual(bad_score["maximum_band_excess_px"], .5)
        self.assertAlmostEqual(bad_score["worst_view_rms"], 1.5)
        keys = ("maximum_band_excess_px", "band_excess_rms_px")
        self.assertLess(tuple(good_score[key] for key in keys),
                        tuple(bad_score[key] for key in keys))

    def test_duplicate_and_reordered_coordinates_keep_selection(self):
        context = measured_context()
        points = vertical_points(.015, np.linspace(-1., 1., 121))
        first = reader.readout(points, .02, [0., 0., 0.], context)
        second = reader.readout(np.concatenate((points[::-1], points[::3], points)),
                                .02, [0., 0., 0.], context)
        self.assertGreater(first["components"], 0)
        np.testing.assert_array_equal(first["segments"], second["segments"])
        self.assertEqual(first["selected_candidate"], second["selected_candidate"])
        self.assertEqual(first["selected_band_score"], second["selected_band_score"])
        self.assertEqual(canonical_hash(first["readouts"]), canonical_hash(second["readouts"]))

    def test_actual_gap_survives_positive_rgb_across_gap(self):
        context = measured_context()
        self.assertTrue(chain_support_mask([[0., 0., 3.]], context)[0])
        points = vertical_points(0., np.r_[np.linspace(-1., -.3, 71), np.linspace(.3, 1., 71)])
        result = reader.readout(points, .02, [0., 0., 0.], context)
        self.assertEqual(result["components"], 2)
        intervals = np.sort(result["segments"][..., 1], axis=1)
        negative = intervals[intervals[:, 1] < 0]
        positive = intervals[intervals[:, 0] > 0]
        self.assertEqual((len(negative), len(positive)), (1, 1))
        self.assertLess(negative[0, 1], -.25)
        self.assertGreater(positive[0, 0], .25)

    def test_same_actual_coordinates_from_base_or_curve_are_identical(self):
        context = measured_context()
        segments = np.array([[[.015, -1., 3.], [.015, 1., 3.]]])
        policy = dict(voxel_size=.02, origin=[0., 0., 0.])
        samples = sample_segments(segments, .01, 1000)
        base = StrokeSelectedAxisReadout(samples, context)(np.empty((0, 2, 3)), policy)
        curve = StrokeSelectedAxisReadout(np.empty((0, 3)), context)(segments, policy)
        self.assertGreater(base["components"], 0)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["selected_candidate"], curve["selected_candidate"])
        self.assertEqual(canonical_hash(base["readouts"]), canonical_hash(curve["readouts"]))
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])
        self.assertTrue(base["base_curve_sampling_and_ranking_identical"])
        self.assertTrue(curve["base_curve_sampling_and_ranking_identical"])


if __name__ == "__main__":
    unittest.main()
