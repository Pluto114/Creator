"""Sparse-stroke adapter preserves geometry and labels added input cost."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rgb_chain_axis_readout import ChainAxisReadout  # noqa: E402
from creator_eval.rgb_stroke_chain_axis_readout import StrokeChainAxisReadout  # noqa: E402
from creator_eval.rgb_stroke_chain_support import build_context  # noqa: E402
from test_rgb_anchored_chain_support import anchors, fixture  # noqa: E402


class StrokeAdapterContracts(unittest.TestCase):
    def test_only_metadata_changes_and_actual_offset_preserved(self):
        frames, cameras = fixture()
        marks = [dict(a, xy=[a["xy"][0], y]) for a in anchors(frames) for y in (90., 110.)]
        context = build_context(frames, cameras, marks)
        points = np.c_[np.full(101, .015), np.linspace(-1., 1., 101), np.full(101, 3.)]
        curves = np.empty((0, 2, 3))
        call = dict(voxel_size=.02, origin=[0., 0., 0.])
        raw = ChainAxisReadout(points, context)(curves, call)
        result = StrokeChainAxisReadout(points, context)(curves, call)
        for name, value in raw.items():
            if name != "measurement_scope":
                np.testing.assert_equal(result[name], value)
        np.testing.assert_allclose(result["segments"][..., 0], .015, atol=1e-12)
        self.assertEqual(result["support_domain_kind"], "stroke_conditioned_positive_only_not_all_possible")
        self.assertEqual(result["anchor_point_count"], 4)
        self.assertEqual(result["anchor_view_count"], 2)
        self.assertEqual(result["anchor_claims_sha256"], context["association"]["anchors_sha256"])
        self.assertFalse(result["stroke_interpolation_performed"])
        self.assertFalse(result["target_identity_confirmed"])


if __name__ == "__main__":
    unittest.main()
