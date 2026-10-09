"""Actual geometry, identical base/curve policy and explicit observation costs."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_required_stroke_axis_readout import RequiredStrokeAxisReadout  # noqa: E402
from creator_eval.rgb_required_view_support import required_stroke_support_mask  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_rgb_stroke_chain_support import marks  # noqa: E402


class RequiredStrokeReadoutContracts(unittest.TestCase):
    def test_output_stays_on_actual_3d_axis_not_annotation_axis(self):
        frames, cameras = fixture()
        context = stroke.build_context(frames, cameras, marks(frames))
        points = np.column_stack((np.full(200, .006), np.linspace(-.4, 1., 200), np.full(200, 3.)))
        reader = RequiredStrokeAxisReadout(points, context)
        graph = reader(np.empty((0, 2, 3)), dict(voxel_size=.01, origin=[0., 0., 0.]))
        self.assertGreater(len(graph["segments"]), 0)
        np.testing.assert_allclose(np.asarray(graph["segments"])[:, :, 0], .006, atol=1e-12)
        self.assertEqual(graph["supported_base_point_count"], int(reader.mask.sum()))
        self.assertTrue(graph["required_support_views_applied"])
        self.assertEqual(graph["required_support_view_ids"], ["0", "3"])
        self.assertFalse(graph["target_identity_confirmed"])
        self.assertFalse(graph["stroke_interpolation_performed"])
        self.assertEqual(graph["anchor_point_count"], 4)

    def test_base_and_curve_samples_share_same_required_mask(self):
        frames, cameras = fixture()
        context = stroke.build_context(frames, cameras, marks(frames))
        segments = np.array([[[.006, -.4, 3.], [.006, 1., 3.]], [[.5, -.4, 3.], [.5, 1., 3.]]])
        samples = sample_segments(segments, .005, 2000000)
        reader = RequiredStrokeAxisReadout(samples, context)
        graph = reader(segments, dict(voxel_size=.01, origin=[0., 0., 0.]))
        self.assertEqual(graph["full_curve_sample_count"], len(samples))
        self.assertEqual(graph["supported_curve_sample_count"], int(reader.mask.sum()))
        self.assertEqual(graph["supported_base_point_count"], graph["supported_curve_sample_count"])
        np.testing.assert_array_equal(reader.mask, required_stroke_support_mask(samples, context))


if __name__ == "__main__":
    unittest.main()
