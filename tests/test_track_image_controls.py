"""Independent projection answers for the synthetic feature positive controls."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.track_image_controls import (
    cast_planes,
    control_camera,
    physical_pair,
    project,
    render_planes,
)


class TrackImageControlTests(unittest.TestCase):
    def test_parallel_camera_ray_hits_nearest_plane_and_roundtrips(self):
        camera = dict(K_index=[[100., 0, 50], [0, 100, 50], [0, 0, 1]],
                      world_to_camera_cv=[[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], size_wh=[100, 100])
        planes = [dict(x0=-2, x1=2, y=y, z0=-2, z1=2) for y in (5., 3.)]
        world, depth, label = cast_planes(np.array([[50., 50], [60, 40]]), camera, planes)
        np.testing.assert_allclose(world, [[0, 3, 0], [.3, 3, .3]])
        np.testing.assert_array_equal(label, [1, 1])
        np.testing.assert_allclose(depth, [3, 3])
        np.testing.assert_allclose(project(world, camera)[0], [[50, 50], [60, 40]])

    def test_projected_correspondence_correct_and_displaced_match_wrong(self):
        cameras = [control_camera(a, [640, 480]) for a in (-11, 7)]
        planes = [dict(x0=-3, x1=3, y=1.35, z0=-2.4, z1=2.4)]
        xy = [project([[0, 1.35, 0], [.4, 1.35, .5]], c)[0] for c in cameras]
        np.testing.assert_array_equal(physical_pair(*xy, *cameras, planes), ["correct", "correct"])
        np.testing.assert_array_equal(physical_pair(xy[0], xy[1] + [12, 0], *cameras, planes), ["wrong", "wrong"])

    def test_render_deterministic_and_background_unknown(self):
        camera = control_camera(0, [160, 120])
        planes = [dict(x0=-.5, x1=.5, y=1.35, z0=-.5, z1=.5)]
        a = render_planes(camera, planes, False, 123)
        np.testing.assert_array_equal(a, render_planes(camera, planes, False, 123))
        self.assertEqual(a.shape, (120, 160, 3))
        self.assertGreater(int(a.max()), 28)
        labels = physical_pair(np.array([[0., 0.]]), np.array([[0., 0.]]), camera, camera, planes)
        np.testing.assert_array_equal(labels, ["indeterminate"])


if __name__ == "__main__":
    unittest.main()
