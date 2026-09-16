"""Known epipolar answers independent of feature detection."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.camera_diagnostics import fundamental_from_cameras, sampson_distances


class CameraBackgroundTests(unittest.TestCase):
    def test_known_parallel_camera_correspondence_and_pixel_error(self):
        k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
        first, second = np.eye(4), np.eye(4)
        second[0, 3] = -2
        f = fundamental_from_cameras(k, first, k, second)
        # z=10, x=1, y=2 gives (60,70) and (40,70); shift y of the latter by 6.
        result = sampson_distances([[60, 70], [60, 70]], [[40, 70], [40, 76]], f)
        np.testing.assert_allclose(result, [0, 6/np.sqrt(2)], atol=1e-12)

    def test_global_rigid_world_change_does_not_change_epipolar_geometry(self):
        k = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
        first, second = np.eye(4), np.eye(4)
        second[0, 3] = -2
        world = np.array([[0., -1, 0, 4], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        np.testing.assert_allclose(
            fundamental_from_cameras(k, first, k, second),
            fundamental_from_cameras(k, first@world, k, second@world), atol=1e-12
        )

    def test_zero_baseline_is_not_perfect_consistency(self):
        with self.assertRaises(ValueError):
            fundamental_from_cameras(np.eye(3), np.eye(4), np.eye(3), np.eye(4))


if __name__ == "__main__":
    unittest.main()
