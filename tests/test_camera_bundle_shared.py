"""Shared-lens assumptions and training-only trimming do not borrow GT labels."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from test_camera_bundle import scene

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.camera_bundle_shared import (
    exceeds_initial_bounds,
    shared_square_intrinsics,
    trim_training,
)


class SharedBundleTests(unittest.TestCase):
    def test_shared_focal_preserves_source_and_principal_points(self):
        k, _, _, _ = scene()
        k[1, 0, 0] *= 1.1
        original = k.copy()
        output = shared_square_intrinsics(k)
        np.testing.assert_array_equal(k, original)
        np.testing.assert_array_equal(output[:, :2, 2], k[:, :2, 2])
        self.assertAlmostEqual(float(output[:, 0, 0].std()), 0)
        np.testing.assert_array_equal(output[:, 0, 0], output[:, 1, 1])

    def test_training_trim_keeps_good_and_removes_corrupted_observation_track(self):
        k, e, xyz, tracks = scene()
        changed = copy.deepcopy(tracks)
        changed[0]["observations"][2]["xy"][0] += 25
        result = dict(intrinsics=k, extrinsics=e, points=xyz, kept_track_ids=list(range(len(tracks))))
        kept, report = trim_training(changed, result, 4., 16)
        self.assertEqual(report["removed_track_ids"], [0])
        self.assertEqual(len(kept), len(tracks) - 1)
        self.assertTrue(report["enough_coverage"])

    def test_second_stage_cannot_hide_motion_beyond_original_bounds(self):
        k, e, _, _ = scene()
        result = dict(extrinsics=e.copy(), focal_scale=1.)
        config = dict(rotation_bound_rad=.26, center_bound_baseline_fraction=.3, focal_scale_bounds=[.7, 1.3])
        self.assertFalse(exceeds_initial_bounds(k, e, result, config))
        result["extrinsics"][1, 0, 3] += 2
        self.assertTrue(exceeds_initial_bounds(k, e, result, config))


if __name__ == "__main__":
    unittest.main()
