"""Known geometry, gauge, withheld observations and corruption boundaries."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.camera_bundle import (
    CameraGauge,
    centers,
    correction_decision,
    corrupt_training,
    optimize_cameras,
    project,
    triangulate,
    validate_heldout,
    validation_plan,
)


def scene():
    k = np.tile([[260., 0, 160], [0, 260, 120], [0, 0, 1]], (4, 1, 1))
    c = np.array([[-1., 0, 0], [-.3, .1, .1], [.4, -.1, 0], [1, .2, .1]])
    e = np.concatenate([np.tile(np.eye(3), (4, 1, 1)), -c[:, :, None]], axis=2)
    rng = np.random.default_rng(321)
    xyz = rng.uniform([-1, -.7, 3], [1, .7, 5], (36, 3))
    xy = [project(xyz, a, b)[0] for a, b in zip(k, e)]
    tracks = [dict(track_id=i, observations=[dict(view=v, xy=xy[v][i].tolist()) for v in range(4)]) for i in range(len(xyz))]
    return k, e, xyz, tracks


class BundleTests(unittest.TestCase):
    def test_dlt_known_point_and_gauge_roundtrip(self):
        k, e, xyz, tracks = scene()
        np.testing.assert_allclose(triangulate(tracks[0]["observations"], k, e), xyz[0], atol=1e-12)
        gauge = CameraGauge(k, e, True)
        a, b = gauge.decode(np.zeros(gauge.parameter_count), native=True)
        np.testing.assert_allclose(a, k)
        np.testing.assert_allclose(b, e, atol=1e-12)
        params = np.full(gauge.parameter_count, .02)
        a, b = gauge.decode(params, native=True)
        np.testing.assert_allclose(b[0], e[0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(centers(b)[-1] - centers(b)[0]), np.linalg.norm(centers(e)[-1] - centers(e)[0]), places=12)

    def test_holdout_scores_unused_view_not_triangulation_views(self):
        k, e, _, tracks = scene()
        plans = validation_plan(tracks[:1], e)
        self.assertFalse({o["view"] for o in plans[0]["triangulation"]} & {o["view"] for o in plans[0]["scoring"]})
        expected = len(plans[0]["scoring"])
        self.assertEqual(validate_heldout(k, e, plans)["summary"]["sample_count"], expected)
        plans[0]["scoring"][0]["xy"][1] += 12
        errors = validate_heldout(k, e, plans)["rows"]
        self.assertAlmostEqual(errors[0]["error_px"], 12., places=9)

    def test_corruption_is_reproducible_and_never_mutates_source(self):
        _, _, _, tracks = scene()
        original = copy.deepcopy(tracks)
        a, changes = corrupt_training(tracks, .2, 12)
        b, repeated = corrupt_training(tracks, .2, 12)
        self.assertEqual(a, b)
        self.assertEqual(changes, repeated)
        self.assertEqual(tracks, original)
        self.assertEqual(len(changes), 7)

    def test_known_pose_error_reduces_on_unoptimized_tracks(self):
        k, truth, _, tracks = scene()
        e = truth.copy()
        e[1, :, :3] = Rotation.from_rotvec([.006, -.008, .004]).as_matrix()
        e[1, :, 3] = -e[1, :, :3] @ centers(truth)[1]
        plans = validation_plan(tracks[24:], e)
        before = validate_heldout(k, e, plans)["summary"]["median_px"]
        config = dict(minimum_initial_tracks=12, rotation_bound_rad=.26, center_bound_baseline_fraction=.3, focal_scale_bounds=[.7, 1.3], robust_scale_px=1.5, max_nfev=150, tolerance=1e-8)
        untouched = copy.deepcopy(tracks)
        result = optimize_cameras(k, e, tracks[:24], config)
        self.assertEqual(tracks, untouched)
        after = validate_heldout(result["intrinsics"], result["extrinsics"], plans)["summary"]["median_px"]
        self.assertLess(after, before * .1)
        self.assertLess(after, .02)
        np.testing.assert_allclose(result["extrinsics"][0], e[0], atol=1e-12)

    def test_empty_support_never_certifies_correction(self):
        summary = dict(sample_count=0, finite_count=0, median_px=None, p95_px=None, fraction_within_2px=None)
        result = correction_decision(dict(success=True, at_parameter_bound=False, negative_depth_observations=0),
            dict(summary=summary), dict(summary=summary), dict(minimum_scored_observations=24, maximum_median_ratio=.8, maximum_p95_px=3, minimum_fraction_within_2px=.8))
        self.assertEqual(result["state"], "withhold_correction")

    def test_zero_baseline_and_duplicate_view_are_rejected(self):
        k, e, _, tracks = scene()
        e[-1] = e[0]
        with self.assertRaises(ValueError):
            CameraGauge(k, e, False)
        with self.assertRaises(ValueError):
            triangulate([tracks[0]["observations"][0]] * 2, k, e)


if __name__ == "__main__":
    unittest.main()
