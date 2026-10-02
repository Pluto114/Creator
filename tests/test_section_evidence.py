"""Independent analytic section contracts, not the scored 44-condition pack."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.section_evidence import DEFAULTS, analyze  # noqa: E402


def surface(radius=.03, *, half=False, offset=0., gap=False, count=40, rotate=False, center=False):
    z = np.linspace(0., 1.5, 301)
    if gap:
        z = z[(z <= .55) | (z >= .95)]
    angles = np.linspace(0., np.pi if half else 2*np.pi, count, endpoint=half)
    if rotate:
        angles = np.linspace(-np.pi/6, np.pi/6, count)[None]+2*np.pi*z[:, None]/1.5
    else:
        angles = np.broadcast_to(angles, (len(z), count))
    points = np.c_[offset+radius*np.cos(angles).ravel(), radius*np.sin(angles).ravel(), np.repeat(z, count)]
    mask = np.indices((len(z), count)).sum(axis=0).ravel() % 2 == 0
    if center:
        points = np.r_[points, np.c_[np.full(len(z), offset), np.zeros(len(z)), z]]
        mask = np.r_[mask, np.arange(len(z)) % 2 == 0]
    return points, mask


class SectionEvidenceTests(unittest.TestCase):
    def check_axis(self, segments, x, count=1):
        self.assertEqual(len(segments), count)
        np.testing.assert_allclose(segments[:, :, 0], x, atol=1e-8)
        np.testing.assert_allclose(segments[:, :, 1], 0., atol=1e-8)

    def test_half_circle_requires_real_curvature_and_dense_local_arc(self):
        points, mask = surface(half=True, count=41)
        segments, consumed, diagnostic = analyze(points, mask, {"voxel_size": .008})
        self.check_axis(segments, 0.)
        self.assertTrue(consumed.all())
        self.assertEqual(diagnostic["accepted_surface_groups"], 1)

    def test_plane_does_not_become_a_circle(self):
        x, z = np.meshgrid(np.linspace(-.06, .06, 31), np.linspace(0, 1.5, 201))
        points = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
        segments, consumed, diagnostic = analyze(points, np.arange(len(points)) % 3 != 0, {"voxel_size": .008})
        self.assertEqual(len(segments), 0)
        self.assertFalse(consumed.any())
        self.assertEqual(diagnostic["accepted_surface_groups"], 0)

    def test_sparse_motherlines_supply_no_accepted_circle(self):
        for half in (False, True):
            with self.subTest(half=half):
                points, mask = surface(half=half, count=4)
                segments, _, diagnostic = analyze(points, mask, {"voxel_size": .008})
                self.assertEqual(len(segments), 0)
                self.assertEqual(diagnostic["accepted_surface_groups"], 0)

    def test_rotating_sparse_arc_is_consumed_as_unresolved_not_ridge_evidence(self):
        points, mask = surface(count=4, rotate=True)
        segments, consumed, diagnostic = analyze(points, mask, {"voxel_size": .008})
        self.assertEqual(len(segments), 0)
        self.assertTrue(consumed.all())
        self.assertEqual(diagnostic["unresolved_surface_groups"], 1)

    def test_two_small_shells_and_observed_centers_have_two_separate_axes(self):
        first, fm = surface(radius=.003, offset=-.02, center=True)
        second, sm = surface(radius=.003, offset=.02, center=True)
        segments, consumed, diagnostic = analyze(np.r_[first, second], np.r_[fm, sm], {"voxel_size": .01})
        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(np.sort(segments.mean(axis=1)[:, 0]), [-.02, .02], atol=1e-8)
        self.assertTrue(consumed.all())
        self.assertEqual(diagnostic["accepted_surface_groups"], 2)

    def test_pure_line_is_not_consumed_as_a_surface(self):
        points = np.c_[np.zeros(301), np.zeros(301), np.linspace(0, 1.5, 301)]
        segments, consumed, _ = analyze(points, np.arange(len(points)) % 2 == 0, {"voxel_size": .008})
        self.assertEqual(len(segments), 0)
        self.assertFalse(consumed.any())

    def test_observed_gap_is_not_closed_by_a_circle_model(self):
        points, mask = surface(gap=True)
        segments, _, _ = analyze(points, mask, {"voxel_size": .008})
        self.check_axis(segments, 0., count=2)
        self.assertTrue(all(np.max(segment[:, 2]) <= .56 or np.min(segment[:, 2]) >= .94 for segment in segments))

    def test_validation_cannot_move_trained_circle_geometry(self):
        points, mask = surface()
        _, _, first = analyze(points, mask, {"voxel_size": .008})
        changed = points.copy()
        changed[~mask, :2] += [.001, -.001]
        _, _, second = analyze(changed, mask, {"voxel_size": .008})
        for key in ("trained_center", "trained_axis", "trained_radius"):
            self.assertEqual(first["groups"][0][key], second["groups"][0][key])

    def test_radial_outliers_are_not_silently_consumed(self):
        points, mask = surface()
        branch = np.c_[np.linspace(.04, .1, 101), np.zeros(101), np.full(101, .75)]
        _, consumed, _ = analyze(np.r_[points, branch], np.r_[mask, np.arange(len(branch)) % 2 == 0], {"voxel_size": .008})
        self.assertFalse(consumed[-len(branch):].any())

    def test_budgets_and_invalid_policy_fail_explicitly(self):
        points, mask = surface()
        segments, consumed, diagnostic = analyze(points, mask, {"section_maximum_points": 2})
        self.assertEqual(diagnostic["state"], "unmeasurable")
        self.assertEqual(len(segments), 0)
        self.assertFalse(consumed.any())
        for policy in ({"section_bad": 1}, {"section_slices": 0}, {"voxel_size": 0}, {"section_cell_voxels": .01}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                analyze(points, mask, policy)
        self.assertLess(DEFAULTS["section_circle_line_rmse_ratio"], 1)


if __name__ == "__main__":
    unittest.main()
