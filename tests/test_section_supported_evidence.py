"""Independent identification/prediction/support contracts; no formal pack IO."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import section_evidence as frozen  # noqa: E402
from creator_eval.section_supported_evidence import (  # noqa: E402
    DEFAULTS,
    _training_identifiability,
    analyze,
    policy,
)


def half_shell(*, missing=False, gap=False, validation_offset=0.):
    z = np.arange(91)*.017
    if gap:
        z = z[(z <= .51) | (z >= .85)]
    angles = np.deg2rad(np.arange(0, 181, 10))
    if missing:
        angles = angles[(angles < np.deg2rad(80)) | (angles > np.deg2rad(120))]
    angle_degrees = np.round(np.rad2deg(angles)).astype(int)
    train_angles = ((angle_degrees <= 30) | ((angle_degrees >= 80) & (angle_degrees <= 110)) | (angle_degrees >= 160))
    # Locally each fold has an angular hole, while across other axial windows
    # both folds establish the same connected training transverse component.
    alternate = np.floor(z/(1.53/6)).astype(int) % 2 == 1
    mask = np.where(alternate[:, None], ~train_angles[None], train_angles[None]).ravel()
    points = np.c_[np.tile(.024*np.cos(angles), len(z)), np.tile(.024*np.sin(angles), len(z)), np.repeat(z, len(angles))]
    if validation_offset:
        points[~mask, :2] *= 1+validation_offset
    return points, mask


class SectionSupportedEvidenceTests(unittest.TestCase):
    def test_both_folds_have_holes_but_actual_half_shell_is_supported(self):
        points, mask = half_shell()
        p = policy({"voxel_size": .006})
        for fold in (mask, ~mask):
            evidence = frozen._angular_evidence(points[fold & (points[:, 2] < .2), :2], np.zeros(2), p)
            self.assertGreater(evidence["maximum_internal_gap_degrees"], 30.)
        segments, consumed, detail = analyze(points, mask, p)
        self.assertEqual(len(segments), 1)
        self.assertTrue(consumed.all())
        np.testing.assert_allclose(segments[:, :, :2], 0., atol=1e-8)
        for run in detail["groups"][0]["runs"]:
            for section in run["slices"]:
                self.assertTrue(section["training_identifiability"]["accepted"])
                self.assertTrue(section["validation_prediction"]["accepted"])
                self.assertLess(section["maximum_internal_gap_degrees"], 11.)

    def test_true_missing_angular_support_is_not_repaired_by_combining_folds(self):
        points, mask = half_shell(missing=True)
        segments, _, _ = analyze(points, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)

    def test_shallow_training_arc_has_poor_parameter_conditioning(self):
        angles = np.deg2rad(np.linspace(0, 20, 9))
        xy = np.c_[np.cos(angles), np.sin(angles)]
        circle = dict(inside=np.ones(len(xy), dtype=bool), center=np.zeros(2), radius=1.)
        evidence = _training_identifiability(xy, circle, policy({}))
        self.assertFalse(evidence["accepted"])

    def test_heldout_residual_failure_cannot_move_model_or_window(self):
        points, mask = half_shell()
        _, _, original = analyze(points, mask, {"voxel_size": .006})
        moved, same_mask = half_shell(validation_offset=.2)
        segments, _, changed = analyze(moved, same_mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)
        first, second = original["groups"][0], changed["groups"][0]
        for key in ("trained_center", "trained_axis", "trained_radius", "training_sampling_evidence"):
            self.assertEqual(first[key], second[key])
        self.assertEqual([r["training_window_edges"] for r in first["training_runs"]],
                         [r["training_window_edges"] for r in second["training_runs"]])

    def test_heldout_endpoint_rings_and_missing_training_layers_keep_actual_extent(self):
        for missing_layers in ((45,), (44, 45, 46)):
            with self.subTest(missing_layers=missing_layers):
                points, mask = half_shell()
                layers = np.rint(points[:, 2]/.017).astype(int)
                mask[np.isin(layers, (0, 90, *missing_layers))] = False
                segments, _, detail = analyze(points, mask, {"voxel_size": .006})
                self.assertEqual(len(segments), 1)
                np.testing.assert_allclose(np.sort(segments[0, :, 2]), [0., 1.53], atol=1e-8)
                group = detail["groups"][0]
                self.assertEqual(group["sampling_evidence"]["observed_run_count"], 1)
                expected_training_runs = 1 if len(missing_layers) == 1 else 2
                self.assertEqual(group["training_sampling_evidence"]["observed_run_count"], expected_training_runs)
                self.assertEqual(len(group["runs"][0]["training_run_indices"]), expected_training_runs)
                self.assertTrue(all(run["accepted"] for run in group["training_runs"]))

    def test_single_heldout_endpoint_ray_cannot_extend_finite_axis(self):
        points, mask = half_shell()
        extra = np.array([[.024, 0., 1.547]])
        segments, _, detail = analyze(np.r_[points, extra], np.r_[mask, False], {"voxel_size": .006})
        self.assertEqual(len(segments), 0)
        self.assertEqual(detail["groups"][0]["runs"][0]["reason"], "added_boundary_support_rejected")

    def test_axial_gap_remains_two_runs(self):
        points, mask = half_shell(gap=True)
        segments, _, detail = analyze(points, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(segment[:, 2].max() <= .511 or segment[:, 2].min() >= .849 for segment in segments))
        self.assertEqual(detail["groups"][0]["sampling_evidence"]["observed_run_count"], 2)

    def test_plane_and_rotating_sparse_arc_remain_unsupported(self):
        x, z = np.meshgrid(np.linspace(-.06, .06, 25), np.arange(91)*.017)
        plane = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
        segments, consumed, _ = analyze(plane, np.arange(len(plane)) % 2 == 0, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)
        self.assertFalse(consumed.any())
        z = np.arange(91)*.017
        angles = np.linspace(-np.pi/6, np.pi/6, 4)[None]+2*np.pi*z[:, None]/z.max()
        arc = np.c_[.024*np.cos(angles).ravel(), .024*np.sin(angles).ravel(), np.repeat(z, 4)]
        segments, consumed, detail = analyze(arc, np.arange(len(arc)) % 3 != 0, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)
        self.assertTrue(consumed.all())
        self.assertEqual(detail["unresolved_surface_groups"], 1)

    def test_one_validation_ray_is_not_sufficient_prediction_evidence(self):
        points, _ = half_shell()
        angles = np.mod(np.rad2deg(np.arctan2(points[:, 1], points[:, 0])), 360.)
        mask = np.abs(angles-90.) > .01
        segments, _, _ = analyze(points, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)

    def test_validation_extending_the_scene_cannot_expand_training_windows(self):
        points, mask = half_shell()
        _, _, first = analyze(points, mask, {"voxel_size": .006})
        extra = points[~mask].copy()
        extra[:, 2] += 2.
        segments, _, second = analyze(np.r_[points, extra], np.r_[mask, np.zeros(len(extra), dtype=bool)], {"voxel_size": .006})
        self.assertEqual([r["training_window_edges"] for r in first["groups"][0]["training_runs"]],
                         [r["training_window_edges"] for r in second["groups"][0]["training_runs"]])
        self.assertEqual([r["measurement_envelope"] for r in first["groups"][0]["training_runs"]],
                         [r["measurement_envelope"] for r in second["groups"][0]["training_runs"]])
        self.assertEqual(len(segments), 1)
        self.assertLessEqual(segments[:, :, 2].max(), 1.53+1e-8)
        self.assertEqual(second["groups"][0]["runs"][-1]["reason"], "no_training_section_support")

    def test_existing_curvature_and_actual_support_thresholds_are_unchanged(self):
        for key, value in frozen.DEFAULTS.items():
            self.assertEqual(DEFAULTS[key], value)
        points, mask = half_shell()
        _, _, result = analyze(points, mask, {"section_maximum_points": 2})
        self.assertEqual(result["state"], "unmeasurable")
        for config in ({"section_supported_bad": 1}, {"section_supported_maximum_training_condition": float("nan")},
                       {"section_supported_minimum_validation_bins": 2}, {"section_supported_minimum_validation_bins": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                policy(config)


if __name__ == "__main__":
    unittest.main()
