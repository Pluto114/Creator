"""Independent local-section geometry contracts; no fixture/formal-pack IO."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import section_supported_evidence as prior  # noqa: E402
from creator_eval.section_local_evidence import DEFAULTS, analyze, policy  # noqa: E402


def shell(*, fragmented=False, gap=False, coverage=180., turns=0.):
    z = np.arange(123)*.011
    if gap:
        z = z[(z < .44) | (z > .77)]
    base = np.deg2rad(np.arange(0, coverage+1, 10.))
    angles = base[None]+turns*2*np.pi*z[:, None]/1.342
    points = np.c_[.023*np.cos(angles).ravel(), .023*np.sin(angles).ravel(), np.repeat(z, len(base))]
    # Alternate angular ownership across layers, so the training transverse
    # component is connected without allowing validation to join components.
    mask = ((np.arange(len(base))[None]+np.arange(len(z))[:, None]) % 3 != 1).ravel()
    if fragmented:
        layers = np.rint(points[:, 2]/.011).astype(int)
        mask[(layers % 6 >= 3) | (layers == 0) | (layers == 122)] = False
    return points, mask


class SectionLocalEvidenceTests(unittest.TestCase):
    def test_real_half_shell_reports_actual_finite_endpoints(self):
        points, mask = shell()
        segments, consumed, detail = analyze(points, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(np.sort(segments[0, :, 2]), [0., 1.342], atol=1e-8)
        np.testing.assert_allclose(segments[0, :, :2], 0., atol=1e-8)
        self.assertTrue(consumed.all())
        self.assertEqual(detail["accepted_surface_groups"], 1)

    def test_short_training_fragments_do_not_veto_complete_observed_axis(self):
        points, mask = shell(fragmented=True)
        segments, _, detail = analyze(points, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(np.sort(segments[0, :, 2]), [0., 1.342], atol=1e-8)
        group = detail["groups"][0]
        self.assertEqual(group["sampling_evidence"]["observed_run_count"], 1)
        self.assertTrue(all(item["accepted"] for item in group["training_contexts"]))
        self.assertTrue(all(item["accepted"] for item in group["runs"][0]["slices"]))

    def test_each_rotating_one_hundred_degree_arc_stays_unresolved(self):
        for turns in (1., 4., 11.):
            with self.subTest(turns=turns):
                points, mask = shell(coverage=100., turns=turns)
                segments, consumed, detail = analyze(points, mask, {"voxel_size": .006})
                self.assertEqual(len(segments), 0)
                self.assertTrue(consumed.all())
                self.assertEqual(detail["unresolved_surface_groups"], 1)
                slices = [item for run in detail["groups"][0]["runs"] for item in run["slices"]]
                self.assertTrue(slices)
                self.assertTrue(all(item["coverage_degrees"] < 150. for item in slices))

    def test_long_actual_gap_is_not_filled_by_context_overlap(self):
        points, mask = shell(gap=True)
        segments, _, detail = analyze(points, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(segment[:, 2].max() <= .44+1e-8 or segment[:, 2].min() >= .77-1e-8 for segment in segments))
        self.assertEqual(detail["groups"][0]["sampling_evidence"]["observed_run_count"], 2)

    def test_full_heldout_endpoint_ring_cannot_lend_angles_to_following_point(self):
        points, mask = shell()
        angles = np.deg2rad(np.arange(0, 181, 10))
        ring = np.c_[.023*np.cos(angles), .023*np.sin(angles), np.full(len(angles), 1.353)]
        extra = np.r_[ring, [[.023, 0., 1.364]]]
        segments, _, detail = analyze(np.r_[points, extra], np.r_[mask, np.zeros(len(extra), dtype=bool)], {"voxel_size": .006})
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(float(segments[:, :, 2].max()), 1.353, places=8)
        slices = detail["groups"][0]["runs"][0]["slices"]
        self.assertTrue(slices[-2]["accepted"])
        self.assertFalse(slices[-1]["accepted"])

    def test_validation_cannot_change_training_contexts_models_or_scale(self):
        points, mask = shell()
        _, _, first = analyze(points, mask, {"voxel_size": .006})
        moved = points.copy()
        moved[~mask, :2] *= 1.2
        segments, _, second = analyze(moved, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)
        left, right = first["groups"][0], second["groups"][0]
        for key in ("trained_center", "trained_axis", "trained_radius", "local_evidence"):
            self.assertEqual(left[key], right[key])
        for key in ("training_layer_positions_m", "training_intervals_m", "typical_intervals_m", "maximum_allowed_gaps_m"):
            self.assertEqual(left["sampling_evidence"][key], right["sampling_evidence"][key])
        for a, b in zip(left["training_contexts"], right["training_contexts"]):
            for key in ("center", "window", "training_points", "training_identifiability", "center_drift", "radius_drift"):
                self.assertEqual(a[key], b[key])

    def test_remote_validation_support_cannot_move_finite_core(self):
        points, mask = shell()
        extra = points.copy()
        extra[:, 2] += 3.
        segments, _, detail = analyze(np.r_[points, extra], np.r_[mask, np.zeros(len(extra), dtype=bool)], {"voxel_size": .006})
        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(np.sort(segments[0, :, 2]), [0., 1.342], atol=1e-8)
        self.assertFalse(detail["groups"][0]["runs"][-1]["accepted"])

    def test_plane_and_four_motherlines_are_not_axes(self):
        points, mask = shell()
        plane = points.copy()
        plane[:, 1] = 0.
        segments, consumed, _ = analyze(plane, mask, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)
        self.assertFalse(consumed.any())
        z = np.arange(123)*.011
        angles = np.arange(4)*np.pi/2
        lines = np.c_[np.tile(.023*np.cos(angles), len(z)), np.tile(.023*np.sin(angles), len(z)), np.repeat(z, 4)]
        segments, _, _ = analyze(lines, np.arange(len(lines)) % 3 != 0, {"voxel_size": .006})
        self.assertEqual(len(segments), 0)

    def test_original_gates_are_not_relaxed_and_budgets_are_explicit(self):
        for key, value in prior.DEFAULTS.items():
            self.assertEqual(DEFAULTS[key], value)
        points, mask = shell()
        for config in ({"section_maximum_points": 2}, {"section_local_maximum_contexts": 1}, {"section_local_maximum_slabs": 1}):
            with self.subTest(config=config):
                segments, _, detail = analyze(points, mask, {"voxel_size": .006, **config})
                self.assertEqual(len(segments), 0)
                self.assertEqual(detail["state"], "unmeasurable")
        for config in ({"section_local_unknown": 1}, {"section_local_resolution_voxels": 2},
                       {"section_local_maximum_slabs": True}, {"section_local_context_radius_radii": float("nan")}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                policy(config)


if __name__ == "__main__":
    unittest.main()
