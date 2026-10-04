"""Generator-only local-support contracts; no reader invocation or output scoring."""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.section_support_controls import (  # noqa: E402
    AXIAL_STEP_M,
    CASE_IDS,
    COHORT,
    DEFAULT_SEED,
    DENSITY,
    GAP_LIMITS_M,
    GRID_SCALES_M,
    HELIX_TURNS,
    LENGTH_M,
    LOCAL_ARC_DEGREES,
    PHASES,
    PLANE_NOISE_BOUND_M,
    RADIAL_NOISE_BOUND_M,
    RADIUS_M,
    ROTATING_ARC_DEGREES,
    ROTATING_ARC_TURNS,
    ROTATION_DEGREES,
    SAMPLING_PHASES,
    generate,
    make_case,
)


def local(case, values):
    p = case["protocol"]
    return (values - p["local_to_world_translation"]) @ np.array(p["local_to_world_rotation"])


class SectionSupportControlTests(unittest.TestCase):
    def test_fixed_32_conditions_and_truth_roles(self):
        cases = list(generate())
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({case["condition_id"] for case in cases}), 32)
        self.assertEqual(Counter(case["case_id"] for case in cases), dict.fromkeys(CASE_IDS, 4))
        self.assertEqual(Counter(case["truth"]["expected"] for case in cases),
                         {"line": 12, "gap": 4, "unresolved": 12, "abstain": 4})
        self.assertEqual({case["cohort"] for case in cases}, {COHORT})

    def test_only_finite_surface_observations_enter_normal_inputs(self):
        for case in generate():
            with self.subTest(case=case["condition_id"]):
                self.assertEqual(set(case["inputs"]), {"points", "segments"})
                points, segments = case["inputs"].values()
                self.assertEqual(points.shape[1:], (3,))
                self.assertEqual(segments.shape, (0, 2, 3))
                self.assertTrue(np.isfinite(points).all())
                self.assertEqual(points.dtype, np.dtype("<f8"))
                self.assertEqual(segments.dtype, np.dtype("<f8"))

    def test_voxel_scale_does_not_change_the_point_cloud(self):
        for case_id in CASE_IDS:
            for phase in PHASES:
                a, b = [make_case(case_id, phase=phase, voxel_size=scale) for scale in GRID_SCALES_M]
                np.testing.assert_array_equal(a["inputs"]["points"], b["inputs"]["points"])
                np.testing.assert_array_equal(a["truth"]["expected_segments"], b["truth"]["expected_segments"])

    def test_sampling_phases_move_axial_observations_not_construction_truth(self):
        for case_id in CASE_IDS:
            a, b = [make_case(case_id, phase=phase) for phase in PHASES]
            az = np.unique(np.round(local(a, a["inputs"]["points"])[:, 2], 12))
            bz = np.unique(np.round(local(b, b["inputs"]["points"])[:, 2], 12))
            self.assertFalse(np.array_equal(az, bz))
            for key in ("expected_segments", "forbidden_segments", "construction_segments"):
                np.testing.assert_array_equal(a["truth"][key], b["truth"][key])

    def test_phased_lattice_uses_fixed_19mm_physical_step(self):
        for phase in PHASES:
            case = make_case("full_noisy_sections", phase=phase)
            z = np.unique(np.round(local(case, case["inputs"]["points"])[:, 2], 12))
            self.assertAlmostEqual(z[0], 0.0)
            self.assertAlmostEqual(z[-1], LENGTH_M)
            np.testing.assert_allclose(np.diff(z[1:-1]), AXIAL_STEP_M, atol=2e-12)
            np.testing.assert_allclose(z[1:-1] / AXIAL_STEP_M - SAMPLING_PHASES[phase],
                                       np.round(z[1:-1] / AXIAL_STEP_M - SAMPLING_PHASES[phase]), atol=1e-9)
            self.assertEqual(case["protocol"]["density"], DENSITY)
            self.assertEqual(case["protocol"]["sampling_phase"], SAMPLING_PHASES[phase])

    def test_new_world_pose_is_fixed_and_proper(self):
        case = make_case("full_noisy_sections")
        rotation = np.array(case["protocol"]["local_to_world_rotation"])
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        self.assertEqual(case["protocol"]["rotation_degrees"], list(ROTATION_DEGREES))
        self.assertNotIn(tuple(ROTATION_DEGREES), ((17.0, -23.0, 11.0), (-29.0, 34.0, 53.0), (38.0, -17.0, 71.0)))

    def test_noise_is_bounded_radial_and_does_not_insert_an_axis(self):
        for case_id in ("full_noisy_sections", "half_noisy_sections", "true_gap_half"):
            case = make_case(case_id)
            points = local(case, case["inputs"]["points"])
            deviations = np.linalg.norm(points[:, :2], axis=1) - RADIUS_M
            self.assertLessEqual(np.abs(deviations).max(), RADIAL_NOISE_BOUND_M + 1e-14)
            self.assertGreater(np.std(deviations), RADIAL_NOISE_BOUND_M / 4)
            self.assertGreater(np.linalg.norm(points[:, :2], axis=1).min(), 0.028)

    def test_half_surface_keeps_true_180_degree_support(self):
        case = make_case("half_noisy_sections")
        points = local(case, case["inputs"]["points"]).reshape(-1, 16, 3)
        angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
        np.testing.assert_allclose(np.ptp(angles, axis=1), np.pi, atol=1e-13)
        self.assertEqual(case["truth"]["expected"], "line")
        self.assertFalse(case["truth"]["abstention_is_recovery"])

    def test_real_gap_is_empty_and_has_observed_positive_guarded_length(self):
        for phase in PHASES:
            case = make_case("true_gap_half", phase=phase)
            z = np.round(local(case, case["inputs"]["points"])[:, 2], 12)
            low, high = GAP_LIMITS_M
            self.assertAlmostEqual(high - low, 0.15)
            self.assertGreater(high - low - 2 * 0.025, 0.0)
            self.assertFalse(np.any((z > low) & (z < high)))
            self.assertIn(low, z)
            self.assertIn(high, z)
            gap = local(case, case["truth"]["forbidden_segments"])[0]
            np.testing.assert_allclose(gap[:, 2], [low, high], atol=1e-14)
            self.assertEqual(len(case["truth"]["expected_segments"]), 2)

    def test_taper_remains_positive_with_a_known_straight_coaxial_target(self):
        case = make_case("tapered_sections")
        points = local(case, case["inputs"]["points"])
        nominal = RADIUS_M * (0.72 + 0.56 * points[:, 2] / LENGTH_M)
        self.assertLessEqual(np.max(np.abs(np.linalg.norm(points[:, :2], axis=1) - nominal)), RADIAL_NOISE_BOUND_M + 1e-14)
        np.testing.assert_allclose(case["protocol"]["nominal_radius_range_m"], [0.02088, 0.03712], atol=1e-14)
        self.assertEqual(case["truth"]["expected"], "line")
        self.assertEqual(len(case["truth"]["expected_segments"]), 1)
        self.assertFalse(case["truth"]["abstention_is_recovery"])

    def test_actual_missing_angles_are_local_not_train_validation_subsampling(self):
        case = make_case("locally_missing_angles")
        points = local(case, case["inputs"]["points"]).reshape(-1, 11, 3)
        angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
        np.testing.assert_allclose(np.ptp(angles, axis=1), np.deg2rad(LOCAL_ARC_DEGREES), atol=1e-13)
        # Four disconnected observed azimuth patches span 315 degrees globally.
        all_angles = np.sort(np.mod(angles.ravel(), 2 * np.pi))
        largest_gap = np.diff(np.r_[all_angles, all_angles[0] + 2 * np.pi]).max()
        self.assertAlmostEqual(np.rad2deg(2 * np.pi - largest_gap), 315.0)
        self.assertEqual(case["truth"]["expected"], "unresolved")
        self.assertTrue(case["truth"]["unresolved_construction_is_not_negative_foreground_truth"])

    def test_rotating_arc_and_helix_do_not_claim_recoverable_straight_axes(self):
        for case_id, width, turns in (("rotating_sparse_arc", 5, ROTATING_ARC_TURNS), ("helical_trace", 1, HELIX_TURNS)):
            case = make_case(case_id)
            points = local(case, case["inputs"]["points"]).reshape(-1, width, 3)
            angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
            np.testing.assert_allclose(np.ptp(angles, axis=1), np.deg2rad(ROTATING_ARC_DEGREES) if width == 5 else 0.0, atol=1e-13)
            trajectory = np.unwrap(angles[:, 0])
            self.assertAlmostEqual(trajectory[-1] - trajectory[0], 2 * np.pi * turns)
            self.assertEqual(case["truth"]["expected"], "unresolved")
            self.assertEqual(len(case["truth"]["expected_segments"]), 0)
            self.assertEqual(len(case["truth"]["construction_segments"]), 1)

    def test_plane_noise_is_normal_to_the_plane_and_is_not_a_positive_axis(self):
        case = make_case("noisy_plane")
        points = local(case, case["inputs"]["points"])
        self.assertLessEqual(np.abs(points[:, 1]).max(), PLANE_NOISE_BOUND_M + 1e-14)
        self.assertAlmostEqual(np.ptp(points[:, 0]), 0.130)
        self.assertEqual(case["truth"]["expected"], "abstain")
        self.assertEqual(len(case["truth"]["expected_segments"]), 0)

    def test_fixed_seed_is_repeatable_without_changing_truth(self):
        a, b = make_case("full_noisy_sections"), make_case("full_noisy_sections")
        np.testing.assert_array_equal(a["inputs"]["points"], b["inputs"]["points"])
        changed = make_case("full_noisy_sections", seed=DEFAULT_SEED + 1)
        self.assertFalse(np.array_equal(a["inputs"]["points"], changed["inputs"]["points"]))
        np.testing.assert_array_equal(a["truth"]["expected_segments"], changed["truth"]["expected_segments"])

    def test_truth_is_independent_and_scope_does_not_invent_physical_samples(self):
        case, reference = make_case("true_gap_half"), make_case("true_gap_half")
        case["inputs"]["points"][:] = 999
        np.testing.assert_array_equal(case["truth"]["expected_segments"], reference["truth"]["expected_segments"])
        self.assertTrue(case["protocol"]["scale_and_phase_are_repeated_conditions_not_objects"])
        self.assertTrue(case["protocol"]["prior_scored_44_36_40_are_separate_development_regression"])
        for record in (case["protocol"], case["truth"]):
            self.assertIn("not_real_photos", record["scope"])
            self.assertIn("not_independent_physical_objects", record["scope"])
            self.assertIn("not_foreground_identity", record["scope"])

    def test_invalid_protocol_values_are_rejected(self):
        invalid = ({"case_id": "unknown"}, {"seed": -1}, {"seed": True}, {"seed": 2**32},
                   {"phase": -1}, {"phase": 2}, {"phase": True},
                   {"voxel_size": 0}, {"voxel_size": float("nan")}, {"voxel_size": 0.006})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_case(**{"case_id": "full_noisy_sections", **kwargs})


if __name__ == "__main__":
    unittest.main()
