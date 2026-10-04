"""Construction-only contracts; no reader, fixture data, or score is used."""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.sampling_branch_controls import (  # noqa: E402
    ARC_COVERAGE_DEGREES,
    ARC_SWEEP_TURNS,
    AXIAL_STEPS_M,
    BRANCH_ANGLES_DEGREES,
    BRANCH_LENGTHS_M,
    CASE_IDS,
    COHORT,
    DEFAULT_SEED,
    DENSITIES,
    GAP_LENGTHS_M,
    GRID_SCALES_M,
    HELIX_TURNS,
    LENGTH_M,
    RADIUS_M,
    ROTATION_DEGREES,
    SAMPLING_PHASE,
    SECOND_TWIN_SAMPLING_PHASE,
    SMALL_RADIUS_M,
    SPARSE_INTERVAL_M,
    TWIN_AXIS_DISTANCE_M,
    generate,
    make_case,
)


def local(case, values):
    p = case["protocol"]
    return (values - p["local_to_world_translation"]) @ np.array(p["local_to_world_rotation"])


class SamplingBranchControlTests(unittest.TestCase):
    def test_fixed_40_condition_inventory_and_separate_cohort(self):
        cases = list(generate())
        self.assertEqual(len(cases), 40)
        self.assertEqual(len({case["condition_id"] for case in cases}), 40)
        self.assertEqual(Counter(case["case_id"] for case in cases), dict.fromkeys(CASE_IDS, 4))
        self.assertEqual({case["cohort"] for case in cases}, {COHORT})
        self.assertEqual(Counter(case["truth"]["expected"] for case in cases),
                         {"line": 8, "two_lines": 4, "gap": 8, "branches": 8, "abstain": 4, "unresolved": 8})

    def test_inputs_only_contain_finite_unlabelled_arrays(self):
        for case in generate():
            with self.subTest(case=case["condition_id"]):
                self.assertEqual(set(case["inputs"]), {"points", "segments"})
                points, segments = case["inputs"].values()
                self.assertEqual(points.shape[1:], (3,))
                self.assertEqual(segments.shape[1:], (2, 3))
                self.assertTrue(np.isfinite(points).all() and np.isfinite(segments).all())
                self.assertEqual(points.dtype, np.dtype("<f8"))
                self.assertEqual(segments.dtype, np.dtype("<f8"))
                self.assertTrue((np.linalg.norm(np.diff(segments, axis=1), axis=2) > 0).all())

    def test_voxel_scale_does_not_change_any_normal_geometry(self):
        for case_id in CASE_IDS:
            for density in DENSITIES:
                a, b = [make_case(case_id, density=density, voxel_size=scale) for scale in GRID_SCALES_M]
                for key in ("points", "segments"):
                    np.testing.assert_array_equal(a["inputs"][key], b["inputs"][key])
                np.testing.assert_array_equal(a["truth"]["expected_segments"], b["truth"]["expected_segments"])

    def test_coarse_density_changes_sampling_not_declared_structure(self):
        for case_id in CASE_IDS:
            dense, coarse = [make_case(case_id, density=density) for density in DENSITIES]
            self.assertGreater(len(dense["inputs"]["points"]), len(coarse["inputs"]["points"]))
            for key in ("expected_segments", "forbidden_segments", "construction_segments"):
                np.testing.assert_array_equal(dense["truth"][key], coarse["truth"][key])

    def test_middle_region_has_twice_the_physical_sampling_pitch(self):
        for density in DENSITIES:
            case = make_case("full_variable_density", density=density)
            z = np.unique(np.round(local(case, case["inputs"]["points"])[:, 2], 12))
            low, high = SPARSE_INTERVAL_M
            step = AXIAL_STEPS_M[density]
            for first, last, pitch in ((0.0, low, step), (low, high, 2 * step), (high, LENGTH_M, step)):
                values = z[(z > first + 1e-10) & (z < last - 1e-10)]
                self.assertGreater(len(values), 3)
                np.testing.assert_allclose(np.diff(values), pitch, atol=2e-12)
                np.testing.assert_allclose(values / pitch - SAMPLING_PHASE,
                                           np.round(values / pitch - SAMPLING_PHASE), atol=1e-9)
            self.assertAlmostEqual(z[0], 0.0)
            self.assertAlmostEqual(z[-1], LENGTH_M)

    def test_new_pose_is_rigid_and_not_either_prior_control_rotation(self):
        case = make_case("full_variable_density")
        rotation = np.array(case["protocol"]["local_to_world_rotation"])
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        self.assertEqual(case["protocol"]["rotation_degrees"], list(ROTATION_DEGREES))
        self.assertNotIn(tuple(ROTATION_DEGREES), ((17.0, -23.0, 11.0), (-29.0, 34.0, 53.0)))

    def test_half_surface_has_180_degree_observations_without_hidden_axis(self):
        case = make_case("half_variable_density")
        points = local(case, case["inputs"]["points"])
        np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), RADIUS_M, atol=1e-14)
        self.assertGreaterEqual(points[:, 0].min(), -1e-14)
        self.assertAlmostEqual(np.ptp(np.arctan2(points[:, 1], points[:, 0])), np.pi)
        self.assertEqual(len(case["inputs"]["segments"]), 0)
        self.assertEqual(case["truth"]["expected"], "line")

    def test_twin_surfaces_have_separate_axes_and_staggered_sampling_layers(self):
        for density in DENSITIES:
            case = make_case("staggered_twin_surfaces", density=density)
            points = local(case, case["inputs"]["points"])
            offsets = np.array([-TWIN_AXIS_DISTANCE_M / 2, TWIN_AXIS_DISTANCE_M / 2])
            distances = np.sqrt((points[:, 0, None] - offsets)**2 + points[:, 1, None]**2)
            np.testing.assert_allclose(distances.min(axis=1), SMALL_RADIUS_M, atol=1e-14)
            first = np.unique(np.round(points[points[:, 0] < 0, 2], 12))
            second = np.unique(np.round(points[points[:, 0] > 0, 2], 12))
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(case["protocol"]["second_surface_sampling_phase"], SECOND_TWIN_SAMPLING_PHASE)
            self.assertEqual(len(case["inputs"]["segments"]), 0)
            self.assertEqual(len(case["truth"]["expected_segments"]), 2)

    def test_short_and_long_true_gaps_are_empty_and_keep_positive_guarded_interiors(self):
        # The protocol's 90/270 mm gaps exceed two declared 25 mm endpoint guards.
        # This checks geometry only, without importing an evaluation module.
        for case_id, length in GAP_LENGTHS_M.items():
            for density in DENSITIES:
                case = make_case(case_id, density=density)
                gap = local(case, case["truth"]["forbidden_segments"])[0]
                low, high = gap[:, 2]
                self.assertAlmostEqual(high - low, length)
                self.assertGreater(high - low - 2 * 0.025, 0.0)
                z = np.round(local(case, case["inputs"]["points"])[:, 2], 12)
                self.assertFalse(np.any((z > low + 1e-10) & (z < high - 1e-10)))
                self.assertTrue(np.any(np.isclose(z, low, atol=1e-12)))
                self.assertTrue(np.any(np.isclose(z, high, atol=1e-12)))
                self.assertEqual(len(case["truth"]["expected_segments"]), 2)
                self.assertEqual(len(case["inputs"]["segments"]), 0)

    def test_two_branch_angles_and_unequal_short_lengths_are_preserved(self):
        for case_id, degrees in BRANCH_ANGLES_DEGREES.items():
            case = make_case(case_id)
            segments = local(case, case["truth"]["expected_segments"])
            main, branch = segments[:, 1] - segments[:, 0]
            angle = np.rad2deg(np.arccos(np.dot(main, branch) / (np.linalg.norm(main) * np.linalg.norm(branch))))
            self.assertAlmostEqual(angle, degrees)
            self.assertAlmostEqual(np.linalg.norm(branch), BRANCH_LENGTHS_M[case_id])
            self.assertLess(np.linalg.norm(branch) / np.linalg.norm(main), 0.15)
            np.testing.assert_allclose(segments[1, 0], [0.0, 0.0, 0.63 * LENGTH_M], atol=1e-14)
            self.assertTrue(case["truth"]["branch_completeness_requires_boundary_metrics_not_only_global_recovery"])
            self.assertFalse(case["truth"]["abstention_is_recovery"])

    def test_both_branches_share_exactly_the_scatter_negative_points(self):
        for density in DENSITIES:
            negative = make_case("scatter_only", density=density)
            for case_id in BRANCH_ANGLES_DEGREES:
                branch = make_case(case_id, density=density)
                np.testing.assert_array_equal(branch["inputs"]["points"], negative["inputs"]["points"])
                self.assertEqual(len(branch["inputs"]["segments"]), 2)
            self.assertEqual(len(negative["inputs"]["segments"]), 0)
            self.assertEqual(len(negative["truth"]["expected_segments"]), 0)

    def test_sparse_rotating_arc_and_helix_do_not_declare_recoverable_cylinder_axes(self):
        for case_id, width, turns in (("rotating_sparse_arc", 3, ARC_SWEEP_TURNS), ("helical_trace", 1, HELIX_TURNS)):
            case = make_case(case_id)
            points = local(case, case["inputs"]["points"]).reshape(-1, width, 3)
            angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
            np.testing.assert_allclose(np.ptp(angles, axis=1), np.deg2rad(ARC_COVERAGE_DEGREES) if width == 3 else 0.0, atol=1e-13)
            trajectory = np.unwrap(angles[:, 0])
            self.assertAlmostEqual(trajectory[-1] - trajectory[0], 2 * np.pi * turns)
            self.assertEqual(case["truth"]["expected"], "unresolved")
            self.assertEqual(len(case["truth"]["expected_segments"]), 0)
            self.assertEqual(len(case["truth"]["construction_segments"]), 1)
            self.assertTrue(case["truth"]["unresolved_construction_is_not_negative_foreground_truth"])

    def test_fixed_seed_is_repeatable_and_only_changes_scatter_realization(self):
        a, b = make_case("shallow_side_branch"), make_case("shallow_side_branch")
        np.testing.assert_array_equal(a["inputs"]["points"], b["inputs"]["points"])
        different = make_case("shallow_side_branch", seed=DEFAULT_SEED + 1)
        self.assertFalse(np.array_equal(a["inputs"]["points"], different["inputs"]["points"]))
        np.testing.assert_array_equal(a["truth"]["expected_segments"], different["truth"]["expected_segments"])

    def test_inputs_cannot_mutate_truth_or_future_generations(self):
        changed, untouched = make_case("steep_side_branch"), make_case("steep_side_branch")
        changed["inputs"]["points"][:] = 123
        changed["inputs"]["segments"][:] = 456
        np.testing.assert_array_equal(changed["truth"]["expected_segments"], untouched["truth"]["expected_segments"])
        np.testing.assert_array_equal(make_case("steep_side_branch")["inputs"]["points"], untouched["inputs"]["points"])

    def test_scope_does_not_count_new_conditions_as_independent_physical_objects(self):
        case = make_case("full_variable_density")
        self.assertTrue(case["protocol"]["scale_and_density_are_repeated_conditions_not_objects"])
        self.assertTrue(case["protocol"]["prior_scored_44_and_36_are_separate_development_regression"])
        for record in (case["protocol"], case["truth"]):
            self.assertIn("not_real_photos", record["scope"])
            self.assertIn("not_independent_physical_objects", record["scope"])
            self.assertIn("not_foreground_identity", record["scope"])

    def test_invalid_protocol_values_are_rejected(self):
        invalid = ({"case_id": "unknown"}, {"seed": -1}, {"seed": True}, {"seed": 2**32},
                   {"density": "medium"}, {"density": True}, {"voxel_size": 0},
                   {"voxel_size": float("nan")}, {"voxel_size": 0.006})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_case(**{"case_id": "full_variable_density", **kwargs})


if __name__ == "__main__":
    unittest.main()
