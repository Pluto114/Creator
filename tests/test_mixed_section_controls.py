"""New control-construction contracts, without readers, fixture IO, or scoring."""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.mixed_section_controls import (  # noqa: E402
    AXIAL_STEPS_M,
    CASE_IDS,
    COHORT,
    DEFAULT_SEED,
    DENSITIES,
    GAP_LIMITS_M,
    GRID_SCALES_M,
    LENGTH_M,
    RADIUS_M,
    ROTATION_DEGREES,
    SAMPLING_PHASE,
    SMALL_RADIUS_M,
    TWIN_AXIS_DISTANCE_M,
    generate,
    make_case,
)


def local(case, values):
    p = case["protocol"]
    return (values - p["local_to_world_translation"]) @ np.array(p["local_to_world_rotation"])


class MixedSectionControlTests(unittest.TestCase):
    def test_fixed_36_condition_inventory_and_new_cohort(self):
        cases = list(generate())
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({case["condition_id"] for case in cases}), 36)
        self.assertEqual(Counter(case["case_id"] for case in cases), dict.fromkeys(CASE_IDS, 4))
        self.assertEqual({case["cohort"] for case in cases}, {COHORT})
        for case in cases:
            self.assertIn("density-" + case["protocol"]["density"], case["condition_id"])
            self.assertEqual(case["protocol"]["cohort"], COHORT)

    def test_inputs_are_only_finite_unlabelled_arrays(self):
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

    def test_reader_scale_never_changes_normal_input_geometry(self):
        for case_id in CASE_IDS:
            for density in DENSITIES:
                a, b = [make_case(case_id, density=density, voxel_size=scale) for scale in GRID_SCALES_M]
                for name in ("points", "segments"):
                    np.testing.assert_array_equal(a["inputs"][name], b["inputs"][name])
                np.testing.assert_array_equal(a["truth"]["expected_segments"], b["truth"]["expected_segments"])

    def test_coarse_and_dense_change_samples_but_preserve_declared_geometry(self):
        for case_id in CASE_IDS:
            dense, coarse = [make_case(case_id, density=density) for density in DENSITIES]
            self.assertGreater(len(dense["inputs"]["points"]), len(coarse["inputs"]["points"]))
            for key in ("expected_segments", "forbidden_segments", "construction_segments"):
                np.testing.assert_array_equal(dense["truth"][key], coarse["truth"][key])

    def test_sampling_phase_and_step_are_physical_not_voxel_dependent(self):
        for density in DENSITIES:
            case = make_case("full_cylinder", density=density)
            z = np.unique(np.round(local(case, case["inputs"]["points"])[:, 2], 12))
            step = AXIAL_STEPS_M[density]
            self.assertAlmostEqual(z[0], 0.0)
            self.assertAlmostEqual(z[-1], LENGTH_M)
            np.testing.assert_allclose(z[1:-1] / step - SAMPLING_PHASE,
                                       np.round(z[1:-1] / step - SAMPLING_PHASE), atol=1e-10)
            self.assertEqual(case["protocol"]["sampling_phase"], 0.37)
            self.assertEqual(case["protocol"]["axial_step_m"], step)

    def test_new_rotation_is_proper_and_fixed_across_conditions(self):
        for case in generate():
            p = case["protocol"]
            self.assertEqual(p["rotation_degrees"], list(ROTATION_DEGREES))
            self.assertNotEqual(p["rotation_degrees"], [17.0, -23.0, 11.0])
            rotation = np.array(p["local_to_world_rotation"])
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_half_surface_has_true_180_degree_support_and_no_hidden_axis(self):
        for density in DENSITIES:
            case = make_case("half_cylinder", density=density)
            points = local(case, case["inputs"]["points"])
            np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), RADIUS_M, atol=1e-14)
            self.assertGreaterEqual(points[:, 0].min(), -1e-14)
            angles = np.arctan2(points[:, 1], points[:, 0])
            self.assertAlmostEqual(np.ptp(angles), np.pi)
            self.assertEqual(len(case["inputs"]["segments"]), 0)
            self.assertEqual(case["truth"]["expected"], "line")
            self.assertFalse(case["truth"]["abstention_is_recovery"])

    def test_small_twins_are_two_surfaces_not_privileged_axis_primitives(self):
        case = make_case("small_twin_cylinders")
        points = local(case, case["inputs"]["points"])
        targets = local(case, case["truth"]["expected_segments"])
        offsets = np.array([-TWIN_AXIS_DISTANCE_M / 2, TWIN_AXIS_DISTANCE_M / 2])
        distances = np.sqrt((points[:, 0, None] - offsets)**2 + points[:, 1, None]**2)
        np.testing.assert_allclose(distances.min(axis=1), SMALL_RADIUS_M, atol=1e-14)
        np.testing.assert_allclose(targets[:, :, 0], np.repeat(offsets[:, None], 2, axis=1), atol=1e-14)
        self.assertEqual(len(case["inputs"]["segments"]), 0)
        self.assertEqual(case["truth"]["expected"], "two_lines")

    def test_gap_has_observed_boundaries_but_no_interior_points(self):
        for density in DENSITIES:
            case = make_case("gapped_half_cylinder", density=density)
            z = np.round(local(case, case["inputs"]["points"])[:, 2], 12)
            low, high = GAP_LIMITS_M
            self.assertAlmostEqual(high - low, 0.320)
            self.assertFalse(np.any((z > low) & (z < high)))
            self.assertIn(low, z)
            self.assertIn(high, z)
            self.assertEqual(len(case["truth"]["expected_segments"]), 2)
            forbidden = local(case, case["truth"]["forbidden_segments"])
            np.testing.assert_allclose(forbidden[0, :, 2], [low, high], atol=1e-14)

    def test_plane_declares_no_positive_centerline(self):
        case = make_case("plane")
        points = local(case, case["inputs"]["points"])
        np.testing.assert_allclose(points[:, 1], 0.0, atol=1e-14)
        self.assertAlmostEqual(np.ptp(points[:, 0]), 0.140)
        self.assertEqual(case["truth"]["expected"], "abstain")
        self.assertEqual(len(case["truth"]["expected_segments"]), 0)

    def test_rotating_arc_and_helix_remain_unresolved_not_negative_truth(self):
        for case_id, width in (("rotating_sparse_arc", 4), ("helical_trace", 1)):
            case = make_case(case_id)
            truth = case["truth"]
            self.assertEqual(truth["expected"], "unresolved")
            self.assertEqual(len(truth["expected_segments"]), 0)
            self.assertEqual(len(truth["construction_segments"]), 1)
            self.assertTrue(truth["unresolved_construction_is_not_negative_foreground_truth"])
            self.assertEqual(len(case["inputs"]["segments"]), 0)
            points = local(case, case["inputs"]["points"]).reshape(-1, width, 3)
            angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
            np.testing.assert_allclose(np.ptp(angles, axis=1), np.pi / 3 if width == 4 else 0.0, atol=1e-13)
            turns = np.unwrap(angles[:, 0])
            self.assertAlmostEqual(turns[-1] - turns[0], 2 * np.pi)

    def test_scatter_pair_is_identical_at_each_density(self):
        for density in DENSITIES:
            positive = make_case("axis_scatter", density=density)
            negative = make_case("scatter_only", density=density)
            np.testing.assert_array_equal(positive["inputs"]["points"], negative["inputs"]["points"])
            self.assertEqual(len(positive["inputs"]["segments"]), 1)
            self.assertEqual(len(negative["inputs"]["segments"]), 0)
            self.assertEqual(negative["truth"]["expected"], "abstain")

    def test_determinism_and_seed_does_not_change_construction_truth(self):
        a, b = make_case("axis_scatter"), make_case("axis_scatter")
        np.testing.assert_array_equal(a["inputs"]["points"], b["inputs"]["points"])
        c = make_case("axis_scatter", seed=DEFAULT_SEED + 1)
        self.assertFalse(np.array_equal(a["inputs"]["points"], c["inputs"]["points"]))
        np.testing.assert_array_equal(a["truth"]["expected_segments"], c["truth"]["expected_segments"])

    def test_input_mutation_cannot_change_truth_or_later_generation(self):
        changed, untouched = make_case("axis_scatter"), make_case("axis_scatter")
        changed["inputs"]["points"][:] = 123
        changed["inputs"]["segments"][:] = 456
        np.testing.assert_array_equal(changed["truth"]["expected_segments"], untouched["truth"]["expected_segments"])
        np.testing.assert_array_equal(make_case("axis_scatter")["inputs"]["points"], untouched["inputs"]["points"])

    def test_repeated_conditions_and_development_cohort_are_explicit(self):
        case = make_case("full_cylinder")
        self.assertTrue(case["protocol"]["scale_and_density_are_repeated_conditions_not_objects"])
        self.assertTrue(case["protocol"]["old_44_scored_conditions_are_separate_development_regression"])
        for record in (case["protocol"], case["truth"]):
            self.assertIn("not_real_photos", record["scope"])
            self.assertIn("not_independent_physical_objects", record["scope"])
            self.assertIn("not_foreground_identity", record["scope"])

    def test_invalid_protocol_values_are_rejected(self):
        invalid = ({"case_id": "old_scored_case"}, {"seed": -1}, {"seed": True},
                   {"seed": 2**32}, {"density": "medium"}, {"density": True},
                   {"voxel_size": 0}, {"voxel_size": float("nan")}, {"voxel_size": 0.007})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_case(**{"case_id": "full_cylinder", **kwargs})


if __name__ == "__main__":
    unittest.main()
