"""Generator invariants only: no reader tuning, fixture replay, or truth-file IO."""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.mixed_readout_controls import (  # noqa: E402
    CASE_IDS,
    DEFAULT_SEED,
    GAP_LIMITS_M,
    GRID_SCALES_M,
    LENGTH_M,
    SUBVOXEL_PHASES,
    generate,
    make_case,
)


def local(case, values):
    p = case["protocol"]
    return (values - p["local_to_world_translation"]) @ np.array(p["local_to_world_rotation"])


class MixedReadoutControlTests(unittest.TestCase):
    def test_fixed_factorial_design_and_unique_condition_ids(self):
        cases = list(generate())
        self.assertEqual(len(cases), 44)
        self.assertEqual(len({case["condition_id"] for case in cases}), 44)
        self.assertEqual(Counter(case["case_id"] for case in cases), dict.fromkeys(CASE_IDS, 4))
        self.assertEqual(len(GRID_SCALES_M), 2)
        self.assertEqual(len(SUBVOXEL_PHASES), 2)

    def test_reader_inputs_have_only_finite_unlabelled_arrays(self):
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

    def test_determinism_and_seed_affects_scatter_not_truth(self):
        first, second = list(generate()), list(generate())
        for a, b in zip(first, second):
            for key in ("points", "segments"):
                np.testing.assert_array_equal(a["inputs"][key], b["inputs"][key])
        a, b = make_case("axis_scatter"), make_case("axis_scatter", seed=DEFAULT_SEED + 1)
        self.assertFalse(np.array_equal(a["inputs"]["points"], b["inputs"]["points"]))
        np.testing.assert_array_equal(a["truth"]["expected_segments"], b["truth"]["expected_segments"])

    def test_scatter_pair_changes_only_presence_of_finite_curve(self):
        a, b = make_case("axis_scatter"), make_case("scatter_only")
        np.testing.assert_array_equal(a["inputs"]["points"], b["inputs"]["points"])
        self.assertEqual(len(a["inputs"]["segments"]), 1)
        self.assertEqual(len(b["inputs"]["segments"]), 0)
        self.assertEqual(b["truth"]["expected"], "abstain")

    def test_scales_and_phases_preserve_metric_geometry(self):
        for case_id in CASE_IDS:
            reference = make_case(case_id)
            for scale in GRID_SCALES_M:
                for phase in range(2):
                    case = make_case(case_id, voxel_size=scale, phase=phase)
                    for key in ("points", "segments"):
                        np.testing.assert_allclose(
                            local(case, case["inputs"][key]),
                            local(reference, reference["inputs"][key]), atol=2e-15,
                        )
                    rotation = np.array(case["protocol"]["local_to_world_rotation"])
                    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
                    self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_actual_subvoxel_translation_matches_declared_phase(self):
        for scale in GRID_SCALES_M:
            a = make_case("axis_scatter", voxel_size=scale, phase=0)
            b = make_case("axis_scatter", voxel_size=scale, phase=1)
            delta = (np.array(SUBVOXEL_PHASES[1]) - SUBVOXEL_PHASES[0]) * scale
            np.testing.assert_allclose(b["inputs"]["points"], a["inputs"]["points"] + delta, atol=1e-15)
            self.assertTrue(np.all((delta > 0) & (delta < scale)))

    def test_full_and_half_surfaces_have_no_hidden_axis_samples(self):
        full, half = make_case("full_cylinder"), make_case("half_cylinder")
        for case in (full, half):
            points = local(case, case["inputs"]["points"])
            self.assertEqual(len(case["inputs"]["segments"]), 0)
            np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 0.025, atol=1e-14)
            self.assertAlmostEqual(points[:, 2].min(), 0.0)
            self.assertAlmostEqual(points[:, 2].max(), LENGTH_M)
        self.assertGreater(len(full["inputs"]["points"]), len(half["inputs"]["points"]))
        self.assertGreaterEqual(local(half, half["inputs"]["points"])[:, 1].min(), -1e-14)

    def test_sheet_is_planar_but_does_not_supply_an_axis_target(self):
        case = make_case("thin_sheet")
        points = local(case, case["inputs"]["points"])
        np.testing.assert_allclose(points[:, 1], 0.0, atol=1e-14)
        self.assertAlmostEqual(np.ptp(points[:, 0]), 0.140)
        # Count robustly after inverse rotation (which has float roundoff).
        self.assertEqual(len(np.unique(np.round(points[:, 0], 12))), 9)
        self.assertEqual(len(case["truth"]["expected_segments"]), 0)

    def test_sparse_surface_construction_does_not_imply_recoverable_axis(self):
        for case_id in ("sparse_cylinder", "sparse_half_cylinder", "rotating_sparse_arc"):
            case = make_case(case_id)
            self.assertEqual(case["truth"]["expected"], "unresolved")
            self.assertEqual(len(case["truth"]["expected_segments"]), 0)
            self.assertEqual(len(case["truth"]["construction_segments"]), 1)
            self.assertEqual(len(case["inputs"]["segments"]), 0)
            points = local(case, case["inputs"]["points"])
            self.assertEqual(points.shape, (241 * 4, 3))
            np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 0.025, atol=1e-14)
            self.assertTrue(np.all(np.unique(np.round(points[:, 2], 12), return_counts=True)[1] == 4))

    def test_rotating_partial_sections_cannot_borrow_whole_cloud_coverage(self):
        case = make_case("rotating_sparse_arc")
        points = local(case, case["inputs"]["points"]).reshape(241, 4, 3)
        angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
        np.testing.assert_allclose(np.ptp(angles, axis=1), np.pi / 3, atol=1e-13)
        first_angle = np.unwrap(angles[:, 0])
        self.assertAlmostEqual(first_angle[-1] - first_angle[0], 2 * np.pi)

    def test_close_parallel_lines_are_not_replaced_by_their_midline(self):
        case = make_case("nearby_lines")
        segments = local(case, case["truth"]["expected_segments"])
        self.assertEqual(case["truth"]["expected"], "two_lines")
        np.testing.assert_allclose(segments[:, :, 0], [[-0.018, -0.018], [0.018, 0.018]], atol=1e-14)
        np.testing.assert_allclose(segments[:, 1] - segments[:, 0], [[0, 0, LENGTH_M]] * 2, atol=1e-14)

    def test_true_gap_has_no_point_or_segment_interior_support(self):
        case = make_case("true_gap")
        z = local(case, case["inputs"]["points"])[:, 2]
        low, high = GAP_LIMITS_M
        self.assertFalse(np.any((z > low) & (z < high)))
        segments = local(case, case["inputs"]["segments"])
        self.assertAlmostEqual(segments[0, 1, 2], low)
        self.assertAlmostEqual(segments[1, 0, 2], high)
        forbidden = local(case, case["truth"]["forbidden_segments"])
        np.testing.assert_allclose(forbidden[0, :, 2], [low, high], atol=1e-14)

    def test_side_branch_is_a_real_orthogonal_target_not_a_spur_label(self):
        case = make_case("side_branch")
        segments = local(case, case["truth"]["expected_segments"])
        self.assertEqual(case["truth"]["expected"], "branches")
        directions = segments[:, 1] - segments[:, 0]
        self.assertAlmostEqual(np.dot(directions[0], directions[1]), 0.0)
        self.assertAlmostEqual(np.linalg.norm(directions[1]), 0.36)
        self.assertFalse(case["truth"]["abstention_is_recovery"])

    def test_input_mutation_cannot_change_truth_or_next_generation(self):
        case, untouched = make_case("axis_scatter"), make_case("axis_scatter")
        case["inputs"]["segments"][:] = 123
        case["inputs"]["points"][:] = 456
        np.testing.assert_array_equal(case["truth"]["expected_segments"], untouched["truth"]["expected_segments"])
        np.testing.assert_array_equal(make_case("axis_scatter")["inputs"]["points"], untouched["inputs"]["points"])

    def test_scope_does_not_count_phases_or_synthetic_shapes_as_objects(self):
        case = make_case("axis_scatter")
        for record in (case["protocol"], case["truth"]):
            self.assertIn("not_real_photos", record["scope"])
            self.assertIn("not_independent_physical_objects", record["scope"])
            self.assertIn("not_foreground_identity", record["scope"])
        self.assertTrue(case["truth"]["negative_labels_are_construction_assumptions"])
        self.assertTrue(case["protocol"]["scale_and_phase_are_repeated_conditions_not_objects"])

    def test_unknown_or_unbounded_protocol_inputs_are_rejected(self):
        invalid = [
            {"case_id": "not_a_family"}, {"seed": -1}, {"seed": True},
            {"seed": 2**32}, {"phase": -1}, {"phase": 2}, {"phase": True},
            {"voxel_size": 0.0}, {"voxel_size": float("nan")}, {"voxel_size": 0.007},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_case(**{"case_id": "axis_scatter", **kwargs})


if __name__ == "__main__":
    unittest.main()
