"""Author-only geometry contracts; do not import or invoke a tested reader."""

import ast
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import local_evidence_controls as controls  # noqa: E402


def local(case, values):
    p = case["protocol"]
    return (values - p["local_to_world_translation"]) @ np.array(p["local_to_world_rotation"])


class LocalEvidenceControlTests(unittest.TestCase):
    def test_fixed_inventory_roles_and_order(self):
        cases = list(controls.generate())
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({c["condition_id"] for c in cases}), 32)
        self.assertEqual(Counter(c["case_id"] for c in cases), dict.fromkeys(controls.CASE_IDS, 4))
        self.assertEqual(Counter(c["truth"]["expected"] for c in cases),
                         {"line": 8, "gap": 4, "two_lines": 4, "branches": 4, "unresolved": 8, "abstain": 4})
        self.assertEqual({c["cohort"] for c in cases}, {"new_local_evidence_controls_v1"})
        self.assertEqual(cases[0]["condition_id"], "full_shell-grid-0-phase-0")
        self.assertEqual(cases[-1]["condition_id"], "jittered_sparse_plane-grid-1-phase-1")

    def test_unlabelled_finite_point_only_arrays_and_separate_truth(self):
        for case in controls.generate():
            self.assertEqual(set(case["inputs"]), {"points", "segments"})
            self.assertEqual(case["inputs"]["segments"].shape, (0, 2, 3))
            self.assertEqual(case["inputs"]["points"].shape[1:], (3,))
            for array in [*case["inputs"].values(), *(case["truth"][k] for k in ("expected_segments", "forbidden_segments", "construction_segments"))]:
                self.assertEqual(array.dtype, np.dtype("<f8"))
                self.assertTrue(np.isfinite(array).all())
            saved = case["truth"]["expected_segments"].copy()
            case["inputs"]["points"][:] = 999
            np.testing.assert_array_equal(saved, case["truth"]["expected_segments"])

    def test_voxel_changes_only_protocol_not_geometry(self):
        for case_id in controls.CASE_IDS:
            for phase in controls.PHASES:
                a, b = [controls.make_case(case_id, phase=phase, voxel_size=v) for v in controls.GRID_SCALES_M]
                np.testing.assert_array_equal(a["inputs"]["points"], b["inputs"]["points"])
                for key in ("expected_segments", "forbidden_segments", "construction_segments"):
                    np.testing.assert_array_equal(a["truth"][key], b["truth"][key])

    def test_phase_changes_samples_and_keeps_physical_truth(self):
        for case_id in controls.CASE_IDS:
            a, b = [controls.make_case(case_id, phase=phase) for phase in controls.PHASES]
            self.assertFalse(np.array_equal(a["inputs"]["points"], b["inputs"]["points"]))
            for key in ("expected_segments", "forbidden_segments", "construction_segments"):
                np.testing.assert_array_equal(a["truth"][key], b["truth"][key])

    def test_fixed_13mm_physical_lattice_and_endpoints(self):
        for phase in controls.PHASES:
            c = controls.make_case("full_shell", phase=phase)
            z = np.unique(np.round(local(c, c["inputs"]["points"])[:, 2], 12))
            np.testing.assert_allclose(z[[0, -1]], [0.0, controls.LENGTH_M], atol=1e-12)
            np.testing.assert_allclose(np.diff(z[1:-1]), controls.AXIAL_STEP_M, atol=2e-12)
            values = z[1:-1] / controls.AXIAL_STEP_M - controls.SAMPLING_PHASES[phase]
            np.testing.assert_allclose(values, np.round(values), atol=1e-9)
            self.assertEqual(c["protocol"]["density"], "fixed_13mm")

    def test_proper_new_pose_and_point_roundtrip(self):
        c = controls.make_case("full_shell")
        r = np.array(c["protocol"]["local_to_world_rotation"])
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(r), 1.0)
        self.assertEqual(c["protocol"]["rotation_degrees"], [23.0, -47.0, 62.0])
        np.testing.assert_allclose(local(c, c["inputs"]["points"]) @ r.T + c["protocol"]["local_to_world_translation"], c["inputs"]["points"], atol=1e-14)

    def test_full_and_half_shell_have_actual_local_angular_support(self):
        for case_id, width, degrees in (("full_shell", 28, 360.0), ("half_shell", 15, 180.0)):
            c = controls.make_case(case_id)
            points = local(c, c["inputs"]["points"]).reshape(-1, width, 3)
            radial = np.linalg.norm(points[:, :, :2], axis=2)
            self.assertLessEqual(np.abs(radial - controls.RADIUS_M).max(), controls.RADIAL_NOISE_BOUND_M + 1e-14)
            angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
            span = 360.0 * 27 / 28 if width == 28 else degrees
            np.testing.assert_allclose(np.rad2deg(np.ptp(angles, axis=1)), span, atol=1e-10)
            self.assertEqual(c["protocol"]["local_angular_support_degrees"], degrees)
            self.assertGreater(radial.min(), 0.023)

    def test_true_gap_is_210mm_and_clear_of_observations(self):
        for phase in controls.PHASES:
            c = controls.make_case("true_gap_half", phase=phase)
            z = np.round(local(c, c["inputs"]["points"])[:, 2], 12)
            low, high = controls.GAP_LIMITS_M
            self.assertAlmostEqual(high - low, .210)
            self.assertGreater(high - low - .050, 0.0)
            self.assertFalse(np.any((z > low) & (z < high)))
            self.assertIn(low, z)
            self.assertIn(high, z)
            self.assertEqual(len(c["truth"]["expected_segments"]), 2)
            np.testing.assert_allclose(local(c, c["truth"]["forbidden_segments"])[0, :, 2], [low, high], atol=1e-14)

    def test_two_bare_lines_are_real_positive_observations(self):
        c = controls.make_case("twin_bare_lines")
        truth = local(c, c["truth"]["expected_segments"])
        points = local(c, c["inputs"]["points"])
        self.assertEqual(c["truth"]["expected"], "two_lines")
        self.assertAlmostEqual(np.linalg.norm(truth[0, 0, :2] - truth[1, 0, :2]), .043)
        distances = np.linalg.norm(points[:, None, :2] - truth[None, :, 0, :2], axis=2)
        self.assertLessEqual(distances.min(axis=1).max(), np.sqrt(2) * controls.LINE_TRANSVERSE_NOISE_BOUND_M + 1e-14)
        self.assertGreater(np.bincount(np.argmin(distances, axis=1)).min(), 100)
        self.assertFalse(c["truth"]["abstention_is_recovery"])

    def test_three_axes_are_noncoplanar_and_short_branch_remains_truth(self):
        c = controls.make_case("noncoplanar_rods_with_branch")
        truth = local(c, c["truth"]["expected_segments"])
        points = local(c, c["inputs"]["points"])
        centers = truth[:3, 0, :2]
        self.assertGreater(abs(np.linalg.det(centers[1:] - centers[0])), .001)
        np.testing.assert_allclose(truth[:3, 1] - truth[:3, 0], np.tile([0., 0., controls.LENGTH_M], (3, 1)), atol=1e-14)
        branch = truth[3, 1] - truth[3, 0]
        self.assertAlmostEqual(np.linalg.norm(branch), .190)
        self.assertAlmostEqual(np.rad2deg(np.arccos(branch[2] / np.linalg.norm(branch))), 55.0)
        self.assertEqual(c["truth"]["expected"], "branches")
        self.assertEqual(len(truth), 4)
        self.assertLess(np.linalg.norm(points - truth[3, 0], axis=1).min(), 1e-14)
        self.assertLess(np.linalg.norm(points - truth[3, 1], axis=1).min(), 1e-14)

    def test_sparse_arc_is_locally_70_degrees_despite_global_rotation(self):
        c = controls.make_case("rotating_sparse_arc")
        points = local(c, c["inputs"]["points"]).reshape(-1, 4, 3)
        angles = np.unwrap(np.arctan2(points[:, :, 1], points[:, :, 0]), axis=1)
        np.testing.assert_allclose(np.rad2deg(np.ptp(angles, axis=1)), 70.0, atol=1e-10)
        winding = np.unwrap(angles[:, 0])
        self.assertAlmostEqual(winding[-1] - winding[0], 2 * np.pi * 2.6)
        self.assertEqual(c["truth"]["expected"], "unresolved")
        self.assertEqual(len(c["truth"]["expected_segments"]), 0)

    def test_short_pitch_helix_is_one_point_per_layer_and_unresolved(self):
        c = controls.make_case("short_pitch_helix")
        points = local(c, c["inputs"]["points"])
        angles = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
        self.assertAlmostEqual(angles[-1] - angles[0], 2 * np.pi * 8.25)
        self.assertLess(controls.LENGTH_M / controls.HELIX_TURNS, .210)
        self.assertEqual(len(points), c["protocol"]["axial_layer_count_before_jitter"])
        self.assertEqual(c["truth"]["expected"], "unresolved")
        self.assertTrue(c["truth"]["unresolved_construction_is_not_negative_foreground_truth"])
        self.assertEqual(len(c["truth"]["construction_segments"]), 1)

    def test_sparse_plane_has_seven_columns_and_bounded_transverse_axial_jitter(self):
        for phase in controls.PHASES:
            c = controls.make_case("jittered_sparse_plane", phase=phase)
            points = local(c, c["inputs"]["points"]).reshape(-1, 7, 3)
            base_z = controls._samples(controls.LENGTH_M, phase)
            x = np.linspace(-.066, .066, 7)
            self.assertLessEqual(np.abs(points[:, :, 0] - x).max(), controls.PLANE_TRANSVERSE_NOISE_BOUND_M + 1e-14)
            self.assertLessEqual(np.abs(points[:, :, 1]).max(), controls.PLANE_TRANSVERSE_NOISE_BOUND_M + 1e-14)
            deviation = points[:, :, 2] - base_z[:, None]
            self.assertLessEqual(np.abs(deviation).max(), controls.PLANE_AXIAL_NOISE_BOUND_M + 1e-14)
            self.assertGreater(np.std(deviation[1:-1]), .0001)
            self.assertEqual(c["truth"]["expected"], "abstain")
            self.assertIn("not proof", c["truth"]["negative_scope"])

    def test_seed_repeatability_and_no_reader_dependencies(self):
        for case_id in controls.CASE_IDS:
            a, b = controls.make_case(case_id), controls.make_case(case_id)
            np.testing.assert_array_equal(a["inputs"]["points"], b["inputs"]["points"])
            changed = controls.make_case(case_id, seed=controls.DEFAULT_SEED + 1)
            self.assertFalse(np.array_equal(a["inputs"]["points"], changed["inputs"]["points"]))
            np.testing.assert_array_equal(a["truth"]["expected_segments"], changed["truth"]["expected_segments"])
        tree = ast.parse(Path(controls.__file__).read_text())
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(modules, {"__future__", "mixed_readout_controls", "readout_background_controls"})

    def test_scope_and_representation_do_not_claim_independent_evidence(self):
        for case in controls.generate():
            p, t = case["protocol"], case["truth"]
            self.assertTrue(p["representation_pairs_are_identical_inputs_not_independent_evidence"])
            self.assertTrue(p["scale_and_phase_are_repeated_conditions_not_objects"])
            self.assertTrue(p["prior_scored_44_36_40_32_are_separate_development_regression"])
            self.assertFalse(t["abstention_is_recovery"])
            for record in (p, t):
                self.assertIn("not_real_photos", record["scope"])
                self.assertIn("not_independent_physical_objects", record["scope"])

    def test_invalid_protocol_values_fail_closed(self):
        invalid = ({"case_id": "unknown"}, {"seed": -1}, {"seed": True}, {"seed": 2**32},
                   {"phase": -1}, {"phase": 2}, {"phase": True}, {"voxel_size": 0},
                   {"voxel_size": float("nan")}, {"voxel_size": .0055}, {"voxel_size": True})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                controls.make_case(**{"case_id": "full_shell", **kwargs})


if __name__ == "__main__":
    unittest.main()
