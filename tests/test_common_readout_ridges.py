"""Independent analytic contracts; these are development tests, not blind evidence.

No fixture images, held-out controls, model output, or scoring artefacts are read.
"""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import common_readout_ridges as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402

EMPTY_POINTS = np.empty((0, 3))
EMPTY_SEGMENTS = np.empty((0, 2, 3))
VOXEL = .01


def sampled(segments):
    """Use precisely the public primitive sampling contract, not an approximation."""
    return sample_segments(np.asarray(segments, float), VOXEL / 2, 2000000)


def canonical(segments):
    lines = np.asarray(segments).copy()
    for line in lines:
        if tuple(line[1]) < tuple(line[0]):
            line[:] = line[::-1]
    return np.asarray(sorted(lines, key=lambda line: tuple(line.ravel()))).reshape(-1, 2, 3)


class RidgeReadoutContracts(unittest.TestCase):
    def test_slanted_finite_segment_and_identically_sampled_points_agree(self):
        segment = np.array([[[.013, .017, .019], [.213, .317, 1.019]]])
        as_segment = reader.readout(EMPTY_POINTS, segment, {})
        as_points = reader.readout(sampled(segment), EMPTY_SEGMENTS, {})
        self.assertEqual(as_segment["components"], 1)
        np.testing.assert_allclose(canonical(as_segment["segments"]), segment, atol=1e-12)
        np.testing.assert_array_equal(as_segment["segments"], as_points["segments"])
        self.assertEqual(as_segment["occupied_voxels"], as_points["occupied_voxels"])
        self.assertEqual(as_segment["unique_points"], as_points["unique_points"])

    def test_order_multiplicity_and_primitive_mix_do_not_change_geometry(self):
        segment = np.array([[[.013, .017, .019], [.213, .317, 1.019]]])
        points = sampled(segment)
        first = reader.readout(points, EMPTY_SEGMENTS, {})
        shuffled = np.repeat(points[np.random.default_rng(321).permutation(len(points))], 4, axis=0)
        repeated = reader.readout(shuffled, segment, {})
        for key in ("occupied_voxels", "unique_points", "components", "training_points", "validation_points"):
            self.assertEqual(first[key], repeated[key], key)
        np.testing.assert_array_equal(first["segments"], repeated["segments"])

    def test_diagonal_line_has_independent_validation_cells(self):
        # A linear parity hash gives every (k, 0, k) cell the same fold.
        segment = np.array([[[.003, .003, .003], [.803, .003, .803]]])
        result = reader.readout(EMPTY_POINTS, segment, {})
        self.assertEqual(result["components"], 1, result.get("reason"))
        self.assertGreater(result["training_points"], 3)
        self.assertGreater(result["validation_points"], 3)

    def test_three_collinear_runs_retain_two_explicit_long_gaps(self):
        segments = np.array([[[.013, .017, start], [.013, .017, end]]
                             for start, end in ((.019, .419), (.719, 1.119), (1.419, 1.819))])
        result = reader.readout(EMPTY_POINTS, segments, {})
        self.assertEqual(result["components"], 3, result.get("reason"))
        np.testing.assert_allclose(canonical(result["segments"]), segments, atol=1e-12)
        for low, high in ((.419, .719), (1.119, 1.419)):
            for line in result["segments"]:
                ends = np.sort(line[:, 2])
                self.assertFalse(ends[0] < low + 1e-9 and ends[1] > high - 1e-9)

    def test_parallel_subvoxel_members_cannot_support_an_empty_middle_axis(self):
        segments = np.array([[[-.004, .013, .017], [-.004, .013, 1.017]],
                             [[.004, .013, .017], [.004, .013, 1.017]]])
        result = reader.readout(EMPTY_POINTS, segments, {})
        # Resolving members or abstaining is permissible; an unsupported x=0 axis is not.
        for line in result["segments"]:
            lateral = np.min(np.abs(line[:, 0, None] - np.array([-.004, .004])), axis=1)
            self.assertTrue(np.all(lateral <= .0015 + 1e-12), line)

    def test_three_coextensive_separate_generators_remain_surface_ambiguous(self):
        segments = np.array([[[x, .013, .017], [x, .013, 1.017]] for x in (-.06, 0., .06)])
        result = reader.readout(EMPTY_POINTS, segments, {})
        self.assertEqual(result["components"], 0, result["segments"])

    def test_unrelated_nonparallel_line_neither_cancels_nor_inherits_parallel_ambiguity(self):
        lines = np.array([[[x, .013, .017], [x, .013, 1.017]] for x in (-.06, 0., .06)]
                         + [[[.2, .2, .2], [.7, .2, .7]]])
        original = lines.copy()
        rejected = reader._parallel_ambiguity(lines, reader.policy({}))
        np.testing.assert_array_equal(rejected, [True, True, True, False])
        np.testing.assert_array_equal(lines, original)

    def test_outer_radius_neighbor_does_not_fill_an_inner_support_gap(self):
        segments = np.array([[[0., 0., .019], [0., 0., .419]],
                             [[0., 0., .719], [0., 0., 1.119]],
                             [[.004, 0., .419], [.004, 0., .719]]])
        points = np.unique(sampled(segments), axis=0)
        train_mask = reader._training_mask(np.floor(points / VOXEL).astype(np.int64))
        lines, _ = reader._ridge_runs(points, train_mask, np.zeros(3), np.array([0., 0., 1.]), reader.policy({}))
        self.assertEqual(len(lines), 2)
        np.testing.assert_allclose(canonical(lines), segments[:2], atol=1e-12)

    def test_overlap_merging_cannot_extend_a_parallel_axis_beyond_its_actual_support(self):
        segments = np.array([[[0., .013, 0.], [0., .013, .4]],
                             [[.004, .013, .3], [.004, .013, 1.4]],
                             [[0., .013, 1.3], [0., .013, 1.7]]])
        result = reader.readout(EMPTY_POINTS, segments, {})
        output_points = sampled(result["segments"])
        minimum_distance = np.full(len(output_points), np.inf)
        for start, end in segments:
            axis = end - start
            fraction = np.clip((output_points - start) @ axis / (axis @ axis), 0, 1)
            distance = np.linalg.norm(output_points - start - fraction[:, None] * axis, axis=1)
            minimum_distance = np.minimum(minimum_distance, distance)
        self.assertTrue(np.all(minimum_distance <= .0015 + 1e-12), result["segments"])

    def test_circle_requires_angular_coverage_not_only_sector_count(self):
        # Both arcs occupy many sectors, but 260 degrees leaves an excessive gap.
        for coverage, expect_circle in ((260., False), (320., True)):
            with self.subTest(coverage=coverage):
                angles = np.deg2rad(np.linspace(-coverage / 2, coverage / 2, 81))
                theta, z = np.meshgrid(angles, np.linspace(.017, 1.017, 121))
                points = np.c_[.03 * np.cos(theta.ravel()), .03 * np.sin(theta.ravel()), z.ravel()]
                train_mask = np.repeat(np.arange(121) % 2 == 0, len(angles))
                lines, evidence = reader._sliced_circle(points, train_mask, reader.policy({}))
                self.assertEqual(bool(lines), expect_circle)
                if evidence:
                    self.assertTrue(all(part["angular_coverage_degrees"] >= 270 for part in evidence["slices"]))

    def test_all_points_in_each_voxel_share_exactly_one_fold(self):
        segments = np.array([[[.013, .017, .019], [.013, .017, 1.019]]])
        points = sampled(segments)
        # More than one distinct coordinate in each occupied cell; no exact duplicates.
        points = np.r_[points, points + [.0003, .0002, 0]]
        captured = []
        original = reader._ridge_runs

        def observe(values, train_mask, anchor, axis, config):
            grid = np.floor((values - config["origin"]) / config["voxel_size"]).astype(np.int64)
            train_cells = {tuple(cell) for cell in grid[train_mask]}
            validation_cells = {tuple(cell) for cell in grid[~train_mask]}
            self.assertTrue(train_cells)
            self.assertTrue(validation_cells)
            self.assertFalse(train_cells & validation_cells)
            captured.append((train_cells, validation_cells))
            return original(values, train_mask, anchor, axis, config)

        with patch.object(reader, "_ridge_runs", side_effect=observe):
            result = reader.readout(points, EMPTY_SEGMENTS, {})
        self.assertTrue(captured, "The test must actually exercise held-out ridge validation")
        self.assertEqual(result["components"], 1)
        self.assertEqual(result["training_points"] + result["validation_points"], result["unique_points"])

    def test_regular_sheet_and_dense_sheet_motherline_are_not_accepted_as_rods(self):
        x, z = np.meshgrid(np.linspace(-.06, .06, 25), np.linspace(.017, 1.017, 101))
        sheet = np.c_[x.ravel(), np.full(x.size, .013), z.ravel()]
        dense_edge = np.c_[np.full(1001, .06), np.full(1001, .013), np.linspace(.017, 1.017, 1001)]
        for points in (sheet, np.r_[sheet, dense_edge]):
            with self.subTest(dense=len(points) > len(sheet)):
                result = reader.readout(points, EMPTY_SEGMENTS, {})
                self.assertEqual(result["components"], 0, result["segments"])

    def test_densely_sampled_cylinder_generator_cannot_replace_the_axis(self):
        # A complete regular cylinder plus a more densely sampled surface generator.
        theta, z = np.meshgrid(np.arange(32) * (2 * np.pi / 32), np.linspace(.017, 1.017, 101))
        shell = np.c_[.03 * np.cos(theta.ravel()), .03 * np.sin(theta.ravel()), z.ravel()]
        generator = np.c_[np.full(1001, .03), np.zeros(1001), np.linspace(.017, 1.017, 1001)]
        result = reader.readout(np.r_[shell, generator], EMPTY_SEGMENTS, {})
        # Abstention is allowed; any output must be the geometrically evidenced center.
        for line in result["segments"]:
            self.assertTrue(np.all(np.linalg.norm(line[:, :2], axis=1) < .002), line)

    def test_invalid_policies_are_rejected(self):
        invalid = ({"voxel_size": 0}, {"voxel_size": np.nan}, {"origin": [0, 0]},
                   {"origin": [0, np.inf, 0]}, {"trials": True}, {"trials": 4097},
                   {"maximum_voxels": 2000001}, {"maximum_unique_points": -1},
                   {"minimum_core_fraction": 1.1}, {"expanded_radius_ratio": 1},
                   {"circle_slices": 1}, {"circle_sectors": 65},
                   {"minimum_circle_sectors": 2}, {"merge_angle_degrees": 90},
                   {"candidate_identity": "rod"})
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                reader.readout(EMPTY_POINTS, EMPTY_SEGMENTS, config)

    def test_nonfinite_malformed_and_unrepresentable_geometry_are_rejected(self):
        invalid = (([[np.nan, 0, 0]], EMPTY_SEGMENTS), ([[np.inf, 0, 0]], EMPTY_SEGMENTS),
                   (np.zeros((2, 2)), EMPTY_SEGMENTS), (EMPTY_POINTS, np.zeros((2, 3))),
                   (EMPTY_POINTS, [[[0, 0, 0], [0, np.inf, 1]]]))
        for points, segments in invalid:
            with self.subTest(points=np.shape(points), segments=np.shape(segments)), self.assertRaises(ValueError):
                reader.readout(points, segments, {})
        for points, segments in (([[1e18, 0, 0]], EMPTY_SEGMENTS),
                                 (EMPTY_POINTS, [[[0, 0, 0], [1e18, 0, 0]]])):
            with self.subTest(kind="precision_preflight"), np.errstate(over="raise", invalid="raise"):
                with self.assertRaisesRegex(ValueError, "precision"):
                    reader.readout(points, segments, {})

    def test_resource_exhaustion_is_explicit_and_returns_no_geometry(self):
        segment = np.array([[[.013, .017, .019], [.013, .017, 1.019]]])
        points = sampled(segment)
        cases = ((points, EMPTY_SEGMENTS, {"maximum_voxels": 2}),
                 (points, EMPTY_SEGMENTS, {"maximum_unique_points": 2}),
                 (EMPTY_POINTS, segment, {"maximum_curve_samples": 2}),
                 (EMPTY_POINTS, np.array([[[0., 0, 0], [1e8, 0, 0]]]), {}))
        for cloud, curves, config in cases:
            with self.subTest(config=config, endpoint=curves.tolist()):
                result = reader.readout(cloud, curves, config)
                self.assertEqual(result["state"], "unmeasurable")
                self.assertIn("budget_exceeded", result["reason"])
                self.assertEqual(result["segments"].shape, (0, 2, 3))

    def test_finite_output_and_inputs_and_policy_remain_unchanged(self):
        segment = np.array([[[.013, .017, .019], [.213, .317, 1.019]]])
        points = sampled(segment)
        config = {"voxel_size": VOXEL, "origin": [-.001, -.003, -.007]}
        expected_points, expected_segment, expected_config = points.copy(), segment.copy(), copy.deepcopy(config)
        result = reader.readout(points, segment, config)
        self.assertTrue(np.isfinite(result["segments"]).all())
        self.assertTrue(np.all(np.linalg.norm(np.diff(result["segments"], axis=1), axis=2) > 0))
        np.testing.assert_array_equal(points, expected_points)
        np.testing.assert_array_equal(segment, expected_segment)
        self.assertEqual(config, expected_config)


if __name__ == "__main__":
    unittest.main()
