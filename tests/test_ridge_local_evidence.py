"""Independent local parallel-array contracts; no formal controls or artifacts."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import ridge_local_evidence as reader  # noqa: E402
from creator_eval.common_readout_sampling import _training_mask  # noqa: E402


def rotation():
    axis = np.array([.3, -.7, .2])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = .83
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew @ skew)


def rows(offsets, *, length=.21, noise=.00012, rotate=True):
    random = np.random.default_rng(81733)
    x = np.linspace(0., length, 97)
    cloud = np.concatenate([np.c_[x, np.full(len(x), y), random.uniform(-noise, noise, len(x))] for y in offsets])
    line = np.array([[0., 0., 0.], [length, 0., 0.]])
    transform = rotation() if rotate else np.eye(3)
    translation = np.array([1.713, -.643, .127])
    return cloud @ transform.T+translation, line @ transform.T+translation


def partition(points, voxel):
    return _training_mask(np.floor(points/voxel).astype(np.int64))


class RidgeLocalEvidenceContracts(unittest.TestCase):
    def test_sparse_plane_rows_rejected_at_two_scales_rotated_and_axis_aligned(self):
        for voxel in (.006, .010):
            for rotate in (False, True):
                for target in (0, 3):
                    with self.subTest(voxel=voxel, rotate=rotate, target=target):
                        points, line = rows((np.arange(7)-target)*.022, rotate=rotate)
                        result = reader.analyze(points, partition(points, voxel), line, dict(voxel_size=voxel))
                        self.assertEqual(result["state"], "complete")
                        self.assertTrue(result["parallel_array"], result)
                        self.assertTrue(result["training_only_proposals"])
                        self.assertFalse(result["validation_refits_model"])
                        self.assertTrue(all(len(family) >= 3 for family in result["training_model"]["families"]))

    def test_single_and_two_parallel_rods_always_remain_not_array(self):
        for voxel in (.006, .010):
            for offsets in ([0.], [0., .007], [0., .022], [0., .19]):
                with self.subTest(voxel=voxel, offsets=offsets):
                    points, line = rows(offsets)
                    result = reader.analyze(points, partition(points, voxel), line, dict(voxel_size=voxel))
                    self.assertEqual(result["state"], "complete")
                    self.assertFalse(result["parallel_array"], result)

    def test_short_branch_is_not_a_persistent_parallel_row_array(self):
        x = np.linspace(0., .24, 121)
        main = np.c_[x, np.zeros((len(x), 2))]
        y = np.linspace(0., .093, 51)
        branch = np.c_[np.full(len(y), .113), y, np.zeros(len(y))]
        cloud = np.r_[main, branch] @ rotation().T
        segments = np.array([[[0., 0., 0.], [.24, 0., 0.]], [[.113, 0., 0.], [.113, .093, 0.]]]) @ rotation().T
        for voxel in (.006, .010):
            for line in segments:
                with self.subTest(voxel=voxel, line=line.tolist()):
                    result = reader.analyze(cloud, partition(cloud, voxel), line, dict(voxel_size=voxel))
                    self.assertFalse(result["parallel_array"], result)

    def test_short_parallel_fragments_do_not_reject_a_long_member(self):
        points, line = rows([0.], rotate=False)
        short = np.linspace(.08, .12, 21)
        stubs = np.concatenate([np.c_[short, np.full(len(short), y), np.zeros(len(short))] for y in (.018, .036)])
        points = np.r_[points, stubs+np.array([1.713, -.643, .127])]
        result = reader.analyze(points, partition(points, .006), line, dict(voxel_size=.006))
        self.assertFalse(result["parallel_array"], result)

    def test_scatter_neither_creates_an_array_nor_erases_repeated_rows(self):
        random = np.random.default_rng(53119)
        outliers = random.uniform([-.01, -.09, -.055], [.23, .09, .055], (900, 3))
        outliers = outliers @ rotation().T+np.array([1.713, -.643, .127])
        for offsets, expected in (([0.], False), ([-.044, -.022, 0., .022, .044], True)):
            with self.subTest(offsets=offsets):
                points, line = rows(offsets)
                points = np.r_[points, outliers]
                result = reader.analyze(points, partition(points, .006), line, dict(voxel_size=.006))
                self.assertEqual(result["parallel_array"], expected, result)

    def test_heldout_points_cannot_nominate_new_rows_or_expand_training_model(self):
        core, line = rows([0.], rotate=False)
        mask = np.arange(len(core)) % 2 == 0
        a = reader.analyze(core, mask, line, dict(voxel_size=.006))
        extra, _ = rows([.022, .044], rotate=False)
        # Even remote validation coordinates must not inflate training numeric
        # tolerances, bins, neighborhoods, or candidate-row geometry.
        extra = np.r_[extra, [[1e12, -1e12, 1e12]]]
        combined, fold = np.r_[core, extra], np.r_[mask, np.zeros(len(extra), bool)]
        b = reader.analyze(combined, fold, line, dict(voxel_size=.006))
        self.assertEqual(a["training_model"], b["training_model"])
        self.assertFalse(b["parallel_array"])

    def test_missing_heldout_neighbor_support_cannot_refit_a_frozen_training_array(self):
        cloud, line = rows([-.022, 0., .022], rotate=False)
        fold = np.tile(np.arange(97) % 2 == 0, 3)
        a = reader.analyze(cloud, fold, line, dict(voxel_size=.006))
        self.assertTrue(a["parallel_array"], a)
        central = np.repeat([False, True, False], 97)
        keep = fold | central
        b = reader.analyze(cloud[keep], fold[keep], line, dict(voxel_size=.006))
        self.assertEqual(a["training_model"], b["training_model"])
        self.assertFalse(b["parallel_array"])
        self.assertEqual(b["reason"], "heldout_array_not_confirmed")

    def test_input_order_duplicates_and_reversed_segment_are_invariant(self):
        points, line = rows([-.044, -.022, 0., .022, .044])
        fold = partition(points, .006)
        a = reader.analyze(points, fold, line, dict(voxel_size=.006))
        order = np.random.default_rng(47329).permutation(len(points))
        b = reader.analyze(np.repeat(points[order], 3, axis=0), np.repeat(fold[order], 3), line[::-1], dict(voxel_size=.006))
        self.assertEqual(a, b)

    def test_three_noncollinear_parallel_members_do_not_form_a_row_array(self):
        x = np.linspace(0., .24, 121)
        points = np.concatenate([np.c_[x, np.full(len(x), y), np.full(len(x), z)] for y, z in ((0., 0.), (.027, 0.), (0., .031))])
        line = np.array([[0., 0., 0.], [.24, 0., 0.]])
        fold = np.tile(np.arange(len(x)) % 2 == 0, 3)
        result = reader.analyze(points, fold, line, dict(voxel_size=.006))
        self.assertFalse(result["parallel_array"], result)
        self.assertEqual(result["reason"], "no_local_collinear_repeated_spacing")

    def test_far_parallel_array_does_not_change_isolated_local_member(self):
        points, line = rows([0., .5, .522, .544])
        result = reader.analyze(points, partition(points, .006), line, dict(voxel_size=.006))
        self.assertFalse(result["parallel_array"], result)
        self.assertEqual(len(result["training_model"]["row_centers"]), 1)

    def test_budgets_are_explicit_unmeasurable_not_negative_evidence(self):
        points, line = rows((np.arange(7)-3)*.018, noise=0., rotate=False)
        fold = np.tile(np.arange(97) % 2 == 0, 7)
        for key, reason in (("ridge_array_maximum_points", "point"), ("ridge_array_maximum_seeds", "seed"),
                            ("ridge_array_maximum_rows", "row"), ("ridge_array_maximum_families", "family")):
            with self.subTest(key=key):
                result = reader.analyze(points, fold, line, dict(voxel_size=.006, **{key: 1}))
                self.assertEqual(result["state"], "unmeasurable", result)
                self.assertFalse(result["parallel_array"])
                self.assertEqual(result["reason"], f"ridge_array_{reason}_budget_exceeded")

    def test_malformed_inputs_and_policy_are_rejected(self):
        points, line = rows([0.])
        fold = partition(points, .006)
        for change in ({"ridge_array_unknown": 1}, {"ridge_array_minimum_axial_bins": 2},
                       {"ridge_array_maximum_rows": True}, {"ridge_array_context_span_fraction": 0.},
                       {"ridge_array_maximum_spacing_ratio": np.nan}, {"ridge_array_support_fraction": 1.1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.analyze(points, fold, line, dict(voxel_size=.006, **change))
        for bad_points, bad_fold, bad_line in ((points, fold.astype(int), line),
                                             (points*np.nan, fold, line), (points, fold, np.zeros((2, 3)))):
            with self.assertRaises(ValueError):
                reader.analyze(bad_points, bad_fold, bad_line, dict(voxel_size=.006))


if __name__ == "__main__":
    unittest.main()
