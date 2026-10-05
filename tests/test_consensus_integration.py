"""Supplement contracts independent of scored replay data."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import common_readout_bundle as old  # noqa: E402
from creator_eval import common_readout_consensus as reader  # noqa: E402
from creator_eval import common_readout_fold as folds  # noqa: E402
from creator_eval.ridge_fold_consensus import overlaps_existing  # noqa: E402


def result(segments):
    return dict(state="complete", segments=np.asarray(segments, float).reshape(-1, 2, 3),
        components=len(segments), resolution_state="resolved", ridge_fit_audits=[], section_evidence={})


class ConsensusContracts(unittest.TestCase):
    def test_fixed_partitions_preserve_original_and_are_distinct(self):
        grid = np.c_[np.arange(-100, 100), np.zeros((200, 2))].astype(np.int64)
        np.testing.assert_array_equal(folds._training_mask(grid), old._training_mask(grid))
        for index in (1, 2):
            mask = folds._training_mask(grid, index)
            self.assertTrue(np.any(mask != old._training_mask(grid)))
            np.testing.assert_array_equal(mask[::-1], folds._training_mask(grid[::-1], index))
            np.testing.assert_array_equal(np.repeat(mask, 2), folds._training_mask(np.repeat(grid, 2, axis=0), index))

    def test_fold_indices_cannot_be_searched_by_caller(self):
        for index in (-1, 3, True, 1.5):
            with self.assertRaises(ValueError):
                folds.policy({"ridge_fold_index": index})
        with self.assertRaises(ValueError):
            reader.policy({"ridge_fold_index": 1})

    def test_agreed_missing_branch_is_added_without_moving_baseline(self):
        base = [[[0, 0, 0], [1, 0, 0]]]
        branch = [[.5, 0, 0], [.5, .2, 0]]
        with mock.patch.object(reader.baseline, "readout", return_value=result(base)), mock.patch.object(
                reader.fold_reader, "readout", side_effect=[result(base+[branch]), result(base+[branch])]) as call:
            actual = reader.readout([], [], {})
        np.testing.assert_array_equal(actual["segments"], base+[branch])
        self.assertEqual([args.args[2]["ridge_fold_index"] for args in call.call_args_list], [1, 2])
        self.assertEqual(actual["fold_consensus"]["added_segments"], 1)

    def test_one_fold_cannot_add_a_branch(self):
        branch = [[[0, 0, 0], [0, .2, 0]]]
        with mock.patch.object(reader.baseline, "readout", return_value=result([])), mock.patch.object(
                reader.fold_reader, "readout", side_effect=[result(branch), result([])]):
            actual = reader.readout([], [], {})
        self.assertEqual(len(actual["segments"]), 0)

    def test_shorter_or_extended_same_member_is_not_duplicated(self):
        config = {"voxel_size": .01, "merge_distance_voxels": .5}
        base = np.array([[[0., 0., 0.], [1., 0., 0.]]])
        self.assertTrue(overlaps_existing(np.array([[.1, 0, 0], [.8, 0, 0]]), base, config))
        self.assertTrue(overlaps_existing(np.array([[-.1, 0, 0], [1.1, 0, 0]]), base, config))
        self.assertFalse(overlaps_existing(np.array([[1.1, 0, 0], [1.4, 0, 0]]), base, config))
        self.assertFalse(overlaps_existing(np.array([[.5, 0, 0], [.5, .2, 0]]), base, config))

    def test_unmeasurable_fold_is_not_silently_success(self):
        with mock.patch.object(reader.baseline, "readout", return_value=result([])), mock.patch.object(
                reader.fold_reader, "readout", return_value=dict(state="unmeasurable", reason="budget", segments=[])):
            actual = reader.readout([], [], {})
        self.assertEqual(actual["state"], "unmeasurable")

    def test_output_budget_remains_explicit(self):
        base = [[[0, 0, 0], [1, 0, 0]]]
        branch = [[.5, 0, 0], [.5, .2, 0]]
        with mock.patch.object(reader.baseline, "readout", return_value=result(base)), mock.patch.object(
                reader.fold_reader, "readout", side_effect=[result(base+[branch]), result(base+[branch])]):
            actual = reader.readout([], [], {"maximum_output_segments": 1})
        self.assertEqual(actual["state"], "unmeasurable")
