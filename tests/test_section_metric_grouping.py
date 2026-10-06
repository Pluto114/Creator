"""Geometry contracts for section metric grouping, without GT or pack IO."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import section_evidence as evidence  # noqa: E402
from creator_eval import section_metric_grouping as grouping  # noqa: E402

ANCHOR = np.zeros(3)
AXIS = np.array([0, 0, 1])


def policy(**extra):
    config = {"voxel_size": 1.0}
    config.update(extra)
    return evidence.policy(config)


def grouped(points, train_mask, **extra):
    return grouping.groups(points, train_mask, ANCHOR, AXIS, policy(**extra))


def training_partition(groups, train_mask):
    """Set-of-sets of original training indices, independent of group numbering."""
    return frozenset(frozenset(int(i) for i in group if train_mask[i]) for group in groups)


class SectionMetricGroupingTests(unittest.TestCase):
    def test_diagonal_training_points_within_1_5_voxel_share_one_group(self):
        points = np.array([[0.49, 0.49, 0.0], [1.51, 1.01, 1.0]])
        train = np.array([True, True])
        groups, detail = grouped(points, train)
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(np.sort(groups[0]), np.array([0, 1]))
        self.assertEqual(detail["training_group_count"], 1)

    def test_distant_training_points_form_two_groups(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]])
        train = np.array([True, True])
        groups, detail = grouped(points, train)
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), 2)
        self.assertEqual(detail["training_group_count"], 2)
        self.assertEqual(training_partition(groups, train),
                         frozenset({frozenset({0}), frozenset({1})}))

    def test_heldout_midpoint_cannot_bridge_training_groups(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0], [1.5, 0.0, 0.5]])
        train = np.array([True, True, False])
        groups, detail = grouped(points, train)
        self.assertIsNotNone(groups)
        self.assertEqual(detail["training_group_count"], 2)
        present = [2 in group for group in groups]
        self.assertLessEqual(sum(present), 1)  # heldout attaches to at most one group
        for group in groups:
            self.assertLessEqual(sum(bool(train[i]) for i in group), 1)
        self.assertEqual(training_partition(groups, train),
                         frozenset({frozenset({0}), frozenset({1})}))

    def test_validation_point_changes_leave_training_partition_unchanged(self):
        base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]])
        reference_points = np.r_[base, [[1.5, 0.0, 0.5]]]
        reference_train = np.array([True, True, False])
        reference_groups, _ = grouped(reference_points, reference_train)
        self.assertIsNotNone(reference_groups)
        reference = training_partition(reference_groups, reference_train)
        self.assertEqual(reference, frozenset({frozenset({0}), frozenset({1})}))
        variants = [
            base,                                            # heldout removed
            np.r_[base, [[1.5, 0.0, 0.5]]],                  # heldout added back
            np.r_[base, [[0.5, 0.0, 0.0]]],                  # heldout shifted
            np.r_[base, [[1.5, 0.0, 0.5], [0.5, 0.0, 0.0]]],  # second heldout added
        ]
        for points in variants:
            train = np.zeros(len(points), dtype=bool)
            train[:2] = True
            with self.subTest(n_points=len(points)):
                groups, _ = grouped(points, train)
                self.assertIsNotNone(groups)
                self.assertEqual(training_partition(groups, train), reference)

    def test_shuffled_order_preserves_geometric_grouping(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]])
        train = np.array([True, True])
        perm = np.array([1, 0])
        shuffled = points[perm]
        shuffled_train = train[perm]

        def canonical(group_list, mask, mapping):
            return frozenset(
                frozenset(int(mapping(i)) for i in group if mask[i])
                for group in group_list
            )

        groups, _ = grouped(points, train)
        shuffled_groups, _ = grouped(shuffled, shuffled_train)
        self.assertIsNotNone(groups)
        self.assertIsNotNone(shuffled_groups)
        self.assertEqual(canonical(groups, train, lambda i: i),
                         canonical(shuffled_groups, shuffled_train, lambda i: int(perm[i])))

    def test_cell_budget_overflow_returns_unmeasurable_not_empty_success(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]])
        train = np.array([True, True])
        groups, detail = grouped(points, train, section_maximum_cells=1)
        self.assertIsNone(groups)
        self.assertEqual(detail["state"], "unmeasurable")


if __name__ == "__main__":
    unittest.main()
