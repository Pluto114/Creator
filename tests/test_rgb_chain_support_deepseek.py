"""Contract tests for creator_eval.rgb_chain_support.consistent_chain_mask.

Known-answer checks with hand-built small matrices; no model weights or Blender
required. The production API is implemented elsewhere; these tests only pin the
documented pure-API contract. Tests were not run in the worker workspace.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rgb_chain_support import consistent_chain_mask  # noqa: E402


class ConsistentChainMaskContractTests(unittest.TestCase):
    """Six pure-API contract tests for consistent_chain_mask."""

    def test_same_chain_three_view_votes_accept_and_fewer_reject(self):
        # 同一 chain 内逐 view 各投 1 票，满 3 票通过；少于 3 票的负对照必须拒绝。
        memberships = [
            np.array([[True], [True], [False]], bool),   # view0: c0 = {0, 1}
            np.array([[True], [True], [False]], bool),   # view1: c0 = {0, 1}
            np.array([[True], [False], [False]], bool),  # view2: c0 = {0}
        ]
        np.testing.assert_array_equal(
            consistent_chain_mask(memberships, [[(0,), (0,), (0,)]]),
            np.array([True, False, False]),
        )
        # 5 个 view 时恰好 3 票达到阈值仍接受，2 票拒绝。
        five_view_memberships = [
            np.array([[True], [True]], bool),
            np.array([[True], [True]], bool),
            np.array([[True], [False]], bool),
            np.array([[False], [False]], bool),
            np.array([[False], [False]], bool),
        ]
        np.testing.assert_array_equal(
            consistent_chain_mask(five_view_memberships, [[(0,)] * 5], minimum_views=3),
            np.array([True, False]),
        )

    def test_votes_split_across_chains_do_not_add_up(self):
        # chain A 给点0投2票、chain B 投1票：合计3票但分属两条 chain，必须拒绝。
        memberships = [
            np.array([[True, False], [False, True], [True, False]], bool),  # view0: c0={0,2}, c1={1}
            np.array([[True, False], [False, True], [True, False]], bool),  # view1: c0={0,2}, c1={1}
            np.array([[False, True], [False, True], [True, False]], bool),  # view2: c0={2}, c1={0,1}
        ]
        chain_a = [(0,), (0,), (0,)]
        chain_b = [(1,), (1,), (1,)]
        np.testing.assert_array_equal(
            consistent_chain_mask(memberships, [chain_a, chain_b]),
            np.array([False, True, True]),
        )

    def test_two_candidates_same_view_count_once(self):
        # 同一 view 的两个 candidate 都覆盖同一点也只能算该 view 的 1 票。
        memberships = [
            np.array([[True, True], [True, True]], bool),    # view0: 两个 candidate 都覆盖两点
            np.array([[True, False], [True, False]], bool),  # view1: 仅 c0 覆盖
            np.array([[False, False], [True, False]], bool),  # view2: 仅覆盖点1
        ]
        np.testing.assert_array_equal(
            consistent_chain_mask(memberships, [[(0, 1), (0,), (0,)]]),
            np.array([False, True]),
        )

    def test_all_feasible_chains_are_or_combined(self):
        # 两条 chain 分别支持不同点，最终必须都保留，不能只取第一条或最高排名。
        memberships = [
            np.array([[True, False], [False, False], [False, True], [False, False]], bool),
            np.array([[True, False], [False, False], [False, True], [False, False]], bool),
            np.array([[True, False], [False, False], [False, True], [False, False]], bool),
        ]
        chain_a = [(0,), (0,), (0,)]
        chain_b = [(1,), (1,), (1,)]
        expected = np.array([True, False, True, False])
        np.testing.assert_array_equal(
            consistent_chain_mask(memberships, [chain_a, chain_b]), expected
        )
        # 调整 chain 顺序不改变并集结果。
        np.testing.assert_array_equal(
            consistent_chain_mask(memberships, [chain_b, chain_a]), expected
        )
        # 单条 chain 各自的支持集合互不干扰。
        np.testing.assert_array_equal(
            consistent_chain_mask(memberships, [chain_a]),
            np.array([True, False, False, False]),
        )
        np.testing.assert_array_equal(
            consistent_chain_mask(memberships, [chain_b]),
            np.array([False, False, True, False]),
        )

    def test_permutation_of_candidates_and_chain_order_is_invariant(self):
        # 重排每 view 的 candidate 列并正确重映射 assignments 索引，再重排 chain 列表，
        # 结果必须不变；原结果同时含 True 与 False。
        view0 = np.array(
            [
                [True, False, False],
                [True, False, False],
                [False, True, False],
                [False, True, False],
                [False, False, True],
                [False, False, True],
            ],
            bool,
        )  # c0={0,1}, c1={2,3}, c2={4,5}
        view1 = np.array(
            [
                [True, False],
                [True, False],
                [True, False],
                [False, True],
                [False, True],
                [False, True],
            ],
            bool,
        )  # c0={0,1,2}, c1={3,4,5}
        view2 = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, False, True],
                [False, False, True],
            ],
            bool,
        )  # c0={0}, c1={1,2,3}, c2={4,5}
        memberships = [view0, view1, view2]
        chain_a = [(0,), (0,), (0,)]
        chain_b = [(2,), (1,), (2,)]
        chain_c = [(1,), (0,), (1,)]
        expected = np.array([True, False, True, False, True, True])
        original = consistent_chain_mask(memberships, [chain_a, chain_b, chain_c])
        np.testing.assert_array_equal(original, expected)
        self.assertTrue(original.any())
        self.assertFalse(original.all())
        # 逐 view 重排 candidate 列并重映射 assignments 索引，再重排 chain 顺序。
        permuted = [view0[:, [2, 0, 1]], view1[:, [1, 0]], view2[:, [0, 2, 1]]]
        remapped_a = [(1,), (1,), (0,)]
        remapped_b = [(0,), (0,), (1,)]
        remapped_c = [(2,), (1,), (2,)]
        np.testing.assert_array_equal(
            consistent_chain_mask(permuted, [remapped_c, remapped_a, remapped_b]),
            expected,
        )

    def test_empty_assignments_return_all_false(self):
        # assignments 为空时返回 shape=(N,) 且全 false 的 bool 数组。
        memberships = [
            np.array([[True, False], [False, True], [False, False]], bool),
            np.array([[True, False], [False, True], [False, False]], bool),
            np.array([[True, False], [False, True], [False, False]], bool),
        ]
        result = consistent_chain_mask(memberships, [])
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, np.bool_)
        self.assertFalse(result.any())
        np.testing.assert_array_equal(result, np.array([False, False, False]))


if __name__ == "__main__":
    unittest.main()
