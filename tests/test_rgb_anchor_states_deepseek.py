"""Known-answer contracts for RGB anchor row states; no model weights or Blender required.

classify_anchor_rows(rows, left, right) is a pure function over the anchor x
uncertainty interval [left, right] and the per-pixel-row observation bands in
`rows`. It returns exactly one of "supported", "contradicted" or "unresolved".
These tests pin only the mechanical contract with small hand-written intervals
and expected strings; they do not mock or reimplement the API.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rgb_anchored_chain_support import classify_anchor_rows  # noqa: E402


class RgbAnchorRowStateTests(unittest.TestCase):
    def test_all_rows_fully_contain_anchor_interval_is_supported(self):
        result = classify_anchor_rows([[(1, 6)], [(0, 8)], [(-2, 7)]], 2, 5)
        self.assertEqual(result, "supported")

    def test_all_rows_disjoint_are_contradicted(self):
        # Codex correction: the entire uncertainty box is contradicted only when every row is disjoint.
        result = classify_anchor_rows([[(-2, 1)], [(8, 9), (10, 11)]], 2, 5)
        self.assertEqual(result, "contradicted")

    def test_empty_row_blocks_supported_and_yields_unresolved(self):
        # 空行 [] 表示该行无已选候选观测，属于 unknown，不能产生 supported。
        result = classify_anchor_rows([[], [(1, 6)], [(0, 8)]], 2, 5)
        self.assertEqual(result, "unresolved")

    def test_partial_overlap_and_boundary_touch_yield_unresolved(self):
        cases = [
            # 部分重叠但不完整包含 [2,5]。
            ([[(0, 4)], [(1, 6)]], 2, 5),
            # 边界接触属于 overlap，但未完整包含。
            ([[(0, 2)]], 2, 5),
            ([[(5, 9)]], 2, 5),
        ]
        for rows, left, right in cases:
            with self.subTest(rows=rows, left=left, right=right):
                self.assertEqual(classify_anchor_rows(rows, left, right), "unresolved")

    def test_nested_and_reordered_bands_keep_supported(self):
        variants = [
            [[(0, 9), (1, 6)], [(1, 7), (2, 5)]],
            [[(1, 6), (0, 9)], [(2, 5), (1, 7)]],
            [[(1, 7), (2, 5)], [(1, 6), (0, 9)]],
        ]
        for rows in variants:
            with self.subTest(rows=rows):
                self.assertEqual(classify_anchor_rows(rows, 2, 5), "supported")

    def test_separated_bands_with_anchor_in_gap_are_contradicted(self):
        # 两个分离 band 位于两侧，anchor 区间 [4,6] 落在中间空隙。
        result = classify_anchor_rows([[(0, 1), (8, 9)]], 4, 6)
        self.assertEqual(result, "contradicted")


if __name__ == "__main__":
    unittest.main()
