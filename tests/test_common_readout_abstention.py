"""Abstention contracts on known development geometry, not efficacy qualification."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_abstention import DEFAULTS, readout  # noqa: E402
from creator_eval.common_readout_valley_closed import DEFAULTS as CLOSED_DEFAULTS  # noqa: E402
from creator_eval.common_readout_valley_closed import readout as closed_readout  # noqa: E402
from creator_eval.line_controls import curve_metrics  # noqa: E402
from creator_eval.readout_background_controls import member, panel  # noqa: E402

EMPTY = np.empty((0, 2, 3))
TRUTH = np.array([[[.007, .013, .011], [.007, .013, 1.241]]])


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class AbstentionReadoutTests(unittest.TestCase):
    def test_thresholds_are_exactly_the_frozen_closed_policy(self):
        self.assertEqual(DEFAULTS, CLOSED_DEFAULTS)

    def test_unresolved_empty_output_still_fails_positive_recovery(self):
        result = readout(panel(1.23, .24), EMPTY, {"voxel_size": .02})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["resolution_state"], "unresolved")
        self.assertGreater(result["unresolved_component_count"], 0)
        self.assertEqual(result["components"], 0)
        score = curve_metrics(result["segments"], TRUTH, tolerance=.025, spacing=.005)
        self.assertEqual(score["recovery_fraction"], 0.)
        self.assertIsNone(score["precision_fraction"])
        self.assertFalse(score["recovery_fraction"] >= .9)
        # 同一片点云可以少画假线，但未决不能包装成“认出了背景”。
        self.assertTrue(all(item["state"] == "unresolved" for item in result["unresolved_components"]))

    def test_partial_result_retains_independent_supported_component(self):
        rod = member(TRUTH[0] + [2., 0., 0.], .05, noise=.0015, seed=502)
        standalone = readout(rod, EMPTY, {"voxel_size": .018})
        result = readout(np.concatenate((panel(1.23, .24), rod)), EMPTY, {"voxel_size": .018})
        self.assertEqual(result["resolution_state"], "partial")
        self.assertGreater(result["unresolved_component_count"], 0)
        self.assertEqual(result["components"], 1)
        np.testing.assert_array_equal(result["segments"], standalone["segments"])

    def test_circle_and_half_circle_without_split_proposal_are_unchanged(self):
        for angular_range in ((0., 2 * np.pi), (-np.pi / 2, np.pi / 2)):
            with self.subTest(angular_range=angular_range):
                cloud = member(TRUTH[0], .05, noise=.0015, seed=502, angular_range=angular_range)
                result = readout(cloud, EMPTY, {"voxel_size": .018})
                previous = closed_readout(cloud, EMPTY, {"voxel_size": .018})
                self.assertEqual(result["resolution_state"], "resolved")
                self.assertEqual(result["unresolved_component_count"], 0)
                self.assertEqual(result["components"], 1)
                np.testing.assert_array_equal(result["segments"], previous["segments"])

    def test_resolvable_parallel_members_keep_both_axes(self):
        truth = np.concatenate((TRUTH, TRUTH + [.18, 0., 0.]))
        cloud = np.concatenate([member(axis, .032, noise=.001, seed=500+i) for i, axis in enumerate(truth)])
        result = readout(cloud, EMPTY, {"voxel_size": .02})
        score = curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005)
        self.assertEqual(result["resolution_state"], "resolved")
        self.assertEqual(result["components"], 2)
        self.assertGreater(score["recovery_fraction"], .95)
        self.assertGreater(score["precision_fraction"], .95)

    def test_exact_binary_grid_plane_preserves_closed_boundary_safety(self):
        voxel = 1 / 32
        x, z = np.meshgrid((np.arange(8)-3.5)*voxel, (np.arange(40)+.5)*voxel, indexing="ij")
        cloud = np.c_[x.ravel(), np.full(x.size, voxel/2), z.ravel()]
        result = readout(cloud, EMPTY, {"voxel_size": voxel})
        self.assertEqual(result["components"], 0)
        self.assertEqual(result["resolution_state"], "unresolved")
        self.assertTrue(all(row["gap_points"] > 0 for check in result["split_valley_checks"] for row in check["axial_slices"]))

    def test_resolution_and_component_evidence_survive_json_roundtrip(self):
        result = readout(panel(1.23, .24), EMPTY, {"voxel_size": .02})
        saved = json.loads(json.dumps(result, default=serializable, allow_nan=False))
        self.assertEqual(saved["resolution_state"], "unresolved")
        self.assertEqual(saved["unresolved_component_count"], len(saved["unresolved_components"]))
        self.assertEqual(saved["unresolved_components"], result["unresolved_components"])
        for item in saved["unresolved_components"]:
            self.assertEqual(item["emitted_segments"], 0)
            self.assertEqual(item["proposed_parts"], 2)
            self.assertEqual(item["valley"]["decision"], "abstain_unresolved")
            self.assertTrue(item["component_id"].startswith("component-"))
            self.assertEqual(len(item["bounds_min"]), 3)

    def test_duplicate_order_does_not_change_unresolved_evidence(self):
        cloud = panel(1.23, .24)
        first = readout(cloud, EMPTY, {"voxel_size": .02})
        second = readout(np.repeat(cloud[::-1], 3, axis=0), EMPTY, {"voxel_size": .02})
        self.assertEqual(first["unresolved_components"], second["unresolved_components"])
        self.assertEqual(first["resolution_state"], second["resolution_state"])

    def test_empty_input_has_no_invented_split_uncertainty(self):
        result = readout(np.empty((0, 3)), EMPTY, {"voxel_size": .02})
        self.assertEqual(result["resolution_state"], "no_supported_curve")
        self.assertEqual(result["unresolved_component_count"], 0)
        self.assertEqual(result["components"], 0)

    def test_depth_budget_cannot_silently_reenable_whole_axis(self):
        cloud = member(TRUTH[0], .05, noise=.0015, seed=502)
        # 隔离递归契约：无论深层提议来自什么几何，预算耗尽都不能输出替代轴。
        def valid_proposal(points, *args):
            return [points.copy(), points.copy()], "persistent_transverse_valley", {"supported_slices": 6, "required_slices": 4}
        with patch("creator_eval.common_readout_abstention.guarded_split_component", side_effect=valid_proposal):
            result = readout(cloud, EMPTY, {"voxel_size": .018, "maximum_split_depth": 1})
        self.assertEqual(result["accepted_splits"], 1)
        self.assertEqual(result["unresolved_component_count"], 2)
        self.assertEqual(result["components"], 0)
        self.assertEqual(result["resolution_state"], "unresolved")
        self.assertTrue(all(item["reason"] == "supported_split_exceeds_depth_budget" for item in result["unresolved_components"]))


if __name__ == "__main__":
    unittest.main()
