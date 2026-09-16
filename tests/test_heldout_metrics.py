"""Synthetic projection checks only: never read held-out predictions or labels."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.heldout_metrics import evaluate_projected_candidate


def labels(xs, surface_ids=None, visible=None, fractions=None, gaps=None):
    count = len(xs)
    gap = np.zeros(count, bool) if gaps is None else np.asarray(gaps, bool)
    return {
        "uv_index": np.c_[np.asarray(xs) + 10.0, np.full(count, 10.0)],
        "rod_id": np.full(count, 6),
        "segment_surface_id": np.ones(count, int)
        if surface_ids is None
        else np.asarray(surface_ids),
        "segment_fraction": np.linspace(0, 1, count)
        if fractions is None
        else np.asarray(fractions),
        "visible": ~gap if visible is None else np.asarray(visible, bool),
        "inside_image": np.ones(count, bool),
        "within_clip": np.ones(count, bool),
        "intentional_gap": gap,
        "gap_clear": gap.copy(),
    }


def segment(a, b, z=1.0):
    return np.array([[a, 0.0, z], [b, 0.0, z]])


class HeldoutMetricsTests(unittest.TestCase):
    k = np.array([[1.0, 0, 10], [0, 1, 10], [0, 0, 1]])

    def score(self, candidate, truth, **kwargs):
        return evaluate_projected_candidate(
            candidate,
            self.k,
            np.eye(4),
            [30, 30],
            truth,
            6,
            tolerance_px=kwargs.pop("tolerance_px", 0.01),
            **kwargs,
        )

    def gap_truth(self):
        return labels(
            [-4, -3, -2, 2, 3, 4, -2, -1, 0, 1, 2],
            surface_ids=[6, 6, 6, 7, 7, 7, 0, 0, 0, 0, 0],
            fractions=[0, 0.5, 1, 0, 0.5, 1, 0, 0.25, 0.5, 0.75, 1],
            gaps=[False] * 6 + [True] * 5,
        )

    def test_correct_finite_pieces_keep_real_gap(self):
        result = self.score([segment(-4, -2), segment(2, 4)], self.gap_truth())
        self.assertEqual(result["visible_gt_to_candidate"]["recall_fraction"], 1)
        self.assertEqual(result["candidate_to_visible_gt"]["precision_fraction"], 1)
        self.assertEqual(result["gap_clear_interior_to_candidate"]["sample_count"], 3)
        self.assertEqual(result["gap_clear_interior_to_candidate"]["false_coverage_fraction"], 0)
        self.assertEqual(result["visible_gt_to_candidate"]["distance_max_px"], 0)

    def test_bridging_gap_does_not_gain_precision_from_infinite_gt_lines(self):
        result = self.score([segment(-4, 4)], self.gap_truth())
        self.assertEqual(result["visible_gt_to_candidate"]["recall_fraction"], 1)
        self.assertAlmostEqual(result["candidate_to_visible_gt"]["precision_fraction"], 0.5)
        self.assertEqual(
            result["gap_clear_guarded_interior_to_candidate"]["false_coverage_fraction"], 1
        )

    def test_finite_candidate_does_not_recover_gt_beyond_its_endpoints(self):
        result = self.score([segment(-2, 2)], self.gap_truth())
        self.assertAlmostEqual(result["visible_gt_to_candidate"]["recall_fraction"], 2 / 6)
        self.assertAlmostEqual(result["visible_gt_to_candidate"]["distance_mean_px"], 1)
        self.assertAlmostEqual(result["visible_gt_to_candidate"]["distance_max_px"], 2)

    def test_occluded_points_do_not_get_connected_after_filtering(self):
        truth = labels(
            np.arange(-4, 5), visible=[True, True, True, False, False, False, True, True, True]
        )
        result = self.score([segment(-1, 1)], truth)
        self.assertEqual(result["visible_gt_to_candidate"]["sample_count"], 6)
        self.assertEqual(result["candidate_to_visible_gt"]["precision_fraction"], 0)

    def test_single_visible_sample_remains_point_support(self):
        result = self.score([segment(-1, 1)], labels([0]), tolerance_px=0.3)
        self.assertEqual(result["visible_gt_to_candidate"]["recall_fraction"], 1)
        self.assertEqual(result["candidate_to_visible_gt"]["precision_fraction"], 0.5)

    def test_empty_candidate_is_zero_recall_and_undefined_precision(self):
        result = self.score([], self.gap_truth())
        self.assertEqual(result["visible_gt_to_candidate"]["recall_fraction"], 0)
        self.assertEqual(result["gap_clear_interior_to_candidate"]["false_coverage_fraction"], 0)
        self.assertIsNone(result["candidate_to_visible_gt"]["precision_fraction"])
        self.assertIsNone(result["visible_gt_to_candidate"]["distance_mean_px"])
        json.dumps(result, allow_nan=False)

    def test_behind_camera_and_crossing_near_plane_are_counted_and_rejected(self):
        candidates = [segment(-1, 1, z=-1), [[-1, 0, -0.1], [1, 0, 1]], segment(-1, 1, z=0.05)]
        result = self.score(candidates, labels([-1, 0, 1]), clip_start=0.1)
        self.assertEqual(result["candidate_projection"]["rejected_at_or_crossing_near_plane"], 3)
        self.assertTrue(result["candidate_projection"]["has_depth_rejections"])
        self.assertEqual(result["visible_gt_to_candidate"]["recall_fraction"], 0)

    def test_offscreen_segments_and_finite_image_clipping(self):
        result = self.score([segment(-30, -20), segment(-20, 0)], labels([-5, 0]))
        self.assertEqual(result["candidate_projection"]["rejected_outside_image"], 1)
        self.assertEqual(result["candidate_projection"]["clipped_to_image"], 1)
        self.assertAlmostEqual(result["candidate_to_visible_gt"]["projected_length_px"], 10.5)

    def test_gap_endpoint_tolerance_guard_does_not_call_correct_endpoints_a_bridge(self):
        result = self.score([segment(-4, -2), segment(2, 4)], self.gap_truth(), tolerance_px=1.1)
        self.assertAlmostEqual(
            result["gap_clear_interior_to_candidate"]["false_coverage_fraction"], 2 / 3
        )
        self.assertEqual(result["gap_clear_guarded_interior_to_candidate"]["sample_count"], 1)
        self.assertEqual(
            result["gap_clear_guarded_interior_to_candidate"]["false_coverage_fraction"], 0
        )

    def test_index_intrinsics_and_camera_transform(self):
        truth = labels([0])
        truth["uv_index"] = np.array([[9.5, 9.5]])
        intrinsic = self.k.copy()
        intrinsic[:2, 2] = 9.5
        camera = np.eye(4)
        camera[:3, 3] = [-5, 0, 0]
        result = evaluate_projected_candidate(
            [[[5, 0, 1], [5, 1, 1]]], intrinsic, camera, [30, 30], truth, 6, tolerance_px=0
        )
        self.assertEqual(result["visible_gt_to_candidate"]["distance_max_px"], 0)

    def test_empty_visible_gt_has_no_recall_claim(self):
        truth = labels([-1, 0, 1], visible=[False, False, False])
        result = self.score([segment(-1, 1)], truth)
        self.assertIsNone(result["visible_gt_to_candidate"]["recall_fraction"])
        self.assertEqual(result["candidate_to_visible_gt"]["precision_fraction"], 0)

    def test_arc_weights_do_not_reward_splitting_the_good_segment(self):
        truth = labels([-4, -3, -2])
        original = self.score([segment(-4, 4)], truth)
        split = self.score([segment(-4, -3), segment(-3, -2), segment(-2, 4)], truth)
        self.assertAlmostEqual(original["candidate_to_visible_gt"]["precision_fraction"], 0.25)
        self.assertAlmostEqual(
            original["candidate_to_visible_gt"]["precision_fraction"],
            split["candidate_to_visible_gt"]["precision_fraction"],
        )


if __name__ == "__main__":
    unittest.main()
