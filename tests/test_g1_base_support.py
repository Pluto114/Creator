"""Synthetic-only contracts for the normal-data base-support diagnostic."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_g1_base_support as diagnostic  # noqa: E402


def views():
    return [dict(view_id=f"v{i}", K_index=np.eye(3), world_to_camera_cv=np.c_[np.eye(3), np.zeros(3)],
                 left=np.array([[0.], [0.], [0.]]), right=np.array([[1.], [1.], [1.]])) for i in range(3)]


def frames():
    return [dict(frame_id=f"v{i}", image_sha256=str(i)*64, prediction_size_wh=[3, 3]) for i in range(3)]


class BaseSupportContracts(unittest.TestCase):
    def test_chunked_per_view_votes_equal_original_and_input_unchanged(self):
        points = np.array([[.5, 1., 1.], [20., 1., 1.], [.5, 1., -1.], [1., 2., 1.]])
        before = points.copy()
        matrix = diagnostic.view_support_matrix(points, views(), diagnostic.POLICY, chunk_points=1)
        np.testing.assert_array_equal(matrix, [[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(matrix, diagnostic.view_support_matrix(points, views(), diagnostic.POLICY, 3))
        np.testing.assert_array_equal(points, before)

    def test_duplicate_views_and_changed_policy_are_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic.view_support_matrix(np.zeros((1, 3)), views()[:1]*3, diagnostic.POLICY)
        with self.assertRaises(ValueError):
            diagnostic.view_support_matrix(np.zeros((1, 3)), views(), dict(diagnostic.POLICY, minimum_views=2))

    def test_ids_decode_frame_row_column_and_validate_raster(self):
        ids = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.uint32)
        schema, sources = diagnostic.point_id_schema(ids, frames())
        self.assertEqual(schema["columns"], ["frame_index", "row", "column"])
        np.testing.assert_array_equal(sources, [0, 2])
        for bad in (np.array([[3, 0, 0]]), np.array([[0, 3, 0]]), np.array([[0, 0, 3]]), ids.astype(float)):
            schema, sources = diagnostic.point_id_schema(bad, frames())
            self.assertEqual(schema["state"], "unknown")
            self.assertIsNone(sources)
            self.assertEqual(schema["actual"]["shape"], list(bad.shape))

    def test_spatial_statistics_hand_computable_and_duplicates_retained(self):
        result = diagnostic.spatial_summary(np.array([[0., 0., 0.], [3., 0., 0.], [7., 0., 0.]]))
        self.assertEqual(result["extent"]["span_xyz_m"], [7., 0., 0.])
        self.assertEqual(result["nearest_neighbor_m"]["quantiles"]["0.5"], 3.)
        duplicates = diagnostic.spatial_summary(np.zeros((2, 3)))
        self.assertEqual(duplicates["nearest_neighbor_m"]["zero_distance_count"], 2)
        for n in (0, 1):
            self.assertEqual(diagnostic.spatial_summary(np.zeros((n, 3)))["nearest_neighbor_m"]["reason"], "fewer_than_two_points")

    def test_source_histograms_account_for_every_point_and_exact_mask(self):
        points = np.array([[0., 0., 0.], [2., 0., 0.], [6., 0., 0.], [10., 0., 0.]])
        ids = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0]], dtype=np.uint32)
        matrix = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        result = diagnostic.support_summary(points, ids, frames(), matrix, [1, 1, 0, 2], 2)
        self.assertEqual([r["full_point_count"] for r in result["sources"]], [2, 1, 1])
        self.assertEqual([r["mask_geometry"]["point_count"] for r in result["sources"]], [1, 1, 0])
        self.assertEqual(result["sources"][0]["supported_vote_histogram"], [0, 0, 0, 1])
        self.assertEqual(sum(p["point_count"] for p in result["support_patterns"]), len(points))
        for histogram, count in (([1, 2, 0, 1], 2), ([1, 1, 0, 2], 3)):
            with self.assertRaisesRegex(ValueError, "frozen normal"):
                diagnostic.support_summary(points, ids, frames(), matrix, histogram, count)

    def test_fold_summary_keeps_rejections_not_as_success(self):
        record = dict(unique_fits=2, accepted_ridge_runs=0, ridge_fit_audits=[
            dict(gate_counts={"short": 3}, sampling_scale=dict(reason="few"), measured_runs=[]),
            dict(gate_counts={"short": 2}, sampling_scale=dict(reason=None), measured_runs=[{}])])
        summary = diagnostic.fold_summary(record)
        self.assertEqual(summary["gate_counts"], {"short": 5})
        self.assertEqual(summary["sampling_reason_counts"], {"few": 1, "none": 1})
        self.assertEqual(summary["accepted_ridge_runs"], 0)
        self.assertEqual(summary["measured_run_count"], 1)

    def test_report_is_exclusive_and_strict_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            diagnostic.exclusive_publish(path, {"gt_read": False})
            self.assertEqual(json.loads(path.read_text()), {"gt_read": False})
            with self.assertRaises(FileExistsError):
                diagnostic.exclusive_publish(path, {"overwrite": True})
            with self.assertRaises(ValueError):
                diagnostic.exclusive_publish(Path(folder) / "invalid.json", {"bad": float("nan")})
            self.assertFalse((Path(folder) / "invalid.json").exists())

    def test_truth_guard_allows_only_exclusive_report_write(self):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        diagnostic.block_truth("open", (str(diagnostic.PUBLIC), "x", flags))
        for path, mode, bits in ((diagnostic.PUBLIC, "r", 0), (diagnostic.PUBLIC, "w", os.O_WRONLY),
                                (ROOT / "data/eval_gt/fake.json", "r", 0),
                                (ROOT / "docs/experiments/results/other.json", "x", flags)):
            with self.assertRaises(PermissionError):
                diagnostic.block_truth("open", (str(path), mode, bits))

    def test_changed_receipt_rejected_without_real_pack(self):
        with patch.object(diagnostic, "file_hash", return_value="b"*64):
            with self.assertRaisesRegex(ValueError, "Changed"):
                diagnostic.verify_receipts({"scripts/fake.py": "a"*64})
            with self.assertRaisesRegex(ValueError, "Changed"):
                diagnostic.bind(ROOT / "scripts/fake.py", {}, "a"*64)


if __name__ == "__main__":
    unittest.main()
