"""Six contract checks for the naive fixture controls adapter (single-span TLS/RANSAC).

Public API under test::

    fit_controls(points, point_ids, camera_span, mask) -> list[dict]

    points       finite Nx3 float array of world points.
    point_ids    uint32 Nx3 array of unique (frame, row, col) source IDs.
    camera_span  camera span, fixed to 1.0 throughout this contract.
    mask         length-N bool array selecting the points that may contribute.

The result is a list with exactly two records in method order; the test helper
indexes these records by their method fields::

    "depth_tls_single_span", "depth_ransac_single_span"

Every method record carries: state, outcome, reason, segments (Nx2x3),
input_support_count, input_support_sha256, model, elapsed_seconds.

TLS consumes every selected point. RANSAC is fixed at 256 trials, seed 0,
0.005 distance threshold, 0.03 minimum pair distance, 12 minimum inliers and
3 required views. Both methods clip the span endpoints to the observed
min/max projections and emit a single span; no automatic gap inference is
expected, so a gap is naively bridged.

Only synthetic axis-aligned geometry is used here: no model weights, no
Blender, no GT/score files, no real repository data.
"""

import sys
import unittest
from numbers import Integral, Real
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.fixture_naive_controls import fit_controls  # noqa: E402

TLS = "depth_tls_single_span"
RANSAC = "depth_ransac_single_span"
METHODS = (TLS, RANSAC)

RECORD_FIELDS = (
    "state",
    "outcome",
    "reason",
    "segments",
    "input_support_count",
    "input_support_sha256",
    "model",
    "elapsed_seconds",
)


def make_points():
    """30 collinear points along x, unique uint32 (frame, row, col) IDs, full mask."""
    points = np.c_[np.linspace(0.0, 1.0, 30), np.zeros(30), np.zeros(30)]
    ids = np.c_[np.arange(30) % 3, np.arange(30), np.zeros(30)].astype(np.uint32)
    mask = np.ones(30, dtype=bool)
    return points, ids, mask


def make_cluster_points():
    """30 collinear points in two clusters [0, .3] and [.7, 1] with a real gap."""
    x = np.concatenate([np.linspace(0.0, 0.3, 15), np.linspace(0.7, 1.0, 15)])
    points = np.c_[x, np.zeros(30), np.zeros(30)]
    ids = np.c_[np.arange(30) % 3, np.arange(30), np.zeros(30)].astype(np.uint32)
    mask = np.ones(30, dtype=bool)
    return points, ids, mask


def by_method(rows):
    assert [row["method"] for row in rows] == list(METHODS)
    return {row["method"]: row for row in rows}


def assert_record_contract(test_case, record):
    """Structural contract shared by every method record (values checked per test)."""
    for field in RECORD_FIELDS:
        test_case.assertIn(field, record)
    for field in ("state", "outcome"):
        test_case.assertIsInstance(record[field], str)
    if record["reason"] is not None:
        test_case.assertIsInstance(record["reason"], str)
    segments = np.asarray(record["segments"])
    test_case.assertEqual(segments.ndim, 3)
    test_case.assertEqual(segments.shape[1:], (2, 3))
    test_case.assertIsInstance(record["input_support_count"], Integral)
    test_case.assertIsInstance(record["input_support_sha256"], str)
    test_case.assertIsInstance(record["elapsed_seconds"], Real)


def ordered_endpoints(record):
    """The two endpoints of the single span, ordered by x."""
    endpoints = np.asarray(record["segments"][0], dtype=float)
    return endpoints[endpoints[:, 0].argsort()]


def assert_single_span(test_case, record, x_min, x_max):
    """One span whose observed-projection endpoints are exactly x_min..x_max on x."""
    test_case.assertEqual(np.asarray(record["segments"]).shape[0], 1)
    endpoints = ordered_endpoints(record)
    test_case.assertAlmostEqual(endpoints[0, 0], x_min)
    test_case.assertAlmostEqual(endpoints[1, 0], x_max)
    np.testing.assert_allclose(endpoints[:, 1:], 0.0, atol=1e-9)


class FixtureNaiveFitContractTests(unittest.TestCase):
    def test_01_clean_line_fits_one_complete_span(self):
        points, ids, mask = make_points()
        result = by_method(fit_controls(points, ids, 1.0, mask))
        self.assertEqual(list(result), list(METHODS))
        for name in METHODS:
            record = result[name]
            assert_record_contract(self, record)
            self.assertEqual(record["state"], "complete")
            assert_single_span(self, record, 0.0, 1.0)

    def test_02_fully_false_mask_reports_no_supported_change(self):
        points, ids, _ = make_points()
        result = by_method(fit_controls(points, ids, 1.0, np.zeros(30, dtype=bool)))
        self.assertEqual(list(result), list(METHODS))
        for name in METHODS:
            record = result[name]
            assert_record_contract(self, record)
            self.assertEqual(np.asarray(record["segments"]).shape, (0, 2, 3))
            self.assertEqual(record["outcome"], "no_supported_change")
            self.assertTrue(record["reason"].strip())

    def test_03_tls_fits_three_points_ransac_stays_empty(self):
        points, ids, _ = make_points()
        mask = np.zeros(30, dtype=bool)
        mask[:3] = True
        result = by_method(fit_controls(points, ids, 1.0, mask))
        tls, ransac = result[TLS], result[RANSAC]
        assert_record_contract(self, tls)
        assert_record_contract(self, ransac)
        self.assertEqual(tls["state"], "complete")
        assert_single_span(self, tls, 0.0, 2.0 / 29.0)
        self.assertEqual(np.asarray(ransac["segments"]).shape, (0, 2, 3))

    def test_04_two_clusters_keep_naive_bridge(self):
        points, ids, mask = make_cluster_points()
        result = by_method(fit_controls(points, ids, 1.0, mask))
        self.assertEqual(list(result), list(METHODS))
        for name in METHODS:
            record = result[name]
            assert_record_contract(self, record)
            assert_single_span(self, record, 0.0, 1.0)

    def test_05_shuffled_inputs_give_identical_geometry_and_sha(self):
        points, ids, mask = make_points()
        baseline = by_method(fit_controls(points, ids, 1.0, mask))
        perm = np.random.default_rng(0).permutation(30)
        shuffled = by_method(fit_controls(points[perm], ids[perm], 1.0, mask[perm]))
        self.assertEqual(list(shuffled), list(METHODS))
        for name in METHODS:
            base, shuf = baseline[name], shuffled[name]
            assert_record_contract(self, base)
            assert_record_contract(self, shuf)
            self.assertEqual(shuf["input_support_sha256"], base["input_support_sha256"])
            self.assertEqual(
                np.asarray(shuf["segments"]).shape, np.asarray(base["segments"]).shape
            )
            if np.asarray(base["segments"]).shape[0]:
                np.testing.assert_allclose(
                    ordered_endpoints(shuf), ordered_endpoints(base)
                )
        # elapsed_seconds is deliberately not compared.

    def test_06_fit_controls_does_not_mutate_inputs(self):
        points, ids, mask = make_points()
        points_before, ids_before, mask_before = points.copy(), ids.copy(), mask.copy()
        fit_controls(points, ids, 1.0, mask)
        np.testing.assert_array_equal(points, points_before)
        np.testing.assert_array_equal(ids, ids_before)
        self.assertEqual(ids.dtype, np.uint32)
        np.testing.assert_array_equal(mask, mask_before)


if __name__ == "__main__":
    unittest.main()
