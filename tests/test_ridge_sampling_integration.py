"""Synthetic cadence/continuity mechanisms; no experiment inputs or scores."""

import unittest
from unittest.mock import patch

import numpy as np
import test_context_integration as prior
from creator_eval import common_readout_ridge_sampling as reader

EMPTY = np.empty((0, 2, 3))


def sparse_line():
    return np.c_[np.full(80, .021), np.full(80, .033), .017+np.arange(80)*.013]


class RidgeSamplingIntegrationTests(unittest.TestCase):
    def setUp(self):
        for module in (prior, prior.prior):
            active = patch.object(module, "reader", reader)
            active.start()
            self.addCleanup(active.stop)

    def test_sparse_line_recovers_at_both_voxel_scales(self):
        points = sparse_line()
        for voxel in (.0045, .0075):
            with self.subTest(voxel=voxel):
                result = reader.readout(points, EMPTY, {"voxel_size": voxel})
                self.assertEqual(result["state"], "complete")
                self.assertEqual(len(result["segments"]), 1)
                line = result["segments"][0]
                np.testing.assert_allclose(line[np.argsort(line[:, 2])], points[[0, -1]], atol=1e-10)
                self.assertTrue(any(a["sampling_scale"]["learning_state"] == "training_cadence"
                                    for a in result["ridge_fit_audits"]))

    def test_large_observed_gap_splits_actual_endpoints(self):
        points = sparse_line()
        points = np.r_[points[:30], points[50:]]
        result = reader.readout(points, EMPTY, {"voxel_size": .0045})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 2)
        endpoints = np.sort(np.asarray(result["segments"])[:, :, 2].ravel())
        np.testing.assert_allclose(endpoints, points[[0, 29, 30, -1], 2], atol=1e-10)

    def test_small_axial_jitter_cannot_collapse_adjacent_sampling_stations(self):
        points = sparse_line()
        points[:, 2] += np.random.default_rng(53129).uniform(-.00001, .00001, len(points))
        mask = np.arange(len(points)) % 3 != 0
        lines, detail, audit = reader._ridge_runs(points, mask, points[0],
            np.array([0., 0., 1.]), reader.policy({"voxel_size": .0045}))
        self.assertEqual(audit["sampling_scale"]["learning_state"], "training_cadence")
        self.assertEqual(len(lines), 1)
        self.assertEqual(detail[0]["axial_coverage"], 1.)
        self.assertAlmostEqual(detail[0]["validation_coverage"], 27/80)

    def test_noncollinear_three_rods_are_not_a_planar_array(self):
        points = sparse_line()
        points = np.r_[points, points+[.08, 0, 0], points+[.039, .072, 0]]
        result = reader.readout(points, EMPTY, {"voxel_size": .0045})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 3)
        self.assertEqual(result["rejected_parallel_arrays"], 0)

    def test_sparse_collinear_array_remains_unresolved(self):
        points = np.concatenate([sparse_line()+[.02*i, 0, 0] for i in range(-3, 4)])
        result = reader.readout(points, EMPTY, {"voxel_size": .0045})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 0)

    def test_training_api_never_receives_validation_projection(self):
        points = sparse_line()
        p = reader.policy({"voxel_size": .0045})
        mask = reader._training_mask(np.floor(points/p["voxel_size"]).astype(np.int64))
        original = reader.ridge_sampling_evidence.train_scale
        with patch.object(reader.ridge_sampling_evidence, "train_scale", wraps=original) as fit:
            reader._ridge_runs(points, mask, points[0], np.array([0., 0., 1.]), p)
        np.testing.assert_array_equal(fit.call_args.args[0], (points-points[0])[mask, 2])

    def test_validation_change_cannot_refit_the_training_scale(self):
        points = sparse_line()
        mask = np.arange(len(points)) % 2 == 0
        p = reader.policy({"voxel_size": .0045})
        anchor, axis = points[0].copy(), np.array([0., 0., 1.])
        _, _, first = reader._ridge_runs(points, mask, anchor, axis, p)
        points[~mask, 2] += .003
        _, _, changed = reader._ridge_runs(points, mask, anchor, axis, p)
        self.assertEqual(first["sampling_scale"], changed["sampling_scale"])

    def test_cadence_budget_failure_propagates_without_partial_geometry(self):
        result = reader.readout(sparse_line(), EMPTY,
                                {"voxel_size": .0045, "ridge_sampling_maximum_points": 6})
        self.assertEqual(result["state"], "unmeasurable")
        self.assertEqual(result["reason"], "ridge_sampling_point_budget_exceeded")
        self.assertEqual(len(result["segments"]), 0)

    def test_rejected_long_run_has_explicit_fold_gate_diagnostics(self):
        points = sparse_line()
        _, _, audit = reader._ridge_runs(points, np.ones(len(points), bool), points[0],
                                         np.array([0., 0., 1.]), reader.policy({"voxel_size": .0045}))
        self.assertEqual(audit["gate_counts"], {"independent_fold_coverage": 1})
        self.assertEqual(audit["measured_runs"][0]["validation_coverage"], 0.)
        self.assertFalse(audit["measured_runs"][0]["accepted"])


for _name in dir(prior.ContextIntegrationTests):
    if _name.startswith("test_"):
        setattr(RidgeSamplingIntegrationTests, _name, getattr(prior.ContextIntegrationTests, _name))


if __name__ == "__main__":
    unittest.main()
