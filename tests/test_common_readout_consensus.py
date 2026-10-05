"""Replay the unchanged geometry contracts against the consensus supplement."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import test_common_readout_bundle as prior
import test_common_readout_sampling as sampling_tests
from creator_eval import common_readout_bundle as baseline_impl
from creator_eval import common_readout_consensus as consensus
from creator_eval import common_readout_fold as folds


class ConsensusGeometryContracts(prior.BundleContracts):
    def setUp(self):
        facade = SimpleNamespace(**vars(folds))
        facade.readout, facade.policy, facade.DEFAULTS = consensus.readout, consensus.policy, consensus.DEFAULTS
        module = prior
        while module is not None:
            for name, value in (("reader", facade), ("ridge_proposals", folds.ridge_proposals)):
                if hasattr(module, name):
                    active = mock.patch.object(module, name, value)
                    active.start()
                    self.addCleanup(active.stop)
            module = getattr(module, "prior", None)

    def test_all_points_in_each_voxel_share_exactly_one_fold(self):
        line = np.array([[[.013, .017, .019], [.013, .017, 1.019]]])
        points = folds.sample_segments(line, .005, 2000000)
        points = np.r_[points, points+[.0003, .0002, 0]]
        captured = set()
        original = folds._ridge_runs

        def observe(values, mask, anchor, axis, config):
            grid = np.floor((values-config["origin"])/config["voxel_size"]).astype(np.int64)
            training = {tuple(cell) for cell in grid[mask]}
            validation = {tuple(cell) for cell in grid[~mask]}
            self.assertTrue(training)
            self.assertTrue(validation)
            self.assertFalse(training & validation)
            captured.add(config.get("ridge_fold_index", 0))
            return original(values, mask, anchor, axis, config)

        with mock.patch.object(baseline_impl, "_ridge_runs", side_effect=observe), \
                mock.patch.object(folds, "_ridge_runs", side_effect=observe):
            actual = consensus.readout(points, np.empty((0, 2, 3)), {})
        self.assertEqual(captured, {0, 1, 2})
        self.assertEqual(actual["components"], 1)
        self.assertEqual(actual["training_points"]+actual["validation_points"], actual["unique_points"])

    def test_proposal_api_receives_only_remaining_training_points(self):
        cloud, lines = sampling_tests.independent_branch()
        original = folds.ridge_proposals.propose
        captured = set()

        def observe(training, config):
            grid = np.floor((training-config["origin"])/config["voxel_size"]).astype(np.int64)
            index = config.get("ridge_fold_index", 0)
            self.assertTrue(folds._training_mask(grid, index).all())
            captured.add(index)
            return original(training, config)

        with mock.patch.object(folds.ridge_proposals, "propose", side_effect=observe):
            actual = consensus.readout(cloud, lines, {"voxel_size": sampling_tests.VOXEL, "trials": 1})
        self.assertEqual(captured, {0, 1, 2})
        self.assertTrue(sampling_tests.matched_endpoints(actual["segments"], lines))
