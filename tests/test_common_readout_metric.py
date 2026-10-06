"""Replay the unchanged geometry contracts against the consensus supplement."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import test_common_readout_bundle as prior
import test_common_readout_sampling as sampling_tests
from creator_eval import common_readout_metric as consensus
from creator_eval import common_readout_metric_base as baseline_impl
from creator_eval import common_readout_metric_fold as folds


class MetricGeometryContracts(prior.BundleContracts):
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


import test_section_context_evidence as context_tests  # noqa: E402
import test_section_local_evidence as local_tests  # noqa: E402
from creator_eval import section_metric_evidence as metric_sections  # noqa: E402


class MetricSectionContracts(local_tests.SectionLocalEvidenceTests):
    def setUp(self):
        active = mock.patch.multiple(local_tests, analyze=metric_sections.analyze,
                                     policy=metric_sections.policy, DEFAULTS=metric_sections.DEFAULTS)
        active.start()
        self.addCleanup(active.stop)


class MetricContextContracts(context_tests.SectionContextEvidenceTests):
    def setUp(self):
        active = mock.patch.object(context_tests, "subject", metric_sections)
        active.start()
        self.addCleanup(active.stop)


class MetricSpecificContracts(local_tests.unittest.TestCase):
    def test_all_acceptance_and_sampling_defaults_are_unchanged(self):
        from creator_eval import common_readout_consensus, section_context_evidence
        self.assertEqual(consensus.DEFAULTS, common_readout_consensus.DEFAULTS)
        self.assertEqual(metric_sections.DEFAULTS, section_context_evidence.DEFAULTS)

    def test_grouping_uses_actual_training_distance_and_preserves_real_axial_gap(self):
        # Different geometry from scored replay; no normal-pack or GT IO.
        z = np.r_[np.arange(30)*.013, .71+np.arange(30)*.013]
        angle = np.deg2rad(np.arange(0, 181, 15))
        points = np.c_[np.tile(.024*np.cos(angle), len(z)),
                       np.tile(.024*np.sin(angle), len(z)), np.repeat(z, len(angle))]
        mask = ((np.arange(len(z))[:, None]+np.arange(len(angle))[None]) % 3 != 1).ravel()
        segments, _, detail = metric_sections.analyze(points, mask, {"voxel_size": .0045})
        self.assertEqual(len(segments), 2)
        self.assertEqual(detail["training_group_count"], 1)
        self.assertTrue(all(s[:, 2].max() <= .378 or s[:, 2].min() >= .709 for s in segments))
        self.assertTrue(detail["metric_grouping"]["training_only"])
        self.assertFalse(detail["metric_grouping"]["validation_can_bridge"])
        self.assertTrue(all(row["training_distance_m"] <= .00675
                            for row in detail["metric_grouping"]["repairs"]))
