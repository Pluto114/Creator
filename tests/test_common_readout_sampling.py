"""Independent training-local proposal contracts; no scored controls are read."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
import test_common_readout_evidence as prior  # noqa: E402
from creator_eval import common_readout_sampling as reader  # noqa: E402
from creator_eval import ridge_proposals  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402

EMPTY = np.empty((0, 2, 3))
VOXEL = .008
LINES = np.array([[[.011, .017, .023], [.011, .017, 1.423]],
                  [[.011, .017, .763], [.191, .059, .797]]])


def independent_branch():
    cloud = np.random.default_rng(935771).uniform([-.04, -.035, .023], [.06, .065, 1.423], (2400, 3))
    return cloud, LINES.copy()


def matched_endpoints(actual, expected, tolerance=.003):
    for target in expected:
        errors = [min(np.linalg.norm(line-target, axis=1).max(),
                      np.linalg.norm(line[::-1]-target, axis=1).max()) for line in actual]
        if not errors or min(errors) > tolerance:
            return False
    return True


class SamplingContracts(unittest.TestCase):
    def setUp(self):
        for module in (prior, prior.prior):
            active = patch.object(module, "reader", reader)
            active.start()
            self.addCleanup(active.stop)

    def test_minority_branch_recovered_without_random_pair_trials(self):
        cloud, lines = independent_branch()
        # One retained old trial cannot explain deterministic local coverage.
        result = reader.readout(cloud, lines, {"voxel_size": VOXEL, "trials": 1})
        self.assertEqual(result["state"], "complete", result.get("reason"))
        self.assertTrue(matched_endpoints(result["segments"], lines), result["segments"])
        self.assertTrue(result["local_proposal_evidence"]["training_only"])
        self.assertGreater(result["local_proposal_evidence"]["proposed"], 0)

    def test_point_order_duplicates_and_primitive_representation_do_not_nominate_branches(self):
        cloud, lines = independent_branch()
        config = {"voxel_size": VOXEL, "trials": 1}
        native = reader.readout(cloud, lines, config)
        sampled = np.r_[cloud, sample_segments(lines, VOXEL/2, 2000000)]
        shuffled = sampled[np.random.default_rng(6193).permutation(len(sampled))]
        changed = reader.readout(np.repeat(shuffled, 2, axis=0), EMPTY, config)
        np.testing.assert_array_equal(native["segments"], changed["segments"])
        self.assertEqual(native["local_proposal_evidence"], changed["local_proposal_evidence"])

    def test_proposal_api_receives_only_remaining_training_points(self):
        cloud, lines = independent_branch()
        original = ridge_proposals.propose
        captured = []

        def observe(training, config):
            grid = np.floor((training-config["origin"])/config["voxel_size"]).astype(np.int64)
            self.assertTrue(reader._training_mask(grid).all())
            captured.append(training.copy())
            return original(training, config)

        with patch.object(ridge_proposals, "propose", side_effect=observe):
            result = reader.readout(cloud, lines, {"voxel_size": VOXEL, "trials": 1})
        self.assertTrue(captured)
        self.assertTrue(matched_endpoints(result["segments"], lines))

    def test_local_pool_is_invariant_to_training_order_and_multiplicity(self):
        points = sample_segments(LINES, VOXEL/2, 2000000)
        config = reader.policy({"voxel_size": VOXEL})
        a, detail_a = ridge_proposals.propose(points, config)
        b, detail_b = ridge_proposals.propose(np.repeat(points[::-1], 3, axis=0), config)
        self.assertEqual(detail_a, detail_b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_new_seed_and_proposal_budget_failures_do_not_return_partial_geometry(self):
        points = sample_segments(LINES, VOXEL/2, 2000000)
        for config, reason in (
            ({"local_proposal_maximum_seeds": 1}, "local_proposal_seed_budget_exceeded"),
            ({"local_proposal_maximum_count": 1}, "local_proposal_count_budget_exceeded"),
        ):
            with self.subTest(config=config):
                result = reader.readout(points, EMPTY, {"voxel_size": VOXEL, **config})
                self.assertEqual(result["state"], "unmeasurable")
                self.assertEqual(result["reason"], reason)
                self.assertEqual(len(result["segments"]), 0)

    def test_new_policy_rejects_invalid_local_budgets(self):
        for config in ({"local_proposal_neighbors": 2}, {"local_proposal_neighbors": 129},
                       {"local_proposal_maximum_count": True}, {"local_proposal_maximum_count": 4097},
                       {"local_proposal_maximum_seeds": 0}, {"local_proposal_unknown": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                reader.policy(config)

    def test_true_surface_plus_radial_branch_retains_both_kinds_of_geometry(self):
        theta, z = np.meshgrid(np.arange(36)*(2*np.pi/36), np.linspace(.021, 1.421, 281))
        shell = np.c_[.021*np.cos(theta.ravel()), .021*np.sin(theta.ravel()), z.ravel()]
        branch = np.array([[[.031, .009, .753], [.211, .051, .787]]])
        result = reader.readout(shell, branch, {"voxel_size": VOXEL, "trials": 1})
        self.assertEqual(result["state"], "complete", result.get("reason"))
        self.assertGreaterEqual(result["section_evidence"]["accepted_surface_groups"], 1)
        self.assertTrue(matched_endpoints(result["segments"], branch), result["segments"])
        surface = np.array([[[0., 0., .021], [0., 0., 1.421]]])
        self.assertTrue(matched_endpoints(result["segments"], surface, tolerance=.003), result["segments"])


# Existing geometric refusal, provenance, gaps and budgets remain required.
for _name in dir(prior.EvidenceContracts):
    if _name.startswith("test_"):
        setattr(SamplingContracts, _name, getattr(prior.EvidenceContracts, _name))


if __name__ == "__main__":
    unittest.main()
