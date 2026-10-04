"""Small training-only neighborhoods may nominate, never accept, short members."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import common_readout_ridge_sampling as reader  # noqa: E402
from creator_eval import ridge_sampling_proposals as proposals  # noqa: E402


class RidgeSamplingProposalTests(unittest.TestCase):
    def test_small_training_set_can_nominate_a_short_axis(self):
        points = np.c_[np.arange(7)*.013, np.zeros((7, 2))]
        axes, detail = proposals.propose(points, reader.policy({"voxel_size": .0045}))
        self.assertEqual(detail["state"], "complete")
        self.assertEqual(detail["neighborhood_sizes"], [6, 12])
        self.assertTrue(axes)
        self.assertTrue(all(abs(axis[0]) > .9999 for _, axis in axes))
        self.assertTrue(detail["training_only"])

    def test_invalid_small_neighborhood_and_global_budget_are_explicit(self):
        for size in (True, 2, 129):
            with self.assertRaises(ValueError):
                proposals.policy({"local_proposal_small_neighbors": size})
        points = np.c_[np.arange(40)*.013, np.zeros((40, 2))]
        axes, detail = proposals.propose(points, reader.policy({"local_proposal_maximum_count": 1}))
        self.assertEqual(axes, [])
        self.assertEqual(detail["state"], "unmeasurable")
        self.assertEqual(detail["reason"], "local_proposal_count_budget_exceeded")

    def test_defaults_still_keep_training_sampling_policy_separate(self):
        self.assertEqual(reader.ridge_sampling_evidence.policy({})["voxel_size"], .01)
        self.assertEqual(set(proposals.DEFAULTS), {"local_proposal_neighbors", "local_proposal_small_neighbors",
            "local_proposal_maximum_count", "local_proposal_maximum_seeds"})


if __name__ == "__main__":
    unittest.main()
