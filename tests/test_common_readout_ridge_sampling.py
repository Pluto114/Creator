"""Inherited geometric contracts executed against the cadence reader itself."""

import unittest
from unittest.mock import patch

import numpy as np
import test_common_readout_context as prior
from creator_eval import common_readout_ridge_sampling as reader


class RidgeSamplingContracts(unittest.TestCase):
    def setUp(self):
        active = patch.object(prior.prior, "ridge_proposals", reader.ridge_proposals)
        active.start()
        self.addCleanup(active.stop)
        for module in (prior, prior.prior, prior.prior.prior, prior.prior.prior.prior):
            active = patch.object(module, "reader", reader)
            active.start()
            self.addCleanup(active.stop)


for _name in dir(prior.ContextContracts):
    if _name.startswith("test_"):
        setattr(RidgeSamplingContracts, _name, getattr(prior.ContextContracts, _name))


def _outer_gap(self):
    # Same frozen geometry/assertion; new private API also returns gate audit.
    base = prior.prior.prior.prior
    segments = np.array([[[0., 0., .019], [0., 0., .419]],
                         [[0., 0., .719], [0., 0., 1.119]],
                         [[.004, 0., .419], [.004, 0., .719]]])
    points = np.unique(base.sampled(segments), axis=0)
    mask = reader._training_mask(np.floor(points/base.VOXEL).astype(np.int64))
    lines, _, audit = reader._ridge_runs(points, mask, np.zeros(3), np.array([0., 0., 1.]), reader.policy({}))
    self.assertEqual(len(lines), 2)
    self.assertEqual(audit["state"], "complete")
    np.testing.assert_allclose(base.canonical(lines), segments[:2], atol=1e-12)


RidgeSamplingContracts.test_outer_radius_neighbor_does_not_fill_an_inner_support_gap = _outer_gap


if __name__ == "__main__":
    unittest.main()
