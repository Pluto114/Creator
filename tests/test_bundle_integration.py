"""Retain cadence, geometry, finite support and propagation contracts."""

import unittest
from unittest.mock import patch

import test_ridge_sampling_integration as prior
from creator_eval import common_readout_bundle as reader


class BundleIntegrationTests(unittest.TestCase):
    def setUp(self):
        for module in (prior, prior.prior, prior.prior.prior):
            active = patch.object(module, "reader", reader)
            active.start()
            self.addCleanup(active.stop)


for _name in dir(prior.RidgeSamplingIntegrationTests):
    if _name.startswith("test_"):
        setattr(BundleIntegrationTests, _name, getattr(prior.RidgeSamplingIntegrationTests, _name))


if __name__ == "__main__":
    unittest.main()
