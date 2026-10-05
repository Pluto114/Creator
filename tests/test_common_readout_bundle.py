"""All inherited ridge contracts run on the new bundle reader."""

import unittest
from unittest.mock import patch

import test_common_readout_ridge_sampling as prior
from creator_eval import common_readout_bundle as reader


class BundleContracts(unittest.TestCase):
    def setUp(self):
        active = patch.object(prior.prior.prior, "ridge_proposals", reader.ridge_proposals)
        active.start()
        self.addCleanup(active.stop)
        for module in (prior, prior.prior, prior.prior.prior, prior.prior.prior.prior, prior.prior.prior.prior.prior):
            active = patch.object(module, "reader", reader)
            active.start()
            self.addCleanup(active.stop)


for _name in dir(prior.RidgeSamplingContracts):
    if _name.startswith("test_"):
        setattr(BundleContracts, _name, getattr(prior.RidgeSamplingContracts, _name))


if __name__ == "__main__":
    unittest.main()
