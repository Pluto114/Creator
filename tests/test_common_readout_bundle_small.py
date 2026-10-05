"""Run all inherited geometry assertions with the declared small-arm defaults."""

import unittest
from unittest.mock import patch

import test_common_readout_bundle as prior
from creator_eval import common_readout_bundle_small as small


class SmallBundleContracts(prior.BundleContracts):
    def setUp(self):
        super().setUp()
        active = patch.object(prior.reader, "DEFAULTS", small.DEFAULTS)
        active.start()
        self.addCleanup(active.stop)


if __name__ == "__main__":
    unittest.main()
