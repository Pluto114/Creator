"""Run integration mechanisms on the implementation with small-arm defaults."""

import unittest
from unittest.mock import patch

import test_bundle_integration as prior
from creator_eval import common_readout_bundle_small as small


class SmallBundleIntegrationTests(prior.BundleIntegrationTests):
    def setUp(self):
        super().setUp()
        active = patch.object(prior.reader, "DEFAULTS", small.DEFAULTS)
        active.start()
        self.addCleanup(active.stop)


if __name__ == "__main__":
    unittest.main()
