"""Unchanged surface/ridge integration contracts against supported readout."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
import test_evidence_integration as prior  # noqa: E402
from creator_eval import common_readout_supported as reader  # noqa: E402


class SupportedIntegrationTests(unittest.TestCase):
    def setUp(self):
        active = patch.object(prior, "reader", reader)
        active.start()
        self.addCleanup(active.stop)


for _name in dir(prior.EvidenceIntegrationTests):
    if _name.startswith("test_"):
        setattr(SupportedIntegrationTests, _name, getattr(prior.EvidenceIntegrationTests, _name))


if __name__ == "__main__":
    unittest.main()
