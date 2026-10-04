"""Reuse all frozen sampling-reader contracts against local-support sections."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
import test_common_readout_sampling as prior  # noqa: E402
from creator_eval import common_readout_local as reader  # noqa: E402


class LocalContracts(unittest.TestCase):
    def setUp(self):
        # Reused methods retain their defining module globals. Patch every
        # level: sampling -> evidence -> ridges, never silently test old code.
        for module in (prior, prior.prior, prior.prior.prior):
            active = patch.object(module, "reader", reader)
            active.start()
            self.addCleanup(active.stop)


for _name in dir(prior.SamplingContracts):
    if _name.startswith("test_"):
        setattr(LocalContracts, _name, getattr(prior.SamplingContracts, _name))


if __name__ == "__main__":
    unittest.main()
