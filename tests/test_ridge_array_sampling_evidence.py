"""Retain the full local array contract suite with train-cadence counting."""

import unittest
from unittest.mock import patch

import test_ridge_local_evidence as prior
from creator_eval import ridge_array_sampling_evidence as reader


class RidgeArraySamplingContracts(unittest.TestCase):
    def setUp(self):
        active = patch.object(prior, "reader", reader)
        active.start()
        self.addCleanup(active.stop)


for _name in dir(prior.RidgeLocalEvidenceContracts):
    if _name.startswith("test_"):
        setattr(RidgeArraySamplingContracts, _name, getattr(prior.RidgeLocalEvidenceContracts, _name))


if __name__ == "__main__":
    unittest.main()
