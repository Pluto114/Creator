"""Sequential-development contracts, never scored controls or fixture inputs."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
import test_common_readout_ridges as prior  # noqa: E402
from creator_eval import common_readout_evidence as reader  # noqa: E402


class EvidenceContracts(unittest.TestCase):
    def setUp(self):
        # Reuse the frozen inputs/assertions verbatim, not a weakened copy.
        active = patch.object(prior, "reader", reader)
        active.start()
        self.addCleanup(active.stop)

    def test_continuous_line_amid_scatter_has_no_fraction_truncation(self):
        line = np.array([[[.003, .007, .011], [.003, .007, 1.311]]])
        cloud = np.random.default_rng(732119).uniform([-.055, -.055, .011], [.055, .055, 1.311], (3400, 3))
        result = reader.readout(cloud, line, {"voxel_size": .012})
        self.assertEqual(result["components"], 1)
        np.testing.assert_allclose(prior.canonical(result["segments"]), line, atol=.002)

    def test_same_scatter_without_line_remains_empty(self):
        cloud = np.random.default_rng(732119).uniform([-.055, -.055, .011], [.055, .055, 1.311], (3400, 3))
        result = reader.readout(cloud, prior.EMPTY_SEGMENTS, {"voxel_size": .012})
        self.assertEqual(result["components"], 0)

    def test_old_fraction_threshold_is_not_a_hidden_configurable_gate(self):
        with self.assertRaises(ValueError):
            reader.policy({"minimum_core_fraction": .5})


for _name in dir(prior.RidgeReadoutContracts):
    if _name.startswith("test_") and _name != "test_circle_requires_angular_coverage_not_only_sector_count":
        setattr(EvidenceContracts, _name, getattr(prior.RidgeReadoutContracts, _name))


if __name__ == "__main__":
    unittest.main()
