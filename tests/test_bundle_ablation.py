"""Declared proposal-size ablation changes only one reader policy field."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import common_readout_bundle as base  # noqa: E402
from creator_eval import common_readout_bundle_small as small  # noqa: E402


class BundleAblationTests(unittest.TestCase):
    def test_only_small_neighborhood_size_differs(self):
        self.assertEqual([k for k in base.DEFAULTS if base.DEFAULTS[k] != small.DEFAULTS[k]],
                         ["local_proposal_small_neighbors"])
        self.assertEqual(base.DEFAULTS["local_proposal_small_neighbors"], 12)
        self.assertEqual(small.DEFAULTS["local_proposal_small_neighbors"], 6)

    def test_wrapper_transfers_geometry_without_modifying_input(self):
        points, lines = np.zeros((4, 3)), np.empty((0, 2, 3))
        with patch.object(base, "readout", return_value={"sentinel": True}) as call:
            result = small.readout(points, lines, {"voxel_size": .006})
        self.assertEqual(result, {"sentinel": True})
        self.assertIs(call.call_args.args[0], points)
        self.assertEqual(call.call_args.args[2]["local_proposal_small_neighbors"], 6)
        self.assertEqual(call.call_args.args[2]["voxel_size"], .006)

    def test_invalid_overrides_are_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            small.policy({"unknown": 1})


if __name__ == "__main__":
    unittest.main()
