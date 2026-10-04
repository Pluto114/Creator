"""Unchanged surface/ridge integration contracts against supported readout."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
import test_evidence_integration as prior  # noqa: E402
from creator_eval import common_readout_local as reader  # noqa: E402


class LocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        active = patch.object(prior, "reader", reader)
        active.start()
        self.addCleanup(active.stop)

    def test_parallel_array_evidence_veto_is_applied_after_ridge_merging(self):
        x = np.linspace(0., 1.2, 301)
        points = np.c_[x, .2+.05*x, -.3+.02*x]
        evidence = dict(state="complete", parallel_array=True)
        with patch.object(reader.ridge_local_evidence, "analyze", return_value=evidence) as analyze:
            result = reader.readout(points, np.empty((0, 2, 3)), {"voxel_size": .01})
        self.assertTrue(analyze.called)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 0)
        self.assertGreater(result["rejected_parallel_arrays"], 0)
        self.assertEqual(analyze.call_args.args[1].dtype, np.dtype(bool))

    def test_array_budget_failure_cannot_be_reported_as_empty_rejection(self):
        x = np.linspace(0., 1.2, 301)
        points = np.c_[x, .2+.05*x, -.3+.02*x]
        evidence = dict(state="unmeasurable", parallel_array=False, reason="mock_array_budget")
        with patch.object(reader.ridge_local_evidence, "analyze", return_value=evidence) as analyze:
            result = reader.readout(points, np.empty((0, 2, 3)), {"voxel_size": .01})
        self.assertTrue(analyze.called)
        self.assertEqual(result["state"], "unmeasurable")
        self.assertEqual(result["reason"], "mock_array_budget")

    def test_non_array_preserves_the_fitted_finite_ridge(self):
        x = np.linspace(0., 1.2, 301)
        points = np.c_[x, .2+.05*x, -.3+.02*x]
        evidence = dict(state="complete", parallel_array=False)
        with patch.object(reader.ridge_local_evidence, "analyze", return_value=evidence):
            result = reader.readout(points, np.empty((0, 2, 3)), {"voxel_size": .01})
        self.assertEqual(result["state"], "complete")
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["rejected_parallel_arrays"], 0)
        actual = np.asarray(result["segments"])[0]
        np.testing.assert_allclose(actual[np.argsort(actual[:, 0])], points[[0, -1]], atol=1e-8)


for _name in dir(prior.EvidenceIntegrationTests):
    if _name.startswith("test_"):
        setattr(LocalIntegrationTests, _name, getattr(prior.EvidenceIntegrationTests, _name))


if __name__ == "__main__":
    unittest.main()
