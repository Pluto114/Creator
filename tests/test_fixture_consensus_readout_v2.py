"""Retain all fixture contracts and pin immutable storage paths in the repair."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_fixture_consensus_readout_v2 as runner  # noqa: E402
import test_fixture_consensus_readout as prior  # noqa: E402


class FixtureConsensusRepairContracts(prior.FixtureConsensusContracts):
    def setUp(self):
        active = patch.object(prior, "runner", runner)
        active.start()
        self.addCleanup(active.stop)
        super().setUp()

    def test_algorithm_name_cannot_rename_original_storage(self):
        folder = Path("original-case")
        paths = runner.bundle_paths(folder, ["base", "baseline", "cylinder_support"])
        self.assertEqual(paths["base"], folder/"bundle/base")
        for variant in ("baseline", "cylinder_support"):
            self.assertEqual(paths["enabled"][variant], folder/f"bundle/{variant}-enabled.json")
            self.assertEqual(paths["withdrawn"][variant], folder/f"bundle/{variant}-withdrawn.json")
        self.assertEqual(self.config["repair_of_run_id"], "fixture-consensus-readout-v1-20261005")
