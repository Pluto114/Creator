"""Appearance control must keep its frozen settings and a pixel-identical replica."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rod_appearance_control import frozen_config, locations, require_exact_replica
from thin_pack_gt import sha256, write_json


class AppearanceBoundaryTests(unittest.TestCase):
    def test_changed_mapping_or_method_rejected_after_prepare(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            config = {"run_id": "demo", "method": "frozen", "target": 5}
            write_json(run / "protocol.json", config)
            status = {"protocol_sha256": sha256(run / "protocol.json")}
            self.assertEqual(frozen_config(config, run, status), config)
            for key, value in (("target", 6), ("method", "changed")):
                edited = copy.deepcopy(config)
                edited[key] = value
                with self.assertRaisesRegex(ValueError, "frozen protocol"):
                    frozen_config(edited, run, status)
            write_json(run / "protocol.json", {**config, "target": 6})
            with self.assertRaisesRegex(ValueError, "frozen protocol"):
                frozen_config(config, run, status)

    def test_replica_mismatch_or_missing_replica_stops_inference(self):
        require_exact_replica([{"decoded_exact": True, "max_channel_difference": 0}])
        for values in ([], [{"decoded_exact": False, "max_channel_difference": 1}], [{"decoded_exact": True, "max_channel_difference": 1}]):
            with self.assertRaisesRegex(ValueError, "exactly reproduce"):
                require_exact_replica(values)

    def test_run_id_cannot_escape_workspace_output(self):
        for rid in ("../escape", "D:/elsewhere", "", "."):
            with self.assertRaises(ValueError):
                locations({"run_id": rid})


if __name__ == "__main__":
    unittest.main()
