"""The inference boundary must fail loudly if code, settings, or reads drift."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_rod_identity_stress as stress  # noqa: E402
from run_rod_identity_blender import digest  # noqa: E402


class FrozenStressTests(unittest.TestCase):
    def test_source_and_method_mutations_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "method.py"
            source.write_text("original", encoding="utf-8")
            method = root / "method_config.json"
            method.write_text("{}", encoding="utf-8")
            prepared = {"source_sha256": {"method.py": digest(source)},
                        "method_config_sha256": digest(method)}
            with patch.object(stress, "ROOT", root):
                stress.frozen_source_check(root, prepared)
                source.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    stress.frozen_source_check(root, prepared)
                source.write_text("original", encoding="utf-8")
                method.write_text('{"threshold":999}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Method changed"):
                    stress.frozen_source_check(root, prepared)

    def test_inference_must_belong_to_exact_frozen_inputs(self):
        prepared = {"run_id": "case", "source_sha256": {"method": "a"},
                    "input_manifest_sha256": "b", "method_config_sha256": "c"}
        inference = {**prepared, "state": "inferred", "gt_read_during_inference": False,
                     "truth_read_tripwire_enabled": True}
        stress.validate_inference_identity(inference, prepared)
        for key in ("run_id", "source_sha256", "input_manifest_sha256", "method_config_sha256", "state", "truth_read_tripwire_enabled"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                stress.validate_inference_identity({**inference, key: "changed"}, prepared)

    def test_truth_and_full_protocol_reads_are_blocked(self):
        for path in (stress.ROOT / "data/eval_gt/case/depth.npy",
                     stress.ROOT / "data/evaluation/case/summary.json",
                     stress.ROOT / ".runtime/experiments/case/protocol.json",
                     stress.ROOT / ".runtime/experiments/case/render_request.json"):
            with self.assertRaises(PermissionError):
                stress.reject_truth_open("open", (str(path), "r", 0))
        stress.reject_truth_open("open", (str(stress.ROOT / "data/inputs/case/image.png"), "r", 0))
        stress.reject_truth_open("open", (42, "w", 0))


if __name__ == "__main__":
    unittest.main()
