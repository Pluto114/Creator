"""Mock-only freeze/guard contracts; no real experiment is prepared or inferred.

All receipt values below are in-memory sentinels, never generated run receipts.
The two JSON-write tests use disposable directories on the project's D: drive.
"""

import ast
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_readout_ridges as runner  # noqa: E402


def mock_protocol():
    manifest = {"inputs": [dict(input_id=f"control-{i:04d}", sha256="mock-sha") for i in range(44)]}
    config = {"readers": ["ridges", "abstention"], "representations": ["native", "sampled_points"]}
    rows = [dict(input_id=key[0], reader=key[1], representation=key[2],
                 path=f"records/mock-{i:04d}.json", sha256="mock-sha")
            for i, key in enumerate(runner.inventory(manifest, config))]
    normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False,
                  pre_sha256="mock-sha", input_manifest_sha256="mock-sha", rows=rows)
    return manifest, config, normal


class MixedReadoutRunnerTests(unittest.TestCase):
    def test_normal_guard_blocks_truth_scores_and_reports_but_allows_inputs(self):
        blocked = (
            ROOT / "data/eval_gt/mock/manifest.json",
            ROOT / "data/evaluation/mock/metrics.json",
            ROOT / "docs/experiments/results/mock.json",
            ROOT / "docs/experiments/mock-report.md",
            ROOT / ".runtime/mock/evaluation.json",
            ROOT / ".runtime/mock/post.json",
            ROOT / ".runtime/mock/protocol.json",
            ROOT / ".runtime/mock/generation-checks.json",
            ROOT / ".runtime/mock/case-render_request.json",
            ROOT / ".runtime/mock/rendered-rgb/image.png",
        )
        for path in blocked:
            with self.subTest(path=path), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(path), "r", 0))
        for path in (ROOT / "data/inputs/mock/control-0000.npz",
                     ROOT / "experiments/src/creator_eval/common_readout_ridges.py"):
            runner.block_truth("open", (str(path), "r", 0))
        runner.block_truth("unrelated_event", (str(blocked[0]),))
        runner.block_truth("open", (3, "r", 0))

    def test_source_inventory_freezes_readers_metrics_and_runner_contracts(self):
        sources = runner.source_inventory()
        self.assertEqual(sources, sorted(set(sources)))
        expected = {
            "scripts/run_mixed_readout_ridges.py", "configs/mixed_readout_ridges_v1.json",
            "experiments/src/creator_eval/common_readout_ridges.py",
            "experiments/src/creator_eval/common_readout_abstention.py",
            "experiments/src/creator_eval/common_readout_components.py",
            "experiments/src/creator_eval/common_readout_sections.py",
            "experiments/src/creator_eval/common_readout_split.py",
            "experiments/src/creator_eval/common_readout.py",
            "experiments/src/creator_eval/mixed_readout_controls.py",
            "experiments/src/creator_eval/fixture_physical_metrics.py",
            "tests/test_mixed_readout_controls.py", "tests/test_mixed_readout_runner.py",
        }
        self.assertTrue(expected <= set(sources), sorted(expected - set(sources)))

    def test_inference_representation_imports_no_generator_or_truth_metric(self):
        # Inspect the normal-stage imports without executing infer or installing
        # an irreversible audit hook in the test process.
        tree = ast.parse(inspect.getsource(runner.infer))
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 1)
        self.assertIsInstance(imports[0], ast.ImportFrom)
        self.assertEqual(imports[0].module, "creator_eval.common_readout")
        self.assertEqual([name.name for name in imports[0].names], ["sample_segments"])
        modules = runner.readers()
        self.assertEqual(set(modules), {"ridges", "abstention"})
        self.assertTrue(all("mixed_readout_controls" not in module.__name__ for module in modules.values()))

    def test_write_is_exclusive_and_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory(prefix="mixed-runner-tests-", dir=ROOT / ".runtime") as directory:
            path = Path(directory) / "record.json"
            runner.write(path, {"array": np.array([1.0, 2.0]), "scalar": np.int64(3)})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(runner.read(path), {"array": [1.0, 2.0], "scalar": 3})

    def test_nan_write_fails_without_silently_replacing_the_failed_attempt(self):
        with tempfile.TemporaryDirectory(prefix="mixed-runner-tests-", dir=ROOT / ".runtime") as directory:
            path = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                runner.write(path, {"invalid": np.array([np.nan])})
            self.assertTrue(path.exists(), "The failed attempt must not silently disappear")
            with self.assertRaises(FileExistsError):
                runner.write(path, {"retry": "would overwrite an earlier attempt"})

    def test_normal_inventory_has_exactly_176_distinct_ordered_rows(self):
        manifest, config, _ = mock_protocol()
        rows = runner.inventory(manifest, config)
        self.assertEqual(len(rows), 176)
        self.assertEqual(len(set(rows)), 176)
        self.assertEqual(rows[:4], [
            ("control-0000", "ridges", "native"),
            ("control-0000", "ridges", "sampled_points"),
            ("control-0000", "abstention", "native"),
            ("control-0000", "abstention", "sampled_points"),
        ])
        self.assertEqual(rows[-1], ("control-0043", "abstention", "sampled_points"))

    def test_missing_duplicate_or_reordered_normal_rows_prevent_truth_access(self):
        manifest, config, original = mock_protocol()
        rows = original["rows"]
        variants = (rows[:-1], rows + rows[-1:], rows[:-1] + rows[:1], rows[1:] + rows[:1])
        for variant in variants:
            normal = {**original, "rows": variant}
            with self.subTest(length=len(variant)):
                def fake_read(path):
                    self.assertEqual(Path(path), runner.RUN / "inference.json")
                    return normal

                def fake_digest(path):
                    self.assertNotEqual(Path(path), runner.TRUTH, "Truth must remain unopened")
                    return "mock-sha"

                with patch.object(runner, "checked", return_value=({}, manifest, config)), \
                        patch.object(runner, "read", side_effect=fake_read), \
                        patch.object(runner, "digest", side_effect=fake_digest), \
                        self.assertRaisesRegex(ValueError, "176"):
                    runner.evaluation_payload()

    def test_bad_last_normal_record_prevents_truth_even_with_176_manifest_rows(self):
        manifest, config, normal = mock_protocol()
        entries = {runner.RUN / row["path"]: row for row in normal["rows"]}
        opened = []

        def fake_read(path):
            path = Path(path)
            self.assertNotEqual(path, runner.TRUTH, "Truth must remain unopened")
            if path == runner.RUN / "inference.json":
                return normal
            row = entries[path]
            opened.append(path)
            return {**row, "input_sha256": "mock-sha", "gt_read": len(opened) == 176}

        def fake_digest(path):
            self.assertNotEqual(Path(path), runner.TRUTH, "Truth hash access is also a forbidden read")
            return "mock-sha"

        with patch.object(runner, "checked", return_value=({}, manifest, config)), \
                patch.object(runner, "read", side_effect=fake_read), \
                patch.object(runner, "digest", side_effect=fake_digest), \
                self.assertRaisesRegex(ValueError, "identity"):
            runner.evaluation_payload()
        self.assertEqual(len(opened), 176)


if __name__ == "__main__":
    unittest.main()
