"""Mock-only normal/reference provenance and evaluation-order contracts."""

import ast
import copy
import inspect
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_mixed_readout_consensus as runner  # noqa: E402


def protocol():
    config = runner.validated_config()
    cohorts = [runner.REPLAY_COHORT]*44 + [runner.SECTION_REPLAY_COHORT]*36 + [runner.SAMPLING_REPLAY_COHORT]*40 + [runner.SUPPORT_REPLAY_COHORT]*32 + [runner.LOCAL_REPLAY_COHORT]*32
    items = [dict(input_id=f"mock-{i:04d}", cohort=cohort, sha256="hash", voxel_size=.006)
             for i, cohort in enumerate(cohorts)]
    manifest = dict(inputs=items)
    by_id = {item["input_id"]: item for item in items}
    keys = runner.inventory(manifest, config)
    rows = [dict(input_id=key[0], reader=key[1], representation=key[2], cohort=by_id[key[0]]["cohort"],
                 path=f"records/mock-{i:04d}.json", sha256="hash") for i, key in enumerate(keys)]
    references = {key: {"sentinel": key[0]} for key in keys if key[1] == "bundle_small" and int(key[0][-4:]) < 184}
    normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, pre_sha256="hash",
                  input_manifest_sha256="hash", rows=rows, fresh_inference_rows=368, frozen_reference_rows=368)
    return manifest, config, normal, references


class ConsensusRunnerTests(unittest.TestCase):
    def test_worker_blocks_truth_in_fresh_process(self):
        code = ("import sys;sys.path.insert(0,'scripts');import run_mixed_readout_consensus as r;"
                "r.initialize_normal_worker();open(r.ROOT/'data/eval_gt/forbidden.json')")
        done = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Normal mixed-reader inference cannot access truth or scores", done.stderr)

    def test_changed_worker_input_is_rejected_before_array_load(self):
        with patch.object(runner, "digest", return_value="wrong"), patch.object(runner.np, "load") as load:
            with self.assertRaisesRegex(ValueError, "worker input changed"):
                runner.infer_input(dict(path="unused.npz", sha256="expected"))
        load.assert_not_called()

    def test_parallel_workers_preserve_sequential_geometry(self):
        with tempfile.TemporaryDirectory(prefix="consensus-workers-", dir=ROOT/".runtime") as directory:
            path = Path(directory)/"input.npz"
            np.savez(path, points=np.c_[np.linspace(.017, .417, 81), np.full((81, 2), .019)],
                     segments=np.empty((0, 2, 3)))
            item = dict(path=path.relative_to(ROOT).as_posix(), sha256=runner.digest(path), voxel_size=.01)
            expected = runner.infer_input(item)
            with ProcessPoolExecutor(max_workers=3, initializer=runner.initialize_normal_worker) as pool:
                results = list(pool.map(runner.infer_input, [item]*3))
            for result in results:
                for representation in ("native", "sampled_points"):
                    self.assertEqual(runner.json_ready(result[representation]["result"]),
                                     runner.json_ready(expected[representation]["result"]))

    def test_inventory_is_736_pairs_with_368_executions_and_368_exact_references(self):
        manifest, config, normal, references = protocol()
        inventory = runner.inventory(manifest, config)
        self.assertEqual(len(inventory), 736)
        self.assertEqual(len(set(inventory)), 736)
        self.assertEqual(len(references), 368)
        self.assertEqual(len(inventory)-len(references), 368)
        self.assertEqual(normal["fresh_inference_rows"], 368)
        self.assertEqual(config["new_conditions"], 0)
        self.assertTrue(config["all_inputs_prior_outputs_and_scores_seen"])
        self.assertEqual(config["development_replay_conditions"], 184)

    def test_normal_guard_blocks_truth_and_reports(self):
        for path in (ROOT/"data/eval_gt/mock/manifest.json", ROOT/"docs/experiments/results/mock.json",
                     ROOT/".runtime/mock/evaluation.json", ROOT/".runtime/mock/protocol.json"):
            with self.subTest(path=path), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(path), "r", 0))
        runner.block_truth("open", (str(ROOT/".runtime/experiments/mock/records/input.json"), "r", 0))

    def test_normal_stage_imports_no_generator_or_truth(self):
        tree = ast.parse(inspect.getsource(runner.infer_input))
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "creator_eval.common_readout")
        self.assertEqual([name.name for name in imports[0].names], ["sample_segments"])

    def test_inventory_uses_parent_frozen_list_not_mutable_source_glob(self):
        with patch.object(runner, "read", return_value={"source_files": ["scripts/run_mixed_readout_bundle.py"]}):
            sources = runner.source_inventory()
        self.assertEqual(sources, sorted(set(sources)))
        for name in ("scripts/run_mixed_readout_bundle.py", "scripts/run_mixed_readout_consensus.py",
                     "configs/mixed_readout_consensus_v1.json", "tests/test_mixed_consensus_runner.py",
                     "experiments/src/creator_eval/common_readout_consensus.py",
                     "experiments/src/creator_eval/common_readout_fold.py",
                     "experiments/src/creator_eval/ridge_fold_consensus.py",
                     "tests/test_common_readout_consensus.py", "tests/test_ridge_fold_consensus.py",
                     "tests/test_consensus_integration.py"):
            self.assertIn(name, sources)
        self.assertNotIn("experiments/src/creator_eval/rgb_fixture_context_readout.py", sources)

    def test_json_writes_are_exclusive(self):
        with tempfile.TemporaryDirectory(prefix="context-contract-", dir=ROOT/".runtime") as directory:
            path = Path(directory)/"record.json"
            runner.write(path, {"values": np.arange(3)})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write(path, {"replace": True})
            self.assertEqual(path.read_bytes(), before)

    def test_reference_reader_requires_same_input_representation_and_policy(self):
        item = dict(input_id="replay-0000", sha256="hash")
        policy = dict(voxel_size=.006, origin=[0., 0., 0.])
        defaults = dict(trials=192, **policy)
        reference = dict(input_id=item["input_id"], input_sha256="hash", reader="bundle_small", representation="native",
                         original_input_id="old-0000", path="mock.json", sha256="hash")
        record = dict(input_id="old-0000", input_sha256="hash", reader="bundle_small", representation="native",
                      gt_read=False, call_policy=policy, effective_policy=defaults, result={"segments": []})
        with patch.object(runner, "digest", return_value="hash"), patch.object(runner, "read", return_value=record):
            self.assertEqual(runner.reference_result(reference, item, "native", policy, defaults), record["result"])
        for change in ({"input_sha256": "wrong"}, {"gt_read": True}, {"reader": "evidence"},
                       {"representation": "sampled_points"}, {"call_policy": {}}, {"effective_policy": {}},
                       {"input_id": "wrong"}):
            with self.subTest(change=change), patch.object(runner, "digest", return_value="hash"), \
                    patch.object(runner, "read", return_value={**record, **change}), self.assertRaisesRegex(ValueError, "identity/policy"):
                runner.reference_result(reference, item, "native", policy, defaults)

    def test_changed_reference_hash_is_rejected_before_loading_record(self):
        with patch.object(runner, "digest", return_value="changed"), patch.object(runner, "read") as read:
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.reference_result({"path": "mock", "sha256": "hash"}, {}, "native", {}, {})
            read.assert_not_called()

    def test_reference_manifest_order_and_reader_are_fixed(self):
        rows = [dict(input_id=f"replay-{i:04d}", reader="bundle_small", representation=representation)
                for i in range(184) for representation in ("native", "sampled_points")]
        manifest = dict(run_id=runner.RUN_ID, parent_run_id=runner.PARENT_RUN_ID, rows=rows)
        with patch.object(runner, "read", return_value=manifest):
            self.assertEqual(len(runner.reference_rows()), 368)
        for changed in (rows[:-1], rows[::-1], rows+rows[:1]):
            with patch.object(runner, "read", return_value={**manifest, "rows": changed}), self.assertRaises(ValueError):
                runner.reference_rows()

    def test_missing_or_reordered_paired_rows_prevent_truth_access(self):
        manifest, config, normal, _ = protocol()
        for rows in (normal["rows"][:-1], normal["rows"][::-1]):
            def read(path):
                self.assertEqual(path, runner.RUN/"inference.json")
                return {**normal, "rows": rows}
            with patch.object(runner, "checked", return_value=({}, manifest, config)), \
                    patch.object(runner, "read", side_effect=read), patch.object(runner, "digest", return_value="hash"), \
                    self.assertRaisesRegex(ValueError, "736"):
                runner.evaluation_payload()

    def _assert_pretruth_failure(self, mutation, message):
        manifest, config, normal, references = protocol()
        by_path = {runner.RUN/row["path"]: row for row in normal["rows"]}
        opened = []

        def read(path):
            self.assertNotEqual(path, runner.TRUTH)
            if path == runner.RUN/"inference.json":
                return normal
            row = by_path[path]
            key = row["input_id"], row["reader"], row["representation"]
            reference = references.get(key)
            record = dict(**row, input_sha256="hash", gt_read=False,
                          call_policy=dict(voxel_size=.006, origin=[0., 0., 0.]),
                          effective_policy=dict(voxel_size=.006, origin=[0., 0., 0.]),
                          execution_kind="frozen_parent_reference" if reference else "fresh_inference",
                          parent_reference=reference, result={})
            opened.append(path)
            mutation(record, len(opened), reference)
            return record

        def digest(path):
            self.assertNotEqual(path, runner.TRUTH, "Truth cannot even be hashed before all records pass")
            return "hash"

        with ExitStack() as stack:
            for name, kwargs in (("checked", dict(return_value=({}, manifest, config))),
                                 ("read", dict(side_effect=read)), ("digest", dict(side_effect=digest)),
                                 ("reference_rows", dict(return_value=references)),
                                 ("reference_result", dict(return_value={})),
                                 ("readers", dict(return_value={"consensus": SimpleNamespace(DEFAULTS={}),
                                                                "bundle_small": SimpleNamespace(DEFAULTS={})}))):
                stack.enter_context(patch.object(runner, name, **kwargs))
            with self.assertRaisesRegex(ValueError, message):
                runner.evaluation_payload()
        return len(opened)

    def test_bad_last_record_keeps_truth_closed(self):
        def mutation(record, count, reference):
            record["gt_read"] = count == 736
        self.assertEqual(self._assert_pretruth_failure(mutation, "identity"), 736)

    def test_cached_result_cannot_be_replaced_by_claimed_fresh_execution(self):
        def mutation(record, count, reference):
            if reference:
                record["execution_kind"] = "fresh_inference"
        self._assert_pretruth_failure(mutation, "provenance")

    def test_cached_result_cannot_be_modified(self):
        def mutation(record, count, reference):
            if reference:
                record["result"] = {"changed": True}
        self._assert_pretruth_failure(mutation, "modified")

    def test_five_cohorts_are_not_pooled_as_new_holdout(self):
        _, config, normal, _ = protocol()
        rows = [{**row, "positive_truth_present": True, "positive_recovery_precision_at_least_90_percent": True,
                 "positive_finite_structure_display_pass": True, "state": "complete", "scoring_error": None,
                 "output_segment_count": 1} for row in normal["rows"]]
        summaries = runner.cohort_summaries(rows, config)
        self.assertEqual([r["repeated_conditions"] for r in summaries], [44, 36, 40, 32, 32])
        self.assertEqual([r["prior_outputs_and_scores_seen"] for r in summaries], [True, True, True, True, True])

    def test_finite_display_still_rejects_a_missed_short_branch(self):
        score = dict(segment_count=2, boundary_bijection_state="scored", boundary_max_error_m=.001,
                     recovery_fraction=.95, precision_fraction=1., guarded_gap_false_length_m=0.)
        self.assertTrue(runner.finite_completeness(score, [0, 0], .9))
        changed = copy.deepcopy(score)
        changed["segment_count"] = 1
        self.assertFalse(runner.finite_completeness(changed, [0, 0], .9))

    def test_prepare_verifies_all_parent_normal_before_truth_or_writes(self):
        import run_mixed_readout_bundle as legacy

        config = runner.validated_config()
        with patch.object(Path, "exists", return_value=False), \
                patch.object(runner, "validated_config", return_value=config), \
                patch.object(legacy, "normal_records", side_effect=ValueError("mock parent verification")) as parent, \
                patch.object(runner, "digest") as digest, patch.object(runner, "read") as read, \
                patch.object(runner, "write") as write, self.assertRaisesRegex(ValueError, "parent verification"):
            runner.prepare()
        parent.assert_called_once_with()
        digest.assert_not_called()
        read.assert_not_called()
        write.assert_not_called()

    def test_development_prepare_imports_no_generator(self):
        tree = ast.parse(inspect.getsource(runner.prepare))
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(modules, {"creator_eval.fixture_physical_metrics"})
        self.assertNotIn("generate", inspect.getsource(runner.prepare))
        config = runner.validated_config()
        for key in ("seed", "new_local_conditions", "new_local_family_count", "grid_scales_m", "sampling_phases"):
            self.assertNotIn(key, config)
        self.assertTrue(config["hypothesis_development_replay"])

    def test_nonfinite_json_is_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix="context-nan-", dir=ROOT/".runtime") as directory:
            with self.assertRaises(ValueError):
                runner.write(Path(directory)/"invalid.json", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()


