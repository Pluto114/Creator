"""Supplemental barrier contracts, including the unchanged live legacy guard."""

import copy
import hashlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_foreground_consensus_barrier_supplement as supplement  # noqa: E402


class SupplementalBarrierContracts(unittest.TestCase):
    def isolated(self, body):
        command = "import sys\nfrom pathlib import Path\n"
        command += f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        command += "import run_foreground_consensus_barrier_supplement as s\n"
        result = subprocess.run([sys.executable, "-B", "-c", command + textwrap.dedent(body)],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_uses_exact_approved_helper_and_canonical_sorted_paths(self):
        hashes = {"z/file.json": "b" * 64, "a/file.py": "a" * 64}
        with patch.object(supplement.approved, "digest", side_effect=["a" * 64, "b" * 64]) as digest:
            self.assertEqual(supplement.verify_receipts(ROOT, hashes), 2)
        self.assertEqual([call.args[0] for call in digest.call_args_list],
                         [ROOT / "a/file.py", ROOT / "z/file.json"])

    def test_ancestor_changed_bytes_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ancestor.json"
            path.write_bytes(b"old")
            receipts = {"ancestor.json": hashlib.sha256(b"old").hexdigest()}
            supplement.verify_receipts(folder, receipts)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "full receipt changed"):
                supplement.verify_receipts(folder, receipts)

    def test_conflicting_overlapping_receipts_fail(self):
        a, b = {"source.py": "a" * 64}, {"source.py": "b" * 64}
        self.assertEqual(supplement.combine_receipts(a, a), a)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            supplement.combine_receipts(a, b)

    def test_truth_scores_and_noncanonical_paths_rejected_before_hash(self):
        for name in ("data/eval_gt/test.json", "docs/experiments/results/a.json",
                     "records/evaluation.json", "../outside.json", "C:/outside.json"):
            with self.subTest(name=name), patch.object(supplement.approved, "digest") as digest, \
                    self.assertRaises((PermissionError, ValueError)):
                supplement.verify_receipts(ROOT, {name: "a" * 64})
            digest.assert_not_called()

    def test_live_guard_only_approved_hash_can_read_generation_config(self):
        output = self.isolated('''
            import hashlib, tempfile
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "g1_object_pilot_v1.json"
                path.write_bytes(b"synthetic test bytes only")
                expected = hashlib.sha256(b"synthetic test bytes only").hexdigest()
                sys.addaudithook(s.approved.block_truth)
                for operation in (path.read_bytes, lambda: s.frozen.direct.digest(path)):
                    try:
                        operation()
                    except PermissionError:
                        pass
                    else:
                        raise AssertionError("unapproved read escaped legacy guard")
                assert s.verify_receipts(folder, {path.name: expected}) == 1
                assert s.approved._HASH_ONLY_PATH is None
                try:
                    path.read_text()
                except PermissionError:
                    print("GUARD_STILL_ACTIVE")
                else:
                    raise AssertionError("guard was disabled after approved byte hash")
        ''')
        self.assertIn("GUARD_STILL_ACTIVE", output)

    def test_approved_hash_does_not_authorize_gt_or_scores(self):
        self.isolated('''
            sys.addaudithook(s.approved.block_truth)
            for name in ("data/eval_gt/nonexistent.json", "docs/experiments/results/nonexistent.json"):
                try:
                    s.approved.digest(s.ROOT / name)
                except PermissionError:
                    pass
                else:
                    raise AssertionError("approved hash illegally opened truth or score")
            assert s.approved._HASH_ONLY_PATH is None
        ''')

    def manifest_fixture(self):
        path = ROOT / ".runtime/validation/test-supplement/supplement.json"
        body = dict(schema=supplement.SCHEMA, provider=supplement.PROVIDER, gt_read=False,
                    frozen_normal_sources_modified=False,
                    supplemental_source_hashes={name: "a" * 64 for name in supplement.SOURCES},
                    approved_hash_provider_path=supplement.HASH_PROVIDER,
                    approved_hash_provider_sha256="b" * 64,
                    runs={kind: {"run_id": kind} for kind in supplement.frozen.CASES},
                    failed_attempts={kind: {"path": kind} for kind in supplement.frozen.CASES})
        return path, body

    def test_manifest_requires_external_sha_before_read(self):
        path, _ = self.manifest_fixture()
        with patch.object(supplement.approved, "digest", return_value="b" * 64), \
                patch.object(supplement.frozen, "read") as read, self.assertRaisesRegex(ValueError, "external SHA"):
            supplement.verify_manifest(path, "a" * 64)
        read.assert_not_called()

    def test_manifest_closes_sources_both_runs_and_prior_failures(self):
        path, original = self.manifest_fixture()
        for changed in ("source", "suite", "failure", "provider", "gt", "old_sources", "extra"):
            body = copy.deepcopy(original)
            if changed == "source":
                body["supplemental_source_hashes"].pop(supplement.SOURCES[-1])
            elif changed == "suite":
                body["runs"].pop("fixture")
            elif changed == "failure":
                body["failed_attempts"].pop("g1_object")
            elif changed == "provider":
                body["approved_hash_provider_path"] = "scripts/direct_parent_index.py"
            elif changed == "gt":
                body["gt_read"] = True
            elif changed == "old_sources":
                body["frozen_normal_sources_modified"] = True
            else:
                body["extra"] = True
            with self.subTest(changed=changed), patch.object(supplement.approved, "digest", return_value="a" * 64), \
                    patch.object(supplement.frozen, "read", return_value=body), \
                    patch.object(supplement, "verify_receipts") as verify, self.assertRaisesRegex(ValueError, "Exact supplemental"):
                supplement.verify_manifest(path, "a" * 64)
            verify.assert_not_called()

    def test_full_supplement_manifest_uses_source_and_provider_hashes(self):
        path, body = self.manifest_fixture()
        with patch.object(supplement.approved, "digest", return_value="a" * 64), \
                patch.object(supplement.frozen, "read", return_value=body), \
                patch.object(supplement, "verify_receipts") as verify, \
                patch.object(supplement, "binding", side_effect=lambda runner: {"run_id": runner.kind}), \
                patch.object(supplement, "failed_attempt", side_effect=lambda kind: {"path": kind}):
            self.assertEqual(supplement.verify_manifest(path, "a" * 64), body)
        self.assertEqual(verify.call_args.args[1],
                         {**body["supplemental_source_hashes"], supplement.HASH_PROVIDER: "b" * 64})

    def test_manifest_cannot_be_written_into_original_experiment(self):
        with self.assertRaisesRegex(ValueError, "separate runtime"):
            supplement.manifest_path(ROOT / ".runtime/experiments/a/pre.json")

    def test_normal_receipts_include_all_graphs_repeat_associations_and_bindings(self):
        runner = supplement.frozen.Runner("fixture")
        normal = dict(rows=[dict(path="records-r0/graph.json", sha256="a" * 64)],
                      repeats=[dict(path="repeat-0.json", sha256="b" * 64)])
        with patch.object(supplement.approved, "digest", return_value="b" * 64), \
                patch.object(supplement.frozen, "read", return_value={"associations": [dict(path="records-r0/a.json", sha256="c" * 64)]}):
            receipts = supplement.normal_receipts(runner, normal)
        prefix = runner.run.relative_to(ROOT).as_posix() + "/"
        self.assertEqual(set(receipts), {prefix + name for name in ("pre.json", "inference.json", "direct-parent-index.json",
                                                                  "records-r0/graph.json", "repeat-0.json", "records-r0/a.json")})

    def test_full_barrier_keeps_guard_and_checks_complete_receipts_before_after(self):
        self.isolated('''
            from types import SimpleNamespace
            from unittest.mock import Mock, patch
            sha = "a" * 64
            run_path = s.ROOT / ".runtime/experiments/supplement-test-does-not-exist"
            bind = dict(run_id="mock", pre_sha256=sha, inference_sha256=sha, direct_index_sha256=sha)
            source = {"fixture_only.py": sha}
            before, config, index = dict(hashes=source), {}, dict(hashes=source, ancestor_hashes={"old.py": sha})
            normal, graphs = dict(rows=[]), [dict(reader="stroke_chain_axis"), dict(reader="foreground_consensus_axis")]
            runner = SimpleNamespace(kind="fixture", run=run_path, run_id="mock", old=SimpleNamespace(block_truth=s.approved.block_truth),
                                     checked=Mock(return_value=(before, config, index)), normal=Mock(return_value=(normal, graphs)))
            body = dict(runs={"fixture": bind, "g1_object": bind}, supplemental_source_hashes={name: sha for name in s.SOURCES},
                        approved_hash_provider_sha256=sha,
                        failed_attempts={"fixture": dict(path="failure.json", sha256=sha, log_path="failure.log", log_sha256=sha, exit_code=1)})
            events, writes = [], []
            def verify(root, receipts):
                events.append(("receipts", dict(receipts)))
            def validate(config):
                events.append(("normal", None))
                return normal, graphs
            runner.normal.side_effect = validate
            with patch.object(s.frozen, "Runner", return_value=runner), patch.object(s, "verify_manifest", return_value=body) as manifest, \
                    patch.object(s, "normal_receipts", return_value={"graph.json": sha}), patch.object(s, "verify_receipts", side_effect=verify), \
                    patch.object(s, "binding", return_value=bind), patch.object(s.approved, "digest", return_value=sha), \
                    patch.object(s.frozen, "read", side_effect=lambda p: dict(hashes={s.HASH_PROVIDER: sha}) if p.name == "pre.json" else normal), \
                    patch.object(s.frozen, "write", side_effect=lambda p, v: writes.append((p, v))):
                result = s.barrier("fixture", s.ROOT / ".runtime/validation/supplement-test-does-not-exist/freeze.json", sha)
            assert [event[0] for event in events] == ["receipts", "normal", "receipts"]
            assert events[0][1] == events[2][1] == {**source, "old.py": sha, "graph.json": sha}
            assert runner.checked.call_count == 2 and all(c.kwargs == {"ancestors": False} for c in runner.checked.call_args_list)
            assert manifest.call_count == 2
            assert len(writes) == 3 and writes[-1][0] == run_path / "normal-barrier.json"
            assert result["provider"] == s.PROVIDER and result["inference_repeated"] is False
            assert all(writes[i][1]["hashes"] == events[0][1] for i in (0, 1))
            try:
                (s.ROOT / "data/eval_gt/supplement-test-does-not-exist.json").read_bytes()
            except PermissionError:
                pass
            else:
                raise AssertionError("truth guard no longer active")
        ''')


if __name__ == "__main__":
    unittest.main()
