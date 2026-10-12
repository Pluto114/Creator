"""No real ancestry, outputs, images or truth are accessed by these contracts."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import direct_parent_index as direct  # noqa: E402


class DirectParentIndexContracts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        for name, data in (("source.py", b"source"), ("input.json", b"input"), ("ancestor.json", b"ancestor")):
            (self.root / name).write_bytes(data)
        self.ancestors = {name: direct.digest(self.root / name) for name in ("input.json", "ancestor.json")}
        self.index = direct.build_index(self.root, run_id="new", parent_run_id="old",
            roles={"source": ["source.py"], "parent_normal": ["input.json"]}, ancestor_hashes=self.ancestors)
        self.path = self.root / "index.json"
        self.path.write_text(json.dumps(self.index), encoding="utf-8")
        self.sha = direct.digest(self.path)

    def read(self, **kwargs):
        return direct.read_index(self.root, self.path, self.sha, run_id="new", parent_run_id="old", **kwargs)

    def test_worker_checks_only_direct_inputs_not_ancestor_recursion(self):
        original = direct.digest
        visited = []
        def traced(path):
            visited.append(Path(path).name)
            return original(path)
        with patch.object(direct, "digest", side_effect=traced):
            self.read()
        self.assertEqual(visited, ["index.json", "input.json", "source.py"])

    def test_final_barrier_checks_full_ancestor_and_direct_union_once(self):
        original = direct.digest
        visited = []
        def traced(path):
            visited.append(Path(path).name)
            return original(path)
        with patch.object(direct, "digest", side_effect=traced):
            self.read(ancestors=True)
        self.assertEqual(visited, ["index.json", "ancestor.json", "input.json", "source.py"])

    def test_every_direct_byte_change_rejected_even_with_original_mtime_and_size(self):
        path = self.root / "input.json"
        stat = path.stat()
        path.write_bytes(b"other")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            self.read()

    def test_ancestor_change_deferred_but_always_rejected_by_final_barrier(self):
        (self.root / "ancestor.json").write_bytes(b"changed")
        self.read()
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            self.read(ancestors=True)

    def test_index_external_digest_required_before_content_or_roles_are_trusted(self):
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "index changed"):
            self.read()
        with self.assertRaisesRegex(ValueError, "immutable pre"):
            direct.read_index(self.root, self.path, "", run_id="new", parent_run_id="old")

    def test_typed_and_receipt_inventories_must_be_exact(self):
        for change in (lambda i: i["hashes"].pop("source.py"),
                       lambda i: i["roles"].update(unknown=["source.py"]),
                       lambda i: i["roles"].update(source=["source.py", "source.py"]),
                       lambda i: i["hashes"].update(extra="0" * 64),
                       lambda i: i["ancestor_hashes"].update({"input.json": "0" * 64}),
                       lambda i: i.update(gt_read=True)):
            value = copy.deepcopy(self.index)
            change(value)
            with self.assertRaises(ValueError):
                direct.validate_index(value)

    def test_run_identity_cannot_be_rebound(self):
        with self.assertRaisesRegex(ValueError, "different run"):
            direct.read_index(self.root, self.path, self.sha, run_id="wrong", parent_run_id="old")

    def test_truth_results_absolute_and_traversal_paths_cannot_enter_index(self):
        for name in ("../x", "/x", "D:/x", "a\\b", "./x", "a//b", "a/../b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                direct.relative_name(name)
        for name in ("data/eval_gt/x", "data/evaluation/x", "docs/experiments/results/x",
                     ".runtime/foo/evaluation.json", "protocol.json"):
            with self.subTest(name=name), self.assertRaises(PermissionError):
                direct.relative_name(name)

    def test_prepare_rejects_changed_original_input(self):
        (self.root / "input.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "ancestor verification"):
            direct.build_index(self.root, run_id="new", parent_run_id="old",
                roles={"parent_normal": ["input.json"]}, ancestor_hashes=self.ancestors)

    def test_guard_rejects_any_unlisted_input_and_records_allowed_reads(self):
        guard = direct.DirectReadGuard(self.root, self.index, self.root / ".runtime/experiments/new")
        guard("open", (str(self.root / "input.json"), "r"))
        self.assertEqual(guard.opened, {"input.json"})
        with self.assertRaisesRegex(PermissionError, "undeclared"):
            guard("open", (str(self.root / "ancestor.json"), "rb"))
        guard("open", (str(self.root / ".runtime/experiments/new/records-r0/a.json"), "r"))
        with self.assertRaises(PermissionError):
            guard("open", (str(self.root / ".runtime/experiments/new/evaluation.json"), "r"))
        guard.active = False
        guard("open", (str(self.root / "ancestor.json"), "rb"))

    def test_guard_never_exempts_repository_root_as_output(self):
        with self.assertRaisesRegex(ValueError, "exact new"):
            direct.DirectReadGuard(self.root, self.index, self.root)

    def test_one_actual_path_in_multiple_roles_hashed_once(self):
        original, visited = direct.digest, []
        def traced(path):
            visited.append(Path(path).name)
            return original(path)
        with patch.object(direct, "digest", side_effect=traced):
            direct.build_index(self.root, run_id="new", parent_run_id="old",
                roles={"source": ["input.json"], "input_manifest": ["input.json"]}, ancestor_hashes=self.ancestors)
        self.assertEqual(visited, ["input.json"])


if __name__ == "__main__":
    unittest.main()
