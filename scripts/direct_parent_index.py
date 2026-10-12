"""Frozen, typed direct inputs; SHA checks never recurse into producer code.

The caller verifies all ancestors once before building this index.  Workers
verify direct inputs; the final independent barrier verifies the ancestor map
too.  Neither step treats timestamps or an earlier verification as file hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

SCHEMA = "creator.direct-parent-index.v1"
ROLES = frozenset({"source", "input_manifest", "raw_record", "rgb", "cameras", "support",
    "base_snapshot", "candidate_view", "patch", "parent_pre", "parent_normal", "parent_graph",
    "parent_repeat", "parent_association", "control_normal", "control_graph", "control_repeat",
    "control_association", "evaluation_binding"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_name(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Canonical repository-relative path required")
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts or "." in parts or "/".join(parts) != name:
        raise ValueError("Canonical repository-relative path required")
    lower = name.lower()
    if (lower.startswith(("data/eval_gt/", "data/evaluation/", "docs/experiments/results/"))
            or parts[-1].lower() in {"evaluation.json", "protocol.json"}):
        raise PermissionError("Truth and scored results cannot enter a normal input index")
    return name


def checked_path(root, name):
    root = Path(root).resolve()
    target = root / relative_name(name)
    if not target.resolve().is_relative_to(root):
        raise ValueError("Input path resolves outside the repository")
    return target


def validate_hashes(hashes):
    if not isinstance(hashes, dict):
        raise ValueError("Receipt map required")
    for name, sha in hashes.items():
        relative_name(name)
        if not isinstance(sha, str) or _HASH.fullmatch(sha) is None:
            raise ValueError("Full lowercase SHA-256 receipt required")


def verify_hashes(root, hashes):
    validate_hashes(hashes)
    for name, sha in sorted(hashes.items()):
        if digest(checked_path(root, name)) != sha:
            raise ValueError("Frozen direct-parent receipt changed: " + name)
    return len(hashes)


def build_index(root, *, run_id, parent_run_id, roles, ancestor_hashes):
    """Bind exact typed paths to full byte hashes after the caller's ancestor audit."""
    validate_hashes(ancestor_hashes)
    if not isinstance(run_id, str) or not run_id or not isinstance(parent_run_id, str) or not parent_run_id:
        raise ValueError("Explicit current and parent run identities required")
    if not isinstance(roles, dict) or not roles or set(roles) - ROLES:
        raise ValueError("Known nonempty direct-input role map required")
    entries, hashes = {}, {}
    for role, names in sorted(roles.items()):
        if not isinstance(names, (list, tuple)) or not names or len(set(names)) != len(names):
            raise ValueError("Nonempty unique paths required for every role")
        entries[role] = sorted(relative_name(name) for name in names)
        for name in entries[role]:
            if name not in hashes:
                hashes[name] = digest(checked_path(root, name))
            sha = hashes[name]
            if name in ancestor_hashes and sha != ancestor_hashes[name]:
                raise ValueError("Direct input changed since its ancestor verification: " + name)
    result = dict(schema=SCHEMA, run_id=run_id, parent_run_id=parent_run_id, gt_read=False,
        roles=entries, hashes=dict(sorted(hashes.items())), ancestor_hashes=dict(sorted(ancestor_hashes.items())),
        verification_policy="full_sha256_direct_inputs_per_worker_full_ancestors_at_final_barrier")
    validate_index(result)
    return result


def validate_index(index):
    if not isinstance(index, dict) or set(index) != {"schema", "run_id", "parent_run_id", "gt_read",
            "roles", "hashes", "ancestor_hashes", "verification_policy"}:
        raise ValueError("Exact typed direct-parent index schema required")
    if (index["schema"] != SCHEMA or index["gt_read"] is not False
            or not isinstance(index["run_id"], str) or not index["run_id"]
            or not isinstance(index["parent_run_id"], str) or not index["parent_run_id"]
            or index["verification_policy"] != "full_sha256_direct_inputs_per_worker_full_ancestors_at_final_barrier"):
        raise ValueError("Direct-parent index identity or verification policy differs")
    roles = index["roles"]
    if not isinstance(roles, dict) or not roles or set(roles) - ROLES:
        raise ValueError("Unknown direct-parent input role")
    names = set()
    for paths in roles.values():
        if not isinstance(paths, list) or not paths or paths != sorted(set(paths)):
            raise ValueError("Typed paths must be sorted, unique and nonempty")
        names.update(relative_name(path) for path in paths)
    validate_hashes(index["hashes"])
    validate_hashes(index["ancestor_hashes"])
    if names != set(index["hashes"]):
        raise ValueError("Exact typed path and direct SHA inventory required")
    if any(index["ancestor_hashes"][name] != sha for name, sha in index["hashes"].items()
           if name in index["ancestor_hashes"]):
        raise ValueError("Ancestor and direct receipts disagree")
    return index


def read_index(root, path, expected_sha256, *, run_id, parent_run_id, ancestors=False):
    """Verify an externally frozen index digest, then full bytes of selected scope."""
    if not isinstance(expected_sha256, str) or _HASH.fullmatch(expected_sha256) is None:
        raise ValueError("Index SHA must be bound by the new run's immutable pre record")
    if digest(path) != expected_sha256:
        raise ValueError("Frozen direct-parent index changed")
    index = validate_index(json.loads(Path(path).read_text(encoding="utf-8")))
    if index["run_id"] != run_id or index["parent_run_id"] != parent_run_id:
        raise ValueError("Direct-parent index belongs to a different run")
    receipts = dict(index["ancestor_hashes"]) if ancestors else {}
    receipts.update(index["hashes"])
    verify_hashes(root, receipts)
    return index


class DirectReadGuard:
    """Audit hook refusing undeclared repository inputs during a worker's compute.

    The guarded block is installed after imports and initial receipt checks;
    existing Python environments are runtime dependencies, not ancestry data.
    Generated normal outputs under the exact new run may be read back.  Truth
    remains forbidden even there.  Callers still use their existing GT guard.
    """

    def __init__(self, root, index, output_directory, runtime_directories=()):
        validate_index(index)
        self.root = Path(root).resolve()
        self.allowed = {checked_path(root, name).resolve() for name in index["hashes"]}
        self.output = Path(output_directory).resolve()
        if self.output == self.root or not self.output.is_relative_to(self.root / ".runtime/experiments"):
            raise ValueError("Only the exact new experiment output directory is writable")
        self.runtime = tuple(Path(path).resolve() for path in runtime_directories)
        self.active = True
        self.opened = set()

    def __call__(self, event, args):
        if not self.active or event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1]
        if isinstance(mode, str) and all(letter not in mode for letter in ("r", "+")):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(self.root):
            return
        name = path.relative_to(self.root).as_posix()
        relative_name(name)
        if path in self.allowed:
            self.opened.add(name)
        elif path.is_relative_to(self.output) or any(path.is_relative_to(p) for p in self.runtime):
            return
        else:
            raise PermissionError("Worker attempted undeclared repository input: " + name)
