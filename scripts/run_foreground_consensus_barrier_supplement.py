"""Supplemental normal barrier; preserve frozen inference and its truth guards."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import foreground_consensus_runner as frozen
import run_g1_object_rgb as approved

ROOT = frozen.ROOT
PROVIDER = "legacy_approved_hash_supplement_v1"
SCHEMA = "creator.foreground-consensus-barrier-supplement.v1"
SOURCES = ("scripts/run_foreground_consensus_barrier_supplement.py",
           "tests/test_foreground_consensus_barrier_supplement.py")
HASH_PROVIDER = "scripts/run_g1_object_rgb.py"
ATTEMPTS = ".runtime/validation/foreground-consensus-runners-20261012-001"


def verify_receipts(root, hashes):
    """Only the existing approved helper may open generation config bytes."""
    frozen.direct.validate_hashes(hashes)
    for name, expected in sorted(hashes.items()):
        if approved.digest(frozen.direct.checked_path(root, name)) != expected:
            raise ValueError("Supplemental full receipt changed: " + name)
    return len(hashes)


def combine_receipts(*maps):
    result = {}
    for hashes in maps:
        frozen.direct.validate_hashes(hashes)
        for name, sha in hashes.items():
            if name in result and result[name] != sha:
                raise ValueError("Conflicting receipt: " + name)
            result[name] = sha
    return dict(sorted(result.items()))


def binding(runner):
    return dict(run_id=runner.run_id,
                pre_sha256=approved.digest(runner.run / "pre.json"),
                inference_sha256=approved.digest(runner.run / "inference.json"),
                direct_index_sha256=approved.digest(runner.run / "direct-parent-index.json"))


def failed_attempt(kind):
    name = f"{ATTEMPTS}/{kind}-barrier.json"
    record = frozen.read(ROOT / name)
    log_name = f"{ATTEMPTS}/{kind}-barrier.log"
    expected_code = 1073807364 if kind == "fixture" else 1
    if (record.get("kind") != kind or record.get("stage") != "barrier"
            or record.get("exit_code") != expected_code
            or approved.digest(ROOT / log_name) != record["log_sha256"]):
        raise ValueError("Original failed/interrupted attempt identity changed")
    return dict(path=name, sha256=approved.digest(ROOT / name), log_path=log_name,
                log_sha256=record["log_sha256"], exit_code=expected_code)


def manifest_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / ".runtime/validation"):
        raise ValueError("Supplement manifest must be a separate runtime validation artifact")
    return path


def freeze_manifest(path):
    path = manifest_path(path)
    record = dict(schema=SCHEMA, provider=PROVIDER, gt_read=False,
                  frozen_normal_sources_modified=False,
                  supplemental_source_hashes={name: approved.digest(ROOT / name) for name in SOURCES},
                  approved_hash_provider_path=HASH_PROVIDER,
                  approved_hash_provider_sha256=approved.digest(ROOT / HASH_PROVIDER),
                  runs={kind: binding(frozen.Runner(kind)) for kind in frozen.CASES},
                  failed_attempts={kind: failed_attempt(kind) for kind in frozen.CASES})
    frozen.write(path, record)
    return approved.digest(path)


def verify_manifest(path, expected_sha256):
    path = manifest_path(path)
    frozen.direct.validate_hashes({path.relative_to(ROOT).as_posix(): expected_sha256})
    if approved.digest(path) != expected_sha256:
        raise ValueError("Supplement manifest external SHA changed")
    record = frozen.read(path)
    if (set(record) != {"schema", "provider", "gt_read", "frozen_normal_sources_modified",
                       "supplemental_source_hashes", "approved_hash_provider_path",
                       "approved_hash_provider_sha256", "runs", "failed_attempts"}
            or record["schema"] != SCHEMA or record["provider"] != PROVIDER
            or record["gt_read"] is not False or record["frozen_normal_sources_modified"] is not False
            or set(record["supplemental_source_hashes"]) != set(SOURCES)
            or record["approved_hash_provider_path"] != HASH_PROVIDER
            or set(record["runs"]) != set(frozen.CASES)
            or set(record["failed_attempts"]) != set(frozen.CASES)):
        raise ValueError("Exact supplemental source/provider/two-suite manifest required")
    verify_receipts(ROOT, combine_receipts(record["supplemental_source_hashes"],
                    {HASH_PROVIDER: record["approved_hash_provider_sha256"]}))
    for kind in frozen.CASES:
        if record["runs"][kind] != binding(frozen.Runner(kind)):
            raise ValueError("Frozen normal binding changed: " + kind)
        if record["failed_attempts"][kind] != failed_attempt(kind):
            raise ValueError("Prior attempt binding changed: " + kind)
    return record


def normal_receipts(runner, normal):
    prefix = runner.run.relative_to(ROOT).as_posix() + "/"
    maps = [{prefix + name: approved.digest(runner.run / name)
             for name in ("pre.json", "inference.json", "direct-parent-index.json")}]
    maps.extend({prefix + row["path"]: row["sha256"]} for row in normal["rows"])
    for repeat in normal["repeats"]:
        maps.append({prefix + repeat["path"]: repeat["sha256"]})
        path = frozen.direct.checked_path(ROOT, prefix + repeat["path"])
        if approved.digest(path) != repeat["sha256"]:
            raise ValueError("Normal repeat changed before barrier")
        maps.extend({prefix + row["path"]: row["sha256"]}
                    for row in frozen.read(path)["associations"])
    return combine_receipts(*maps)


def barrier(kind, manifest, manifest_sha256):
    runner = frozen.Runner(kind)
    sys.addaudithook(runner.old.block_truth)
    sys.addaudithook(approved.block_truth)
    destination = runner.run / "normal-barrier.json"
    folder = manifest_path(manifest).parent
    receipts_paths = [folder / f"{kind}-receipts-{phase}.json" for phase in ("before", "after")]
    if any(path.exists() for path in (destination, *receipts_paths)):
        raise FileExistsError("Preserve every previous supplemental barrier attempt")
    started = time.perf_counter()
    supplement = verify_manifest(manifest, manifest_sha256)
    before, config, index = runner.checked(ancestors=False)
    provider_run = frozen.Runner("g1_object")
    provider_pre = frozen.read(provider_run.run / "pre.json")
    if (approved.digest(provider_run.run / "pre.json") != supplement["runs"]["g1_object"]["pre_sha256"]
            or provider_pre["hashes"].get(HASH_PROVIDER) != supplement["approved_hash_provider_sha256"]):
        raise ValueError("Hash provider is not bound to the original object-suite source freeze")
    normal = frozen.read(runner.run / "inference.json")
    receipts = combine_receipts(before["hashes"], index["hashes"], index["ancestor_hashes"],
                                normal_receipts(runner, normal))
    verify_receipts(ROOT, receipts)
    before_elapsed = time.perf_counter() - started
    checked_normal, graphs = runner.normal(config)
    if checked_normal != normal:
        raise ValueError("Normal changed during complete semantic validation")
    after, after_config, after_index = runner.checked(ancestors=False)
    if (after, after_config, after_index) != (before, config, index):
        raise ValueError("Frozen source/input inventory changed during barrier")
    verify_receipts(ROOT, receipts)
    if verify_manifest(manifest, manifest_sha256) != supplement:
        raise ValueError("Supplement changed during barrier")
    if binding(runner) != supplement["runs"][kind]:
        raise ValueError("Normal binding changed during barrier")
    for phase, path in zip(("before", "after"), receipts_paths):
        frozen.write(path, dict(provider=PROVIDER, phase=phase,
                               gt_read=False, hashes=receipts, **supplement["runs"][kind]))
    failure = supplement["failed_attempts"][kind]
    result = dict(run_id=runner.run_id, state="passed", gt_read=False,
        pre_sha256=supplement["runs"][kind]["pre_sha256"],
        inference_sha256=supplement["runs"][kind]["inference_sha256"],
        full_ancestor_receipts=len(index["ancestor_hashes"]), direct_receipts=len(index["hashes"]),
        own_receipts=len(before["hashes"]), normal_rows=len(graphs), repeat_pairs=len(graphs)//2,
        controls_equal=sum(g["reader"] in frozen.CONTROLS for g in graphs),
        elapsed_seconds=time.perf_counter()-started, provider=PROVIDER,
        supplemental_manifest_path=manifest_path(manifest).relative_to(ROOT).as_posix(),
        supplemental_manifest_sha256=manifest_sha256,
        supplemental_source_hashes=supplement["supplemental_source_hashes"],
        approved_hash_provider_path=HASH_PROVIDER,
        approved_hash_provider_sha256=supplement["approved_hash_provider_sha256"],
        approved_hash_provider_original_run_id=provider_run.run_id,
        approved_hash_provider_original_pre_sha256=supplement["runs"]["g1_object"]["pre_sha256"],
        failed_attempt_path=failure["path"], failed_attempt_sha256=failure["sha256"],
        failed_attempt_log_path=failure["log_path"], failed_attempt_log_sha256=failure["log_sha256"],
        failed_attempt_exit_code=failure["exit_code"],
        complete_receipts_before=len(receipts), complete_receipts_after=len(receipts),
        before_elapsed_seconds=before_elapsed, receipts_unchanged=True,
        receipt_manifests=[dict(path=path.relative_to(ROOT).as_posix(), sha256=approved.digest(path))
                           for path in receipts_paths],
        inference_repeated=False, frozen_normal_sources_modified=False, **frozen.FLAGS)
    frozen.write(destination, result)
    print("FOREGROUND_SUPPLEMENT_FULL_SHA_BARRIER", kind, len(graphs), "normal rows", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "barrier"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--kind", choices=tuple(frozen.CASES))
    args = parser.parse_args()
    sys.addaudithook(approved.block_truth)
    if args.stage == "freeze":
        if args.kind or args.manifest_sha256:
            parser.error("freeze only accepts the new manifest destination")
        print("SUPPLEMENT_FROZEN", freeze_manifest(args.manifest), flush=True)
    else:
        if not args.kind or not args.manifest_sha256:
            parser.error("barrier requires kind and independently recorded manifest SHA")
        barrier(args.kind, args.manifest, args.manifest_sha256)


if __name__ == "__main__":
    main()
