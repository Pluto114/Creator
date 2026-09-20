"""Freeze, infer and evaluate a conditional cylinder filter on existing RGB pools."""
from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_candidate_extent import bound_selected_candidate  # noqa: E402
from creator_eval.rod_cylinder_gate import (  # noqa: E402
    policy_values,
    recheck_finite_cylinder,
    select_cylinder_hypothesis,
)
from creator_eval.rod_multiview_candidates import apply_guide_identity_guard  # noqa: E402
from run_rod_candidate_ablation import (  # noqa: E402
    artifact_record,
    canonical_hash,
    checked_artifact,
    expected_keys,
    inference_view,
)
from run_rod_identity_blender import digest, locations, read_json, write_json  # noqa: E402
from run_rod_identity_stress import (  # noqa: E402
    frozen_source_check,
    reject_truth_open,
    validate_inference_identity,
)
from run_rod_support_recheck import SOURCES as BASE_SOURCES  # noqa: E402
from run_rod_support_recheck import checked_run as checked_baseline  # noqa: E402
from run_rod_support_recheck import score_record, verify_final_support  # noqa: E402

CONFIG = ROOT / "configs/rod_cylinder_screen_v1.json"
RUN_ID = "rod-cylinder-screen-v1-20260920"
SHARE = ROOT / "docs/experiments/results/2026-09-20-cylinder-screen.json"
SOURCES = [*BASE_SOURCES, "experiments/src/creator_eval/rod_radius_consistency.py",
           "experiments/src/creator_eval/rod_cylinder_gate.py", "scripts/run_rod_cylinder_controls.py",
           "scripts/run_rod_cylinder_screen.py"]


def checked_run(run_id):
    run, inputs, _, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    if prepared["run_id"] != run_id:
        raise ValueError("Run identity mismatch")
    frozen_source_check(run, prepared)
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    method = read_json(run / "method_config.json")
    if digest(ROOT / prepared["analytic_controls"]) != prepared["analytic_controls_sha256"]:
        raise ValueError("Pre-replay analytic controls changed")
    return run, inputs, prepared, method, read_json(inputs / "manifest.json")


def prepare(config_path):
    config = read_json(config_path)
    policy = policy_values(config["cylinder_screen"])
    run, inputs, _, output = locations(config["run_id"])
    if any(p.exists() for p in (run, inputs, output)):
        raise FileExistsError("Preserve every previous cylinder-screen run")
    baseline, _, old, method, _, _ = checked_baseline(config["baseline_run_id"])
    if old["parent_run_id"] != config["source_run_id"] or read_json(baseline / "protocol.json")["evaluation"] != config["evaluation"]:
        raise ValueError("Baseline provenance or evaluation policy mismatch")
    inference = read_json(baseline / "inference.json")
    validate_inference_identity(inference, old)
    controls = read_json(ROOT / config["analytic_controls"])
    if controls["state"] != "complete" or controls["policy"] != policy:
        raise ValueError("Run analytic controls under the same policy first")
    for relative, expected in controls["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Analytic control source changed before freeze")
    pools, pool_records = {}, []
    records = []
    # 先把输入身份核完再落盘。复用的是冻结候选与几何关联，不是假装又做了一次检测。
    for entry in inference["pools"]:
        pool = checked_artifact(baseline, entry)
        key = (pool["case_id"], pool["search_offset_px"])
        if key in pools:
            raise ValueError("Duplicate source pool")
        verify_final_support(pool["frames"], method["image_hypotheses"])
        for frame in pool["frames"]:
            if digest(ROOT / frame["rgb"]) != frame["rgb_sha256"]:
                raise ValueError("Original RGB changed")
        pools[key] = (entry, pool)
    if set(pools) != set(itertools.product(method["case_ids"], method["search_offsets_px"])):
        raise ValueError("Incomplete source pools")
    seen = set()
    for entry in inference["associations"]:
        record = checked_artifact(baseline, entry)
        key = (record["case_id"], record["search_offset_px"], record["cap"])
        if key in seen or record["pool_sha256"] != pools[key[:2]][0]["sha256"]:
            raise ValueError("Duplicate association or wrong source pool")
        seen.add(key)
        if canonical_hash(record["association"]) != record["association_sha256"] or not record["association"]["search_complete"]:
            raise ValueError("Source association changed or was incomplete")
        records.append((entry, record))
    if seen != {k[:3] for k in expected_keys(method)}:
        raise ValueError("Incomplete source associations")
    run.mkdir(parents=True)
    (run / "associations").mkdir()
    (inputs / "pools").mkdir(parents=True)
    (inputs / "associations").mkdir()
    for key, (entry, _) in pools.items():
        path = inputs / "pools" / Path(entry["path"]).name
        shutil.copy2(baseline / entry["path"], path)
        pool_records.append(artifact_record(inputs, path, case_id=key[0], search_offset_px=key[1]))
    association_records = []
    for entry, record in records:
        path = inputs / "associations" / Path(entry["path"]).name
        write_json(path, {k: record[k] for k in ("case_id", "search_offset_px", "cap", "pool_sha256", "association", "association_sha256")})
        association_records.append(artifact_record(inputs, path, baseline_artifact_sha256=entry["sha256"]))
    write_json(inputs / "manifest.json", {"pools": pool_records, "associations": association_records, "truth_excluded": True,
                                         "input_scope": "frozen_RGB_edge_pools_and_stored_geometric_associations_no_old_scores"})
    method["cylinder_screen"] = policy
    write_json(run / "method_config.json", method)
    shutil.copy2(config_path, run / "protocol.json")
    source_hashes = {}
    for relative in SOURCES:
        dest = run / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
        source_hashes[relative] = digest(ROOT / relative)
    write_json(run / "prepared.json", {
        "state": "prepared", "run_id": config["run_id"], "source_sha256": source_hashes,
        "input_manifest_sha256": digest(inputs / "manifest.json"), "method_config_sha256": digest(run / "method_config.json"),
        "protocol_sha256": digest(run / "protocol.json"), "baseline_run_id": config["baseline_run_id"],
        "baseline_prepared_sha256": digest(baseline / "prepared.json"), "baseline_inference_sha256": digest(baseline / "inference.json"),
        "parent_run_id": config["source_run_id"], "parent_prepared_sha256": old["parent_prepared_sha256"],
        "analytic_controls": config["analytic_controls"], "analytic_controls_sha256": digest(ROOT / config["analytic_controls"]),
        "gt_read_during_prepare": False,
    })
    print("PREPARED_CYLINDER_SCREEN", len(records), "associations", len(expected_keys(method)), "conditions", flush=True)


def infer(run_id):
    run, inputs, prepared, method, manifest = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    start = time.perf_counter()
    pools = {}
    for entry in manifest["pools"]:
        pool = checked_artifact(inputs, entry)
        pools[(pool["case_id"], pool["search_offset_px"])] = (entry["sha256"], pool)
    records = []
    for entry in manifest["associations"]:
        source = checked_artifact(inputs, entry)
        sha, pool = pools[(source["case_id"], source["search_offset_px"])]
        if source["pool_sha256"] != sha or canonical_hash(source["association"]) != source["association_sha256"]:
            raise ValueError("Source pool or association changed")
        frames, cap = pool["frames"], source["cap"]
        observations = [f["observations"] for f in frames]
        views = [inference_view(f, f["pool"]["candidates"][:cap], 0) for f in frames]
        begin = time.perf_counter()
        association = select_cylinder_hypothesis(source["association"], views, observations, method["extent"], method["cylinder_screen"])
        seconds = time.perf_counter() - begin
        association_hash = canonical_hash(association)
        variants = []
        for identity in method["identity_offsets_px"]:
            views = [inference_view(f, f["pool"]["candidates"][:cap], identity) for f in frames]
            guarded = apply_guide_identity_guard(association, views, method["guide_guard"])
            finite = bound_selected_candidate(guarded, views, observations, method["extent"])
            finite = recheck_finite_cylinder(finite, guarded["selected"], views, observations, method["cylinder_screen"])
            variants.append({"identity_offset_px": identity, "association_sha256": association_hash,
                             "guarded": {k: guarded[k] for k in ("state", "reason", "guide_identity_guard")}, "finite": finite})
        if canonical_hash(association) != association_hash:
            raise ValueError("Shared filtered association was mutated")
        path = run / "associations" / Path(entry["path"]).name
        write_json(path, {**{k: source[k] for k in ("case_id", "search_offset_px", "cap", "pool_sha256")},
                          "source_association_sha256": source["association_sha256"], "association": association,
                          "association_sha256": association_hash, "association_seconds": seconds, "variants": variants})
        records.append(artifact_record(run, path))
        screen = association["cylinder_screen"]
        print("SCREENED", source["case_id"], source["search_offset_px"], cap,
              screen["input_count"], "->", screen["survivor_count"], association["state"], [v["finite"]["state"] for v in variants], flush=True)
    write_json(run / "inference.json", {
        **{k: prepared[k] for k in ("run_id", "source_sha256", "input_manifest_sha256", "method_config_sha256")},
        "state": "inferred", "gt_read_during_inference": False, "truth_read_tripwire_enabled": True,
        "elapsed_seconds": time.perf_counter() - start, "associations": records,
        "scope": "only_incremental_screen_and_finite_stage_timing_excludes_cached_detection_and_association",
    })


def evaluate(config_path, share):
    config = read_json(config_path)
    run, inputs, prepared, method, manifest = checked_run(config["run_id"])
    _, _, _, output = locations(config["run_id"])
    if output.exists() or share.exists():
        raise FileExistsError("Preserve previous cylinder-screen scores")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Frozen scoring protocol changed")
    inference = read_json(run / "inference.json")
    validate_inference_identity(inference, prepared)
    baseline, _, _, _ = locations(prepared["baseline_run_id"])
    if digest(baseline / "prepared.json") != prepared["baseline_prepared_sha256"] or digest(baseline / "inference.json") != prepared["baseline_inference_sha256"]:
        raise ValueError("Baseline provenance changed")
    parent, _, truth, _ = locations(prepared["parent_run_id"])
    if digest(parent / "prepared.json") != prepared["parent_prepared_sha256"]:
        raise ValueError("Truth parent changed")
    parent_meta = read_json(parent / "prepared.json")
    for filename, key in (("manifest.json", "truth_manifest_sha256"), ("artifact_hashes.json", "truth_artifacts_sha256")):
        if digest(truth / filename) != parent_meta[key]:
            raise ValueError("Truth index changed")
    for relative, expected in read_json(truth / "artifact_hashes.json").items():
        if digest(truth / relative) != expected:
            raise ValueError("Truth artifact changed")
    targets = {c["case_id"]: read_json(truth / c["path"]) for c in read_json(truth / "manifest.json")["cases"]}
    frozen = {}
    for entry in manifest["pools"]:
        checked_artifact(inputs, entry)
    for entry in manifest["associations"]:
        source = checked_artifact(inputs, entry)
        frozen[(source["case_id"], source["search_offset_px"], source["cap"])] = source
    audits, scored = [], []
    for source_run, listing in ((run, inference), (baseline, read_json(baseline / "inference.json"))):
        rows = []
        for entry in listing["associations"]:
            record = checked_artifact(source_run, entry)
            if source_run == run:
                key = (record["case_id"], record["search_offset_px"], record["cap"])
                source = frozen[key]
                if record["source_association_sha256"] != source["association_sha256"] or record["pool_sha256"] != source["pool_sha256"]:
                    raise ValueError("Filtered result was detached from its input association")
                audit = {k: record[k] for k in ("case_id", "search_offset_px", "cap")}
                audit.update(record["association"]["cylinder_screen"])
                audits.append(audit)
                for variant in record["variants"]:
                    finite = variant["finite"]
                    if finite["state"] == "accepted" and finite.get("final_cylinder_screen", {}).get("passed") is not True:
                        raise ValueError("Accepted finite axis lacks final cylinder validation")
            rows.extend(score_record(record, targets[record["case_id"]], config["evaluation"]))
        keys = [(r["case_id"], r["search_offset_px"], r["cap"], r["identity_offset_px"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys(method):
            raise ValueError("Missing or duplicate paired conditions")
        scored.append(rows)
    result = {"state": "complete", "run_id": config["run_id"], "baseline_run_id": prepared["baseline_run_id"],
              "scope": config["scope"], "protocol_sha256": prepared["protocol_sha256"], "inference_sha256": digest(run / "inference.json"),
              "analytic_controls_sha256": prepared["analytic_controls_sha256"], "elapsed_seconds": inference["elapsed_seconds"],
              "timing_scope": inference["scope"], "row_count": len(scored[0]), "rows": scored[0], "baseline_rows_rescored": scored[1],
              "screen_audit": audits, "limitations": ["Five already-seen synthetic development scenes; exact cameras and synthetic guides.",
              "Only stored best/support competitors are screened, not all generated unique lines or full image evidence.",
              "One-pixel localization and circular silhouette are assumptions; coherent wrong objects can pass.",
              "Radius is a nuisance fit, not a validated physical diameter; no estimated-camera or 0.01-pack success claim."]}
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share, result)
    print("EVALUATED_CYLINDER_SCREEN", len(scored[0]), "paired conditions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--share-json", type=Path, default=SHARE)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.config, args.share_json)
