"""Freeze explicit RGB foreground claims and replay both geometry baselines fairly."""
from __future__ import annotations

import argparse
import copy
import itertools
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from creator_eval.rod_foreground_identity import (  # noqa: E402
    build_anchor_proposals,
    select_foreground_identity,
    validate_anchors,
)
from run_rod_candidate_ablation import (  # noqa: E402
    artifact_record,
    canonical_hash,
    checked_artifact,
    inference_view,
)
from run_rod_cylinder_validation import checked_run as checked_parent  # noqa: E402
from run_rod_identity_blender import (  # noqa: E402
    classify,
    digest,
    locations,
    read_json,
    write_json,
)
from run_rod_identity_stress import (  # noqa: E402
    frozen_source_check,
    reject_truth_open,
    validate_inference_identity,
)

CONFIG = ROOT / "configs/rod_foreground_development_v1.json"
RUN_ID = "rod-foreground-development-v1-20260921"
SHARE = ROOT / "docs/experiments/results/2026-09-21-foreground-development.json"
EXTRA_SOURCES = ["experiments/src/creator_eval/rod_foreground_identity.py", "scripts/run_rod_foreground_identity.py"]


def make_views(frames, cap):
    return [{**inference_view(f, f["pool"]["candidates"][:cap], 0), "rgb_sha256": f["rgb_sha256"]} for f in frames]


def profile_anchors(query, profile):
    anchors = copy.deepcopy(query["anchors"])
    if len(anchors) != len(profile["dx_per_anchor"]):
        raise ValueError("Jitter profile must align with the recorded clicks")
    for a, dx in zip(anchors, profile["dx_per_anchor"]):
        a["xy"][0] += dx
        a["uncertainty_xy_px"] = [profile["uncertainty_px"]] * 2
    return anchors[:profile["keep_anchors"]]


def checked_run(run_id):
    run, inputs, _, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    if prepared["run_id"] != run_id or prepared["state"] != "prepared":
        raise ValueError("Run identity changed")
    frozen_source_check(run, prepared)
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Frozen inputs changed")
    manifest = read_json(inputs / "manifest.json")
    annotations = checked_artifact(inputs, manifest["annotations"])
    return run, inputs, prepared, read_json(run / "method_config.json"), manifest, annotations


def prepare(config_path):
    config = read_json(config_path)
    run, inputs, _, output = locations(config["run_id"])
    if any(p.exists() for p in (run, inputs, output)):
        raise FileExistsError("Keep every earlier identity experiment")
    parent, _, parent_meta, method, parent_inputs = checked_parent(config["source_run_id"])
    inference = read_json(parent / "inference.json")
    validate_inference_identity(inference, parent_meta)
    annotations = read_json(ROOT / config["annotation_file"])
    if annotations["source_run_id"] != config["source_run_id"] or annotations["ground_truth_used_for_coordinates"] or annotations["predicted_geometry_used_for_coordinates"]:
        raise ValueError("Annotations must refer to the exact RGB source without GT/prediction-derived coordinates")
    queries = annotations["queries"]
    if len({q["query_id"] for q in queries}) != len(queries) or not queries:
        raise ValueError("Unique nonempty annotation queries required")
    planned = {(q["query_id"], q["case_id"]) for q in config["evaluation_queries"]}
    if planned != {(q["query_id"], q["case_id"]) for q in queries} or len(planned) != len(config["evaluation_queries"]):
        raise ValueError("Annotation and evaluation query plan mismatch")
    if len({p["name"] for p in config["profiles"]}) != len(config["profiles"]):
        raise ValueError("Duplicate perturbation profile")
    if not 1 <= config["parallel_workers"] <= 4:
        raise ValueError("One to four isolated worker processes required")
    source_pools, source_records = [], []
    for entry in inference["pools"]:
        pool = checked_artifact(parent, entry)
        views = make_views(pool["frames"], max(method["candidate_caps"]))
        for frame in pool["frames"]:
            if digest(ROOT / frame["rgb"]) != frame["rgb_sha256"]:
                raise ValueError("RGB changed before annotation freeze")
        for query in queries:
            if query["case_id"] == pool["case_id"]:
                for profile in config["profiles"]:
                    validate_anchors(profile_anchors(query, profile), views, config["identity_policy"])
        source_pools.append((entry, pool))
    expected_cases = {c["case_id"] for c in parent_inputs["cases"]}
    if any(q["case_id"] not in expected_cases for q in queries):
        raise ValueError("Annotation query refers to an unknown case")
    expected = set(itertools.product(method["case_ids"], method["search_offsets_px"], method["candidate_caps"], ("baseline", "cylinder_support")))
    seen = set()
    for entry in inference["associations"]:
        record = checked_artifact(parent, entry)
        key = (record["case_id"], record["search_offset_px"], record["cap"], record["method"])
        if key in seen or canonical_hash(record["association"]) != record["association_sha256"]:
            raise ValueError("Source association identity changed")
        seen.add(key)
        source_records.append((entry, record))
    if seen != expected:
        raise ValueError("Incomplete source method/condition plan")
    run.mkdir(parents=True)
    (run / "records").mkdir()
    (inputs / "pools").mkdir(parents=True)
    (inputs / "associations").mkdir()
    pools, records = [], []
    for entry, pool in source_pools:
        path = inputs / "pools" / Path(entry["path"]).name
        shutil.copy2(parent / entry["path"], path)
        pools.append(artifact_record(inputs, path, case_id=pool["case_id"], search_offset_px=pool["search_offset_px"]))
    for entry, record in source_records:
        path = inputs / "associations" / Path(entry["path"]).name
        write_json(path, {k: record[k] for k in ("case_id", "search_offset_px", "cap", "method", "pool_sha256", "association", "association_sha256")})
        records.append(artifact_record(inputs, path))
    # Query labels and truth selectors stay in protocol.json; inference gets only image claims.
    write_json(inputs / "anchors.json", {"queries": [{k: q[k] for k in ("query_id", "case_id", "anchors")} for q in queries],
                                         "origin": annotations["annotation_origin"]})
    write_json(inputs / "manifest.json", {"pools": pools, "associations": records,
               "annotations": artifact_record(inputs, inputs / "anchors.json"), "truth_excluded": True})
    method.update(identity_policy=config["identity_policy"], profiles=config["profiles"], parallel_workers=config["parallel_workers"])
    write_json(run / "method_config.json", method)
    shutil.copy2(config_path, run / "protocol.json")
    source_hashes = {}
    for relative in dict.fromkeys([*parent_meta["source_sha256"], *EXTRA_SOURCES]):
        destination = run / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        source_hashes[relative] = digest(ROOT / relative)
    write_json(run / "prepared.json", {"state": "prepared", "run_id": config["run_id"],
        "source_sha256": source_hashes, "input_manifest_sha256": digest(inputs / "manifest.json"),
        "method_config_sha256": digest(run / "method_config.json"), "protocol_sha256": digest(run / "protocol.json"),
        "parent_run_id": config["source_run_id"], "parent_prepared_sha256": digest(parent / "prepared.json"),
        "parent_inference_sha256": digest(parent / "inference.json"), "annotation_file_sha256": digest(ROOT / config["annotation_file"]),
        "gt_read_during_prepare": False})
    print("PREPARED_FOREGROUND", len(queries), "queries", len(records), "source records", flush=True)


def compact_bundle(bundle):
    for proposal in bundle["proposals"]:
        full = proposal["finite"]
        proposal["finite_sha256"] = canonical_hash(full)
        proposal["finite"] = {k: full[k] for k in ("state", "segments", "candidate_selection", "rejection_reasons", "line") if k in full}
    return bundle


def infer_record(run_id, entry):
    run, inputs, _, method, manifest, annotations = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    start = time.perf_counter()
    source = checked_artifact(inputs, entry)
    pool_entry = next(p for p in manifest["pools"] if (p["case_id"], p["search_offset_px"]) == (source["case_id"], source["search_offset_px"]))
    pool = checked_artifact(inputs, pool_entry)
    if source["pool_sha256"] != pool_entry["sha256"] or canonical_hash(source["association"]) != source["association_sha256"]:
        raise ValueError("Geometry is detached from its frozen pool")
    frames = pool["frames"]
    views, raw = make_views(frames, source["cap"]), [f["observations"] for f in frames]
    cylinder = method["cylinder_screen"] if source["method"] == "cylinder_support" else None
    bundle = compact_bundle(build_anchor_proposals(source["association"], views, raw, method["extent"], cylinder))
    bundle_sha = canonical_hash(bundle)
    rows = []
    for query in annotations["queries"]:
        if query["case_id"] != source["case_id"]:
            continue
        for profile in method["profiles"]:
            anchors = profile_anchors(query, profile)
            result = select_foreground_identity(bundle, views, raw, anchors, method["identity_policy"])
            rows.append({"query_id": query["query_id"], "profile": profile["name"], "anchors": anchors, "identity": result})
    if canonical_hash(bundle) != bundle_sha:
        raise ValueError("Anchors modified geometric candidates")
    path = run / "records" / Path(entry["path"]).name
    write_json(path, {**{k: source[k] for k in ("case_id", "search_offset_px", "cap", "method", "pool_sha256", "association_sha256")},
                      "input_record_sha256": entry["sha256"], "bundle": bundle, "bundle_sha256": bundle_sha, "rows": rows,
                      "elapsed_seconds": time.perf_counter() - start})
    return artifact_record(run, path)


def infer(run_id):
    run, _, prepared, method, manifest, _ = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=method["parallel_workers"]) as executor:
        jobs = [executor.submit(infer_record, run_id, entry) for entry in manifest["associations"]]
        for job in as_completed(jobs):
            record = job.result()
            records.append(record)
            print("FOREGROUND_RECORD", record["path"], flush=True)
    records.sort(key=lambda r: r["path"])
    write_json(run / "inference.json", {**{k: prepared[k] for k in ("run_id", "source_sha256", "input_manifest_sha256", "method_config_sha256")},
        "state": "inferred", "gt_read_during_inference": False, "truth_read_tripwire_enabled": True,
        "elapsed_seconds": time.perf_counter() - start, "records": records,
        "timing_scope": "incremental finite proposals and anchor screening; excludes frozen RGB detection and geometric association"})


def query_truth(query, target, truth_root):
    if query["target_selector"] == "declared":
        return target
    if query["target_selector"] == "empty":
        return {**target, "target": {"present": False}, "gap_segment": None}
    if isinstance(query["target_selector"], bool) or not isinstance(query["target_selector"], int):
        raise ValueError("Truth selector must be declared, empty, or a surface ID")
    geometry = read_json(truth_root / target["geometry_path"])
    pieces = [o["world_centerline_endpoints"] for o in geometry["objects"] if o["surface_id"] == query["target_selector"] and "world_centerline_endpoints" in o]
    if len(pieces) != 1:
        raise ValueError("Only one finite straight member per alternate-object query")
    return {**target, "target": {"present": True, "endpoints": pieces[0]}, "gap_segment": None}


def evaluate(config_path, share):
    config = read_json(config_path)
    run, inputs, prepared, method, manifest, annotations = checked_run(config["run_id"])
    _, _, _, output = locations(config["run_id"])
    if output.exists() or share.exists():
        raise FileExistsError("Keep old foreground-identity scores")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Protocol changed")
    parent, _, truth, _ = locations(prepared["parent_run_id"])
    if digest(parent / "prepared.json") != prepared["parent_prepared_sha256"] or digest(parent / "inference.json") != prepared["parent_inference_sha256"]:
        raise ValueError("Parent reference changed")
    parent_meta = read_json(parent / "prepared.json")
    for filename, key in (("manifest.json", "truth_manifest_sha256"), ("artifact_hashes.json", "truth_artifacts_sha256")):
        if digest(truth / filename) != parent_meta[key]:
            raise ValueError("Truth index changed")
    for relative, expected in read_json(truth / "artifact_hashes.json").items():
        if digest(truth / relative) != expected:
            raise ValueError("Truth artifact changed")
    targets = {c["case_id"]: read_json(truth / c["path"]) for c in read_json(truth / "manifest.json")["cases"]}
    queries = {q["query_id"]: q for q in config["evaluation_queries"]}
    annotation_lookup = {q["query_id"]: q for q in annotations["queries"]}
    profiles = {p["name"]: p for p in method["profiles"]}
    input_hashes = {entry["sha256"] for entry in manifest["associations"]}
    for entry in manifest["pools"] + manifest["associations"]:
        checked_artifact(inputs, entry)
    inference = read_json(run / "inference.json")
    validate_inference_identity(inference, prepared)
    rows, policy = [], config["evaluation"]
    for entry in inference["records"]:
        record = checked_artifact(run, entry)
        if record["input_record_sha256"] not in input_hashes or canonical_hash(record["bundle"]) != record["bundle_sha256"]:
            raise ValueError("Prediction lost source or finite-bundle identity")
        for variant in record["rows"]:
            query = queries[variant["query_id"]]
            if query["case_id"] != record["case_id"] or variant["anchors"] != profile_anchors(annotation_lookup[query["query_id"]], profiles[variant["profile"]]):
                raise ValueError("Prediction used different annotation coordinates")
            target = query_truth(query, targets[query["case_id"]], truth)
            result = variant["identity"]
            selected = next((p for p in record["bundle"]["proposals"] if p["ordinal"] == result["selected_ordinal"]), None)
            if result["state"] == "accepted":
                if selected is None or selected["finite"]["state"] != "accepted" or canonical_hash(result["segments"]) != canonical_hash(selected["finite"]["segments"]):
                    raise ValueError("Identity selector fabricated or changed geometry")
            label, error = classify({"state": "ambiguous" if result["state"] == "unresolved" else result["state"],
                                     "selected": {"model": selected["finite"]["line"]} if selected else None}, target, policy)
            segments = target["target"].get("segments", [target["target"]["endpoints"]]) if target["target"]["present"] else []
            row = {**{k: record[k] for k in ("case_id", "search_offset_px", "cap", "method", "bundle_sha256")},
                   "query_id": query["query_id"], "query_role": query["role"], "profile": variant["profile"],
                   "state": result["state"], "reason": result["reason"], "selected_ordinal": result["selected_ordinal"],
                   "anchor_view_count": result["anchor_view_count"], "classification": label, "geometry_error": error,
                   "segments": result["segments"], "candidate_count": len(record["bundle"]["proposals"]),
                   "curve_metrics": curve_metrics(result["segments"], segments, tolerance=policy["curve_tolerance_m"], spacing=policy["curve_spacing_m"])}
            if target.get("gap_segment"):
                row["gap"] = gap_coverage(result["segments"], target["gap_segment"], tolerance=policy["curve_tolerance_m"], spacing=policy["curve_spacing_m"])
            rows.append(row)
    expected = set(itertools.product(queries, method["search_offsets_px"], method["candidate_caps"], ("baseline", "cylinder_support"), profiles))
    keys = [(r["query_id"], r["search_offset_px"], r["cap"], r["method"], r["profile"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate query/method/perturbation plan")
    result = {"state": "complete", "run_id": config["run_id"], "parent_run_id": prepared["parent_run_id"],
              "scope": config["scope"], "protocol_sha256": prepared["protocol_sha256"], "inference_sha256": digest(run / "inference.json"),
              "annotation_file_sha256": prepared["annotation_file_sha256"], "row_count": len(rows), "rows": rows,
              "elapsed_seconds": inference["elapsed_seconds"], "timing_scope": inference["timing_scope"],
              "limitations": ["Foreground claims are additional annotation information, not automatic recognition.",
                 "Both baselines receive identical clicks; coarse-guide-only historical results have a different input budget.",
                 "Geometry unchanged and no click triangulation; consistently wrong object claims can still be accepted.",
                 "Stored candidate subset, exact cameras and procedural scenes; no independent-object or real-user claim."]}
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share, result)
    print("EVALUATED_FOREGROUND", len(rows), "query conditions", flush=True)


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
