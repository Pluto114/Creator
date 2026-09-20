"""Replay a frozen candidate experiment with final-support validation enabled."""
from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))

from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from creator_eval.rod_candidate_association import associate_multiview_lines_cached  # noqa: E402
from creator_eval.rod_candidate_extent import bound_selected_candidate  # noqa: E402
from creator_eval.rod_candidate_pool import enumerate_image_line_pool  # noqa: E402
from creator_eval.rod_multiview_candidates import apply_guide_identity_guard  # noqa: E402
from run_rod_candidate_ablation import (  # noqa: E402
    SOURCES as BASE_SOURCES,
)
from run_rod_candidate_ablation import (  # noqa: E402
    artifact_record,
    canonical_hash,
    checked_artifact,
    checked_inputs,
    expected_keys,
    inference_view,
    validate_plan,
)
from run_rod_identity_blender import (  # noqa: E402
    classify,
    digest,
    locations,
    read_json,
    write_json,
)
from run_rod_identity_stress import reject_truth_open, validate_inference_identity  # noqa: E402

CONFIG = ROOT / "configs/rod_refit_support_v1.json"
RUN_ID = "rod-refit-support-v1-20260920"
SHARE = ROOT / "docs/experiments/results/2026-09-20-refit-support.json"
SOURCES = [*BASE_SOURCES, "scripts/run_rod_support_recheck.py"]


def verify_final_support(frames, config):
    total = 0
    for frame in frames:
        for candidate in frame["pool"]["candidates"]:
            matches = candidate["row_matches"]
            ys = [frame["observations"]["rows"][i]["y"] for i, _, _ in matches]
            if (len(ys) != candidate["support_rows"] or len({i for i, _, _ in matches}) != len(ys)
                    or len(ys) < config["minimum_rows"] or np.ptp(ys) < config["minimum_y_span"]):
                raise ValueError("A final candidate violates the support contract")
            total += 1
    return total


def prepare(config_path):
    config = read_json(config_path)
    run, inputs, _, output = locations(config["run_id"])
    if any(path.exists() for path in (run, inputs, output)):
        raise FileExistsError("Keep old support-recheck artifacts")
    if not 1 <= config["parallel_workers"] <= 4:
        raise ValueError("Use one to four independent pool processes")
    baseline, _, _, _ = locations(config["baseline_run_id"])
    old = read_json(baseline / "prepared.json")
    if old["run_id"] != config["baseline_run_id"] or old["parent_run_id"] != config["source_run_id"]:
        raise ValueError("Baseline identity mismatch")
    # 旧实验已经完成；核验它当时存下的源码，不能要求修bug后的工作树仍等于旧版。
    for relative, expected in old["source_sha256"].items():
        if digest(baseline / "source_snapshot" / relative) != expected:
            raise ValueError("Baseline source snapshot changed")
    for filename, key in (("method_config.json", "method_config_sha256"), ("input_index.json", "input_manifest_sha256")):
        if digest(baseline / filename) != old[key]:
            raise ValueError("Baseline input or method changed")
    old_inference = read_json(baseline / "inference.json")
    validate_inference_identity(old_inference, old)
    method, index = read_json(baseline / "method_config.json"), read_json(baseline / "input_index.json")
    cases = {case["case_id"]: case["frames"] for case in index["cases"]}
    validate_plan(config, cases)
    if expected_keys(method) != expected_keys(config):
        raise ValueError("Keep the exact sixty baseline conditions")
    if config["association_budget"] != method["association_common"]["maximum_fit_attempts"] or config["association_budget"] != method["association_common"]["maximum_hypotheses"]:
        raise ValueError("Do not change search budgets in this support control")
    if digest(baseline / "protocol.json") != old["protocol_sha256"]:
        raise ValueError("Baseline protocol changed")
    if read_json(baseline / "protocol.json")["evaluation"] != config["evaluation"]:
        raise ValueError("Keep baseline scoring thresholds")
    method["require_refit_support"] = True
    run.mkdir(parents=True)
    (run / "pools").mkdir()
    (run / "associations").mkdir()
    inputs.mkdir(parents=True)
    raw_records, seen = [], set()
    for entry in old_inference["pools"]:
        pool = checked_artifact(baseline, entry)
        key = (pool["case_id"], pool["search_offset_px"])
        if key in seen:
            raise ValueError("Duplicate baseline pool")
        seen.add(key)
        expected_frames = {frame["view_id"]: frame for frame in cases[key[0]]}
        if [f["view_id"] for f in pool["frames"]] != [f["view_id"] for f in cases[key[0]]]:
            raise ValueError("Baseline view order changed")
        frames = []
        for frame in pool["frames"]:
            base = expected_frames[frame["view_id"]]
            if any(frame.get(k) != v for k, v in base.items()) or digest(ROOT / base["rgb"]) != base["rgb_sha256"]:
                raise ValueError("Cached observations differ from the frozen RGB input identity")
            # 只带原始像素观测。旧候选本体和三维预测不会成为新推理的输入。
            frames.append({**base, "observations": frame["observations"],
                           "legacy_candidate_control_sha256": canonical_hash(frame["pool"]["candidates"])})
        path = inputs / f"{key[0]}-search-{key[1]}.json"
        write_json(path, {"case_id": key[0], "search_offset_px": key[1], "frames": frames})
        raw_records.append(artifact_record(inputs, path, case_id=key[0], search_offset_px=key[1], baseline_pool_sha256=entry["sha256"]))
    if seen != set(itertools.product(method["case_ids"], method["search_offsets_px"])):
        raise ValueError("Incomplete baseline pool plan")
    write_json(inputs / "manifest.json", {"pools": raw_records, "truth_excluded": True})
    write_json(run / "input_index.json", index)
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
        "protocol_sha256": digest(run / "protocol.json"), "method_config_sha256": digest(run / "method_config.json"),
        "input_manifest_sha256": digest(run / "input_index.json"), "observation_manifest_sha256": digest(inputs / "manifest.json"),
        "baseline_run_id": config["baseline_run_id"], "baseline_prepared_sha256": digest(baseline / "prepared.json"),
        "baseline_inference_sha256": digest(baseline / "inference.json"), "parent_run_id": config["source_run_id"],
        "parent_prepared_sha256": old["parent_prepared_sha256"], "parallel_workers": config["parallel_workers"],
        "gt_read_during_prepare": False,
    })
    print("PREPARED_SUPPORT_RECHECK", len(expected_keys(method)), flush=True)


def checked_run(run_id):
    run, prepared, method, index = checked_inputs(run_id)
    _, inputs, _, _ = locations(run_id)
    if method.get("require_refit_support") is not True or digest(inputs / "manifest.json") != prepared["observation_manifest_sha256"]:
        raise ValueError("Support policy or frozen observations changed")
    return run, inputs, prepared, method, index, read_json(inputs / "manifest.json")


def infer_pool(run_id, entry):
    run, inputs, _, method, _, _ = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    begin = time.perf_counter()
    raw = checked_artifact(inputs, entry)
    if (raw["case_id"], raw["search_offset_px"]) != (entry["case_id"], entry["search_offset_px"]):
        raise ValueError("Frozen observation identity mismatch")
    frames = []
    for frame in raw["frames"]:
        key = {"case_id": raw["case_id"], "search_offset_px": raw["search_offset_px"], "view_id": frame["view_id"]}
        legacy_equal = None
        if key in method["legacy_pool_checks"]:
            control = enumerate_image_line_pool(frame["observations"], method["image_hypotheses"])
            legacy_equal = canonical_hash(control["candidates"]) == frame["legacy_candidate_control_sha256"]
            if not legacy_equal:
                raise ValueError("Raw replay does not reproduce the frozen legacy candidate control")
        pool = enumerate_image_line_pool(frame["observations"], method["image_hypotheses"], require_refit_support=True)
        frames.append({**frame, "pool": pool, "legacy_full_pool_exact_match": legacy_equal})
    verify_final_support(frames, method["image_hypotheses"])
    case_id, offset = raw["case_id"], raw["search_offset_px"]
    path = run / "pools" / f"{case_id}-search-{offset}.json"
    write_json(path, {"case_id": case_id, "search_offset_px": offset, "frames": frames,
                      "raw_observation_sha256": entry["sha256"], "elapsed_seconds": time.perf_counter() - begin})
    pool_record = artifact_record(run, path, case_id=case_id, search_offset_px=offset)
    associations, brief = [], []
    for cap in method["candidate_caps"]:
        start = time.perf_counter()
        views = [inference_view(f, f["pool"]["candidates"][:cap], 0) for f in frames]
        association = associate_multiview_lines_cached(views, method["association_common"])
        seconds = time.perf_counter() - start
        if not association["search_complete"]:
            raise ValueError("Retained-pool search did not complete")
        sha = canonical_hash(association)
        variants = []
        for identity in method["identity_offsets_px"]:
            views = [inference_view(f, f["pool"]["candidates"][:cap], identity) for f in frames]
            guarded = apply_guide_identity_guard(association, views, method["guide_guard"])
            finite = bound_selected_candidate(guarded, views, [f["observations"] for f in frames], method["extent"])
            variants.append({"identity_offset_px": identity, "association_sha256": sha,
                             "guarded": {k: guarded[k] for k in ("state", "reason", "guide_identity_guard")}, "finite": finite})
        if canonical_hash(association) != sha:
            raise ValueError("Identity guard mutated the shared geometry result")
        path = run / "associations" / f"{case_id}-search-{offset}-cap-{cap}.json"
        write_json(path, {"case_id": case_id, "search_offset_px": offset, "cap": cap, "pool_sha256": pool_record["sha256"],
                          "association": association, "association_sha256": sha, "association_seconds": seconds, "variants": variants})
        associations.append(artifact_record(run, path))
        brief.append([cap, association["state"], association["unique_hypothesis_count"], round(seconds, 2)])
    return {"pool": pool_record, "associations": associations, "sizes": [len(f["pool"]["candidates"]) for f in frames], "brief": brief}


def infer(run_id):
    run, _, prepared, _, _, raw = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    start = time.perf_counter()
    workers = min(prepared["parallel_workers"], os.cpu_count() or 1)
    # 每个进程只负责一个窗口，不共享可变候选，也不共享输出文件。
    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(infer_pool, run_id, entry) for entry in raw["pools"]]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print("FINISHED_POOL", record["pool"]["case_id"], record["pool"]["search_offset_px"], record["sizes"], record["brief"], flush=True)
    records.sort(key=lambda r: (r["pool"]["case_id"], r["pool"]["search_offset_px"]))
    write_json(run / "inference.json", {
        **{k: prepared[k] for k in ("run_id", "source_sha256", "input_manifest_sha256", "method_config_sha256", "observation_manifest_sha256")},
        "state": "inferred", "gt_read_during_inference": False, "truth_read_tripwire_enabled": True,
        "workers": workers, "elapsed_seconds": time.perf_counter() - start,
        "pools": [r["pool"] for r in records], "associations": [a for r in records for a in r["associations"]],
    })


def score_record(record, target, policy):
    association = record["association"]
    if canonical_hash(association) != record["association_sha256"] or not association["search_complete"]:
        raise ValueError("Association hash or completion mismatch")
    geometry_class, error = classify(association, target, policy)
    target_segments = target["target"].get("segments", [target["target"]["endpoints"]]) if target["target"]["present"] else []
    rows = []
    for variant in record["variants"]:
        if variant["association_sha256"] != record["association_sha256"]:
            raise ValueError("Identity variants do not share the same geometry")
        guarded = {**association, **variant["guarded"]}
        finite = variant["finite"]
        label, _ = classify(guarded, target, policy)
        row = {"case_id": record["case_id"], "search_offset_px": record["search_offset_px"], "cap": record["cap"],
               "identity_offset_px": variant["identity_offset_px"], "association_sha256": record["association_sha256"],
               "association_state": association["state"], "geometry_classification": geometry_class, "geometry_error": error,
               "guide_classification": label, "finite_state": finite["state"], "finite_reasons": finite["rejection_reasons"],
               "unique_hypothesis_count": association["unique_hypothesis_count"], "search_complete": association["search_complete"],
               "combinations": association["attempted_combination_count"], "association_seconds": record["association_seconds"],
               "segments": finite["segments"], "finite_sha256": canonical_hash(finite),
               "curve_metrics": curve_metrics(finite["segments"], target_segments, tolerance=policy["curve_tolerance_m"], spacing=policy["curve_spacing_m"])}
        if target.get("gap_segment"):
            row["gap"] = gap_coverage(finite["segments"], target["gap_segment"], tolerance=policy["curve_tolerance_m"], spacing=policy["curve_spacing_m"])
        rows.append(row)
    return rows


def evaluate(config_path, share):
    config = read_json(config_path)
    run, _, prepared, method, index, observation_manifest = checked_run(config["run_id"])
    _, _, _, output = locations(config["run_id"])
    if output.exists() or share.exists():
        raise FileExistsError("Keep old support-recheck evaluation")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    inference = read_json(run / "inference.json")
    validate_inference_identity(inference, prepared)
    baseline, _, _, _ = locations(prepared["baseline_run_id"])
    if digest(baseline / "prepared.json") != prepared["baseline_prepared_sha256"] or digest(baseline / "inference.json") != prepared["baseline_inference_sha256"]:
        raise ValueError("Baseline reference changed")
    old_inference = read_json(baseline / "inference.json")
    parent, _, truth, _ = locations(prepared["parent_run_id"])
    if digest(parent / "prepared.json") != prepared["parent_prepared_sha256"]:
        raise ValueError("Parent truth provenance changed")
    parent_meta = read_json(parent / "prepared.json")
    for filename, key in (("manifest.json", "truth_manifest_sha256"), ("artifact_hashes.json", "truth_artifacts_sha256")):
        if digest(truth / filename) != parent_meta[key]:
            raise ValueError("Truth index changed")
    for relative, expected in read_json(truth / "artifact_hashes.json").items():
        if digest(truth / relative) != expected:
            raise ValueError("Truth artifact changed")
    targets = {c["case_id"]: read_json(truth / c["path"]) for c in read_json(truth / "manifest.json")["cases"]}
    raw_hashes = {(e["case_id"], e["search_offset_px"]): e["sha256"] for e in observation_manifest["pools"]}
    expected_views = {c["case_id"]: [f["view_id"] for f in c["frames"]] for c in index["cases"]}
    pools, audits = {}, []
    for entry in inference["pools"]:
        pool = checked_artifact(run, entry)
        key = (pool["case_id"], pool["search_offset_px"])
        if key in pools:
            raise ValueError("Duplicate corrected pool")
        pools[key] = entry["sha256"]
        if pool["raw_observation_sha256"] != raw_hashes[key] or [f["view_id"] for f in pool["frames"]] != expected_views[key[0]]:
            raise ValueError("Corrected pool observation hash or frame plan changed")
        if any(f["pool"].get("refit_support_validated") is not True for f in pool["frames"]):
            raise ValueError("A pool did not enable final support validation")
        verify_final_support(pool["frames"], method["image_hypotheses"])
        for f in pool["frames"]:
            audits.append({"case_id": key[0], "search_offset_px": key[1], "view_id": f["view_id"],
                           "size": len(f["pool"]["candidates"]), "sampling": f["pool"]["sampling"],
                           "legacy_full_pool_exact_match": f["legacy_full_pool_exact_match"], "final_support_violations": 0})
    if set(pools) != set(itertools.product(method["case_ids"], method["search_offsets_px"])):
        raise ValueError("Incomplete corrected pool plan")
    scored = []
    for source, listing in ((run, inference), (baseline, old_inference)):
        rows = []
        for entry in listing["associations"]:
            record = checked_artifact(source, entry)
            if source == run and record["pool_sha256"] != pools[(record["case_id"], record["search_offset_px"])]:
                raise ValueError("Corrected association pool hash mismatch")
            rows.extend(score_record(record, targets[record["case_id"]], config["evaluation"]))
        keys = [(r["case_id"], r["search_offset_px"], r["cap"], r["identity_offset_px"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys(method):
            raise ValueError("Missing or duplicate paired conditions")
        scored.append(rows)
    result = {"state": "complete", "run_id": config["run_id"], "baseline_run_id": prepared["baseline_run_id"],
              "scope": config["scope"], "protocol_sha256": prepared["protocol_sha256"], "inference_sha256": digest(run / "inference.json"),
              "elapsed_seconds": inference["elapsed_seconds"], "workers": inference["workers"],
              "row_count": len(scored[0]), "rows": scored[0], "baseline_rows_rescored": scored[1], "pool_audit": audits,
              "limitations": ["Same five development scenes and exact cameras; no independent-object claim.",
                              "Only final support validation changed; full pools remain finite random samples.",
                              "Parallel wall time and sums of worker times are different quantities."]}
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share, result)
    print("EVALUATED_SUPPORT_RECHECK", len(scored[0]), "paired conditions", flush=True)


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
