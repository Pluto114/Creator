"""Separate search windows, retained candidate caps, and identity guides on old RGB."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from creator_eval.rod_candidate_extent import bound_selected_candidate  # noqa: E402
from creator_eval.rod_multiview_candidates import (  # noqa: E402
    apply_guide_identity_guard,
    enumerate_image_lines,
    image_line_from_endpoints,
)
from creator_eval.rod_observations import extract_rod_observations  # noqa: E402
from run_rod_identity_blender import (  # noqa: E402
    classify,
    clean,
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

CONFIG = ROOT / "configs/rod_candidate_ablation_v1.json"
DEFAULT_RUN_ID = "rod-candidate-ablation-v1-20260920r1"
SHARE = ROOT / "docs/experiments/results/2026-09-20-candidate-ablation.json"
SOURCES = [
    "scripts/run_rod_candidate_ablation.py", "scripts/run_rod_identity_blender.py",
    "scripts/run_rod_identity_stress.py", "experiments/src/creator_eval/rod_candidate_pool.py",
    "experiments/src/creator_eval/rod_candidate_audit.py",
    "experiments/src/creator_eval/rod_candidate_association.py",
    "experiments/src/creator_eval/rod_multiview_candidates.py",
    "experiments/src/creator_eval/rod_observations.py",
    "experiments/src/creator_eval/rod_candidate_extent.py",
    "experiments/src/creator_eval/rod_evidence.py",
    "experiments/src/creator_eval/line_controls.py",
    "experiments/src/creator_eval/native_diagnostics.py",
]


def canonical_hash(value):
    encoded = json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def expected_keys(method):
    return set(itertools.product(method["case_ids"], method["search_offsets_px"],
                                 method["candidate_caps"], method["identity_offsets_px"]))


def validate_plan(config, frames_by_case):
    for key in ("case_ids", "search_offsets_px", "identity_offsets_px", "candidate_caps"):
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Plan needs nonempty unique {key}")
    caps = config["candidate_caps"]
    if any(isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0 for cap in caps):
        raise ValueError("Candidate caps must be positive integers")
    if caps != sorted(caps):
        raise ValueError("Candidate caps must be increasing nested prefixes")
    for case_id in config["case_ids"]:
        frames = frames_by_case[case_id]
        if len(frames) < 4 or len({frame["view_id"] for frame in frames}) != len(frames):
            raise ValueError("Each case needs distinct views for finite leave-one-out")
        needed = math.comb(len(frames), 3) * max(caps) ** 3
        if config["association_budget"] < needed:
            raise ValueError("Both association budgets must cover the largest retained pool")


def prepare(config_path):
    config = read_json(config_path)
    run, _, _, output = locations(config["run_id"])
    if run.exists() or output.exists():
        raise FileExistsError("Keep existing candidate ablation output")
    parent_run, parent_inputs, _, _ = locations(config["source_run_id"])
    parent = read_json(parent_run / "prepared.json")
    if parent["state"] != "prepared" or parent["run_id"] != config["source_run_id"]:
        raise ValueError("Parent identity mismatch")
    for path, key in ((parent_inputs / "manifest.json", "input_manifest_sha256"),
                      (parent_run / "method_config.json", "method_config_sha256")):
        if digest(path) != parent[key]:
            raise ValueError("Frozen source inputs or method changed")
    manifest = read_json(parent_inputs / "manifest.json")
    cases = {case["case_id"]: case["frames"] for case in manifest["cases"]}
    validate_plan(config, cases)
    method = read_json(parent_run / "method_config.json")
    method = {key: method[key] for key in ("observation", "image_hypotheses", "association_common", "guide_guard", "extent")}
    for key in ("case_ids", "search_offsets_px", "identity_offsets_px", "candidate_caps", "legacy_pool_checks"):
        method[key] = config[key]
    method["association_common"] = {
        **method["association_common"], "minimum_support_views": 4,
        "maximum_hypotheses": config["association_budget"],
        "maximum_fit_attempts": config["association_budget"],
    }
    selected = []
    for case_id in config["case_ids"]:
        frames = []
        for frame in cases[case_id]:
            rgb = parent_inputs / frame["rgb"]
            if digest(rgb) != frame["rgb_sha256"]:
                raise ValueError("Source RGB changed")
            frames.append({**frame, "rgb": rgb.relative_to(ROOT).as_posix()})
        selected.append({"case_id": case_id, "frames": frames})
    run.mkdir(parents=True)
    (run / "pools").mkdir()
    (run / "associations").mkdir()
    source_hashes = {}
    for relative in SOURCES:
        source = ROOT / relative
        destination = run / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative] = digest(source)
    shutil.copy2(config_path, run / "protocol.json")
    write_json(run / "method_config.json", method)
    write_json(run / "input_index.json", {"cases": selected, "truth_excluded": True})
    write_json(run / "prepared.json", {
        "state": "prepared", "run_id": config["run_id"],
        "protocol_sha256": digest(run / "protocol.json"),
        "method_config_sha256": digest(run / "method_config.json"),
        "input_manifest_sha256": digest(run / "input_index.json"),
        "source_sha256": source_hashes,
        "parent_run_id": config["source_run_id"],
        "parent_prepared_sha256": digest(parent_run / "prepared.json"),
        "parent_inference_sha256": digest(parent_run / "inference.json"),
        "development_reuse_declared": True,
        "gt_read_during_prepare": False,
    })
    print("PREPARED_ABLATION", len(expected_keys(method)), "planned rows", flush=True)


def checked_inputs(run_id):
    run, _, _, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    if prepared["state"] != "prepared" or prepared["run_id"] != run_id:
        raise ValueError("Prepared identity mismatch")
    frozen_source_check(run, prepared)
    if digest(run / "input_index.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Input index changed")
    return run, prepared, read_json(run / "method_config.json"), read_json(run / "input_index.json")


def inference_view(frame, candidates, identity_offset):
    # 搜图窗口已经固定。这里挪的只是一条身份提示，不再重提取像素。
    guide = np.asarray(frame["guide_xyxy"], float).copy()
    guide[:, 0] += identity_offset
    return {
        "view_id": frame["view_id"], "K_index": frame["K_index"],
        "world_to_camera_cv": frame["world_to_camera_cv"], "size_wh": frame["size_wh"],
        "y_range": guide[:, 1].tolist(), "guide_line": image_line_from_endpoints(guide),
        "candidates": candidates,
    }


def artifact_record(run, path, **metadata):
    return {**metadata, "path": path.relative_to(run).as_posix(), "sha256": digest(path)}


def infer(run_id):
    from creator_eval.rod_candidate_association import associate_multiview_lines_cached
    from creator_eval.rod_candidate_pool import enumerate_image_line_pool

    run, prepared, method, index = checked_inputs(run_id)
    sys.addaudithook(reject_truth_open)
    started = time.perf_counter()
    pool_records, association_records = [], []
    for case in index["cases"]:
        images = []
        for frame in case["frames"]:
            path = ROOT / frame["rgb"]
            if digest(path) != frame["rgb_sha256"]:
                raise ValueError("Input RGB changed")
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")))
        for search_offset in method["search_offsets_px"]:
            pool_start = time.perf_counter()
            frames = []
            for frame, rgb in zip(case["frames"], images):
                guide = np.asarray(frame["guide_xyxy"], float).copy()
                guide[:, 0] += search_offset
                raw = extract_rod_observations(rgb, guide, method["observation"])
                pool = enumerate_image_line_pool(raw, method["image_hypotheses"])
                check_key = {"case_id": case["case_id"], "search_offset_px": search_offset, "view_id": frame["view_id"]}
                legacy_equal = None
                if check_key in method["legacy_pool_checks"]:
                    legacy = enumerate_image_lines(raw, {**method["image_hypotheses"], "maximum_models": method["image_hypotheses"]["trials"]})
                    legacy_equal = canonical_hash(legacy) == canonical_hash(pool["candidates"])
                    if not legacy_equal:
                        raise ValueError("Accelerated pool differs from legacy full pool")
                frames.append({**frame, "observations": raw, "pool": pool, "legacy_full_pool_exact_match": legacy_equal})
            pool_path = run / "pools" / f"{case['case_id']}-search-{search_offset}.json"
            write_json(pool_path, {"case_id": case["case_id"], "search_offset_px": search_offset,
                                   "elapsed_seconds": time.perf_counter() - pool_start, "frames": frames})
            pool_records.append(artifact_record(run, pool_path, case_id=case["case_id"], search_offset_px=search_offset))
            print("POOL", case["case_id"], search_offset, [len(frame["pool"]["candidates"]) for frame in frames], flush=True)
            for cap in method["candidate_caps"]:
                start = time.perf_counter()
                views = [inference_view(frame, frame["pool"]["candidates"][:cap], 0) for frame in frames]
                association = associate_multiview_lines_cached(views, method["association_common"])
                if not association["search_complete"]:
                    raise RuntimeError("Planned exhaustive retained-pool search did not complete")
                association_seconds = time.perf_counter() - start
                association_hash = canonical_hash(association)
                variants = []
                for identity_offset in method["identity_offsets_px"]:
                    identity_views = [inference_view(frame, frame["pool"]["candidates"][:cap], identity_offset) for frame in frames]
                    guarded = apply_guide_identity_guard(association, identity_views, method["guide_guard"])
                    finite = bound_selected_candidate(guarded, identity_views, [frame["observations"] for frame in frames], method["extent"])
                    variants.append({
                        "identity_offset_px": identity_offset, "association_sha256": association_hash,
                        "guarded": {key: guarded[key] for key in ("state", "reason", "guide_identity_guard")},
                        "finite": finite,
                    })
                if canonical_hash(association) != association_hash:
                    raise ValueError("Identity variants mutated their shared association")
                path = run / "associations" / f"{case['case_id']}-search-{search_offset}-cap-{cap}.json"
                write_json(path, {"case_id": case["case_id"], "search_offset_px": search_offset, "cap": cap,
                                  "pool_sha256": pool_records[-1]["sha256"], "association": association,
                                  "association_sha256": association_hash, "association_seconds": association_seconds,
                                  "variants": variants})
                association_records.append(artifact_record(run, path))
                print("ASSOCIATION", case["case_id"], search_offset, cap, association["state"],
                      association["unique_hypothesis_count"], round(association_seconds, 2), flush=True)
    write_json(run / "inference.json", {
        "state": "inferred", "run_id": run_id, "source_sha256": prepared["source_sha256"],
        "input_manifest_sha256": prepared["input_manifest_sha256"], "method_config_sha256": prepared["method_config_sha256"],
        "gt_read_during_inference": False, "truth_read_tripwire_enabled": True,
        "elapsed_seconds": time.perf_counter() - started, "pools": pool_records, "associations": association_records,
    })


def checked_artifact(run, entry):
    path = (run / entry["path"]).resolve()
    if not path.is_relative_to(run.resolve()) or digest(path) != entry["sha256"]:
        raise ValueError("Frozen derived artifact changed")
    return read_json(path)



def require_legacy_regression(records):
    # 如果连旧的相同条件都复现不了，这轮就没有资格把差异归因于新增候选。
    fields = ("association_exact_match", "guarded_state_equal", "finite_exact_match")
    if not records or any(row.get(field) is not True for row in records for field in fields):
        raise ValueError("Legacy regression failed; do not publish this ablation as complete")


def evaluate(config_path, share_path):
    from creator_eval.rod_candidate_audit import audit_candidate_edges

    config = read_json(config_path)
    run, prepared, method, _ = checked_inputs(config["run_id"])
    _, _, _, output = locations(config["run_id"])
    if output.exists() or share_path.exists():
        raise FileExistsError("Keep previous ablation evaluation")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    inference = read_json(run / "inference.json")
    validate_inference_identity(inference, prepared)
    parent_run, _, truth, _ = locations(prepared["parent_run_id"])
    if digest(parent_run / "prepared.json") != prepared["parent_prepared_sha256"] or digest(parent_run / "inference.json") != prepared["parent_inference_sha256"]:
        raise ValueError("Parent provenance changed")
    parent = read_json(parent_run / "prepared.json")
    for filename, key in (("manifest.json", "truth_manifest_sha256"), ("artifact_hashes.json", "truth_artifacts_sha256")):
        if digest(truth / filename) != parent[key]:
            raise ValueError("Parent truth index changed")
    for relative, expected in read_json(truth / "artifact_hashes.json").items():
        if digest(truth / relative) != expected:
            raise ValueError("Parent truth changed")
    truths = {entry["case_id"]: read_json(truth / entry["path"]) for entry in read_json(truth / "manifest.json")["cases"]}
    reference = {(row["case_id"], row["guide_offset_px"]): row for row in read_json(parent_run / "inference.json")["rows"]}
    pools, pool_audits = {}, []
    for entry in inference["pools"]:
        pool = checked_artifact(run, entry)
        key = (pool["case_id"], pool["search_offset_px"])
        if key in pools:
            raise ValueError("Duplicate observation pool")
        pools[key] = pool
        truth_frames = {frame["view_id"]: frame for frame in truths[key[0]]["frames"]}
        for frame in pool["frames"]:
            ids = np.load(truth / truth_frames[frame["view_id"]]["arrays"]["surface_id"]["path"], allow_pickle=False)
            audits = [{"rank": rank + 1, "support_rows": hypothesis["support_rows"],
                       "width_median_px": hypothesis["width_median_px"],
                       **audit_candidate_edges(frame["observations"], hypothesis, ids,
                                               probe_offset_px=config["edge_probe_offset_px"])}
                      for rank, hypothesis in enumerate(frame["pool"]["candidates"][:max(method["candidate_caps"])])]
            pool_audits.append({"case_id": key[0], "search_offset_px": key[1], "view_id": frame["view_id"],
                               "sampled_pool_size": len(frame["pool"]["candidates"]),
                               "legacy_full_pool_exact_match": frame["legacy_full_pool_exact_match"], "candidates": audits})
    if set(pools) != set(itertools.product(method["case_ids"], method["search_offsets_px"])):
        raise ValueError("Observation pool plan mismatch")
    rows, seen, regression = [], set(), []
    pool_hashes = {(entry["case_id"], entry["search_offset_px"]): entry["sha256"] for entry in inference["pools"]}
    for entry in inference["associations"]:
        record = checked_artifact(run, entry)
        case_id, search_offset, cap = record["case_id"], record["search_offset_px"], record["cap"]
        association = record["association"]
        if record["pool_sha256"] != pool_hashes[(case_id, search_offset)] or canonical_hash(association) != record["association_sha256"]:
            raise ValueError("Association provenance mismatch")
        target = truths[case_id]
        geometry_class, geometry_error = classify(association, target, config["evaluation"])
        for variant in record["variants"]:
            identity_offset = variant["identity_offset_px"]
            key = (case_id, search_offset, cap, identity_offset)
            if key in seen or variant["association_sha256"] != record["association_sha256"]:
                raise ValueError("Duplicate row or inconsistent shared association")
            seen.add(key)
            guarded = {**association, **variant["guarded"]}
            finite = variant["finite"]
            guide_class, _ = classify(guarded, target, config["evaluation"])
            segments = target["target"].get("segments", [target["target"]["endpoints"]]) if target["target"]["present"] else []
            metrics = curve_metrics(finite["segments"], segments, tolerance=config["evaluation"]["curve_tolerance_m"], spacing=config["evaluation"]["curve_spacing_m"])
            row = {"case_id": case_id, "search_offset_px": search_offset, "cap": cap, "identity_offset_px": identity_offset,
                   "split": "development_regression", "geometry_classification": geometry_class, "guide_classification": guide_class,
                   "geometry_error": geometry_error, "association_state": association["state"],
                   "search_complete": association["search_complete"], "combinations": association["attempted_combination_count"],
                   "unique_hypothesis_count": association["unique_hypothesis_count"], "association_sha256": record["association_sha256"],
                   "association_seconds": record["association_seconds"], "finite_state": finite["state"],
                   "finite_reasons": finite["rejection_reasons"], "segments": finite["segments"], "curve_metrics": metrics}
            if target.get("gap_segment"):
                row["gap"] = gap_coverage(finite["segments"], target["gap_segment"], tolerance=config["evaluation"]["curve_tolerance_m"], spacing=config["evaluation"]["curve_spacing_m"])
            rows.append(row)
            if cap == 8 and identity_offset == search_offset:
                old = reference[(case_id, identity_offset)]
                regression.append({"case_id": case_id, "offset_px": identity_offset,
                                   "association_exact_match": canonical_hash(association) == canonical_hash(old["association"]),
                                   "guarded_state_equal": guarded["state"] == old["guarded"]["state"],
                                   "finite_exact_match": canonical_hash(finite) == canonical_hash(old["finite"])})
    if seen != expected_keys(method):
        raise ValueError("Missing or unexpected ablation rows")
    require_legacy_regression(regression)
    result = {"state": "complete", "run_id": config["run_id"], "scope": config["scope"],
              "protocol_sha256": prepared["protocol_sha256"], "inference_sha256": digest(run / "inference.json"),
              "elapsed_seconds": inference["elapsed_seconds"], "row_count": len(rows), "association_count": len(inference["associations"]),
              "rows": rows, "legacy_regression": regression, "image_pool_audit": pool_audits,
              "limitations": ["Reused development scenes, exact cameras and synthetic guides; no independent-object performance.",
                              "Full pool means the deduplicated output of a fixed random sample, not all possible image lines.",
                              "The original post-association guide guard does not resolve ambiguous associations.",
                              "One-pixel surface-ID probes are posthoc diagnostics, not RGB antialias or silhouette certification."]}
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share_path, result)
    print("EVALUATED_ABLATION", len(rows), "rows; old cap8 regression:", regression, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
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
