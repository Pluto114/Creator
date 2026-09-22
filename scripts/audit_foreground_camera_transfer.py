"""Evaluation-only camera substitutions; does not mutate normal inputs or patches."""
from __future__ import annotations

import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from run_foreground_estimated_pilot import (
    ROOT,
    apply_similarity,
    associate_multiview_lines_cached,
    build_anchor_proposals,
    canonical_hash,
    checked,
    compact_bundle,
    curve_metrics,
    digest,
    gap_coverage,
    locations,
    make_views,
    query_truth,
    read_json,
    select_foreground_identity,
    select_supported_cylinders,
    write_json,
)

CONFIG = ROOT / "configs/foreground_camera_counterfactual_v1.json"


def substitute_cameras(estimated, oracle, similarity, variant):
    frames = copy.deepcopy(estimated)
    matrix = np.asarray(similarity, float)
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    rotation, translation = matrix[:3, :3] / scale, matrix[:3, 3]
    if variant not in ("oracle_intrinsics_only", "oracle_pose_only", "oracle_intrinsics_and_pose"):
        raise ValueError("Unknown camera counterfactual")
    if [f["view_id"] for f in frames] != [f["view_id"] for f in oracle]:
        raise ValueError("Camera comparison view order changed")
    for frame, truth in zip(frames, oracle):
        if frame["rgb_sha256"] != truth["rgb_sha256"]:
            raise ValueError("Counterfactual changed the source image")
        if variant in ("oracle_intrinsics_only", "oracle_intrinsics_and_pose"):
            frame["K_index"] = truth["K_index"]
        if variant in ("oracle_pose_only", "oracle_intrinsics_and_pose"):
            gt = np.asarray(truth["world_to_camera_cv"])[:3]
            # GT相机只用于这里的诊断。换回估计坐标单位后再算，别顺手把尺度也换掉。
            frame["world_to_camera_cv"] = np.c_[gt[:, :3] @ rotation, (gt[:, :3] @ translation + gt[:, 3]) / scale]
        frame["camera_source"] = "evaluation_only_" + variant
    return frames, scale, rotation, translation


def audit_case(config, case_id, destination):
    run, _, frozen, method = checked(config["source_run_id"])
    protocol = read_json(run / "protocol.json")
    original = read_json(ROOT / "data/evaluation" / config["source_run_id"] / "summary.json")
    case = next(c for c in original["cases"] if c["case_id"] == case_id)
    geometry = read_json(run / case_id / "geometry.json")
    assert digest(run / case_id / "geometry.json") == case["geometry_sha256"]
    parent, parent_inputs = locations(protocol["source_run_id"])
    parent_meta = read_json(parent / "prepared.json")
    assert digest(parent / "prepared.json") == frozen["parent_prepared_sha256"]
    assert digest(parent_inputs / "manifest.json") == parent_meta["input_manifest_sha256"]
    oracle = next(c["frames"] for c in read_json(parent_inputs / "manifest.json")["cases"] if c["case_id"] == case_id)
    truth_root = ROOT / "data/eval_gt" / protocol["source_run_id"]
    assert digest(truth_root / "artifact_hashes.json") == parent_meta["truth_artifacts_sha256"]
    for path, sha in read_json(truth_root / "artifact_hashes.json").items():
        assert digest(truth_root / path) == sha
    target_entry = next(c for c in read_json(truth_root / "manifest.json")["cases"] if c["case_id"] == case_id)
    target = read_json(truth_root / target_entry["path"])
    assert digest(ROOT / protocol["evaluation_queries_file"]) == frozen["evaluation_queries_sha256"]
    queries = {q["query_id"]: q for q in read_json(ROOT / protocol["evaluation_queries_file"])["evaluation_queries"]}
    rows, started = [], time.perf_counter()
    for variant in config["variants"]:
        frames, scale, rotation, translation = substitute_cameras(geometry["frames"], oracle, case["alignment"]["prediction_world_to_gt_world"], variant)
        views, raw = make_views(frames, method["candidate_cap"]), [f["observations"] for f in frames]
        association = associate_multiview_lines_cached(views, method["association_common"])
        if not association["search_complete"]:
            raise ValueError("Incomplete diagnostic association")
        cylinder = select_supported_cylinders(association, views, raw, method["extent"], method["cylinder_screen"])
        for name, proposal in (("baseline", association), ("cylinder_support", cylinder)):
            bundle = compact_bundle(build_anchor_proposals(proposal, views, raw, method["extent"], method["cylinder_screen"] if name == "cylinder_support" else None))
            bundle_sha = canonical_hash(bundle)
            for query in method["annotations"]["queries"]:
                if query["case_id"] != case_id:
                    continue
                result = select_foreground_identity(bundle, views, raw, query["anchors"], method["identity_policy"])
                selected_truth = query_truth(queries[query["query_id"]], target, truth_root)
                truth = selected_truth["target"].get("segments", [selected_truth["target"].get("endpoints")]) if selected_truth["target"]["present"] else []
                aligned = apply_similarity(np.asarray(result["segments"]).reshape(-1, 2, 3), scale, rotation, translation)
                score = curve_metrics(aligned, truth, tolerance=protocol["curve_tolerance_m"], spacing=protocol["curve_spacing_m"])
                row = dict(case_id=case_id, query_id=query["query_id"], method=name, variant=variant, state=result["state"], reason=result["reason"],
                           target_present=selected_truth["target"]["present"], role=queries[query["query_id"]]["role"], segments=aligned, metrics=score)
                if selected_truth.get("gap_segment"):
                    row["gap"] = gap_coverage(aligned, selected_truth["gap_segment"], tolerance=protocol["curve_tolerance_m"], spacing=protocol["curve_spacing_m"])
                rows.append(row)
            if canonical_hash(bundle) != bundle_sha:
                raise ValueError("Identity mutated diagnostic candidates")
            write_json(destination / f"{case_id}-{variant}-{name}.json", dict(bundle=bundle, association=proposal))
    path = destination / (case_id + "-rows.json")
    write_json(path, dict(rows=rows, elapsed_seconds=time.perf_counter() - started))
    return dict(path=path.name, sha256=digest(path))


def run(path, output):
    config = read_json(path)
    parent, _, frozen, method = checked(config["source_run_id"])
    original_path = ROOT / "data/evaluation" / config["source_run_id"] / "summary.json"
    original_sha = digest(original_path)
    destination = ROOT / "data/evaluation" / config["source_run_id"] / config["audit_id"]
    destination.mkdir(exist_ok=False)
    write_json(destination / "protocol.json", config)
    sources = {**frozen["source_sha256"], "scripts/audit_foreground_camera_transfer.py": digest(Path(__file__))}
    write_json(destination / "prepared.json", dict(scope=config["scope"], config_sha256=digest(path), source_sha256=sources,
        source_summary_sha256=original_sha, normal_inference_sha256=digest(parent / "inference.json")))
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=config["parallel_workers"]) as pool:
        pending = [pool.submit(audit_case, config, c["case_id"], destination) for c in method["cases"]]
        for item in as_completed(pending):
            record = item.result()
            records.append(record)
            print("CAMERA_COUNTERFACTUAL", record["path"], flush=True)
    rows = [{**r, "variant": "estimated"} for r in read_json(original_path)["rows"]]
    for record in sorted(records, key=lambda r: r["path"]):
        assert digest(destination / record["path"]) == record["sha256"]
        rows.extend(read_json(destination / record["path"])["rows"])
    expected = {(q["query_id"], name, variant) for q in method["annotations"]["queries"] for name in ("baseline", "cylinder_support") for variant in ["estimated", *config["variants"]]}
    assert {(r["query_id"], r["method"], r["variant"]) for r in rows} == expected and len(rows) == len(expected)
    for name, sha in sources.items():
        assert digest(ROOT / name) == sha
    assert digest(original_path) == original_sha
    report = dict(state="complete", source_run_id=config["source_run_id"], scope=config["scope"], rows=rows, source_sha256=sources,
                  config_sha256=digest(path), source_summary_sha256=original_sha, normal_inputs_unchanged=True,
                  elapsed_seconds=time.perf_counter() - started, records=records)
    checked(config["source_run_id"])
    write_json(destination / "summary.json", report)
    write_json(output, report)
    print("CAMERA_COUNTERFACTUAL_COMPLETE", len(rows), "query/method/camera conditions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-22-camera-counterfactual.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run(args.config, args.output)
