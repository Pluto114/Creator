"""Freeze and audit all-view held-out camera residuals, without changing old gates."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from run_camera_bundle_pilot import (
    ROOT,
    checked_run,
    digest,
    load_case,
    locations,
    read_json,
    write_json,
)
from run_rod_candidate_ablation import canonical_hash
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.camera_all_view_validation import all_view_plan, score_all_views
from creator_eval.camera_bundle import observations_from_record, validate_heldout, validation_plan

RUN_ID = "camera-all-view-validation-v1-20260923r2"
PARENTS = ("rod-reference-bundle-v1-20260923", "rod-height-bundle-v1-20260923")
OUTPUT = ROOT / "docs/experiments/results/2026-09-23-camera-all-view-validation-r2.json"
SOURCES = ("scripts/audit_camera_all_views.py", "experiments/src/creator_eval/camera_all_view_validation.py")


def run(run_id, output):
    sys.addaudithook(reject_truth_open)
    run_root, inputs = locations(run_id)
    if run_root.exists() or inputs.exists() or output.exists():
        raise FileExistsError("Keep previous diagnostics; choose a new run and output")
    cases, parent_receipts, sources = [], {}, set(SOURCES)
    # 全部父案例都收进来，包括旧变焦反例。别只挑看起来有希望的三根。
    for parent_id in PARENTS:
        parent, _, frozen, manifest = checked_run(parent_id)
        inference = read_json(parent / "inference.json")
        if inference["state"] != "complete" or inference["input_sha256"] != frozen["input_sha256"] or inference["source_sha256"] != frozen["source_sha256"]:
            raise ValueError("Detached parent camera inference")
        parent_receipts[parent_id] = {name: digest(parent / name) for name in ("prepared.json", "inference.json", "predictions.json")}
        sources.update(frozen["source_sha256"])
        for case in manifest["cases"]:
            cid = case["case_id"]
            _, _, tracks, _, initial_k, initial_e = load_case(parent_id, cid)
            train = observations_from_record(tracks, "train")
            validation = observations_from_record(tracks, "validation")
            plan = all_view_plan(validation, initial_e, training_track_ids=[t["track_id"] for t in train])
            original = validation_plan(validation, initial_e)
            cameras = [dict(variant="initial", intrinsics=initial_k.tolist(), extrinsics=initial_e.tolist(), decision=None)]
            for receipt in inference["records"]:
                if receipt["case_id"] != cid:
                    continue
                path = parent / receipt["path"]
                if digest(path) != receipt["sha256"]:
                    raise ValueError("Parent camera receipt changed")
                record = read_json(path)
                if record["validation_plan"] != original:
                    raise ValueError("Original held-out observations changed")
                numerical = "extrinsics" in record["result"]
                cameras.append(dict(variant=receipt["variant"], receipt=receipt, decision=record["decision"],
                                    intrinsics=record["result"].get("intrinsics"), extrinsics=record["result"].get("extrinsics"),
                                    original_summary=record["after"]["summary"], numerical=numerical))
            cases.append(dict(parent_run_id=parent_id, case_id=cid, track_sha256=case["tracks_sha256"], plan=plan,
                              plan_sha256=canonical_hash(plan), original_plan=original, cameras=cameras,
                              view_ids=[f["view_id"] for f in case["frames"]]))
    inputs.mkdir(parents=True)
    run_root.mkdir(parents=True)
    write_json(inputs / "plans-and-cameras.json", dict(cases=cases, parents=parent_receipts))
    source_sha = {}
    for name in sorted(sources):
        target = run_root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        source_sha[name] = digest(target)
    prepared = dict(run_id=run_id, input_sha256=digest(inputs / "plans-and-cameras.json"), source_sha256=source_sha,
                    parents=parent_receipts, gt_read_during_inference=False,
                    selection_policy="All parent cases and numerical cameras, including withheld candidates; no score-based filtering")
    write_json(run_root / "prepared.json", prepared)
    frozen = read_json(inputs / "plans-and-cameras.json")
    if digest(inputs / "plans-and-cameras.json") != prepared["input_sha256"]:
        raise ValueError("New plan changed before scoring")
    rows = []
    for case in frozen["cases"]:
        plan = case["plan"]
        for camera in case["cameras"]:
            common = dict(parent_run_id=case["parent_run_id"], case_id=case["case_id"], variant=camera["variant"],
                          plan_sha256=case["plan_sha256"], camera_decision=camera["decision"])
            if camera["extrinsics"] is None:
                rows.append({**common, "state": "no_numerical_camera"})
                continue
            k, e = np.asarray(camera["intrinsics"]), np.asarray(camera["extrinsics"])
            old = validate_heldout(k, e, case["original_plan"])
            if camera.get("original_summary") is not None and old["summary"] != camera["original_summary"]:
                raise ValueError("Original primary scores changed")
            scored = score_all_views(k, e, plan)
            if canonical_hash(plan) != case["plan_sha256"]:
                raise ValueError("Scoring mutated the frozen plan")
            full = {**common, "state": "complete", "diagnostic": scored, "original": old}
            filename = case["parent_run_id"] + "-" + case["case_id"] + "-" + camera["variant"] + ".json"
            write_json(run_root / filename, full)
            original_counts = [sum(row["view"] == v for row in old["rows"]) for v in range(len(k))]
            rows.append({**common, "state": "complete", "summary": scored["summary"], "per_view": scored["per_view"],
                         "original_summary": old["summary"], "original_scoring_counts": original_counts,
                         "view_ids": case["view_ids"], "path": (run_root / filename).relative_to(ROOT).as_posix(), "sha256": digest(run_root / filename)})
            print("ALL_VIEW_DIAGNOSTIC", case["parent_run_id"], case["case_id"], camera["variant"], flush=True)
    for name, sha in source_sha.items():
        if digest(ROOT / name) != sha or digest(run_root / "source_snapshot" / name) != sha:
            raise ValueError("Diagnostic source changed during scoring: " + name)
    for parent_id, receipts in parent_receipts.items():
        parent, _ = locations(parent_id)
        for name, sha in receipts.items():
            if digest(parent / name) != sha:
                raise ValueError("Parent changed during diagnostic")
    result = dict(state="complete", run_id=run_id, prepared_sha256=digest(run_root / "prepared.json"),
                  case_count=len(cases), camera_condition_count=len(rows),
                  plan_sha256={c["parent_run_id"] + "/" + c["case_id"]: c["plan_sha256"] for c in cases}, rows=rows,
                  gt_read_during_inference=False, old_inputs_and_results_unchanged=True,
                  scope="Post-hoc RGB-only coverage audit, not a preregistered new independent test or replacement gate. Reuses validation tracks; within-track rows correlated.")
    write_json(run_root / "summary.json", result)
    write_json(output, result)
    print("ALL_VIEW_COMPLETE", len(cases), len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run(args.run_id, args.output)
