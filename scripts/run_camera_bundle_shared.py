"""New frozen calibration hypothesis; reuse predictions without touching pilot v1."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import run_camera_bundle_pilot as base
from run_camera_bundle_pilot import (
    ROOT,
    checked_run,
    correction_decision,
    corrupt_training,
    digest,
    load_case,
    locations,
    observations_from_record,
    read_json,
    reject_truth_open,
    validate_heldout,
    validation_plan,
    write_json,
)

# isort: split
from creator_eval.camera_bundle_shared import fit_shared, shared_square_intrinsics


def prepare(path):
    config = read_json(path)
    parent, parent_inputs, old, source = checked_run(config["parent_run_id"])
    run, inputs = locations(config["run_id"])
    if run.exists() or inputs.exists():
        raise FileExistsError("Preserve earlier camera attempts")
    index = read_json(parent / "predictions.json")
    if index["state"] != "complete" or index["source_sha256"] != old["source_sha256"]:
        raise ValueError("Parent predictions incomplete")
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    for case in source["cases"]:
        path = parent_inputs / case["tracks_path"]
        assert digest(path) == case["tracks_sha256"]
        shutil.copy2(path, inputs / case["tracks_path"])
        for frame in case["frames"]:
            assert digest(ROOT / frame["rgb"]) == frame["rgb_sha256"]
    for row in index["records"]:
        assert digest(ROOT / row["path"]) == row["prediction_sha256"]
        assert digest(ROOT / row["report"]) == row["report_sha256"]
    method = {key: config[key] for key in source["method"]}
    write_json(inputs / "manifest.json", dict(cases=source["cases"], method=method))
    write_json(run / "protocol.json", config)
    sources = {}
    for name in sorted(set(old["source_sha256"]) | {"scripts/run_camera_bundle_shared.py", "experiments/src/creator_eval/camera_bundle_shared.py"}):
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        sources[name] = digest(target)
    write_json(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=sources, parents=old["parents"],
        parent_prepared_sha256=digest(parent / "prepared.json"), parent_predictions_sha256=digest(parent / "predictions.json"),
        input_sha256=digest(inputs / "manifest.json"), protocol_sha256=digest(run / "protocol.json")))
    write_json(run / "predictions.json", dict(state="complete", source_sha256=sources, input_sha256=digest(inputs / "manifest.json"),
        records=[{**r, "reused": True} for r in index["records"]]))
    print("PREPARED_SHARED_CAMERA_BUNDLE", len(source["cases"]), "unchanged image/initial-camera cases", flush=True)


def infer_variant(run_id, case_id, condition, variant):
    sys.addaudithook(reject_truth_open)
    run, _, tracks, method, k, e = load_case(run_id, case_id)
    if tracks["availability"] != "eligible_for_further_validation":
        raise ValueError("No eligible RGB evidence")
    training = observations_from_record(tracks, "train")
    validation = observations_from_record(tracks, "validation")
    training, changes = corrupt_training(training, condition["corrupt_fraction"], method["seed"])
    plan = validation_plan(validation, e)
    before = validate_heldout(k, e, plan)
    common_before = validate_heldout(shared_square_intrinsics(k), e, plan)
    started = time.perf_counter()
    try:
        result = fit_shared(k, e, training, method["optimizer"], variant)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result = dict(state="numerical_failure", success=False, error=repr(error))
    after = validate_heldout(result["intrinsics"], result["extrinsics"], plan) if "extrinsics" in result else before
    decision = correction_decision(result, before, after, method["decision"])
    path = run / case_id / (condition["id"] + "-" + variant["id"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(case_id=case_id, condition=condition, variant=variant, initial_intrinsics=k, initial_extrinsics=e,
        result=result, before=before, common_intrinsics_only_before=common_before, after=after, decision=decision, corruption=changes,
        validation_plan=plan, training_count=len(training), validation_count=len(validation), elapsed_seconds=time.perf_counter() - started,
        gt_read_during_inference=False, dense_base_changed=False,
        acquisition_assumption="same fixed lens and square pixels" if variant["common_intrinsics"] else "original per-view model K times one scalar"))
    return dict(case_id=case_id, condition=condition["id"], variant=variant["id"], path=path.relative_to(run).as_posix(), sha256=digest(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/camera_bundle_shared_v1.json")
    parser.add_argument("--run-id", default="camera-bundle-shared-v1-20260923")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-23-camera-bundle-shared.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        base.infer_variant = infer_variant
        base.infer(args.run_id)
    else:
        base.evaluate(args.run_id, args.output)
