"""DA3-initialized camera adjustment: frozen RGB inputs, then separate GT evaluation."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from run_foreground_estimated_pilot import (
    ROOT,
    checked,
    digest,
    locations,
    original_intrinsics,
    read_json,
    write_json,
)
from run_foreground_tracks import checked_tracks
from run_rod_identity_stress import reject_truth_open

# isort: split
from creator_eval.camera_bundle import (
    correction_decision,
    corrupt_training,
    observations_from_record,
    optimize_cameras,
    validate_heldout,
    validation_plan,
)
from creator_eval.native_diagnostics import align_cameras

CONFIG = ROOT / "configs/camera_bundle_pilot_v1.json"
EXTRA = ["scripts/run_camera_bundle_pilot.py", "experiments/src/creator_eval/camera_bundle.py",
         "scripts/smoke_da3.py", "scripts/environment_paths.py"]


def prepare(path):
    config = read_json(path)
    run, inputs = locations(config["run_id"])
    if run.exists() or inputs.exists():
        raise FileExistsError("Keep prior bundle experiments")
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    cases, source_names, parents = [], set(EXTRA), {}
    reuse, _, reuse_frozen, _ = checked(config["reuse_prediction_run_id"])
    reuse_index = read_json(reuse / "predictions.json")
    if reuse_index["state"] != "complete" or reuse_index["source_sha256"] != reuse_frozen["source_sha256"]:
        raise ValueError("Incomplete existing estimates")
    for source_id in config["sources"]:
        source, frozen, source_inputs = checked_tracks(source_id)
        index = read_json(source / "inference.json")
        if index["state"] != "complete" or index["input_sha256"] != frozen["input_sha256"] or index["source_sha256"] != frozen["source_sha256"]:
            raise ValueError("Detached track inference")
        parents[source_id] = dict(prepared_sha256=digest(source / "prepared.json"), inference_sha256=digest(source / "inference.json"))
        source_names.update(frozen["source_sha256"])
        for case in source_inputs["cases"]:
            cid = case["case_id"]
            if any(c["case_id"] == cid for c in cases):
                raise ValueError("Case IDs must be unique across sources")
            entry = next(r for r in index["records"] if r["case_id"] == cid and r["detector"] == config["detector"])
            if digest(source / entry["path"]) != entry["sha256"]:
                raise ValueError("Frozen track file changed")
            raw = read_json(source / entry["path"])
            selected = {key: raw[key] for key in ("case_id", "feature_xy", "graph", "band_labels", "availability", "coverage", "split", "nonplanar_validated_pairs")}
            record_path = inputs / (cid + "-tracks.json")
            write_json(record_path, selected)
            frames = []
            for frame in case["frames"]:
                original = ROOT / frame["rgb"]
                if digest(original) != frame["rgb_sha256"]:
                    raise ValueError("RGB changed")
                target = inputs / cid / (frame["view_id"] + ".png")
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(original, target)
                frames.append(dict(view_id=frame["view_id"], rgb=target.relative_to(ROOT).as_posix(), rgb_sha256=frame["rgb_sha256"], size_wh=frame["size_wh"]))
            previous = next((r for r in reuse_index["records"] if r["case_id"] == cid), None)
            reused = None
            if previous is not None:
                prediction = reuse / cid / "prediction/prediction.npz"
                report = reuse / cid / "prediction/report.json"
                if digest(prediction) != previous["prediction_sha256"] or digest(report) != previous["report_sha256"]:
                    raise ValueError("Existing prediction changed")
                if [f["rgb_sha256"] for f in frames] != [f["sha256"] for f in read_json(report)["images"]]:
                    raise ValueError("Cannot reuse a prediction of other images")
                reused = dict(path=prediction.relative_to(ROOT).as_posix(), report=report.relative_to(ROOT).as_posix(),
                              prediction_sha256=previous["prediction_sha256"], report_sha256=previous["report_sha256"])
            cases.append(dict(case_id=cid, source_run_id=source_id, frames=frames, tracks_path=record_path.name,
                              tracks_sha256=digest(record_path), reuse=reused))
    method = {k: config[k] for k in ("model", "process_res", "use_ray_pose", "conditions", "variants", "optimizer", "decision", "seed", "parallel_workers")}
    write_json(inputs / "manifest.json", dict(cases=cases, method=method))
    write_json(run / "protocol.json", config)
    sources = {}
    for name in sorted(source_names):
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        sources[name] = digest(target)
    write_json(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=sources, parents=parents,
        input_sha256=digest(inputs / "manifest.json"), protocol_sha256=digest(run / "protocol.json")))
    print("PREPARED_CAMERA_BUNDLE", len(cases), "cases", flush=True)


def checked_run(run_id):
    run, inputs = locations(run_id)
    frozen = read_json(run / "prepared.json")
    if frozen["run_id"] != run_id or digest(inputs / "manifest.json") != frozen["input_sha256"]:
        raise ValueError("Bundle input identity changed")
    for name, sha in frozen["source_sha256"].items():
        if digest(ROOT / name) != sha or digest(run / "source_snapshot" / name) != sha:
            raise ValueError("Frozen bundle source changed: " + name)
    return run, inputs, frozen, read_json(inputs / "manifest.json")


def predict(run_id):
    from smoke_da3 import run as model_run
    run, _, frozen, inputs = checked_run(run_id)
    method = inputs["method"]
    sys.addaudithook(reject_truth_open)
    rows = []
    for case in inputs["cases"]:
        for frame in case["frames"]:
            if digest(ROOT / frame["rgb"]) != frame["rgb_sha256"]:
                raise ValueError("RGB changed")
        if case["reuse"]:
            receipt = case["reuse"]
            prediction, report = ROOT / receipt["path"], ROOT / receipt["report"]
            if digest(prediction) != receipt["prediction_sha256"] or digest(report) != receipt["report_sha256"]:
                raise ValueError("Reusable prediction changed")
        else:
            folder = run / case["case_id"] / "prediction"
            folder.mkdir(parents=True, exist_ok=False)
            args = argparse.Namespace(model=method["model"], process_res=method["process_res"], use_ray_pose=method["use_ray_pose"], images=[ROOT / f["rgb"] for f in case["frames"]])
            try:
                result = model_run(args, ROOT, folder)
                write_json(folder / "report.json", result)
            except Exception as error:
                write_json(folder / "failure.json", dict(error=repr(error)))
                raise
            prediction, report = folder / "prediction.npz", folder / "report.json"
        rows.append(dict(case_id=case["case_id"], path=prediction.relative_to(ROOT).as_posix(), report=report.relative_to(ROOT).as_posix(),
                         prediction_sha256=digest(prediction), report_sha256=digest(report), reused=bool(case["reuse"])))
        print("BUNDLE_INITIAL_PREDICTION", case["case_id"], flush=True)
    write_json(run / "predictions.json", dict(state="complete", source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"], records=rows))


def load_case(run_id, case_id):
    run, inputs, frozen, manifest = checked_run(run_id)
    case = next(c for c in manifest["cases"] if c["case_id"] == case_id)
    if digest(inputs / case["tracks_path"]) != case["tracks_sha256"]:
        raise ValueError("Track input changed")
    record = read_json(inputs / case["tracks_path"])
    index = read_json(run / "predictions.json")
    if index["state"] != "complete" or index["source_sha256"] != frozen["source_sha256"] or index["input_sha256"] != frozen["input_sha256"]:
        raise ValueError("Prediction index detached")
    prediction = next(r for r in index["records"] if r["case_id"] == case_id)
    if digest(ROOT / prediction["path"]) != prediction["prediction_sha256"] or digest(ROOT / prediction["report"]) != prediction["report_sha256"]:
        raise ValueError("Prediction artifact changed")
    report = read_json(ROOT / prediction["report"])
    if [r["sha256"] for r in report["images"]] != [f["rgb_sha256"] for f in case["frames"]]:
        raise ValueError("Prediction view order changed")
    with np.load(ROOT / prediction["path"], allow_pickle=False) as native:
        _, height, width = native["depth"].shape
        k = original_intrinsics(native["intrinsics"], case["frames"][0]["size_wh"], [width, height], False)
        e = native["extrinsics"].astype(float)
    return run, case, record, manifest["method"], k, e


def infer_variant(run_id, case_id, condition, variant):
    sys.addaudithook(reject_truth_open)
    run, case, tracks, method, k, e = load_case(run_id, case_id)
    if tracks["availability"] != "eligible_for_further_validation":
        raise ValueError("Camera optimizer called without eligible RGB evidence")
    training = observations_from_record(tracks, "train")
    validation = observations_from_record(tracks, "validation")
    training, changes = corrupt_training(training, condition["corrupt_fraction"], method["seed"])
    plan = validation_plan(validation, e)
    before = validate_heldout(k, e, plan)
    started = time.perf_counter()
    try:
        result = optimize_cameras(k, e, training, method["optimizer"], focal=variant["focal"], loss=variant["loss"])
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        # A broken numerical attempt is still an experimental outcome. Preserve
        # it beside successful candidates rather than quietly dropping the row.
        result = dict(state="numerical_failure", success=False, error=repr(error))
    after = validate_heldout(result["intrinsics"], result["extrinsics"], plan) if "extrinsics" in result else before
    decision = correction_decision(result, before, after, method["decision"])
    path = run / case_id / (condition["id"] + "-" + variant["id"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(case_id=case_id, condition=condition, variant=variant, initial_intrinsics=k, initial_extrinsics=e,
        result=result, before=before, after=after, decision=decision, corruption=changes, validation_plan=plan,
        training_count=len(training), validation_count=len(validation), elapsed_seconds=time.perf_counter() - started,
        gt_read_during_inference=False, dense_base_changed=False))
    return dict(case_id=case_id, condition=condition["id"], variant=variant["id"], path=path.relative_to(run).as_posix(), sha256=digest(path))


def infer(run_id):
    run, inputs, frozen, manifest = checked_run(run_id)
    sys.addaudithook(reject_truth_open)
    tasks, skipped, records = [], [], []
    for case in manifest["cases"]:
        _, _, tracks, method, _, _ = load_case(run_id, case["case_id"])
        if tracks["availability"] != "eligible_for_further_validation":
            skipped.append(dict(case_id=case["case_id"], reason=tracks["availability"], coverage=tracks["coverage"]))
        else:
            tasks.extend((case["case_id"], condition, variant) for condition in method["conditions"] for variant in method["variants"])
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=manifest["method"]["parallel_workers"]) as pool:
        pending = [pool.submit(infer_variant, run_id, *task) for task in tasks]
        for item in as_completed(pending):
            receipt = item.result()
            records.append(receipt)
            print("CAMERA_BUNDLE", receipt["path"], flush=True)
    if len(records) != len(tasks):
        raise ValueError("Incomplete optimizer plan")
    checked_run(run_id)
    write_json(run / "inference.json", dict(state="complete", source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"],
        predictions_sha256=digest(run / "predictions.json"), records=sorted(records, key=lambda r: r["path"]), skipped=skipped,
        gt_read_during_inference=False, elapsed_seconds=time.perf_counter() - started))


def evaluate(run_id, output):
    run, _, frozen, inputs = checked_run(run_id)
    if digest(run / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Protocol changed")
    protocol = read_json(run / "protocol.json")
    inference = read_json(run / "inference.json")
    if inference["state"] != "complete" or inference["source_sha256"] != frozen["source_sha256"] or inference["input_sha256"] != frozen["input_sha256"] or digest(run / "predictions.json") != inference["predictions_sha256"]:
        raise ValueError("Incomplete/detached camera inference")
    skipped_ids = {c["case_id"] for c in inference["skipped"]}
    expected = {(c["case_id"], condition["id"], variant["id"]) for c in inputs["cases"] if c["case_id"] not in skipped_ids
                for condition in inputs["method"]["conditions"] for variant in inputs["method"]["variants"]}
    actual = [(r["case_id"], r["condition"], r["variant"]) for r in inference["records"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing or duplicate optimizer condition")
    truth, consumed = {}, {}
    for source_id in protocol["sources"]:
        source, meta, source_inputs = checked_tracks(source_id)
        parent = frozen["parents"][source_id]
        if digest(source / "prepared.json") != parent["prepared_sha256"] or digest(source / "inference.json") != parent["inference_sha256"]:
            raise ValueError("Frozen parent changed")
        if "truth_sha256" in meta:
            truth_path = ROOT / "data/eval_gt" / source_id / "manifest.json"
            if digest(truth_path) != meta["truth_sha256"]:
                raise ValueError("Analytic truth changed")
            values = read_json(truth_path)
            assert values["input_sha256"] == meta["input_sha256"]
            truth.update({c["case_id"]: c["cameras"] for c in values["cases"]})
            consumed[truth_path.relative_to(ROOT).as_posix()] = digest(truth_path)
        else:
            # Original rod cameras stay evaluation-only; all ordinary geometry
            # runs above read sanitized RGB and actual model estimates instead.
            legacy, _, legacy_meta, _ = checked(protocol["reuse_prediction_run_id"])
            gt_id = read_json(legacy / "protocol.json")["source_run_id"]
            pack = ROOT / ".runtime/experiments" / gt_id
            assert digest(pack / "prepared.json") == legacy_meta["parent_prepared_sha256"]
            pack_meta = read_json(pack / "prepared.json")
            gt_root = ROOT / "data/eval_gt" / gt_id
            assert digest(gt_root / "artifact_hashes.json") == pack_meta["truth_artifacts_sha256"]
            hashes = read_json(gt_root / "artifact_hashes.json")
            for case in source_inputs["cases"]:
                cameras = []
                for frame in case["frames"]:
                    path = gt_root / case["case_id"] / (frame["view_id"] + "-camera.json")
                    assert digest(path) == hashes[path.relative_to(gt_root).as_posix()]
                    cameras.append(read_json(path))
                    consumed[path.relative_to(ROOT).as_posix()] = digest(path)
                truth[case["case_id"]] = cameras
    initial = {}
    for case in inputs["cases"]:
        _, _, _, _, k, e = load_case(run_id, case["case_id"])
        initial[case["case_id"]] = camera_score(k, e, truth[case["case_id"]], protocol["evaluation"]["alignment"])
    rows = []
    for entry in inference["records"]:
        if digest(run / entry["path"]) != entry["sha256"]:
            raise ValueError("Optimizer output changed")
        record = read_json(run / entry["path"])
        result = record["result"]
        score = camera_score(np.array(result["intrinsics"]), np.array(result["extrinsics"]), truth[entry["case_id"]], protocol["evaluation"]["alignment"]) if "extrinsics" in result else None
        rows.append({**entry, "decision": record["decision"], "solver": {k: v for k, v in result.items() if k not in ("intrinsics", "extrinsics", "points", "kept_track_ids")},
            "before": record["before"]["summary"], "after": record["after"]["summary"], "initial_camera": initial[entry["case_id"]], "candidate_camera": score,
            "corrupted_tracks": len(record["corruption"]), "training_count": record["training_count"], "validation_count": record["validation_count"], "elapsed_seconds": record["elapsed_seconds"]})
    report = dict(state="complete", run_id=run_id, scope=protocol["scope"], rows=rows, skipped=inference["skipped"], initial_cameras=initial,
        source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"], protocol_sha256=frozen["protocol_sha256"],
        inference_sha256=digest(run / "inference.json"), consumed_truth_sha256=consumed, inference_seconds=inference["elapsed_seconds"],
        limits=protocol["limits"], dense_bases_and_rod_patches_unchanged=True)
    destination = ROOT / "data/evaluation" / run_id
    destination.mkdir(exist_ok=False)
    write_json(destination / "summary.json", report)
    write_json(output, report)
    print("EVALUATED_CAMERA_BUNDLE", len(rows), "variants;", len(inference["skipped"]), "ineligible cases", flush=True)


def camera_score(k, e, truth, alignment):
    gt_e = np.asarray([c["world_to_camera_cv"] for c in truth])[:, :3]
    _, _, _, score = align_cameras(e, gt_e, alignment)
    gt_k = np.array([c["K_index"] for c in truth])
    score["focal_relative_errors"] = (k[:, [0, 1], [0, 1]] / gt_k[:, [0, 1], [0, 1]] - 1).tolist()
    score["median_orientation_error_deg"] = float(np.median(score["orientation_errors_degrees"]))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default="camera-bundle-pilot-v1-20260923")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-23-camera-bundle-pilot.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "predict":
        predict(args.run_id)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.run_id, args.output)
