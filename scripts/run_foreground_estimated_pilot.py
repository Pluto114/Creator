"""Move frozen RGB observations/anchors onto DA3-estimated cameras and real bases."""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from run_rod_candidate_ablation import canonical_hash
from run_rod_foreground_identity import compact_bundle, make_views, query_truth
from run_rod_identity_blender import digest, read_json, write_json
from run_rod_identity_stress import reject_truth_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction/src"))
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from creator_eval.native_diagnostics import (  # noqa: E402
    AlignmentDegenerate,
    align_cameras,
    apply_similarity,
    unproject,
)
from creator_eval.rod_candidate_association import associate_multiview_lines_cached  # noqa: E402
from creator_eval.rod_cylinder_support import select_supported_cylinders  # noqa: E402
from creator_eval.rod_foreground_identity import (  # noqa: E402
    build_anchor_proposals,
    select_foreground_identity,
)
from creator_recon.domain.point_patch import (  # noqa: E402
    compose,
    open_candidate_view,
    write_candidate_view,
    write_patch,
    write_snapshot,
)
from run_rod_evidence import original_intrinsics  # noqa: E402

CONFIG = ROOT / "configs/foreground_estimated_pilot_v1.json"
RUN_ID = "foreground-estimated-pilot-v1-20260922"
EXTRA_SOURCES = ["scripts/run_foreground_estimated_pilot.py", "scripts/smoke_da3.py", "scripts/environment_paths.py",
                 "scripts/run_rod_evidence.py", "scripts/run_thin_line_controls.py",
                 "reconstruction/src/creator_recon/domain/point_patch.py", "configs/models.lock.json"]
FRAME_FIELDS = ("view_id", "rgb", "rgb_sha256", "size_wh", "guide_xyxy", "guide_source", "observations", "pool")


def require_camera_free(value):
    if isinstance(value, dict):
        if set(value) & {"K_index", "world_to_camera_cv", "camera_source", "intrinsics", "extrinsics", "ground_truth", "gt_axis"}:
            raise ValueError("Privileged camera/geometry field in RGB-only observation input")
        for item in value.values():
            require_camera_free(item)
    elif isinstance(value, list):
        for item in value:
            require_camera_free(item)


def locations(run_id):
    if not run_id or Path(run_id).name != run_id or ":" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("Single-component run ID required")
    return ROOT / ".runtime/experiments" / run_id, ROOT / "data/inputs" / run_id


def prepare(path):
    config = read_json(path)
    run, inputs = locations(config["run_id"])
    if run.exists() or inputs.exists():
        raise FileExistsError("Keep previous camera-transfer runs")
    parent, _ = locations(config["source_run_id"])
    meta, previous = read_json(parent / "prepared.json"), read_json(parent / "inference.json")
    for name, sha in meta["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Parent source changed")
    if previous["state"] != "inferred" or digest(parent / "method_config.json") != meta["method_config_sha256"]:
        raise ValueError("Incomplete parent geometry/pool record")
    annotations = read_json(ROOT / config["annotation_file"])
    if annotations["source_run_id"] != config["source_run_id"] or annotations["ground_truth_used_for_coordinates"] or annotations["predicted_geometry_used_for_coordinates"]:
        raise ValueError("Need existing RGB-only anchors")
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    cases = []
    for case_id in config["case_ids"]:
        entry = next(p for p in previous["pools"] if p["case_id"] == case_id and p["search_offset_px"] == config["search_offset_px"])
        if digest(parent / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen RGB pool changed")
        old = read_json(parent / entry["path"])
        frames = []
        for f in old["frames"]:
            # 明确的白名单：旧池里的真实相机、三维候选，一项都不带进新推理。
            frame = {key: copy.deepcopy(f[key]) for key in FRAME_FIELDS}
            require_camera_free(frame)
            source = ROOT / f["rgb"]
            if digest(source) != frame["rgb_sha256"]:
                raise ValueError("RGB changed")
            destination = inputs / case_id / (frame["view_id"] + ".png")
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
            frame["rgb"] = destination.relative_to(ROOT).as_posix()
            frames.append(frame)
        artifact = inputs / (case_id + ".json")
        write_json(artifact, dict(case_id=case_id, frames=frames, source_pool_sha256=entry["sha256"]))
        cases.append(dict(case_id=case_id, path=artifact.name, sha256=digest(artifact)))
    method = {key: config[key] for key in ("model", "process_res", "use_ray_pose", "candidate_cap", "identity_policy", "parallel_workers")}
    parent_method = read_json(parent / "method_config.json")
    method.update({key: parent_method[key] for key in ("association_common", "extent", "cylinder_screen")})
    method.update(cases=cases, annotations=annotations, camera_source="DA3_estimated_only", oracle_cameras_excluded=True)
    write_json(inputs / "manifest.json", method)
    write_json(run / "protocol.json", config)
    sources = {}
    for name in sorted(set(meta["source_sha256"]) | set(EXTRA_SOURCES)):
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        sources[name] = digest(ROOT / name)
    write_json(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=sources, input_sha256=digest(inputs / "manifest.json"),
        protocol_sha256=digest(run / "protocol.json"), parent_prepared_sha256=digest(parent / "prepared.json"),
        parent_inference_sha256=digest(parent / "inference.json"), evaluation_queries_sha256=digest(ROOT / config["evaluation_queries_file"])))
    print("PREPARED_ESTIMATED_PILOT", len(cases), "camera-free RGB observation pools", flush=True)


def checked(run_id):
    run, inputs = locations(run_id)
    frozen = read_json(run / "prepared.json")
    if frozen["run_id"] != run_id or digest(inputs / "manifest.json") != frozen["input_sha256"]:
        raise ValueError("Frozen identity/input mismatch")
    for name, sha in frozen["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source changed: " + name)
    return run, inputs, frozen, read_json(inputs / "manifest.json")


def checked_case(inputs, entry):
    if digest(inputs / entry["path"]) != entry["sha256"]:
        raise ValueError("Sanitized pool changed")
    case = read_json(inputs / entry["path"])
    if case["case_id"] != entry["case_id"] or any(set(f) != set(FRAME_FIELDS) for f in case["frames"]):
        raise ValueError("Unexpected field in camera-free RGB pool")
    require_camera_free(case)
    for frame in case["frames"]:
        if digest(ROOT / frame["rgb"]) != frame["rgb_sha256"]:
            raise ValueError("RGB input changed")
    return case


def predict(run_id):
    from smoke_da3 import run as model_run
    run, inputs, frozen, method = checked(run_id)
    sys.addaudithook(reject_truth_open)
    records = []
    for entry in method["cases"]:
        case = checked_case(inputs, entry)
        destination = run / entry["case_id"] / "prediction"
        destination.mkdir(parents=True, exist_ok=False)
        args = argparse.Namespace(model=method["model"], process_res=method["process_res"], use_ray_pose=method["use_ray_pose"],
                                  images=[ROOT / f["rgb"] for f in case["frames"]])
        try:
            report = model_run(args, ROOT, destination)
        except Exception as error:
            write_json(destination / "failure.json", dict(state="failed", error=repr(error)))
            raise
        report.update(state="succeeded", camera_source="estimated", gt_read_during_inference=False)
        write_json(destination / "report.json", report)
        records.append(dict(case_id=entry["case_id"], prediction_sha256=digest(destination / "prediction.npz"), report_sha256=digest(destination / "report.json")))
        print("ESTIMATED_PREDICTION", entry["case_id"], report["inference_seconds"], "seconds", flush=True)
    write_json(run / "predictions.json", dict(state="complete", input_sha256=frozen["input_sha256"], source_sha256=frozen["source_sha256"], records=records))


def infer_case(run_id, entry):
    run, inputs, _, method = checked(run_id)
    sys.addaudithook(reject_truth_open)
    case, started = checked_case(inputs, entry), time.perf_counter()
    folder = run / entry["case_id"]
    prediction = folder / "prediction/prediction.npz"
    index = read_json(run / "predictions.json")
    receipt = next(r for r in index["records"] if r["case_id"] == entry["case_id"])
    if digest(prediction) != receipt["prediction_sha256"] or digest(folder / "prediction/report.json") != receipt["report_sha256"]:
        raise ValueError("Estimated prediction changed")
    with np.load(prediction, allow_pickle=False) as native:
        depths, ks, cameras = native["depth"], native["intrinsics"], native["extrinsics"]
    height, width = depths.shape[1:]
    lifted = original_intrinsics(ks, case["frames"][0]["size_wh"], [width, height], False)
    frames = [{**f, "K_index": k, "world_to_camera_cv": e, "camera_source": "estimated"} for f, k, e in zip(case["frames"], lifted, cameras)]
    if len(frames) != len(case["frames"]) or len(cameras) != len(frames):
        raise ValueError("Estimated frame order/cardinality mismatch")
    report = read_json(folder / "prediction/report.json")
    if [f["rgb_sha256"] for f in frames] != [f["sha256"] for f in report["images"]]:
        raise ValueError("Prediction image order changed")
    points, ids, source_frames = [], [], []
    y, x = np.mgrid[:height, :width]
    for i, (depth, k, camera, frame) in enumerate(zip(depths, ks, cameras, frames)):
        valid = np.isfinite(depth) & (depth > 0)
        points.append(unproject(depth, k, camera)[valid])
        ids.append(np.c_[np.full(valid.sum(), i), y[valid], x[valid]].astype(np.uint32))
        source_frames.append(dict(frame_id=frame["view_id"], image_sha256=frame["rgb_sha256"], prediction_size_wh=[width, height]))
    snapshot = folder / "base"
    snapshot_id = write_snapshot(snapshot, np.concatenate(points), np.concatenate(ids), frames=source_frames,
        world_frame_id="da3_prediction:" + receipt["prediction_sha256"], length_unit="reconstruction_unit",
        source_prediction_sha256=receipt["prediction_sha256"], point_policy="all finite positive native depth pixels; native integer pixel centers; no confidence pruning")
    base = compose(snapshot)
    views, raw = make_views(frames, method["candidate_cap"]), [f["observations"] for f in frames]
    association = associate_multiview_lines_cached(views, method["association_common"])
    if not association["search_complete"]:
        raise ValueError("Estimated-camera association search incomplete")
    cylinder = select_supported_cylinders(association, views, raw, method["extent"], method["cylinder_screen"])
    records = []
    for name, proposal in (("baseline", association), ("cylinder_support", cylinder)):
        bundle = compact_bundle(build_anchor_proposals(proposal, views, raw, method["extent"], method["cylinder_screen"] if name == "cylinder_support" else None))
        bundle_sha = canonical_hash(bundle)
        for query in method["annotations"]["queries"]:
            if query["case_id"] != entry["case_id"]:
                continue
            result = select_foreground_identity(bundle, views, raw, query["anchors"], method["identity_policy"])
            records.append(dict(query_id=query["query_id"], method=name, identity=result, bundle_sha256=bundle_sha))
        if canonical_hash(bundle) != bundle_sha:
            raise ValueError("Identity mutated candidate geometry")
        write_json(folder / (name + "-bundle.json"), dict(bundle=bundle, association=proposal, bundle_sha256=bundle_sha))
    output = folder / "geometry.json"
    write_json(output, dict(case_id=entry["case_id"], prediction_sha256=receipt["prediction_sha256"], snapshot_id=snapshot_id,
                           frames=frames, records=records, elapsed_seconds=time.perf_counter() - started))
    for row in records:
        segments = np.asarray(row["identity"]["segments"], float).reshape(-1, 2, 3)
        stem = row["query_id"] + "-" + row["method"]
        evidence = [dict(decision="accept", line_ids=[f"line-{i:04d}" for i in range(len(segments))], suppression_range=None,
                         view_ids=[a["view_id"] for a in next(q for q in method["annotations"]["queries"] if q["query_id"] == row["query_id"])["anchors"]],
                         source_sha256=digest(output), note="Frozen foreground identity applied to DA3-estimated camera geometry")] if len(segments) else []
        patch = folder / ("patch-" + stem)
        write_patch(patch, snapshot, segments, np.empty((0, 3), np.uint32), selection_sha256=canonical_hash(method["annotations"]),
                    method=dict(id="foreground-" + row["method"], version="1-estimated-pilot", config_sha256=canonical_hash(method), seed=0),
                    evidence=evidence, unresolved=[] if len(segments) else [str(row["identity"]["reason"])])
        for enabled in (True, False):
            view = folder / (stem + ("-enabled.json" if enabled else "-withdrawn.json"))
            write_candidate_view(view, snapshot, patch, enabled=enabled)
            opened = open_candidate_view(view)
            assert opened["points"].tobytes() == base["points"].tobytes() and opened["point_ids"].tobytes() == base["point_ids"].tobytes()
            np.testing.assert_array_equal(opened["segments"], segments if enabled else np.empty((0, 2, 3)))
    receipt_path = folder / "result.json"
    write_json(receipt_path, dict(case_id=entry["case_id"], geometry_sha256=digest(output), point_count=len(base["points"]),
        query_method_count=len(records), accepted_count=sum(r["identity"]["state"] == "accepted" for r in records),
        rollback_bytes_equal=True, saved_views_reopened=True, elapsed_seconds=time.perf_counter() - started))
    return dict(case_id=entry["case_id"], path=receipt_path.relative_to(run).as_posix(), sha256=digest(receipt_path))


def infer(run_id):
    run, _, frozen, method = checked(run_id)
    index = read_json(run / "predictions.json")
    if index["state"] != "complete" or index["input_sha256"] != frozen["input_sha256"] or index["source_sha256"] != frozen["source_sha256"] or {r["case_id"] for r in index["records"]} != {c["case_id"] for c in method["cases"]}:
        raise ValueError("Incomplete/detached model predictions")
    sys.addaudithook(reject_truth_open)
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=method["parallel_workers"]) as pool:
        pending = [pool.submit(infer_case, run_id, case) for case in method["cases"]]
        for item in as_completed(pending):
            record = item.result()
            records.append(record)
            print("ESTIMATED_GEOMETRY", record["case_id"], flush=True)
    write_json(run / "inference.json", dict(state="complete", source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"],
        predictions_sha256=digest(run / "predictions.json"), records=sorted(records, key=lambda r: r["case_id"]),
        gt_read_during_inference=False, elapsed_seconds=time.perf_counter() - started))


def evaluate(run_id, output):
    run, _, frozen, method = checked(run_id)
    config = read_json(run / "protocol.json")
    if digest(run / "protocol.json") != frozen["protocol_sha256"] or digest(ROOT / config["evaluation_queries_file"]) != frozen["evaluation_queries_sha256"]:
        raise ValueError("Evaluation protocol changed")
    parent, parent_inputs = locations(config["source_run_id"])
    if digest(parent / "prepared.json") != frozen["parent_prepared_sha256"] or digest(parent / "inference.json") != frozen["parent_inference_sha256"]:
        raise ValueError("Parent changed")
    truth_root = ROOT / "data/eval_gt" / config["source_run_id"]
    parent_meta = read_json(parent / "prepared.json")
    for filename, key in (("manifest.json", "truth_manifest_sha256"), ("artifact_hashes.json", "truth_artifacts_sha256")):
        if digest(truth_root / filename) != parent_meta[key]:
            raise ValueError("GT index changed")
    for relative, sha in read_json(truth_root / "artifact_hashes.json").items():
        if digest(truth_root / relative) != sha:
            raise ValueError("GT artifact changed")
    if digest(parent_inputs / "manifest.json") != parent_meta["input_manifest_sha256"]:
        raise ValueError("Oracle comparison cameras changed")
    truth_cameras = {c["case_id"]: c["frames"] for c in read_json(parent_inputs / "manifest.json")["cases"]}
    targets = {c["case_id"]: read_json(truth_root / c["path"]) for c in read_json(truth_root / "manifest.json")["cases"]}
    queries = {q["query_id"]: q for q in read_json(ROOT / config["evaluation_queries_file"])["evaluation_queries"]}
    inference = read_json(run / "inference.json")
    if inference["state"] != "complete" or inference["source_sha256"] != frozen["source_sha256"] or inference["input_sha256"] != frozen["input_sha256"] or digest(run / "predictions.json") != inference["predictions_sha256"]:
        raise ValueError("Detached geometry inference")
    rows, cases = [], []
    for record in inference["records"]:
        folder = run / record["case_id"]
        if digest(run / record["path"]) != record["sha256"]:
            raise ValueError("Integration receipt changed")
        receipt = read_json(run / record["path"])
        if digest(folder / "geometry.json") != receipt["geometry_sha256"]:
            raise ValueError("Geometry changed")
        geometry = read_json(folder / "geometry.json")
        frames = geometry["frames"]
        try:
            scale, rotation, translation, alignment = align_cameras(np.array([f["world_to_camera_cv"] for f in frames]),
                np.array([f["world_to_camera_cv"] for f in truth_cameras[record["case_id"]]]), config["alignment"])
        except AlignmentDegenerate as error:
            alignment = dict(state="unmeasurable", reason=str(error))
        for item in geometry["records"]:
            query = queries[item["query_id"]]
            target = query_truth(query, targets[record["case_id"]], truth_root)
            truth = target["target"].get("segments", [target["target"].get("endpoints")]) if target["target"]["present"] else []
            segments = np.asarray(item["identity"]["segments"]).reshape(-1, 2, 3)
            measurable = alignment.get("state") != "unmeasurable"
            aligned = apply_similarity(segments, scale, rotation, translation) if measurable else None
            metrics = curve_metrics(aligned, truth, tolerance=config["curve_tolerance_m"], spacing=config["curve_spacing_m"]) if measurable else None
            row = dict(case_id=record["case_id"], query_id=item["query_id"], method=item["method"], role=query["role"],
                       state=item["identity"]["state"], reason=item["identity"]["reason"], segments=aligned,
                       native_segments=segments, metrics=metrics, target_present=target["target"]["present"],
                       evaluation_state="complete" if measurable else "unmeasurable")
            if measurable and target.get("gap_segment"):
                row["gap"] = gap_coverage(aligned, target["gap_segment"], tolerance=config["curve_tolerance_m"], spacing=config["curve_spacing_m"])
            rows.append(row)
        cases.append({**receipt, "alignment": alignment, "prediction": read_json(folder / "prediction/report.json")})
    expected = {(q["query_id"], name) for q in method["annotations"]["queries"] for name in ("baseline", "cylinder_support")}
    if {(r["query_id"], r["method"]) for r in rows} != expected or len(rows) != len(expected) or len(cases) != len(method["cases"]):
        raise ValueError("Missing or duplicate planned output")
    report = dict(state="complete", run_id=run_id, scope=config["scope"], source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"],
        protocol_sha256=frozen["protocol_sha256"], inference_sha256=digest(run / "inference.json"), cases=cases, rows=rows,
        geometry_seconds=inference["elapsed_seconds"], metric_scope="native added curve diagnostics; complete candidate common-readout efficacy remains unqualified")
    evaluation = ROOT / "data/evaluation" / run_id
    evaluation.mkdir(parents=True, exist_ok=False)
    write_json(evaluation / "summary.json", report)
    write_json(output, report)
    print("EVALUATED_ESTIMATED_PILOT", len(rows), "query/method pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-22-foreground-estimated-pilot.json")
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
