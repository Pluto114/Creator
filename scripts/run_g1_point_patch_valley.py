"""Complete historical pair replay with the separately frozen persistent-valley reader."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction/src"))
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_valley import readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from creator_eval.native_diagnostics import (  # noqa: E402
    align_cameras,
    apply_similarity,
    camera_centers,
    unproject,
)
from creator_recon.domain.point_patch import (  # noqa: E402
    compose,
    content_hash,
    file_hash,
    load_bundle,
    open_candidate_view,
    write_candidate_view,
    write_patch,
    write_snapshot,
)
from run_rod_evidence import original_intrinsics  # noqa: E402
from run_rod_identity_blender import clean, read_json, write_json  # noqa: E402
from run_rod_identity_stress import reject_truth_open  # noqa: E402

CONFIG = ROOT / "configs/g1_point_patch_valley_v1.json"
RUN_ID = "g1-point-patch-valley-v1-20260924"
SOURCES = ["scripts/run_g1_point_patch_valley.py", "experiments/src/creator_eval/common_readout_components.py", "reconstruction/src/creator_recon/domain/point_patch.py", "experiments/src/creator_eval/common_readout.py",
           "experiments/src/creator_eval/native_diagnostics.py", "experiments/src/creator_eval/line_controls.py",
           "scripts/run_g1_point_patch.py", "scripts/run_rod_evidence.py", "scripts/run_rod_identity_stress.py",
           "scripts/run_rod_identity_blender.py", "experiments/src/creator_eval/common_readout_split.py",
           "experiments/src/creator_eval/common_readout_sections.py", "experiments/src/creator_eval/common_readout_valley.py"]


def locations(run_id):
    if not run_id or Path(run_id).name != run_id or ":" in run_id or "\\" in run_id:
        raise ValueError("Run ID must be one directory component")
    return ROOT / ".runtime/experiments" / run_id, ROOT / "data/inputs" / run_id


def checked(run_id):
    run, inputs = locations(run_id)
    frozen = read_json(run / "prepared.json")
    if frozen["run_id"] != run_id or file_hash(inputs / "manifest.json") != frozen["input_sha256"]:
        raise ValueError("Frozen run/input identity changed")
    for name, sha in frozen["source_sha256"].items():
        saved = run / "source_snapshot" / name
        if file_hash(saved) != sha or hashlib.sha256(saved.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != frozen["source_lf_sha256"][name]:
            raise ValueError("Frozen source snapshot changed: " + name)
        if file_hash(ROOT / name) not in (sha, frozen["source_lf_sha256"][name]):
            raise ValueError("Frozen source changed beyond declared CRLF mapping: " + name)
    return run, read_json(inputs / "manifest.json"), frozen


def prepare(path):
    config = read_json(path)
    for name in ("scripts/run_g1_point_patch_valley.py", "experiments/src/creator_eval/common_readout_valley.py", "configs/g1_point_patch_valley_v1.json"):
        raw = (ROOT / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n"), name
        command = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "hash-object"]
        assert subprocess.check_output(command + ["--no-filters", name], cwd=ROOT) == subprocess.check_output(command + ["--path=" + name, name], cwd=ROOT), name
    run, inputs = locations(config["run_id"])
    if run.exists() or inputs.exists():
        raise FileExistsError("Keep old point/patch experiments")
    parent = ROOT / ".runtime/experiments" / config["source_run_id"]
    legacy = read_json(parent / "manifest.json")
    selection = read_json(parent / "selection.json")
    if legacy["state"] != "complete" or file_hash(parent / "selection.json") != legacy["selection_sha256"]:
        raise ValueError("Historical fit/selection incomplete or changed")
    jobs = []
    for job in legacy["jobs"]:
        prediction = Path(job["prediction_path"])
        candidate = parent / job["candidate_job_id"] / "candidates.json"
        report = prediction.parent / "report.json"
        if file_hash(prediction) != job["prediction_sha256"] or file_hash(candidate) != job["candidates_sha256"]:
            raise ValueError("Historical prediction/candidate hash mismatch")
        prediction.relative_to(ROOT)
        receipt = read_json(report)
        for image in receipt["images"]:
            if file_hash(image["path"]) != image["sha256"]:
                raise ValueError("Original RGB changed")
        jobs.append({**job, "prediction_path": prediction.relative_to(ROOT).as_posix(), "candidate_path": candidate.relative_to(ROOT).as_posix(),
                     "report_path": report.relative_to(ROOT).as_posix(), "report_sha256": file_hash(report)})
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    method = {k: config[k] for k in ("variant", "voxel_camera_span_fractions", "region_minimum_views", "parallel_workers", "readout")}
    method.update(jobs=jobs, selection=selection["line_control"], selection_sha256=legacy["selection_sha256"],
                  source_manifest_sha256=file_hash(parent / "manifest.json"), source_run_id=config["source_run_id"], truth_excluded=True)
    write_json(inputs / "manifest.json", method)
    shutil.copy2(path, run / "protocol.json")
    hashes, lf_hashes = {}, {}
    for name in SOURCES:
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = file_hash(ROOT / name)
        lf_hashes[name] = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    write_json(run / "prepared.json", dict(run_id=config["run_id"], state="prepared", source_sha256=hashes, source_lf_sha256=lf_hashes,
               source_byte_policy="new sources LF; legacy CRLF preserved with explicit mapping", new_git_filter_bytes_unchanged=True,
               input_sha256=file_hash(inputs / "manifest.json"), protocol_sha256=file_hash(path), gt_read_during_prepare=False))
    print("PREPARED_POINT_PATCH", len(jobs), "historical jobs", flush=True)


def infer_job(run_id, job):
    run, method, _ = checked(run_id)
    sys.addaudithook(reject_truth_open)
    started = time.perf_counter()
    for path_key, hash_key in (("prediction_path", "prediction_sha256"), ("candidate_path", "candidates_sha256"), ("report_path", "report_sha256")):
        if file_hash(ROOT / job[path_key]) != job[hash_key]:
            raise ValueError("Historical input changed")
    candidates = read_json(ROOT / job["candidate_path"])
    if candidates["base_prediction_sha256"] != job["prediction_sha256"]:
        raise ValueError("Candidate is detached from model coordinate frame")
    report = read_json(ROOT / job["report_path"])
    with np.load(ROOT / job["prediction_path"], allow_pickle=False) as native:
        depths, cameras = native["depth"].copy(), native["extrinsics"].copy()
        ks = native["intrinsics"].astype(float)
    oracle = job["camera_mode"] == "oracle_camera"
    height, width = depths.shape[1:]
    target_views = next(iter(candidates["targets"].values()))["views"]
    lifted = original_intrinsics(ks, target_views[0]["size_wh"], [width, height], oracle)
    if len(report["images"]) != len(cameras) or len(target_views) != len(cameras):
        raise ValueError("Frame cardinality changed")
    frames, roi_views, points, ids = [], [], [], []
    index_k = ks.copy()
    if oracle:
        index_k[:, :2, 2] -= .5
    y, x = np.mgrid[:height, :width]
    for i, (depth, k, camera, view, image) in enumerate(zip(depths, index_k, cameras, target_views, report["images"])):
        for target in candidates["targets"].values():
            v = target["views"][i]
            if v["view_id"] != view["view_id"] or not np.allclose(v["K_index"], lifted[i], atol=1e-8, rtol=0) or not np.array_equal(v["world_to_camera_cv"], camera):
                raise ValueError("Stored finite geometry does not use this exact model camera")
        valid = np.isfinite(depth) & (depth > 0)
        points.append(unproject(depth, k, camera)[valid])
        ids.append(np.c_[np.full(valid.sum(), i), y[valid], x[valid]].astype(np.uint32))
        frames.append(dict(frame_id=view["view_id"], image_sha256=image["sha256"], prediction_size_wh=[width, height]))
        guide_y = method["selection"]["guide_y"]
        guides = [dict(xyxy=[[ends[0], guide_y[0]], [ends[1], guide_y[1]]], half_width_px=method["selection"]["half_width"])
                  for ends in method["selection"]["guides"][view["view_id"]].values()]
        roi_views.append(dict(view_id=view["view_id"], K_index=lifted[i], world_to_camera_cv=camera, guides=guides))
    destination = run / job["candidate_job_id"]
    destination.mkdir()
    snapshot_dir, patch_dir = destination / "base", destination / "patch"
    snapshot_id = write_snapshot(snapshot_dir, np.concatenate(points), np.concatenate(ids), frames=frames,
        world_frame_id="da3_prediction:" + job["prediction_sha256"], length_unit="meter" if oracle else "reconstruction_unit",
        source_prediction_sha256=job["prediction_sha256"], point_policy="all finite positive native depth pixels; no confidence filtering; native integer centers, oracle K shifted -0.5")
    segments, evidence, unresolved = [], [], []
    for target_id, target in candidates["targets"].items():
        variant = target["variants"][method["variant"]]
        if variant["state"] == "accepted":
            line_ids = [f"line-{len(segments) + i:04d}" for i in range(len(variant["segments"]))]
            segments.extend(variant["segments"])
            evidence.append(dict(decision="accept", line_ids=line_ids, suppression_range=None,
                view_ids=[v["view_id"] for v in target["views"]], source_sha256=job["candidates_sha256"],
                note=f"Historical {target_id}/{method['variant']} accepted; no new identity claim or GT quality guarantee"))
        else:
            unresolved.append(target_id + ":" + ";".join(variant["rejection_reasons"]))
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    patch_id = write_patch(patch_dir, snapshot_dir, segments, np.empty((0, 3), np.uint32), selection_sha256=method["selection_sha256"],
        method=dict(id="historical_paired_full_replay", version="2026-09-16", config_sha256=content_hash(method), seed=0), evidence=evidence, unresolved=unresolved)
    for enabled, label in ((True, "enabled"), (False, "withdrawn")):
        write_candidate_view(destination / f"view-{label}.json", snapshot_dir, patch_dir, enabled=enabled)
    base = compose(snapshot_dir)
    candidate = open_candidate_view(destination / "view-enabled.json")
    withdrawn = open_candidate_view(destination / "view-withdrawn.json")
    rollback_ok = all(np.array_equal(base[k], withdrawn[k]) for k in ("points", "point_ids", "segments"))
    if not rollback_ok or not np.array_equal(candidate["segments"], segments) or not all(np.array_equal(candidate[k], base[k]) for k in ("points", "point_ids")):
        raise ValueError("Roundtrip/composition altered source geometry")
    centers = camera_centers(cameras)
    span = float(np.linalg.norm(centers[:, None] - centers, axis=2).max())
    region = dict(minimum_views=method["region_minimum_views"], views=roi_views)
    write_json(destination / "region.json", region)
    graphs = []
    for fraction in method["voxel_camera_span_fractions"]:
        config = {**method["readout"], "voxel_size": span * fraction}
        for label, collection in (("base", base), ("candidate", candidate)):
            output = readout(collection["points"], collection["segments"], config, region)
            path = destination / f"readout-{fraction}-{label}.json"
            write_json(path, output)
            graphs.append(dict(fraction=fraction, variant=label, path=path.relative_to(run).as_posix(), sha256=file_hash(path),
                state=output["state"], segment_count=len(output["segments"]), occupied_voxels=output.get("occupied_voxels", 0), region_point_count=output["region_point_count"]))
    result = dict(job=job, snapshot_id=snapshot_id, patch_id=patch_id, base_point_count=len(base["points"]), added_segment_count=len(segments),
        suppressed_point_count=0, rollback_bitwise_equal=rollback_ok, reopened_bitwise_equal=True, candidate_keeps_all_base_points=True,
        region_sha256=file_hash(destination / "region.json"), camera_span=span, graphs=graphs, elapsed_seconds=time.perf_counter() - started)
    path = destination / "result.json"
    write_json(path, result)
    return dict(path=path.relative_to(run).as_posix(), sha256=file_hash(path))


def infer(run_id):
    run, method, frozen = checked(run_id)
    sys.addaudithook(reject_truth_open)
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=method["parallel_workers"]) as pool:
        jobs = [pool.submit(infer_job, run_id, job) for job in method["jobs"]]
        for job in as_completed(jobs):
            entry = job.result()
            records.append(entry)
            print("POINT_PATCH_JOB", entry["path"], flush=True)
    write_json(run / "inference.json", dict(run_id=run_id, input_sha256=frozen["input_sha256"], source_sha256=frozen["source_sha256"],
        state="complete", gt_read_during_inference=False, truth_read_tripwire_enabled=True, records=sorted(records, key=lambda r: r["path"]),
        elapsed_seconds=time.perf_counter() - started))


def evaluate(path, share):
    config = read_json(path)
    run, method, frozen = checked(config["run_id"])
    if file_hash(path) != frozen["protocol_sha256"] or file_hash(run / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Evaluation protocol changed")
    inference = read_json(run / "inference.json")
    if inference["state"] != "complete" or inference["input_sha256"] != frozen["input_sha256"] or inference["source_sha256"] != frozen["source_sha256"]:
        raise ValueError("Incomplete/detached point-patch inference")
    truth_root = ROOT / "data/eval_gt" / config["evaluation"]["gt_bundle"]
    status = read_json(truth_root / "status.json")
    if status["state"] != "complete" or file_hash(truth_root / "artifact_hashes.json") != status["artifact_hashes_sha256"]:
        raise ValueError("GT index integrity failure")
    hashes = read_json(truth_root / "artifact_hashes.json")
    def truth(relative):
        if file_hash(truth_root / relative) != hashes[relative]["sha256"]:
            raise ValueError("GT artifact changed")
        return read_json(truth_root / relative)
    selection = read_json(ROOT / ".runtime/experiments" / method["source_run_id"] / "selection.json")
    if file_hash(ROOT / ".runtime/experiments" / method["source_run_id"] / "selection.json") != method["selection_sha256"]:
        raise ValueError("Source alignment protocol changed")
    rows, job_records = [], []
    for entry in inference["records"]:
        if file_hash(run / entry["path"]) != entry["sha256"]:
            raise ValueError("Inference receipt changed")
        record = read_json(run / entry["path"])
        job = record["job"]
        destination = run / job["candidate_job_id"]
        candidate = open_candidate_view(destination / "view-enabled.json")
        snapshot, _ = load_bundle(destination / "base", "point_snapshot")
        if snapshot["content_id"] != record["snapshot_id"] or candidate["patch_id"] != record["patch_id"]:
            raise ValueError("Bundle identity detached from readout")
        cameras = [truth(job["case_id"] + "/" + f["frame_id"] + "/camera.json")["world_to_camera_cv"] for f in snapshot["metadata"]["frames"]]
        if file_hash(ROOT / job["prediction_path"]) != job["prediction_sha256"]:
            raise ValueError("Base camera source changed")
        with np.load(ROOT / job["prediction_path"], allow_pickle=False) as native:
            ext = native["extrinsics"]
        if job["camera_mode"] == "oracle_camera":
            scale, rotation, translation, alignment = 1., np.eye(3), np.zeros(3), {"state": "oracle_identity"}
        else:
            scale, rotation, translation, alignment = align_cameras(ext, np.asarray(cameras), selection["alignment"])
        geometry = truth(job["case_id"] + "/geometry.json")
        target = np.array([o["world_centerline_endpoints"] for o in geometry["objects"] if o["rod_id"] in config["evaluation"]["target_rod_ids"] and o["duplicate_geometry_of"] is None])
        gaps = [g["world_endpoints"] for g in geometry["gaps"] if g["rod_id"] in config["evaluation"]["target_rod_ids"]]
        expected_graphs = {(fraction, label) for fraction in method["voxel_camera_span_fractions"] for label in ("base", "candidate")}
        actual_graphs = [(g["fraction"], g["variant"]) for g in record["graphs"]]
        if len(actual_graphs) != len(set(actual_graphs)) or set(actual_graphs) != expected_graphs:
            raise ValueError("Incomplete paired readout scales")
        for graph in record["graphs"]:
            if file_hash(run / graph["path"]) != graph["sha256"]:
                raise ValueError("Common readout changed")
            output = read_json(run / graph["path"])
            for tolerance in config["evaluation"]["tolerances_m"]:
                score, gap_scores = None, []
                if output["state"] == "complete":
                    seg = np.asarray(output["segments"]).reshape(-1, 2, 3)
                    aligned = apply_similarity(seg, scale, rotation, translation)
                    score = curve_metrics(aligned, target, tolerance=tolerance, spacing=config["evaluation"]["spacing_m"])
                    gap_scores = [gap_coverage(aligned, gap, tolerance=tolerance, spacing=config["evaluation"]["spacing_m"]) for gap in gaps]
                rows.append(dict(job_id=job["candidate_job_id"], camera_mode=job["camera_mode"], fraction=graph["fraction"], variant=graph["variant"],
                    state=output["state"], reason=output.get("reason"), tolerance_m=tolerance, metrics=score, gaps=gap_scores,
                    readout_segment_count=len(output["segments"]), occupied_voxels=output.get("occupied_voxels"), region_point_count=output["region_point_count"]))
        job_records.append({**record, "alignment": clean(alignment)})
    if len(job_records) != len(method["jobs"]) or {j["job"]["candidate_job_id"] for j in job_records} != {j["candidate_job_id"] for j in method["jobs"]}:
        raise ValueError("Missing or duplicate historical jobs")
    output = dict(run_id=config["run_id"], state="complete", scope=config["scope"], inference_sha256=file_hash(run / "inference.json"),
        protocol_sha256=file_hash(path), source_sha256=frozen["source_sha256"], inference_seconds=inference["elapsed_seconds"], jobs=job_records, rows=rows,
        limitations=["Historical single-asset development replay; no new identity algorithm or independent-object claim.",
          "All model-valid points retained, then one fixed shared RGB-strip region used by the evaluator.",
          "No suppression chosen by this replay; source-ID deletion tested analytically.",
          "Common readout has occupancy/PCA/scale bias; no GT-assisted extraction or candidate-only scoring.",
          "Pilot materialized-point protocol, not complete depth-backed BaseSnapshot/PatchResult or Blender UI."])
    evaluation = ROOT / "data/evaluation" / config["run_id"]
    evaluation.mkdir(exist_ok=False)
    write_json(evaluation / "summary.json", output)
    write_json(share, output)
    print("EVALUATED_POINT_PATCH", len(rows), "paired readout metric rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--share-json", type=Path, default=ROOT / "docs/experiments/results/2026-09-24-point-patch-valley.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.config, args.share_json)
