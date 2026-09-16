"""Read-only truth stage for immutable RGB rod proposals; never used by fit()."""
import sys
from pathlib import Path

import numpy as np
from thin_pack_gt import read_json, sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.line_controls import curve_metrics, gap_coverage
from creator_eval.native_diagnostics import align_cameras, apply_similarity, homogeneous
from run_thin_line_controls import clean

ROOT = Path(__file__).resolve().parents[1]


def project(points, camera):
    ext = homogeneous(camera["world_to_camera_cv"])
    xyz = np.asarray(points) @ ext[:3, :3].T + ext[:3, 3]
    uv = xyz @ np.array(camera["K_index"]).T
    if np.any(xyz[..., 2] <= 0):
        raise ValueError("Evaluation curve extends behind camera")
    return uv[..., :2] / uv[..., 2, None]


def centering_report(xy, truth, camera):
    """Compare only rows intersecting physical finite rods; other responses stay visible."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    projected = project(np.asarray(truth), camera)
    errors = np.full(len(xy), np.inf)
    covered_row = np.zeros(len(xy), bool)
    for first, second in projected:
        if abs(second[1] - first[1]) < 1e-9:
            continue
        fraction = (xy[:, 1] - first[1]) / (second[1] - first[1])
        inside = (fraction >= 0) & (fraction <= 1)
        expected = first[0] + fraction * (second[0] - first[0])
        errors[inside] = np.minimum(errors[inside], abs(xy[inside, 0] - expected[inside]))
        covered_row |= inside
    valid = errors[covered_row]
    return {"selected_rows": len(xy), "finite_truth_rows": int(covered_row.sum()),
            "rows_outside_finite_rod_including_gap": int((~covered_row).sum()),
            "median_horizontal_error_px": float(np.median(valid)) if len(valid) else None,
            "p95_horizontal_error_px": float(np.quantile(valid, .95)) if len(valid) else None,
            "fraction_within_1px": float(np.mean(valid <= 1)) if len(valid) else None,
            "scope": "2D center-only conditional on selected row overlapping a finite physical rod; not detection recall or occlusion-aware"}


def evaluate(run_id):
    from creator_eval.heldout_metrics import evaluate_projected_candidate

    run = ROOT / ".runtime/experiments" / run_id
    record = read_json(run / "manifest.json")
    if record["state"] != "complete":
        raise ValueError("Fit has not completed")
    for name in ("protocol", "selection"):
        if sha256(run / (name + ".json")) != record[name + "_sha256"]:
            raise ValueError("Frozen method configuration changed")
    for name, digest in record["source_hashes"].items():
        if sha256(run / "producer_sources" / name) != digest:
            raise ValueError("Frozen producer changed")
    protocol, selection = read_json(run / "protocol.json"), read_json(run / "selection.json")
    input_path = ROOT / "data/inputs" / protocol["input_bundle"] / "manifest.json"
    if sha256(input_path) != record["input_manifest_sha256"]:
        raise ValueError("Original input manifest changed")
    input_manifest = read_json(input_path)
    settings = protocol["evaluation"]
    gt = ROOT / "data/eval_gt" / settings["gt_bundle"]
    status = read_json(gt / "status.json")
    if status["state"] != "complete" or sha256(gt / "artifact_hashes.json") != status["artifact_hashes_sha256"]:
        raise ValueError("GT integrity gate failed")
    hashes = read_json(gt / "artifact_hashes.json")

    def checked(relative):
        relative = Path(relative)
        if sha256(gt / relative) != hashes[relative.as_posix()]["sha256"]:
            raise ValueError("GT artifact changed: " + str(relative))
        return gt / relative

    if read_json(checked("validation_summary.json"))["input_manifest_sha256"] != record["input_manifest_sha256"]:
        raise ValueError("Truth validation belongs to a different input bundle")
    expected_jobs = [(j["camera_mode"], j["job_id"]) for j in protocol["jobs"]]
    actual_jobs = [(j["camera_mode"], j["job_id"]) for j in record["jobs"]]
    if expected_jobs != actual_jobs or len(set(actual_jobs)) != len(actual_jobs):
        raise ValueError("Incomplete, duplicate or reordered proposal jobs")

    heldout = ROOT / "data/eval_gt" / settings["heldout_bundle"]
    heldout_status = read_json(heldout / "status.json")
    if heldout_status["state"] != "complete" or sha256(heldout / "artifact_hashes.json") != heldout_status["artifact_hashes_sha256"]:
        raise ValueError("Heldout export incomplete or hash index changed")
    heldout_hashes = read_json(heldout / "artifact_hashes.json")
    for relative, identity in heldout_hashes.items():
        if sha256(heldout / relative) != identity["sha256"]:
            raise ValueError("Heldout artifact identity changed: " + relative)
    heldout_config = read_json(heldout / "export_config.json")
    if heldout_config["source_input_manifest_sha256"] != record["input_manifest_sha256"]:
        raise ValueError("Heldout source RGB bundle differs")
    heldout_manifest = read_json(heldout / "manifest.json")
    if not heldout_manifest["eval_only"]:
        raise ValueError("Expected independent evaluator-only bundle")
    output = ROOT / "data/evaluation" / run_id
    output.mkdir(exist_ok=False)
    result = {"run_id": run_id, "state": "running", "rows": [], "heldout_rows": [], "centering": [], "alignments": [],
              "background_audits": [], "proposals": [], "fit_manifest_sha256": sha256(run / "manifest.json"),
              "gt_hash_index_sha256": status["artifact_hashes_sha256"], "heldout_manifest_sha256": sha256(heldout / "manifest.json"),
              "scope": "Same asset development with evaluator-only new angles; no independent-object generalization. Empty/refused proposals included.",
              "alignment_policy": settings["coordinate"]}
    observations = {}
    for case, artifact in record["observations"].items():
        if sha256(run / artifact["path"]) != artifact["sha256"]:
            raise ValueError("RGB observations changed")
        observations[case] = read_json(run / artifact["path"])
    old_run = ROOT / ".runtime/experiments" / protocol["baseline_line_run"]
    if sha256(old_run / "manifest.json") != record["legacy_manifest_sha256"]:
        raise ValueError("Legacy reference manifest changed")
    old_manifest = read_json(old_run / "manifest.json")
    centering_done = set()
    for entry in record["jobs"]:
        case, mode, name = entry["case_id"], entry["camera_mode"], entry["candidate_job_id"]
        path = run / name / "candidates.json"
        if sha256(path) != entry["candidates_sha256"] or sha256(entry["prediction_path"]) != entry["prediction_sha256"]:
            raise ValueError("Proposal or base prediction changed")
        candidates = read_json(path)
        expected_job = next(j for j in protocol["jobs"] if (j["camera_mode"], j["job_id"]) == (mode, entry["job_id"]))
        if candidates["job"] != expected_job or candidates["base_prediction_sha256"] != entry["prediction_sha256"]:
            raise ValueError("Candidate identity differs from frozen job")
        if set(candidates["targets"]) != set(protocol["target_aliases"]):
            raise ValueError("Missing or extra candidate targets")
        native = np.load(entry["prediction_path"], allow_pickle=False)
        geometry = read_json(checked(Path(case) / "geometry.json"))
        input_group = next(g for g in input_manifest["groups"] if g["case_id"] == case)
        frame_order = [f["frame_id"] for f in input_group["frames"]]
        if frame_order != input_group["frame_order"]:
            raise ValueError("Input frame order disagrees with frame list")
        cameras = [read_json(checked(Path(case) / f / "camera.json")) for f in frame_order]
        if mode == "oracle_camera":
            scale, rotation, translation = 1., np.eye(3), np.zeros(3)
            alignment = {"state": "oracle_identity", "fit_inputs": "none; supplied metric cameras"}
        else:
            scale, rotation, translation, alignment = align_cameras(
                native["extrinsics"], np.array([c["world_to_camera_cv"] for c in cameras]), selection["alignment"])
        result["alignments"].append({"candidate_job_id": name, "scale": scale, "rotation": rotation,
                                     "translation": translation, "diagnostics": alignment})
        result["background_audits"].append({"candidate_job_id": name, **candidates["background_camera_audit"]})
        group = next(g for g in heldout_manifest["groups"] if g["case_id"] == case)
        heldout_views = []
        for frame in group["frames"]:
            for key in ("rgb", "camera", "curve_visibility"):
                if sha256(heldout / frame[key]) != frame[key + "_sha256"]:
                    raise ValueError("Heldout artifact changed")
            heldout_views.append((frame, read_json(heldout / frame["camera"]),
                                  dict(np.load(heldout / frame["curve_visibility"], allow_pickle=False))))
        for target, rod_id in settings["target_rod_ids"].items():
            if set(candidates["targets"][target]["variants"]) != set(protocol["variants"]):
                raise ValueError("Missing or extra candidate variants")
            if [v["view_id"] for v in candidates["targets"][target]["views"]] != frame_order or [o["frame_id"] for o in observations[case][target]] != frame_order:
                raise ValueError("Candidate/observation camera order mismatch")
            truth = [obj["world_centerline_endpoints"] for obj in geometry["objects"]
                     if obj["rod_id"] == rod_id and obj["duplicate_geometry_of"] is None]
            gap = next((g["world_endpoints"] for g in geometry["gaps"] if g["rod_id"] == rod_id), None)
            if (case, target) not in centering_done:
                centering_done.add((case, target))
                old_id = "full--" + case + "-504"
                old_entry = next(j for j in old_manifest["jobs"] if j["line_job_id"] == old_id)
                old_path = old_run / old_id / "observations.json"
                if sha256(old_path) != old_entry["observations_sha256"]:
                    raise ValueError("Old RGB centers changed")
                old_obs = read_json(old_path)[target]
                if [o["frame_id"] for o in old_obs] != frame_order:
                    raise ValueError("Legacy observation camera order mismatch")
                for i, camera in enumerate(cameras):
                    obs = observations[case][target][i]
                    selected = []
                    if obs["fitted"]["usable"]:
                        for match in obs["fitted"]["row_matches"]:
                            row = obs["extracted"]["rows"][match["row_index"]]
                            selected.append([row["candidates"][match["candidate_index"]]["center_x"], row["y"]])
                    for detector, xy in (("legacy_ridge", old_obs[i]["source_uv"]), ("paired_robust", selected)):
                        result["centering"].append({"case_id": case, "target": target, "frame_id": obs["frame_id"],
                                                    "detector": detector, **centering_report(xy, truth, camera)})
            for variant, candidate in candidates["targets"][target]["variants"].items():
                raw = np.array(candidate["segments"], float).reshape(-1, 2, 3)
                if candidate["state"] not in ("accepted", "rejected") or (len(raw) > 0) != (candidate["state"] == "accepted"):
                    raise ValueError("Proposal state and published segments disagree")
                aligned = apply_similarity(raw, scale, rotation, translation)
                metadata = {"candidate_job_id": name, "camera_mode": mode, "case_id": case, "process_res": entry["process_res"],
                            "target": target, "variant": variant, "state": candidate["state"],
                            "rejection_reasons": candidate.get("rejection_reasons", []), "segment_count": len(aligned)}
                result["proposals"].append(metadata)
                for tolerance in settings["tolerances_m"]:
                    row = {**metadata, "tolerance_m": tolerance,
                           "metrics": curve_metrics(aligned, truth, tolerance=tolerance, spacing=settings["spacing_m"])}
                    row["missing_truth_length_m"] = row["metrics"]["truth_to_prediction"]["source_length"] - row["metrics"]["truth_to_prediction"]["covered_length"]
                    if gap is not None:
                        row["gap"] = gap_coverage(aligned, gap, tolerance=tolerance, spacing=settings["spacing_m"])
                    result["rows"].append(row)
                for frame, camera, visibility in heldout_views:
                    for tolerance in settings["heldout_tolerance_px"]:
                        metrics = evaluate_projected_candidate(aligned, camera["K_index"], camera["world_to_camera_cv"],
                                                               camera["size_wh"], visibility, rod_id, tolerance_px=tolerance,
                                                               clip_start=camera["clip_start"], clip_end=camera["clip_end"])
                        metrics["eligible_for_success_claim"] = not metrics["candidate_projection"]["has_depth_rejections"]
                        metrics["scope_if_depth_rejected"] = "partial projection diagnostic; dropped segments must not support a success claim"
                        result["heldout_rows"].append({**metadata, "frame_id": frame["frame_id"], "tolerance_px": tolerance, "metrics": metrics})
    result.update(state="complete", accepted_proposals=sum(r["state"] == "accepted" for r in result["proposals"]),
                  refused_proposals=sum(r["state"] != "accepted" for r in result["proposals"]),
                  evaluation_sources={p.relative_to(ROOT).as_posix(): sha256(p) for p in
                                      [Path(__file__), ROOT / "experiments/src/creator_eval/heldout_metrics.py",
                                       ROOT / "experiments/src/creator_eval/line_controls.py", ROOT / "experiments/src/creator_eval/native_diagnostics.py",
                                       ROOT / "scripts/run_thin_line_controls.py", ROOT / "scripts/thin_pack_gt.py"]})
    write_json(output / "summary.json", clean(result))
    print("EVALUATED", output, flush=True)
