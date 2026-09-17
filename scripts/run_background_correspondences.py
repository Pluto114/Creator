"""Freeze RGB-only background matches, then evaluate their physical correspondence."""

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.background_correspondences import (
    detect_features,
    exclusion_mask,
    fit_rgb_geometry,
    mesh_direction,
    mutual_ratio_matches,
    residual_summary,
    spatial_split,
)
from creator_eval.camera_diagnostics import fundamental_from_cameras, sampson_distances
from thin_pack_gt import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "background-correspondences-v1-20260917"


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fit(run_id, config_path):
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(113)
    protocol = read_json(config_path)
    selection_path = ROOT / protocol["selection_config"]
    selection = read_json(selection_path)["line_control"]
    inputs = ROOT / "data/inputs" / protocol["input_bundle"]
    manifest = read_json(inputs / "manifest.json")
    output = ROOT / ".runtime/experiments" / run_id
    output.mkdir(exist_ok=False)
    frozen = output / "producer_sources"
    frozen.mkdir()
    sources = [Path(__file__), ROOT / "experiments/src/creator_eval/background_correspondences.py",
               ROOT / "experiments/src/creator_eval/camera_diagnostics.py",
               ROOT / "experiments/src/creator_eval/native_diagnostics.py", ROOT / "scripts/thin_pack_gt.py"]
    source_hashes = {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in sources}
    for path in sources:
        shutil.copy2(path, frozen / path.name)
    shutil.copy2(inputs / "manifest.json", output / "input_manifest.json")
    shutil.copy2(selection_path, output / "selection.json")
    write(output / "protocol.json", protocol)
    record = {"run_id": run_id, "state": "running", "started_at": now(), "pairs": [],
              "gt_read_in_fit": False, "predictions_read_in_fit": False, "heldout_read_in_fit": False,
              "input_manifest_sha256": sha256(output / "input_manifest.json"),
              "selection_sha256": sha256(output / "selection.json"), "protocol_sha256": sha256(output / "protocol.json"),
              "source_hashes": source_hashes, "opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "opencv_threads": 1, "features": []}
    write(output / "manifest.json", record)
    try:
        for case in protocol["cases"]:
            group = next(g for g in manifest["groups"] if g["case_id"] == case)
            frames = group["frames"]
            if group["frame_order"] != [f["frame_id"] for f in frames] or len(frames) != 5:
                raise ValueError("Unexpected frame order")
            images = []
            for frame in frames:
                path = inputs / frame["rgb"]
                if sha256(path) != frame["sha256"]:
                    raise ValueError("Original RGB identity changed")
                rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                if list(rgb.shape[1::-1]) != frame["size_wh"]:
                    raise ValueError("Original RGB dimensions changed")
                images.append(rgb)
            for detector, settings in protocol["detectors"].items():
                feature_sets = []
                for frame, rgb in zip(frames, images):
                    mask = exclusion_mask(frame["size_wh"], selection["guides"][frame["frame_id"]], selection["guide_y"], protocol["target_exclusion_half_width"])
                    feature = detect_features(rgb, mask, detector, settings)
                    feature_sets.append(feature)
                    filename = f"{case}--{detector}--{frame['frame_id']}--features.npz"
                    arrays = {key: value for key, value in feature.items() if value is not None}
                    np.savez_compressed(output / filename, **arrays)
                    record["features"].append({"case_id": case, "detector": detector, "frame_id": frame["frame_id"],
                                               "count": len(feature["xy"]), "path": filename, "sha256": sha256(output / filename)})
                for first, second in protocol["pairs"]:
                    a, b = feature_sets[first], feature_sets[second]
                    ids, distances = mutual_ratio_matches(a, b, detector, protocol["ratio"])
                    xy_a, xy_b = a["xy"][ids[:, 0]], b["xy"][ids[:, 1]]
                    validation, split = spatial_split(xy_a, xy_b, protocol["split"])
                    geometry = fit_rgb_geometry(xy_a, xy_b, validation, protocol["geometry"])
                    pair_id = f"{case}--{detector}--{first}-{second}"
                    pair = {"pair_id": pair_id, "case_id": case, "detector": detector, "first_view": first, "second_view": second,
                            "frame_ids": [frames[i]["frame_id"] for i in (first, second)], "keypoint_ids": ids,
                            "first_xy": xy_a, "second_xy": xy_b, "descriptor_distances": distances,
                            "validation_mask": validation, "split": split, "geometry": geometry,
                            "exclusion_scope": "keypoint centers outside two strips; descriptor support may cross boundaries; no GT mask"}
                    path = output / (pair_id + ".json")
                    write(path, pair)
                    record["pairs"].append({"pair_id": pair_id, "case_id": case, "detector": detector,
                                             "first_view": first, "second_view": second, "path": path.name, "sha256": sha256(path)})
                    write(output / "manifest.json", record)
                    print("RGB_PAIR", pair_id, len(ids), geometry["state"], flush=True)
        if source_hashes != {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in sources}:
            raise ValueError("Producer changed during fit")
        if len(record["pairs"]) != len(protocol["cases"]) * len(protocol["pairs"]) * len(protocol["detectors"]):
            raise ValueError("Incomplete pair collection")
        record.update(state="complete", completed_at=now())
        write(output / "manifest.json", record)
        print("FIT_FROZEN", output, flush=True)
    except BaseException as error:
        record.update(state="failed", error=str(error), completed_at=now())
        write(output / "manifest.json", record)
        raise


def _subsets(pair):
    validation = np.asarray(pair["validation_mask"], bool)
    subsets = {"all": np.ones(len(validation), bool), "train": ~validation, "validation": validation}
    for kind, model in pair["geometry"]["models"].items():
        if "train_inlier_mask" not in model:
            continue
        selected = np.zeros(len(validation), bool)
        selected[~validation] = model["train_inlier_mask"]
        subsets[kind + "_training_inliers"] = selected
        error = np.asarray(model["all_residuals_px"], float)
        subsets[kind + "_validation_consistent_at_2px"] = validation & np.isfinite(error) & (error <= 2)
    return subsets


def _mesh_report(forward, reverse, selected):
    a, b = forward["source_point_visible_in_target"], reverse["source_point_visible_in_target"]
    both = a & b & selected
    any_visible = (a | b) & selected
    first, second = forward["transfer_error_px"], reverse["transfer_error_px"]
    worst = np.fmax(first, second)
    correct = both & (worst <= 2)
    wrong = any_visible & (worst > 5)
    return {"match_count": int(selected.sum()), "both_directions_visible": int(both.sum()),
            "at_least_one_direction_visible": int(any_visible.sum()),
            "both_visible_fraction_within_2px": float(correct.sum() / both.sum()) if both.any() else None,
            "any_visible_wrong_beyond_5px_count": int(wrong.sum()),
            "indeterminate_count": int(selected.sum() - correct.sum() - wrong.sum()),
            "both_visible_max_transfer": residual_summary(worst[both]),
            "forward_visible_transfer": residual_summary(first[selected & a]),
            "reverse_visible_transfer": residual_summary(second[selected & b]),
            "forward_no_hit": int((selected & ~forward["source_hit"]).sum()),
            "reverse_no_hit": int((selected & ~reverse["source_hit"]).sum()),
            "forward_outside_target_or_clip": int((selected & forward["source_hit"] & ~forward["target_in_frame"]).sum()),
            "reverse_outside_target_or_clip": int((selected & reverse["source_hit"] & ~reverse["target_in_frame"]).sum()),
            "forward_visibility_failed": int((selected & forward["target_in_frame"] & ~a).sum()),
            "reverse_visibility_failed": int((selected & reverse["target_in_frame"] & ~b).sum()),
            **{direction + "_" + key: int((selected & values[key]).sum())
               for direction, values in (("forward", forward), ("reverse", reverse))
               for key in ("occluded", "target_no_hit", "depth_inconsistent", "invalid_projection")},
            "scope": "First-hit geometric keypoint localization diagnostic; 2px consistent / >5px inconsistent, neither used for fitting"}


def evaluate(run_id):
    from run_rod_evidence import original_intrinsics
    from thin_pack_gt import MeshRays

    run = ROOT / ".runtime/experiments" / run_id
    record = read_json(run / "manifest.json")
    if record["state"] != "complete":
        raise ValueError("RGB fit must be complete first")
    for name in ("protocol", "selection", "input_manifest"):
        if sha256(run / (name + ".json")) != record[name + "_sha256"]:
            raise ValueError("Frozen configuration changed")
    for source, digest in record["source_hashes"].items():
        if sha256(run / "producer_sources" / Path(source).name) != digest:
            raise ValueError("Frozen producer changed")
    for entry in record["features"] + record["pairs"]:
        if sha256(run / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen RGB artifact changed")
    protocol, inputs = read_json(run / "protocol.json"), read_json(run / "input_manifest.json")
    gt = ROOT / "data/eval_gt" / protocol["evaluation"]["gt_bundle"]
    status, hashes = read_json(gt / "status.json"), read_json(gt / "artifact_hashes.json")
    if status["state"] != "complete" or sha256(gt / "artifact_hashes.json") != status["artifact_hashes_sha256"]:
        raise ValueError("Incomplete or altered ground truth")
    consumed = {}

    def checked(relative):
        path = gt / relative
        digest = sha256(path)
        if digest != hashes[Path(relative).as_posix()]["sha256"]:
            raise ValueError("GT artifact identity changed")
        consumed[Path(relative).as_posix()] = digest
        return path

    if read_json(checked("validation_summary.json"))["input_manifest_sha256"] != record["input_manifest_sha256"]:
        raise ValueError("GT belongs to different RGB inputs")
    expected = {(case, detector, a, b) for case in protocol["cases"] for detector in protocol["detectors"] for a, b in protocol["pairs"]}
    actual = [(p["case_id"], p["detector"], p["first_view"], p["second_view"]) for p in record["pairs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate match pairs")
    output = ROOT / "data/evaluation" / run_id
    output.mkdir(exist_ok=False)
    result = {"run_id": run_id, "state": "running", "started_at": now(), "fit_manifest_sha256": sha256(run / "manifest.json"),
              "gt_hash_index_sha256": status["artifact_hashes_sha256"], "pairs": [], "camera_rows": [], "artifacts": [],
              "scope": "Same RGB matches for every camera condition; no match selection or camera adjustment from GT",
              "mesh_limitations": "First-hit geometry does not model descriptor support, antialiasing, or view-dependent texture/shading. No-hit/occluded/outside/invalid matches are not called correct."}
    write(output / "summary.json", result)
    pairs_by_case, truth_by_case, frames_by_case = {}, {}, {}
    try:
        for case in protocol["cases"]:
            group = next(g for g in inputs["groups"] if g["case_id"] == case)
            frames = group["frames"]
            frames_by_case[case] = frames
            cameras = [read_json(checked(f"{case}/{f['frame_id']}/camera.json")) for f in frames]
            if any(camera["size_wh"] != frame["size_wh"] for camera, frame in zip(cameras, frames)):
                raise ValueError("GT image size mismatch")
            truth_by_case[case] = cameras
            mesh = MeshRays(checked(f"{case}/mesh.npz"))
            pairs_by_case[case] = []
            for entry in record["pairs"]:
                if entry["case_id"] != case:
                    continue
                pair = read_json(run / entry["path"])
                first, second = pair["first_view"], pair["second_view"]
                if pair["frame_ids"] != [frames[i]["frame_id"] for i in (first, second)]:
                    raise ValueError("Match frame ordering changed")
                a, b = np.asarray(pair["first_xy"], float).reshape(-1, 2), np.asarray(pair["second_xy"], float).reshape(-1, 2)
                if len(a):
                    forward = mesh_direction(a, b, cameras[first], cameras[second], mesh.cast, protocol["evaluation"]["visibility_depth_tolerance_m"])
                    reverse = mesh_direction(b, a, cameras[second], cameras[first], mesh.cast, protocol["evaluation"]["visibility_depth_tolerance_m"])
                else:
                    forward = reverse = {"source_hit": np.zeros(0, bool), "source_point_visible_in_target": np.zeros(0, bool),
                                         "target_in_frame": np.zeros(0, bool), "transfer_error_px": np.empty(0),
                                         **{key: np.zeros(0, bool) for key in ("occluded", "target_no_hit", "depth_inconsistent", "invalid_projection")}}
                gt_f = fundamental_from_cameras(cameras[first]["K_index"], cameras[first]["world_to_camera_cv"], cameras[second]["K_index"], cameras[second]["world_to_camera_cv"])
                error = sampson_distances(a, b, gt_f)
                subsets = _subsets(pair)
                path = output / (pair["pair_id"] + "--mesh.json")
                write(path, {"pair_id": pair["pair_id"], "match_pair_sha256": entry["sha256"], "forward": forward, "reverse": reverse})
                result["artifacts"].append({"path": path.name, "sha256": sha256(path)})
                result["pairs"].append({**entry, "match_count": len(a), "split": pair["split"], "rgb_geometry": pair["geometry"],
                                        "truth_camera_sampson": {name: residual_summary(error[mask]) for name, mask in subsets.items()},
                                        "mesh_correspondence": {name: _mesh_report(forward, reverse, mask) for name, mask in subsets.items()}})
                pairs_by_case[case].append((entry, pair, a, b, subsets))
        for job in protocol["evaluation"]["camera_jobs"]:
            source = ROOT / ".runtime/experiments" / job["basis_run_id"]
            inference = read_json(source / "inference_manifest.json")
            identity = next(j for j in inference["jobs"] if j["job_id"] == job["job_id"])
            prediction, report_path = source / job["job_id"] / "prediction.npz", source / job["job_id"] / "report.json"
            if inference["state"] != "complete" or inference["input_manifest_sha256"] != record["input_manifest_sha256"] or identity["state"] != "succeeded":
                raise ValueError("Incomplete inference or wrong original images")
            if sha256(prediction) != identity["prediction_sha256"] or sha256(report_path) != identity["report_sha256"]:
                raise ValueError("Prediction/report identity changed")
            report = read_json(report_path)
            frames = frames_by_case[job["case_id"]]
            if report["process_res"] != job["process_res"] or [r["sha256"] for r in report["images"]] != [f["sha256"] for f in frames]:
                raise ValueError("Inference image order/resolution mismatch")
            oracle = job["camera_mode"] == "oracle_camera"
            if ("oracle_camera" in report["scope"]) != oracle:
                raise ValueError("Privileged/estimated camera mismatch")
            with np.load(prediction, allow_pickle=False) as native:
                count, height, width = native["depth"].shape
                matrices, cameras = native["intrinsics"], native["extrinsics"]
            if count != len(frames) or matrices.shape != (count, 3, 3) or cameras.shape != (count, 3, 4) or not np.isfinite(matrices).all() or not np.isfinite(cameras).all():
                raise ValueError("Invalid native camera arrays")
            matrices = original_intrinsics(matrices, frames[0]["size_wh"], [width, height], oracle)
            for entry, pair, a, b, subsets in pairs_by_case[job["case_id"]]:
                first, second = pair["first_view"], pair["second_view"]
                f = fundamental_from_cameras(matrices[first], cameras[first], matrices[second], cameras[second])
                errors = sampson_distances(a, b, f)
                result["camera_rows"].append({**job, "pair_id": entry["pair_id"], "match_pair_sha256": entry["sha256"],
                                              "prediction_sha256": identity["prediction_sha256"], "report_sha256": identity["report_sha256"],
                                              "inference_manifest_sha256": sha256(source / "inference_manifest.json"),
                                              "residuals": {name: residual_summary(errors[mask]) for name, mask in subsets.items()},
                                              "units": "original RGB index pixels", "camera_fit_performed": False})
        if len(result["camera_rows"]) != len(protocol["evaluation"]["camera_jobs"]) * len(protocol["pairs"]) * len(protocol["detectors"]):
            raise ValueError("Incomplete camera comparisons")
        result.update(state="complete", completed_at=now(), consumed_gt_sha256=consumed,
                      evaluation_sources={str(p.relative_to(ROOT).as_posix()): sha256(p) for p in
                                          [Path(__file__), ROOT / "scripts/run_rod_evidence.py", ROOT / "scripts/thin_pack_gt.py",
                                           ROOT / "experiments/src/creator_eval/background_correspondences.py", ROOT / "experiments/src/creator_eval/camera_diagnostics.py"]})
        write(output / "summary.json", result)
        print("EVALUATION_COMPLETE", output, flush=True)
    except BaseException as error:
        result.update(state="failed", error=str(error), completed_at=now())
        write(output / "summary.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/background_correspondences_v1.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        parser.error("Invalid run ID")
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "fit":
        fit(args.run_id, args.protocol)
    else:
        evaluate(args.run_id)
