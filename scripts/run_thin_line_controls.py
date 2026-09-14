"""RGB-guided simple fits, then a separate GT evaluator. No rod GT in fit()."""
import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import read_json, sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.line_controls import (
    closest_ray_line_parameters,
    curve_metrics,
    deterministic_ransac_line,
    fit_line_tls,
    fit_multiview_line,
    gap_coverage,
    support_intervals,
)
from creator_eval.native_diagnostics import (
    align_cameras,
    apply_similarity,
    camera_centers,
    confidence_masks,
    homogeneous,
    unproject,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "thin-line-controls-v1-20260914"
PROTOCOL_PATH = ROOT / "configs/thin_line_controls_v1.json"


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(val) for val in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def ridge_observations(rgb, guide, config):
    gray = np.asarray(rgb, dtype=float).mean(axis=2)
    y0, y1 = config["guide_y"]
    rows = np.arange(y0, min(y1 + 1, gray.shape[0]), config["row_stride"])
    expected = guide[0] + (rows-y0)/(y1-y0)*(guide[1]-guide[0])
    offsets = np.arange(-config["half_width"], config["half_width"]+1)
    x = np.rint(expected[:, None]+offsets).astype(int)
    if x.min() < max(config["ridge_offsets"]) or x.max()+max(config["ridge_offsets"]) >= gray.shape[1]:
        raise ValueError("RGB guide outside valid image interior")
    y = rows[:, None]
    center = gray[y, x]
    response = np.zeros_like(center)
    for width in config["ridge_offsets"]:
        response = np.maximum(response, np.abs(center-(gray[y, x-width]+gray[y, x+width])/2))
    index = response.argmax(1)
    contrast = response[np.arange(len(rows)), index]
    selected = contrast >= config["minimum_contrast_255"]
    return np.c_[x[np.arange(len(rows)), index][selected], rows[selected]], contrast[selected]


def descriptors(protocol):
    output = []
    runs = [("full", protocol["baseline_run"]), ("crop", protocol["crop_run"]), ("oracle", protocol["oracle_run"])]
    runs += [("crop", value) for value in protocol.get("additional_crop_runs", [])]
    for kind, run_id in runs:
        run = ROOT / ".runtime/experiments" / run_id
        record = read_json(run / "inference_manifest.json")
        if record["state"] not in ("complete", "complete_with_failures"):
            raise ValueError("Inference still running: " + run_id)
        for entry in record["jobs"]:
            case = entry["case_id"].split("--")[0]
            if case not in protocol["cases"]:
                continue
            crop_box = None
            if kind == "crop":
                crop_protocol = read_json(run / "protocol.json")
                crop_box = crop_protocol["crops"][entry["case_id"].split("--")[1]]
            output.append({
                "crop_box_xyxy": crop_box,
                "kind": kind, "source_case": case, "run_id": run_id,
                "prediction_path": str(run / entry["job_id"] / "prediction.npz"),
                "report_path": str(run / entry["job_id"] / "report.json"),
                "inference_manifest_sha256": sha256(run / "inference_manifest.json"), **entry,
            })
    return output


def fit_job(job, source_group, selection, protocol, output):
    path = Path(job["prediction_path"])
    if sha256(path) != job["prediction_sha256"] or sha256(job["report_path"]) != job["report_sha256"]:
        raise ValueError("Native inference identity mismatch")
    data = np.load(path, allow_pickle=False)
    count, h, w = data["depth"].shape
    if count != len(source_group["frames"]):
        raise ValueError("Frame count mismatch")
    box = [0, 0, 1920, 1080]
    if job["kind"] == "crop":
        box = job["crop_box_xyxy"]
    left, top, right, bottom = box
    matrices = data["intrinsics"].astype(float).copy()
    if job["kind"] == "oracle":
        # 已知相机的主点约定是edge；数学反投影另用index副本，原生NPZ一字不改。
        matrices[:, :2, 2] -= 0.5
    cameras = data["extrinsics"]
    centers = camera_centers(cameras)
    span = float(np.max(np.linalg.norm(centers[:, None]-centers[None], axis=-1)))
    bin_width = span * protocol["support"]["bin_width_camera_span_fraction"]
    masks, threshold = confidence_masks(data["depth"], data["conf"], selection["filters"])
    point_maps = [unproject(data["depth"][i], matrices[i], cameras[i]) for i in range(count)]
    result = {**job, "default_threshold": threshold, "camera_span_model_units": span,
              "support_bin_width_model_units": bin_width, "crop_box_xyxy": box, "targets": {}}
    all_observations = {}
    for target in ("whole", "gap"):
        observations, raw_points, policy_points, ray_sets, lines = [], [], {p: [] for p in masks}, [], []
        for index, frame in enumerate(source_group["frames"]):
            source = ROOT / "data/inputs/thin-pack-v2-20260914r1" / frame["rgb"]
            if sha256(source) != frame["sha256"]:
                raise ValueError("Source RGB changed")
            image = Image.open(source).convert("RGB")
            guide = selection["line_control"]["guides"][frame["frame_id"]][target]
            uv, contrast = ridge_observations(image, guide, selection["line_control"])
            inside = (uv[:, 0] >= left) & (uv[:, 0] < right) & (uv[:, 1] >= top) & (uv[:, 1] < bottom)
            uv, contrast = uv[inside], contrast[inside]
            native = (uv + 0.5 - [left, top]) * [w/(right-left), h/(bottom-top)] - 0.5
            pixels = np.rint(native).astype(int)
            valid = (pixels[:, 0]>=0) & (pixels[:, 0]<w) & (pixels[:, 1]>=0) & (pixels[:, 1]<h)
            native, uv, contrast, pixels = native[valid], uv[valid], contrast[valid], pixels[valid]
            if len(native) < selection["line_control"]["minimum_rows"]:
                raise ValueError(f"{target}/{frame['frame_id']}: insufficient RGB ridge support")
            observations.append({"frame_id": frame["frame_id"], "source_uv": uv, "processed_uv": native, "contrast": contrast})
            # 原图几个采样点可能落进同一个模型像素。3D拟合前去重，不能靠重复点加权。
            _, first = np.unique(pixels[:, 1]*w+pixels[:, 0], return_index=True)
            pixels = pixels[np.sort(first)]
            y, x = pixels[:, 1], pixels[:, 0]
            points = point_maps[index][y, x]
            raw_points.append(points)
            for policy, mask in masks.items():
                policy_points[policy].append(points[mask[index, y, x]])
            slope, intercept = np.linalg.lstsq(np.c_[native[:, 1], np.ones(len(native))], native[:, 0], rcond=None)[0]
            line = np.array([1.0, -slope, -intercept])
            lines.append(line/np.linalg.norm(line[:2]))
            inverse = np.linalg.inv(homogeneous(cameras[index]))
            direction = np.c_[native, np.ones(len(native))] @ np.linalg.inv(matrices[index]).T @ inverse[:3, :3].T
            ray_sets.append((np.broadcast_to(inverse[:3, 3], direction.shape), direction))
        all_observations[target] = observations
        target_result = {"observation_counts": [len(o["source_uv"]) for o in observations],
                         "native_unique_pixel_counts": [len(p) for p in raw_points], "variants": {}}
        for variant in protocol["variants"]:
            started = time.perf_counter()
            try:
                if variant.startswith(("depth_tls_", "depth_ransac_")):
                    policy = variant.removeprefix("depth_tls_").removeprefix("depth_ransac_")
                    points_by_view = policy_points[policy]
                    if variant.startswith("depth_ransac_"):
                        merged = np.concatenate(points_by_view)
                        view_ids = np.concatenate([np.full(len(p), i) for i, p in enumerate(points_by_view)])
                        settings = protocol["ransac"]
                        fitted = deterministic_ransac_line(
                            merged, view_ids, span*settings["distance_camera_span_fraction"],
                            span*settings["minimum_pair_extent_camera_span_fraction"],
                            min_inliers=settings["minimum_inliers"], min_views=settings["minimum_views"],
                            trials=settings["trials"], seed=settings["seed"],
                        )
                        points_by_view = [merged[fitted["inlier_mask"] & (view_ids == i)] for i in range(count)]
                    else:
                        fitted = fit_line_tls(np.concatenate(points_by_view))
                    anchor, direction = fitted["centroid"], fitted["direction"]
                    parameters = [(p-anchor) @ direction for p in points_by_view]
                    diagnostics = fitted
                else:
                    fitted = fit_multiview_line(np.array(lines), matrices, cameras)
                    anchor, direction = fitted["anchor"], fitted["direction"]
                    parameters, separation = [], []
                    for origins, rays in ray_sets:
                        closest = closest_ray_line_parameters(anchor, direction, origins, rays)
                        parameters.append(closest["line_t"][closest["valid"]])
                        separation.extend(closest["closest_separation"][closest["valid"]].tolist())
                    diagnostics = {**fitted, "ray_line_separation_median_model_units": float(np.median(separation))}
                nonempty = [t for t in parameters if len(t)]
                if not nonempty:
                    raise ValueError("No positive camera-ray support")
                all_t = np.concatenate(nonempty)
                extent = np.array([[all_t.min(), all_t.max()]])
                intervals = support_intervals(
                    parameters, bin_width, minimum_views=protocol["support"]["minimum_views"],
                    dilation_bins=protocol["support"]["dilation_bins"],
                    minimum_run_bins=protocol["support"]["minimum_run_bins"],
                )
                modes = {"single_span": extent, "multiview_supported": intervals}
                target_result["variants"][variant] = {
                    "state": "succeeded", "diagnostics": diagnostics,
                    "curves": {name: anchor + bounds[..., None]*direction for name, bounds in modes.items()},
                    "support_t_by_view": parameters, "seconds": time.perf_counter()-started,
                }
            except (ValueError, np.linalg.LinAlgError) as error:
                target_result["variants"][variant] = {"state": "failed", "reason": str(error), "curves": {}}
        result["targets"][target] = target_result
    write_json(output / "observations.json", clean(all_observations))
    write_json(output / "curves.json", clean(result))
    if sha256(path) != job["prediction_sha256"]:
        raise ValueError("Native NPZ was modified")
    return result


def fit():
    protocol = read_json(PROTOCOL_PATH)
    selection = read_json(ROOT / protocol["input_selection_config"])
    manifest = read_json(ROOT / "data/inputs/thin-pack-v2-20260914r1/manifest.json")
    output = ROOT / ".runtime/experiments" / RUN_ID
    output.mkdir(exist_ok=False)
    # 评测映射不进入拟合的冻结配置。方法只知道两份人工对应的图像条带。
    fit_protocol = {k: v for k, v in protocol.items() if k != "target_truth_mapping_evaluation_only"}
    write_json(output / "protocol.json", fit_protocol)
    write_json(output / "selection.json", selection)
    frozen = output / "producer_sources"
    frozen.mkdir()
    for path in [Path(__file__), ROOT / "experiments/src/creator_eval/line_controls.py",
                 ROOT / "experiments/src/creator_eval/native_diagnostics.py"]:
        shutil.copy2(path, frozen / path.name)
    record = {
        "run_id": RUN_ID, "state": "running", "gt_read_in_fit": False,
        "scope": protocol["scope"], "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_hashes": {p.name: sha256(p) for p in frozen.iterdir()},
        "protocol_sha256": sha256(output / "protocol.json"), "selection_sha256": sha256(output / "selection.json"),
        "jobs": [],
    }
    write_json(output / "manifest.json", record)
    for job in descriptors(protocol):
        name = job["kind"] + "--" + job["job_id"]
        folder = output / name
        folder.mkdir()
        group = next(g for g in manifest["groups"] if g["case_id"] == job["source_case"])
        started = time.perf_counter()
        try:
            if job["state"] != "succeeded":
                raise ValueError("Inference failed")
            result = fit_job(job, group, selection, fit_protocol, folder)
            entry = {**job, "line_job_id": name, "fit_state": "succeeded",
                     "curves_sha256": sha256(folder / "curves.json"),
                     "observations_sha256": sha256(folder / "observations.json"),
                     "failed_variants": sum(v["state"] != "succeeded" for t in result["targets"].values() for v in t["variants"].values())}
        except Exception as error:
            entry = {**job, "line_job_id": name, "fit_state": "failed", "reason": str(error)}
            write_json(folder / "failure.json", entry)
        entry["fit_wall_seconds"] = time.perf_counter()-started
        record["jobs"].append(entry)
        write_json(output / "manifest.json", record)
        print("LINE_FIT", name, entry["fit_state"], flush=True)
    record["state"] = "complete" if all(e["fit_state"] == "succeeded" and not e.get("failed_variants", 0) for e in record["jobs"]) else "complete_with_failures"
    write_json(output / "manifest.json", record)


def evaluate():
    protocol = read_json(PROTOCOL_PATH)
    run = ROOT / ".runtime/experiments" / RUN_ID
    record = read_json(run / "manifest.json")
    if record["state"] not in ("complete", "complete_with_failures"):
        raise ValueError("Line fits not complete")
    if sha256(run / "protocol.json") != record["protocol_sha256"] or sha256(run / "selection.json") != record["selection_sha256"]:
        raise ValueError("Fit protocol or selection changed")
    frozen_protocol = read_json(run / "protocol.json")
    if frozen_protocol != {k: v for k, v in protocol.items() if k != "target_truth_mapping_evaluation_only"}:
        raise ValueError("Metric protocol changed after fitting")
    selection = read_json(run / "selection.json")
    gt = ROOT / "data/eval_gt/thin-pack-v2-20260914r1"
    gt_status = read_json(gt / "status.json")
    if gt_status["state"] != "complete" or sha256(gt / "artifact_hashes.json") != gt_status["artifact_hashes_sha256"]:
        raise ValueError("GT not complete")
    hashes = read_json(gt / "artifact_hashes.json")

    def checked(relative):
        if sha256(gt / relative) != hashes[relative.as_posix()]["sha256"]:
            raise ValueError("GT hash mismatch")
        return gt / relative

    output = ROOT / "data/evaluation" / RUN_ID
    output.mkdir(exist_ok=False)
    result = {"scope": protocol["metric_scope"], "state": "running", "run_id": RUN_ID,
              "fit_manifest_sha256": sha256(run / "manifest.json"),
              "gt_artifact_hashes_sha256": gt_status["artifact_hashes_sha256"], "rows": [], "alignments": []}
    for entry in record["jobs"]:
        case = entry["source_case"]
        folder = run / entry["line_job_id"]
        geometry = read_json(checked(Path(case) / "geometry.json"))
        if entry["fit_state"] == "succeeded":
            if sha256(folder / "curves.json") != entry["curves_sha256"] or sha256(folder / "observations.json") != entry["observations_sha256"]:
                raise ValueError("Fit artifacts changed")
            curves = read_json(folder / "curves.json")
            if sha256(entry["prediction_path"]) != entry["prediction_sha256"]:
                raise ValueError("Prediction changed")
            native = np.load(entry["prediction_path"], allow_pickle=False)
            true_cameras = [
                read_json(checked(Path(case) / frame / "camera.json"))["world_to_camera_cv"]
                for frame in selection["line_control"]["guides"]
            ]
            if entry["kind"] == "oracle":
                s, r, t = 1.0, np.eye(3), np.zeros(3)
                alignment = {"state": "oracle_input_camera_frame", "fit_inputs": "none; supplied metric camera coordinates"}
            else:
                s, r, t, alignment = align_cameras(native["extrinsics"], np.array(true_cameras), selection["alignment"])
            result["alignments"].append({"line_job_id": entry["line_job_id"], **alignment,
                                         "support_bin_width_m": s*curves["support_bin_width_model_units"]})
        else:
            curves = {"targets": {}}
            s, r, t = 1.0, np.eye(3), np.zeros(3)
        for target, rod_id in protocol["target_truth_mapping_evaluation_only"].items():
            truth = [o["world_centerline_endpoints"] for o in geometry["objects"]
                     if o["rod_id"] == rod_id and o["duplicate_geometry_of"] is None]
            gap = next((g["world_endpoints"] for g in geometry["gaps"] if g["rod_id"] == rod_id), None)
            for variant in protocol["variants"]:
                candidate = curves["targets"].get(target, {}).get("variants", {}).get(variant, {})
                for mode in protocol["endpoint_modes"]:
                    raw = np.array(candidate.get("curves", {}).get(mode, []), dtype=float).reshape(-1, 2, 3)
                    aligned = apply_similarity(raw, s, r, t)
                    for tolerance in protocol["tolerances_m"]:
                        row = {
                            "line_job_id": entry["line_job_id"], "kind": entry["kind"], "case_id": case,
                            "process_res": entry["process_res"], "target": target, "variant": variant,
                            "mode": mode, "tolerance_m": tolerance,
                            "state": candidate.get("state", "failed"),
                            "failure_reason": candidate.get("reason", entry.get("reason")),
                            "segment_count": len(aligned),
                            "evaluates": "full target GT including endpoints; tight crop has reduced visibility",
                            "metrics": curve_metrics(aligned, truth, tolerance=tolerance, spacing=protocol["sample_spacing_m"]),
                        }
                        if gap is not None:
                            row["gap"] = gap_coverage(aligned, gap, tolerance=tolerance, spacing=protocol["sample_spacing_m"])
                        result["rows"].append(row)
    result["failed_metric_rows"] = sum(row["state"] != "succeeded" for row in result["rows"])
    result["empty_curve_metric_rows"] = sum(row["segment_count"] == 0 for row in result["rows"])
    result["state"] = "complete_with_failures" if result["failed_metric_rows"] else "complete"
    result["evaluation_sources"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [
        Path(__file__), ROOT / "experiments/src/creator_eval/line_controls.py"]}
    write_json(output / "summary.json", clean(result))
    print("LINE_EVALUATION_READY", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        parser.error("Invalid run ID")
    RUN_ID, PROTOCOL_PATH = args.run_id, args.protocol
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    (fit if args.stage == "fit" else evaluate)()
