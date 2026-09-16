"""Freeze RGB-only local proposals first; evaluate against truth in a separate stage."""
import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import read_json, sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.camera_diagnostics import background_camera_audit
from creator_eval.line_controls import (
    closest_ray_line_parameters,
    fit_multiview_line,
    support_intervals,
)
from creator_eval.native_diagnostics import camera_centers, homogeneous
from creator_eval.rod_evidence import build_rod_candidate
from creator_eval.rod_observations import extract_rod_observations, fit_robust_image_line
from run_thin_line_controls import clean

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "rod-evidence-v1-20260916"


def original_intrinsics(native, original_wh, native_wh, oracle=False):
    """Undo the exact pixel-center resize; oracle API supplied edge-convention K."""
    matrices = np.asarray(native, float).copy()
    if oracle:
        matrices[:, :2, 2] -= 0.5
    sx, sy = np.array(native_wh) / np.array(original_wh)
    transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return np.linalg.inv(transform) @ matrices


def observation_view(extracted, fitted, frame, intrinsic, camera):
    matches = {m["row_index"]: m["candidate_index"] for m in fitted["row_matches"]} if fitted["usable"] else {}
    observations = []
    for index, row in enumerate(extracted["rows"]):
        item = {"xy": [row["guide_x"], row["y"]], "width_px": 0, "state": "unknown"}
        if index in matches:
            candidate = row["candidates"][matches[index]]
            item.update(xy=[candidate["center_x"], row["y"]], width_px=candidate["width"], state="accepted")
        elif row["status"] == "absent":
            low, _, high, _ = row["absence_window_xyxy"]
            item.update(state="absent", negative_window_x=[low, high - 1])
        elif row["status"] == "ambiguous" or (row["candidates"] and not fitted["usable"]):
            item["state"] = "ambiguous"
        observations.append(item)
    return {"view_id": frame["frame_id"], "K_index": intrinsic, "world_to_camera_cv": camera,
            "size_wh": frame["size_wh"], "observations": observations,
            "image_line": [1, -fitted["slope"], -fitted["intercept"]] if fitted["usable"] else None}


def occupancy_candidate(views, settings):
    """Keep the old occupancy rule while changing only the observation/2D-fit package."""
    usable = [v for v in views if v["image_line"] is not None]
    if len(usable) < 3:
        return {"state": "rejected", "segments": [], "rejection_reasons": ["fewer_than_three_image_lines"]}
    try:
        fitted = fit_multiview_line([v["image_line"] for v in usable], [v["K_index"] for v in usable],
                                   [v["world_to_camera_cv"] for v in usable])
        anchor, direction = fitted["anchor"], fitted["direction"]
        parameters = []
        for view in views:
            xy = np.array([o["xy"] for o in view["observations"] if o["state"] == "accepted"]).reshape(-1, 2)
            inverse = np.linalg.inv(homogeneous(view["world_to_camera_cv"]))
            rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(view["K_index"]).T @ inverse[:3, :3].T
            nearest = closest_ray_line_parameters(anchor, direction, np.broadcast_to(inverse[:3, 3], rays.shape), rays)
            parameters.append(nearest["line_t"][nearest["valid"]])
        centers = camera_centers([v["world_to_camera_cv"] for v in views])
        span = np.linalg.norm(centers[:, None] - centers[None], axis=-1).max()
        intervals = support_intervals(parameters, span * settings["bin_width_camera_span_fraction"],
                                      minimum_views=settings["minimum_views"], dilation_bins=settings["dilation_bins"],
                                      minimum_run_bins=settings["minimum_run_bins"])
        segments = anchor + intervals[..., None] * direction
        return {"state": "accepted" if len(segments) else "rejected", "segments": segments, "line": fitted,
                "rejection_reasons": [] if len(segments) else ["no_occupied_interval"],
                "scope": "Legacy occupancy with paired-edge observations; no ray-distance or geometry acceptance gate"}
    except ValueError as error:
        return {"state": "rejected", "segments": [], "rejection_reasons": [str(error)]}


def fit(run_id, protocol_path):
    protocol = read_json(protocol_path)
    selection = read_json(ROOT / protocol["selection_config"])
    bundle = ROOT / "data/inputs" / protocol["input_bundle"]
    inputs = read_json(bundle / "manifest.json")
    old_run = ROOT / ".runtime/experiments" / protocol["baseline_line_run"]
    old_manifest = read_json(old_run / "manifest.json")
    output = ROOT / ".runtime/experiments" / run_id
    output.mkdir(exist_ok=False)
    # Freeze before any pilot extraction. Evaluator settings are sealed too, but
    # no truth file is opened here. A failed result gets to stay a failed result.
    write_json(output / "protocol.json", protocol)
    write_json(output / "selection.json", selection)
    frozen = output / "producer_sources"
    frozen.mkdir()
    paths = [Path(__file__), ROOT / "scripts/thin_pack_gt.py", ROOT / "scripts/run_thin_line_controls.py"]
    paths += [ROOT / "experiments/src/creator_eval" / name for name in
              ("rod_observations.py", "rod_evidence.py", "camera_diagnostics.py", "line_controls.py", "native_diagnostics.py")]
    for path in paths:
        shutil.copy2(path, frozen / path.name)
    record = {"run_id": run_id, "state": "running", "gt_read_in_fit": False, "heldout_read_in_fit": False,
              "oracle_exception": "oracle jobs reuse supplied true cameras, explicitly privileged control",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "jobs": [], "observations": {},
              "protocol_sha256": sha256(output / "protocol.json"), "selection_sha256": sha256(output / "selection.json"),
              "input_manifest_sha256": sha256(bundle / "manifest.json"), "legacy_manifest_sha256": sha256(old_run / "manifest.json"),
              "source_hashes": {p.name: sha256(p) for p in frozen.iterdir()}}
    write_json(output / "manifest.json", record)
    groups, image_sets, extracted_sets = {}, {}, {}
    for case in dict.fromkeys(j["case_id"] for j in protocol["jobs"]):
        group = next(g for g in inputs["groups"] if g["case_id"] == case)
        images, extracted = [], {target: [] for target in protocol["target_aliases"]}
        for frame in group["frames"]:
            path = bundle / frame["rgb"]
            if sha256(path) != frame["sha256"]:
                raise ValueError("RGB identity changed: " + str(path))
            rgb = np.array(Image.open(path).convert("RGB"))
            if list(rgb.shape[1::-1]) != frame["size_wh"]:
                raise ValueError("Actual RGB dimensions differ from manifest")
            images.append(rgb)
            for target in protocol["target_aliases"]:
                xs = selection["line_control"]["guides"][frame["frame_id"]][target]
                ys = selection["line_control"]["guide_y"]
                rows = extract_rod_observations(rgb, np.c_[xs, ys], protocol["observation"])
                fitted = fit_robust_image_line(rows, protocol["line_fit"])
                extracted[target].append({"frame_id": frame["frame_id"], "rgb_sha256": frame["sha256"],
                                          "extracted": rows, "fitted": fitted})
        path = output / (case + "--observations.json")
        write_json(path, clean(extracted))
        record["observations"][case] = {"path": path.name, "sha256": sha256(path)}
        groups[case], image_sets[case], extracted_sets[case] = group, images, extracted
        print("OBSERVATIONS", case, {t: [v["fitted"]["state"] for v in extracted[t]] for t in extracted}, flush=True)
    for job in protocol["jobs"]:
        case, mode = job["case_id"], job["camera_mode"]
        source = ROOT / ".runtime/experiments" / job["basis_run_id"]
        inference = read_json(source / "inference_manifest.json")
        if inference["state"] != "complete" or inference["input_manifest_sha256"] != record["input_manifest_sha256"]:
            raise ValueError("Incomplete inference or different input bundle")
        identity = next(j for j in inference["jobs"] if j["job_id"] == job["job_id"])
        prediction = source / job["job_id"] / "prediction.npz"
        if identity["state"] != "succeeded" or sha256(prediction) != identity["prediction_sha256"]:
            raise ValueError("Native inference identity mismatch")
        report_path = source / job["job_id"] / "report.json"
        if sha256(report_path) != identity["report_sha256"]:
            raise ValueError("Native report changed")
        report = read_json(report_path)
        if report["process_res"] != job["process_res"] or ("oracle_camera" in report["scope"]) != (mode == "oracle_camera"):
            raise ValueError("Native process resolution or camera mode mismatch")
        if [i["sha256"] for i in report["images"]] != [f["sha256"] for f in groups[case]["frames"]]:
            raise ValueError("Native image order or identity mismatch")
        native = np.load(prediction, allow_pickle=False)
        count, height, width = native["depth"].shape
        frames = groups[case]["frames"]
        if count != len(frames) or any(f["size_wh"] != frames[0]["size_wh"] for f in frames):
            raise ValueError("Frame dimensions or count differ")
        expected = {"intrinsics": (count, 3, 3), "extrinsics": (count, 3, 4),
                    "processed_images": (count, height, width, 3)}
        for key, shape in expected.items():
            if native[key].shape != shape or not np.isfinite(native[key]).all():
                raise ValueError("Invalid native array: " + key)
        matrices = original_intrinsics(native["intrinsics"], frames[0]["size_wh"], [width, height], mode == "oracle_camera")
        cameras = native["extrinsics"]
        kind = "oracle" if mode == "oracle_camera" else "full"
        old_id = kind + "--" + job["job_id"]
        old_identity = next(j for j in old_manifest["jobs"] if j["line_job_id"] == old_id)
        old_path = old_run / old_id / "curves.json"
        if sha256(old_path) != old_identity["curves_sha256"]:
            raise ValueError("Legacy fit identity mismatch")
        old_curves = read_json(old_path)
        for old in (old_identity, old_curves):
            if old["prediction_sha256"] != identity["prediction_sha256"] or old["inference_manifest_sha256"] != sha256(source / "inference_manifest.json"):
                raise ValueError("Legacy comparison uses a different base prediction")
        result = {"job": job, "base_prediction_sha256": sha256(prediction), "targets": {},
                  "output_contract": "experimental local finite line proposals; not formal PatchResult or modified point cloud",
                  "camera_policy": "fixed base intrinsics/extrinsics lifted to original RGB centers"}
        result["background_camera_audit"] = background_camera_audit(
            image_sets[case], matrices, cameras,
            [selection["line_control"]["guides"][f["frame_id"]] for f in frames], selection["line_control"]["guide_y"],
            protocol["background_camera_audit"])
        for target in protocol["target_aliases"]:
            views = [observation_view(o["extracted"], o["fitted"], frame, matrices[i], cameras[i])
                     for i, (o, frame) in enumerate(zip(extracted_sets[case][target], frames))]
            legacy = old_curves["targets"][target]["variants"]["rgb_multiview_planes"]
            legacy_segments = legacy.get("curves", {}).get("multiview_supported", [])
            variants = {
                "legacy_occupancy": {"state": "accepted" if len(legacy_segments) else "rejected", "segments": legacy_segments,
                                     "source_curves_sha256": sha256(old_path), "rejection_reasons": [] if len(legacy_segments) else ["empty_legacy_curve"]},
                "paired_occupancy": occupancy_candidate(views, protocol["legacy_occupancy"]),
                "paired_geometry": build_rod_candidate(views, protocol["evidence"], negative_veto=False),
                "paired_full": build_rod_candidate(views, protocol["evidence"], negative_veto=True),
            }
            result["targets"][target] = {"views": views, "variants": variants}
        name = mode + "--" + job["job_id"]
        folder = output / name
        folder.mkdir()
        write_json(folder / "candidates.json", clean(result))
        entry = {**job, "candidate_job_id": name, "prediction_path": str(prediction), "prediction_sha256": sha256(prediction),
                 "inference_manifest_sha256": sha256(source / "inference_manifest.json"),
                 "candidates_sha256": sha256(folder / "candidates.json")}
        record["jobs"].append(entry)
        write_json(output / "manifest.json", record)
        print("CANDIDATES", name, {t: {v: c["state"] for v, c in r["variants"].items()} for t, r in result["targets"].items()}, flush=True)
    record.update(state="complete", completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(output / "manifest.json", record)
    print("FIT_FROZEN", output, flush=True)


if __name__ == "__main__":
    import re
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/rod_evidence_v1.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        parser.error("Invalid run ID")
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "fit":
        fit(args.run_id, args.protocol)
    else:
        from evaluate_rod_evidence import evaluate
        evaluate(args.run_id)
