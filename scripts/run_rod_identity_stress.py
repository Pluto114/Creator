"""Frozen oracle-camera guide stress and conditional finite-segment diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from creator_eval.rod_candidate_extent import bound_selected_candidate  # noqa: E402
from creator_eval.rod_multiview_candidates import (  # noqa: E402
    apply_guide_identity_guard,
    associate_multiview_lines,
    enumerate_image_lines,
    image_line_from_endpoints,
)
from creator_eval.rod_observations import extract_rod_observations  # noqa: E402
from run_rod_identity_blender import (  # noqa: E402
    classify,
    digest,
    locations,
    prepare,
    read_json,
    write_json,
)

CONFIG = ROOT / "configs/rod_identity_stress_v2.json"
DEFAULT_RUN_ID = "rod-identity-stress-v2-20260919r4"
SHARE = ROOT / "docs/experiments/results/2026-09-19-identity-stress-v2.json"


def frozen_source_check(run, prepared):
    for relative, expected in prepared["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Source changed since freeze: {relative}")
    if digest(run / "method_config.json") != prepared["method_config_sha256"]:
        raise ValueError("Method changed since freeze")


def reject_truth_open(event, arguments):
    # An auditable tripwire for this process, not an OS security boundary. It
    # catches accidental Path/open/NumPy reads of truth during inference.
    if event != "open" or not isinstance(arguments[0], (str, bytes)):
        return
    path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).resolve()
    for forbidden in (ROOT / "data/eval_gt", ROOT / "data/evaluation"):
        if path.is_relative_to(forbidden):
            raise PermissionError(f"Inference tried to open evaluation truth: {path}")
    if path.name == "protocol.json" or path.name.endswith("render_request.json"):
        raise PermissionError("Inference must not read the generation/evaluation protocol")


def infer(run_id):
    run, inputs, _, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    if prepared["run_id"] != run_id or prepared["state"] != "prepared":
        raise ValueError("Prepared run identity mismatch")
    frozen_source_check(run, prepared)
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Input manifest changed")
    method = read_json(run / "method_config.json")
    manifest = read_json(inputs / "manifest.json")
    # All imports and source verification are done. The inference work below
    # reads only allowed RGB and already-isolated method settings.
    sys.addaudithook(reject_truth_open)
    begin = time.perf_counter()
    rows = []
    for case in manifest["cases"]:
        images = []
        for frame in case["frames"]:
            path = inputs / frame["rgb"]
            if digest(path) != frame["rgb_sha256"]:
                raise ValueError("Frozen RGB changed")
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")))
        for offset in method["guide_offsets_px"]:
            started = time.perf_counter()
            views, raw = [], []
            for frame, rgb in zip(case["frames"], images):
                guide = np.asarray(frame["guide_xyxy"], float).copy()
                guide[:, 0] += offset
                observed = extract_rod_observations(rgb, guide, method["observation"])
                candidates = enumerate_image_lines(observed, method["image_hypotheses"])
                views.append({
                    "view_id": frame["view_id"], "K_index": frame["K_index"],
                    "world_to_camera_cv": frame["world_to_camera_cv"],
                    "size_wh": frame["size_wh"], "y_range": guide[:, 1].tolist(),
                    "guide_line": image_line_from_endpoints(guide),
                    "guide_source": frame["guide_source"], "candidates": candidates,
                })
                raw.append(observed)
            association = associate_multiview_lines(
                views, {**method["association_common"], "minimum_support_views": 4}
            )
            guarded = apply_guide_identity_guard(association, views, method["guide_guard"])
            finite = bound_selected_candidate(guarded, views, raw, method["extent"])
            record = {
                "case_id": case["case_id"], "guide_offset_px": offset,
                "camera_source": "oracle_camera",
                "guide_source": "fixed_screen_coordinates_plus_declared_offset",
                "association": association, "guarded": guarded, "finite": finite,
                "image_candidate_counts": [len(view["candidates"]) for view in views],
                "image_candidate_limit_reached_views": [
                    view["view_id"] for view in views
                    if len(view["candidates"]) == method["image_hypotheses"]["maximum_models"]
                ],
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(record)
            print(case["case_id"], offset, association["state"], guarded["state"],
                  finite["state"], len(finite["segments"]), flush=True)
    write_json(run / "inference.json", {
        "state": "inferred", "run_id": run_id,
        "gt_read_during_inference": False, "truth_read_tripwire_enabled": True,
        "input_manifest_sha256": prepared["input_manifest_sha256"],
        "method_config_sha256": prepared["method_config_sha256"],
        "source_sha256": prepared["source_sha256"],
        "elapsed_seconds": time.perf_counter() - begin, "rows": rows,
    })


def validate_inference_identity(inference, prepared):
    if inference.get("state") != "inferred" or inference.get("gt_read_during_inference") is not False or inference.get("truth_read_tripwire_enabled") is not True:
        raise ValueError("Inference completion or isolation flags failed")
    for key in ("run_id", "source_sha256", "input_manifest_sha256", "method_config_sha256"):
        if inference.get(key) != prepared[key]:
            raise ValueError(f"Inference identity changed: {key}")


def evaluate(config_path, share_path):
    supplied = read_json(config_path)
    run, inputs, truth, output = locations(supplied["run_id"])
    if output.exists() or share_path.exists():
        raise FileExistsError("Keep existing stress evaluation")
    prepared = read_json(run / "prepared.json")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Evaluation protocol changed after freezing")
    frozen_source_check(run, prepared)
    config = read_json(run / "protocol.json")
    policy = config["evaluation"]
    for path, key in ((inputs / "manifest.json", "input_manifest_sha256"),
                      (truth / "manifest.json", "truth_manifest_sha256"),
                      (truth / "artifact_hashes.json", "truth_artifacts_sha256")):
        if digest(path) != prepared[key]:
            raise ValueError(f"Frozen artifact changed: {path}")
    for relative, expected in read_json(truth / "artifact_hashes.json").items():
        if digest(truth / relative) != expected:
            raise ValueError(f"Truth changed: {relative}")
    inference_path = run / "inference.json"
    inference = read_json(inference_path)
    validate_inference_identity(inference, prepared)
    truth_by_case = {}
    for entry in read_json(truth / "manifest.json")["cases"]:
        truth_by_case[entry["case_id"]] = read_json(truth / entry["path"])
    expected_rows = {(case["case_id"], offset) for case in config["cases"] for offset in config["method"]["guide_offsets_px"]}
    actual_rows = [(row["case_id"], row["guide_offset_px"]) for row in inference["rows"]]
    if len(actual_rows) != len(set(actual_rows)) or set(actual_rows) != expected_rows:
        raise ValueError("Missing, duplicate or unexpected planned stress row")
    rows = []
    for record in inference["rows"]:
        target = truth_by_case[record["case_id"]]
        geometry_class, geometry_error = classify(record["association"], target, policy)
        guide_class, _ = classify(record["guarded"], target, policy)
        finite = record["finite"]
        target_segments = target["target"].get("segments", [target["target"]["endpoints"]]) if target["target"]["present"] else []
        curves = None
        # There is deliberately no nearest-rod assignment for unresolved identity.
        if not target["target"].get("identity_ambiguous", False):
            curves = curve_metrics(finite["segments"], target_segments,
                                   tolerance=policy["curve_tolerance_m"], spacing=policy["curve_spacing_m"])
        gap = None
        if target.get("gap_segment"):
            gap = gap_coverage(finite["segments"], target["gap_segment"],
                               tolerance=policy["curve_tolerance_m"], spacing=policy["curve_spacing_m"])
        occlusion = None
        if target.get("occlusion_interval"):
            occlusion = gap_coverage(finite["segments"], target["occlusion_interval"],
                                     tolerance=policy["curve_tolerance_m"], spacing=policy["curve_spacing_m"])
        rows.append({
            "case_id": record["case_id"], "label": target["label"],
            "guide_offset_px": record["guide_offset_px"],
            "geometry_classification": geometry_class,
            "guide_classification": guide_class, "geometry_error": geometry_error,
            "search_complete": record["association"]["search_complete"],
            "combinations": record["association"]["attempted_combination_count"],
            "candidate_limit_reached_views": record["image_candidate_limit_reached_views"],
            "finite_state": finite["state"], "finite_reasons": finite["rejection_reasons"],
            "segment_count": len(finite["segments"]), "segments": finite["segments"],
            "curve_metrics": curves, "gap": gap, "occluded_interval_prediction": occlusion,
            "elapsed_seconds": record["elapsed_seconds"],
        })
    summary = []
    for offset in config["method"]["guide_offsets_px"]:
        subset = [row for row in rows if row["guide_offset_px"] == offset]
        summary.append({
            "guide_offset_px": offset, "case_count": len(subset),
            "geometry_counts": dict(Counter(row["geometry_classification"] for row in subset)),
            "guide_counts": dict(Counter(row["guide_classification"] for row in subset)),
            "finite_state_counts": dict(Counter(row["finite_state"] for row in subset)),
        })
    ray_checks = [frame["blender_ray_check"] for case in truth_by_case.values() for frame in case["frames"]]
    result = {
        "state": "complete", "run_id": config["run_id"], "scope": config["scope"],
        "protocol_sha256": prepared["protocol_sha256"], "inference_sha256": digest(inference_path),
        "elapsed_seconds": inference["elapsed_seconds"], "summaries": summary, "rows": rows,
        "independent_blender_checks": {
            "frames": len(ray_checks), "ray_count": sum(row["samples"] for row in ray_checks),
            "surface_id_mismatches": sum(row["surface_id_mismatches"] for row in ray_checks),
            "max_z_error_m": max(row["max_z_error_m"] for row in ray_checks),
            "projection_max_px": max(frame["blender_projection_error_px"] for case in truth_by_case.values() for frame in case["frames"]),
        },
        "limitations": [
            "Exact oracle cameras; fixed synthetic screen guides are not measured human annotations.",
            "New configurations in the same procedural cylinder family are not independent object generalization.",
            "Search completeness covers retained image hypotheses; RANSAC and the image candidate cap still limit recall.",
            "Finite segments are conditional RGB evidence diagnostics, not a formal patch or a comparison against DA3.",
            "Center rays validate geometry conventions, not subpixel RGB antialias coverage.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share_path, result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--share-json", type=Path, default=SHARE)
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.config, args.share_json)
