"""Evaluate explicit full-frame oracle-camera controls without rescaling saved model depth."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from run_thin_pack_baseline import checked_gt, processed_image_check
from thin_pack_gt import read_json, sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.native_diagnostics import (
    camera_centers,
    confidence_masks,
    fit_sim3,
    region_report,
    unproject,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "thin-oracle-controls-v1-20260914"
INPUT_BUNDLE = "thin-pack-v2-20260914r1"


def pixel_center_intrinsics(native_intrinsics):
    result = np.asarray(native_intrinsics, dtype=np.float64).copy()
    result[..., :2, 2] -= 0.5
    return result


def known_answer_checks():
    # A fake plane is enough to catch the tempting second scale multiplication and
    # half-pixel shortcut. The model's native K must survive this check unchanged.
    depth = np.full((2, 4), 4.0)
    edge_k = np.array([[10.0, 0, 2], [0, 10.0, 1], [0, 0, 1]])
    original = edge_k.copy()
    ext = np.array([[0.0, -1, 0, 2], [1, 0, 0, -3], [0, 0, 1, 1]])
    physical = unproject(depth, pixel_center_intrinsics(edge_k), ext)
    native = unproject(depth, edge_k, ext)
    y, x = np.mgrid[:2, :4]
    camera = np.stack([(x + 0.5 - 2) / 10 * 4, (y + 0.5 - 1) / 10 * 4, depth], axis=-1)
    expected = (camera - ext[:, 3]) @ ext[:, :3]
    np.testing.assert_allclose(physical, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(native - expected, axis=-1), np.sqrt(0.08), atol=1e-12)
    np.testing.assert_array_equal(edge_k, original)
    return {"physical_ray_with_translated_rotated_camera": True, "native_K_unchanged": True}


def point_errors(depth, native_k, returned_ext, truth, cameras):
    physical_errors, native_errors, gt_ray_checks = [], [], []
    physical_k = pixel_center_intrinsics(native_k)
    for index, camera in enumerate(cameras):
        target_k = np.array(camera["K_index"])
        target_ext = np.array(camera["world_to_camera_cv"])
        correct = unproject(truth[index], target_k, target_ext)
        # Both tracks use API-returned metric Z and poses directly. A further Sim3
        # here would silently fit away a failure in the oracle scale restoration.
        native = unproject(depth[index], native_k[index], returned_ext[index])
        physical = unproject(depth[index], physical_k[index], returned_ext[index])
        native_errors.append(np.linalg.norm(native - correct, axis=-1))
        physical_errors.append(np.linalg.norm(physical - correct, axis=-1))
        same_depth = unproject(truth[index], physical_k[index], returned_ext[index])
        valid = np.isfinite(correct).all(axis=-1)
        gt_ray_checks.append(float(np.max(np.linalg.norm(same_depth[valid] - correct[valid], axis=-1))))
    if max(gt_ray_checks) > 1e-4:
        raise ValueError("Oracle K/pose and GT pixel-center ray correspondence disagree")
    return np.stack(physical_errors), np.stack(native_errors), gt_ray_checks


def raw_camera_audit(raw, returned, gt_ext, inference):
    source = camera_centers(gt_ext)
    target = camera_centers(raw["extrinsics"])
    scale, rotation, translation, _ = fit_sim3(source, target)
    reported = inference["oracle_audit"]["raw_input_to_predicted_camera_scale"]
    if not np.isclose(scale, reported, rtol=2e-5, atol=1e-8):
        raise ValueError("Independent input-to-raw-camera scale differs from inference audit")
    # This inverse is ONLY a camera audit. It is never passed to point_errors.
    restored_centers = ((target - translation) @ rotation) / scale
    residual = np.linalg.norm(restored_centers - source, axis=-1)
    returned_residual = np.linalg.norm(camera_centers(returned["extrinsics"]) - source, axis=-1)
    factor = inference["oracle_audit"]["raw_depth_to_input_meter_multiplier"]
    if not np.isclose(factor * reported, 1.0, rtol=1e-10):
        raise ValueError("Inconsistent recorded raw depth multiplier")
    expected_depth = raw["depth"] / reported
    depth_residual = float(np.max(np.abs(returned["depth"] - expected_depth)))
    if depth_residual > 1e-5:
        raise ValueError("API returned depth no longer agrees with its recorded scale restoration")
    return {
        "scope": "raw predicted camera check only; this fit never transforms evaluated depth or points",
        "fit_direction": "input camera centers to raw predicted camera centers; same as actual API",
        "raw_camera_fit_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "raw_camera_residuals_m": residual.tolist(),
        "independent_input_to_raw_scale": scale,
        "raw_input_to_predicted_camera_scale_from_inference_report": reported,
        "raw_depth_to_input_meter_multiplier_from_inference_report": factor,
        "returned_depth_vs_raw_divide_reported_scale_max_abs_m": depth_residual,
        "returned_camera_rmse_m": float(np.sqrt(np.mean(returned_residual**2))),
        "api_replaced_returned_extrinsics_with_input": True,
        "returned_camera_note": "Near-zero returned pose error comes from API replacement, not camera prediction accuracy.",
        "point_or_depth_alignment_applied_in_evaluation": False,
    }


def region_scores(regions, masks, z_error, point_error):
    result = {}
    for policy, kept in masks.items():
        result[policy] = {}
        for name, region in regions.items():
            use_error = name in ("strict_rods", "strict_background", "mixed_rod_center_diagnostic")
            result[policy][name] = region_report(
                region, kept, z_error if use_error else None, point_error if use_error else None
            )
    return result


def frame_scores(group, ids, safe, masks, z_error, point_error):
    result = []
    for index, frame in enumerate(group["frames"]):
        report = {"frame_id": frame["frame_id"], "policies": {}}
        for policy, kept in masks.items():
            report["policies"][policy] = {"rods": {}, "strict_rods": {}}
            for rod in range(1, 7):
                region = ids[index] == rod
                report["policies"][policy]["rods"][str(rod)] = region_report(region, kept[index])
                report["policies"][policy]["strict_rods"][str(rod)] = region_report(
                    region & safe[index], kept[index], z_error[index], point_error[index]
                )
        result.append(report)
    return result


def diagnostic_figure(output, entry, group, data, truth, ids, masks, images):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = group["frame_order"].index("view_+00")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].imshow(images[index])
    axes[0, 0].set_title("Paired original full-frame RGB")
    limits = np.quantile(truth[index][np.isfinite(truth[index])], [0.02, 0.98])
    axes[0, 1].imshow(data["depth"][index], vmin=limits[0], vmax=limits[1], cmap="viridis")
    axes[0, 1].set_title("Oracle API native Z, already metric")
    axes[0, 2].imshow(truth[index], vmin=limits[0], vmax=limits[1], cmap="viridis")
    axes[0, 2].set_title("True center-ray Z")
    axes[1, 0].imshow(ids[index], vmin=0, vmax=6, cmap="tab10")
    axes[1, 0].set_title("Visible rod centers (GT)")
    for axis, policy in zip(axes[1, 1:], ["default_p40", "fixed_1_05"]):
        display = np.zeros_like(ids[index], dtype=np.uint8)
        display[ids[index] > 0] = 1
        display[(ids[index] > 0) & masks[policy][index]] = 2
        axis.imshow(display, vmin=0, vmax=2, cmap="viridis")
        axis.set_title(policy + ": teal removed / yellow kept")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle(entry["job_id"] + " | known-camera diagnostic; retention is not accuracy")
    fig.savefig(output / "diagnostic.png", dpi=125)
    plt.close(fig)


def evaluate_job(run, entry, group, protocol, gt, hashes, output):
    folder = run / entry["job_id"]
    artifact_paths = {name: folder / name for name in ("prediction.npz", "report.json", "request.json")}
    for name, key in (("prediction.npz", "prediction_sha256"), ("report.json", "report_sha256"), ("request.json", "request_sha256")):
        if sha256(artifact_paths[name]) != entry[key]:
            raise ValueError("Inference artifact identity mismatch: " + name)
    inference = read_json(artifact_paths["report.json"])
    request = read_json(artifact_paths["request.json"])
    if inference.get("camera_mode") != "oracle_camera" or not inference["oracle_audit"].get("api_replaced_returned_extrinsics_with_input"):
        raise ValueError("Expected explicitly audited oracle-camera inference")
    for name, key in (("prediction.npz", "prediction_sha256"), ("request.json", "request_sha256")):
        if inference[key] != entry[key]:
            raise ValueError("Inference report and manifest artifact identities disagree: " + name)
    camera_file = Path(request["camera_file"])
    if camera_file.resolve() != (folder / "oracle_cameras.json").resolve():
        raise ValueError("Unexpected oracle camera artifact path")
    camera_hash = sha256(camera_file)
    if camera_hash != request["camera_sha256"] or camera_hash != inference["camera_sha256"]:
        raise ValueError("Oracle camera artifact changed")
    camera_input = read_json(camera_file)
    raw_path = folder / "pre_alignment_prediction.npz"
    if sha256(raw_path) != inference["pre_alignment_prediction_sha256"]:
        raise ValueError("Pre-alignment prediction changed")
    with np.load(artifact_paths["prediction.npz"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    n, h, w = data["depth"].shape
    if (n, h, w) != (5, 280, 504) or entry["process_res"] != 504:
        raise ValueError("This control expects five full-frame da3_504 predictions")
    for key in ("depth", "conf", "intrinsics", "extrinsics"):
        if not np.isfinite(data[key]).all():
            raise ValueError("Nonfinite native prediction: " + key)
    consumed_gt = {}

    def checked(relative):
        path = checked_gt(gt, hashes, relative)
        consumed_gt[relative.as_posix()] = hashes[relative.as_posix()]["sha256"]
        return path

    cameras, truth, ids, safe, coverage, images, image_checks = [], [], [], [], [], [], []
    for index, frame in enumerate(group["frames"]):
        source_path = ROOT / "data/inputs" / INPUT_BUNDLE / frame["rgb"]
        supplied = request["frames"][index]
        if Path(supplied["rgb"]).resolve() != source_path.resolve() or supplied["sha256"] != frame["sha256"] or sha256(source_path) != frame["sha256"]:
            raise ValueError("Source RGB or frame order differs from the paired input manifest")
        if inference["images"][index]["sha256"] != frame["sha256"]:
            raise ValueError("Inference report image identity mismatch")
        full_prefix = Path(entry["case_id"]) / frame["frame_id"]
        full_camera = read_json(checked(full_prefix / "camera.json"))
        supplied_camera = camera_input["frames"][index]
        if supplied_camera["rgb_sha256"] != frame["sha256"]:
            raise ValueError("Oracle camera input has wrong frame identity")
        for key in ("K_edge", "world_to_camera_cv"):
            np.testing.assert_allclose(supplied_camera[key], full_camera[key], atol=1e-10, rtol=0)
        prefix = full_prefix / "da3_504"
        camera = read_json(checked(prefix / "camera.json"))
        if camera["size_wh"] != [w, h]:
            raise ValueError("GT/prediction shape mismatch")
        np.testing.assert_allclose(data["extrinsics"][index], np.array(camera["world_to_camera_cv"])[:3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(data["intrinsics"][index], camera["K_edge"], atol=2e-4, rtol=0)
        cameras.append(camera)

        def array(name):
            return np.load(checked(prefix / f"{name}.npy"), allow_pickle=False)

        truth.append(array("depth_z"))
        ids.append(array("rod_id"))
        safe.append(array("strict_depth_eval_valid"))
        coverage.append(array("segment_coverage_counts").sum(0) > 0)
        with Image.open(checked(prefix / "processed_rgb.png")) as image:
            image = np.array(image)
        images.append(image)
        image_checks.append(processed_image_check(data["processed_images"][index], image))
    truth, ids, safe, coverage = map(np.stack, (truth, ids, safe, coverage))
    gt_ext = np.array([camera["world_to_camera_cv"] for camera in cameras])
    physical_error, native_error, ray_checks = point_errors(data["depth"], data["intrinsics"], data["extrinsics"], truth, cameras)
    z_error = np.abs(data["depth"] - truth)
    masks, threshold = confidence_masks(data["depth"], data["conf"], protocol["filters"])
    regions = {
        "all_pixels": np.ones_like(ids, bool),
        "rod_center_all": ids > 0,
        "strict_rods": (ids > 0) & safe,
        "strict_background": (ids == 0) & safe,
        "mixed_rod_center_diagnostic": (ids > 0) & ~safe,
        "subpixel_coverage_without_rod_center": (ids == 0) & coverage,
    }
    report = {
        **entry,
        "scope": "Known-camera oracle diagnostic on original full frames; not an RGB-only recovery benchmark",
        "camera_mode": "oracle_camera",
        "primary_pixel_policy": "physical_pixel_centers: copied native K with cx/cy minus 0.5, integer grid",
        "point_or_depth_alignment_applied_in_evaluation": False,
        "metric_note": "scaled_camera_z_abs_error_m keeps the baseline field name; only upstream API scale restoration occurred. No evaluation scale fitting. Mixed pixels are diagnostics, not single-surface accuracy.",
        "raw_predicted_camera_audit": raw_camera_audit(raw, data, gt_ext, inference),
        "oracle_inference_audit": inference["oracle_audit"],
        "default_confidence_threshold": threshold,
        "image_correspondence": image_checks,
        "gt_depth_physical_ray_roundtrip_max_displacement_m": ray_checks,
        "filter_policies": region_scores(regions, masks, z_error, physical_error),
        "frames": frame_scores(group, ids, safe, masks, z_error, physical_error),
        "native_integer_K_diagnostic": {
            "scope": "Unmodified returned K with integer grid; preserves upstream convention, not the physical pixel-center primary score",
            "filter_policies": region_scores(regions, masks, z_error, native_error),
            "frames": frame_scores(group, ids, safe, masks, z_error, native_error),
        },
        "cost": {key: inference[key] for key in ("load_seconds", "inference_seconds", "peak_allocated_mib", "peak_reserved_mib")},
        "input_sha256": [frame["sha256"] for frame in group["frames"]],
        "camera_file_sha256": camera_hash,
        "pre_alignment_prediction_sha256": inference["pre_alignment_prediction_sha256"],
        "used_gt_artifact_sha256": consumed_gt,
    }
    diagnostic_figure(output, entry, group, data, truth, ids, masks, images)
    if sha256(artifact_paths["prediction.npz"]) != entry["prediction_sha256"] or sha256(raw_path) != inference["pre_alignment_prediction_sha256"]:
        raise ValueError("Evaluation changed a native or pre-alignment prediction")
    report["native_prediction_hashes_unchanged_after_evaluation"] = True
    write_json(output / "report.json", report)
    return report


def evaluate(evaluation_id, run_id=RUN_ID):
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    run = ROOT / ".runtime/experiments" / run_id
    record_path = run / "inference_manifest.json"
    record = read_json(record_path)
    if record["run_id"] != run_id:
        raise ValueError("Run id and inference manifest disagree")
    if record["state"] != "complete":
        raise ValueError("Wait for the complete oracle inference manifest before evaluating")
    if sha256(run / "protocol.json") != record["protocol_sha256"]:
        raise ValueError("Frozen controls protocol changed")
    protocol = read_json(run / "protocol.json")
    expected_jobs = {case + "-504" for case in protocol["oracle_cases"]}
    if len(record["jobs"]) != len(expected_jobs) or {entry["job_id"] for entry in record["jobs"]} != expected_jobs:
        raise ValueError("Oracle inference job set differs from the frozen protocol")
    for name, digest in record["source_hashes"].items():
        if sha256(run / "producer_sources" / name) != digest:
            raise ValueError("Frozen oracle producer source changed")
    manifest_path = ROOT / "data/inputs" / INPUT_BUNDLE / "manifest.json"
    if sha256(manifest_path) != record["input_manifest_sha256"]:
        raise ValueError("Original full-frame input manifest changed")
    manifest = read_json(manifest_path)
    gt = ROOT / "data/eval_gt" / INPUT_BUNDLE
    status = read_json(gt / "status.json")
    gt_index_hash = sha256(gt / "artifact_hashes.json")
    if status["state"] != "complete" or gt_index_hash != status["artifact_hashes_sha256"] or gt_index_hash != record["gt_artifact_hashes_sha256"]:
        raise ValueError("Original full-frame GT bundle or index changed")
    hashes = read_json(gt / "artifact_hashes.json")
    gt_summary = read_json(checked_gt(gt, hashes, Path("validation_summary.json")))
    if gt_summary["input_manifest_sha256"] != record["input_manifest_sha256"]:
        raise ValueError("GT and inference refer to different original RGB bundles")
    checks = known_answer_checks()
    output = ROOT / "data/evaluation" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run / "protocol.json", output / "protocol.json")
    sources = [Path(__file__), ROOT / "scripts/run_thin_pack_baseline.py", ROOT / "scripts/thin_pack_gt.py", ROOT / "experiments/src/creator_eval/native_diagnostics.py"]
    result = {
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "scope": "Oracle-camera-only diagnostics; primary uses physical pixel centers, native integer-K separate",
        "input_bundle": INPUT_BUNDLE,
        "shared_protocol_input_bundle_note": "Shared controls protocol names cropped RGB inputs; this oracle subset is explicitly bound to original full-frame input_manifest_sha256.",
        "state": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "jobs": [],
        "known_answer_checks": checks,
        "gt_artifact_hashes_sha256": gt_index_hash,
        "gt_validation_summary_sha256": hashes["validation_summary.json"]["sha256"],
        "inference_manifest_sha256": sha256(record_path),
        "input_manifest_sha256": record["input_manifest_sha256"],
        "evaluation_sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    write_json(output / "summary.json", result)
    for entry in record["jobs"]:
        folder = output / entry["job_id"]
        folder.mkdir()
        try:
            if entry["state"] != "succeeded":
                raise ValueError("Inference job was not successful")
            group = next(group for group in manifest["groups"] if group["case_id"] == entry["case_id"])
            report = evaluate_job(run, entry, group, protocol, gt, hashes, folder)
            report["evaluation_state"] = "succeeded"
        except Exception as error:
            report = {**entry, "evaluation_state": "failed", "error": str(error)}
            write_json(folder / "failure.json", report)
        result["jobs"].append(report)
        write_json(output / "summary.json", result)
        print("ORACLE_EVALUATED", entry["job_id"], report["evaluation_state"], flush=True)
    result["state"] = "complete" if all(job["evaluation_state"] == "succeeded" for job in result["jobs"]) else "complete_with_failures"
    if sha256(record_path) != result["inference_manifest_sha256"]:
        raise ValueError("Inference manifest changed during evaluation")
    write_json(output / "summary.json", result)
    print("ORACLE_EVALUATION_READY", output, flush=True)
    if result["state"] != "complete":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID, help="Completed oracle inference run to read")
    parser.add_argument("--evaluation-id", help="New output id; defaults to the input run id")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    for value in (args.run_id, args.evaluation_id):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
            parser.error("Invalid run or evaluation id")
    if args.self_test:
        print(known_answer_checks())
    else:
        evaluate(args.evaluation_id or args.run_id, args.run_id)


if __name__ == "__main__":
    main()
