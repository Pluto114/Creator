"""Separate infer/evaluate stages for the first paired thin-rod baseline diagnostics."""

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import read_json, sha256, write_json
from thin_pack_infer_worker import (
    input_bundle_path,
    rgb_input_path,
    safe_component,
    validate_frame_order,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.native_diagnostics import (
    AlignmentDegenerate,
    align_cameras,
    apply_similarity,
    confidence_masks,
    region_report,
    unproject,
)

ROOT = Path(__file__).resolve().parents[1]


def infer(run_id, protocol_path=None):
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    protocol = read_json(protocol_path or ROOT / "configs/thin_pack_baseline_v1.json")
    if (
        protocol["camera_mode"] != "estimated"
        or protocol["use_ray_pose"]
        or protocol["seed"] != 0
        or protocol["ref_view_strategy"] != "saddle_balanced"
    ):
        raise ValueError("This runner only supports the frozen estimated-camera smoke settings")
    bundle = input_bundle_path(ROOT, protocol["input_bundle"])
    manifest = read_json(bundle / "manifest.json")
    if manifest["known_cameras"]:
        raise ValueError("The ordinary baseline must not receive true cameras")
    if manifest["bundle_id"] != protocol["input_bundle"]:
        raise ValueError("Input bundle identity mismatch")
    group_ids = [safe_component(g["case_id"], "case ID") for g in manifest["groups"]]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Duplicate input groups")
    if not protocol["jobs"]:
        raise ValueError("Empty inference jobs")
    selected_jobs = set()
    for job in protocol["jobs"]:
        safe_component(job["case_id"], "case ID")
        job_key = (job["case_id"], job["process_res"])
        if job["case_id"] not in group_ids or job_key in selected_jobs:
            raise ValueError("Missing or duplicate inference group")
        selected_jobs.add(job_key)
        group = manifest["groups"][group_ids.index(job["case_id"])]
        validate_frame_order(group)
        for frame in group["frames"]:
            rgb_input_path(bundle, frame["rgb"])
    target = ROOT / ".runtime/experiments" / run_id
    target.mkdir(parents=True, exist_ok=False)
    frozen = target / "producer_sources"
    frozen.mkdir()
    for path in [
        Path(__file__),
        ROOT / "scripts/thin_pack_infer_worker.py",
        ROOT / "scripts/smoke_da3.py",
        ROOT / "scripts/thin_pack_gt.py",
        ROOT / "experiments/src/creator_eval/native_diagnostics.py",
        ROOT / "tests/test_native_diagnostics.py",
    ]:
        shutil.copy2(path, frozen / path.name)
    write_json(target / "protocol.json", protocol)
    shutil.copy2(bundle / "manifest.json", target / "input_manifest.json")
    record = {
        "run_id": run_id,
        "state": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_manifest_sha256": sha256(bundle / "manifest.json"),
        "protocol_sha256": sha256(target / "protocol.json"),
        "gt_read_during_inference": False,
        "source_hashes": {p.name: sha256(p) for p in frozen.iterdir()},
        "git_head": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "jobs": [],
    }
    write_json(target / "inference_manifest.json", record)
    for job in protocol["jobs"]:
        group = next(g for g in manifest["groups"] if g["case_id"] == job["case_id"])
        name = f"{job['case_id']}-{job['process_res']}"
        folder = target / name
        folder.mkdir()
        request = {
            "bundle_id": protocol["input_bundle"],
            "model": protocol["model"],
            "process_res": job["process_res"],
            "frames": group["frames"],
        }
        write_json(folder / "request.json", request)
        print("INFER " + name, flush=True)
        with (folder / "inference.log").open("w", encoding="utf-8") as log:
            outcome = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/thin_pack_infer_worker.py"),
                    "--request",
                    str(folder / "request.json"),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        entry = {
            **job,
            "job_id": name,
            "return_code": outcome.returncode,
            "request_sha256": sha256(folder / "request.json"),
        }
        report = read_json(folder / "report.json") if (folder / "report.json").exists() else {}
        entry["state"] = "succeeded" if outcome.returncode == 0 and report.get("ok") else "failed"
        if entry["state"] == "succeeded":
            entry["prediction_sha256"] = sha256(folder / "prediction.npz")
            entry["report_sha256"] = sha256(folder / "report.json")
        record["jobs"].append(entry)
        write_json(target / "inference_manifest.json", record)
        print(f"INFERENCE_{entry['state'].upper()} {name}", flush=True)
    record["state"] = (
        "complete"
        if all(j["state"] == "succeeded" for j in record["jobs"])
        else "complete_with_failures"
    )
    write_json(target / "inference_manifest.json", record)
    print("INFERENCE_READY " + str(target), flush=True)
    if record["state"] != "complete":
        raise SystemExit(1)


def processed_image_check(image, expected):
    # 上游把归一化张量还原为uint8时直接截断，因此不少通道会比resize原图小1。
    # 这里复现这一步，而不是放宽“位置差一点也算对”的图像检查。
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    normalized = (expected.astype(np.float32) / 255 - mean) / std
    restored = np.clip(
        normalized * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1
    )
    visual = (restored * 255).astype(np.uint8)
    difference = np.abs(image.astype(np.int16) - visual.astype(np.int16))
    if difference.max() != 0:
        raise ValueError(f"Processed image mismatch: max={difference.max()}")
    return {
        "native_visualization_max_difference": int(difference.max()),
        "resize_rgb_max_difference": int(
            np.abs(image.astype(np.int16) - expected.astype(np.int16)).max()
        ),
    }


def checked_gt(gt_root, hashes, relative):
    path = gt_root / relative
    if sha256(path) != hashes[relative.as_posix()]["sha256"]:
        raise ValueError("GT artifact identity mismatch: " + str(relative))
    return path


def evaluate_job(run, entry, group, protocol, gt, hashes, output):
    prediction_path = run / entry["job_id"] / "prediction.npz"
    if sha256(prediction_path) != entry["prediction_sha256"]:
        raise ValueError("Prediction changed since inference")
    if sha256(run / entry["job_id"] / "report.json") != entry["report_sha256"]:
        raise ValueError("Inference report identity mismatch")
    data = np.load(prediction_path, allow_pickle=False)
    n, h, w = data["depth"].shape
    if n != len(group["frames"]):
        raise ValueError("View count mismatch")
    cameras = []
    truth = []
    ids = []
    safe = []
    coverage = []
    images = []
    image_checks = []
    for frame in group["frames"]:
        prefix = Path(entry["case_id"]) / frame["frame_id"] / f"da3_{entry['process_res']}"
        camera = read_json(checked_gt(gt, hashes, prefix / "camera.json"))
        if camera["size_wh"] != [w, h]:
            raise ValueError("GT/prediction shape mismatch")
        cameras.append(camera)

        def array(name):
            return np.load(checked_gt(gt, hashes, prefix / f"{name}.npy"), allow_pickle=False)

        truth.append(array("depth_z"))
        ids.append(array("rod_id"))
        safe.append(array("strict_depth_eval_valid"))
        coverage.append(array("segment_coverage_counts").sum(0) > 0)
        image = np.asarray(Image.open(checked_gt(gt, hashes, prefix / "processed_rgb.png")))
        images.append(image)
        image_checks.append(processed_image_check(data["processed_images"][len(images) - 1], image))
    truth = np.stack(truth)
    ids = np.stack(ids)
    safe = np.stack(safe)
    coverage = np.stack(coverage)
    gt_ext = np.array([c["world_to_camera_cv"] for c in cameras])
    masks, threshold = confidence_masks(data["depth"], data["conf"], protocol["filters"])
    # 这里只做相机对齐。杆的位置哪怕错到天边，也不准拿GT杆去偷偷修这个变换。
    try:
        scale, rotation, translation, alignment = align_cameras(
            data["extrinsics"], gt_ext, protocol["alignment"]
        )
        alignment["state"] = "fitted"
        z_error = np.abs(scale * data["depth"] - truth)
        point_error = []
        for index, camera in enumerate(cameras):
            points = unproject(
                data["depth"][index], data["intrinsics"][index], data["extrinsics"][index]
            )
            aligned = apply_similarity(points, scale, rotation, translation)
            correct = unproject(truth[index], np.array(camera["K_index"]), gt_ext[index])
            point_error.append(np.linalg.norm(aligned - correct, axis=-1))
        point_error = np.stack(point_error)
    except AlignmentDegenerate as error:
        alignment = {"state": "degenerate", "reason": str(error)}
        z_error = point_error = None
    regions = {
        "all_pixels": np.ones_like(ids, bool),
        "rod_center_all": ids > 0,
        "strict_rods": (ids > 0) & safe,
        "strict_background": (ids == 0) & safe,
        "mixed_rod_center_diagnostic": (ids > 0) & ~safe,
        "subpixel_coverage_without_rod_center": (ids == 0) & coverage,
    }
    inference = read_json(run / entry["job_id"] / "report.json")
    report = {
        **entry,
        "scope": protocol["scope"],
        "alignment": alignment,
        "default_confidence_threshold": threshold,
        "image_correspondence": image_checks,
        "filter_policies": {},
        "frames": [],
        "cost": {
            k: inference[k]
            for k in [
                "load_seconds",
                "inference_seconds",
                "peak_allocated_mib",
                "peak_reserved_mib",
            ]
        },
        "input_sha256": [f["sha256"] for f in group["frames"]],
    }
    for policy, kept in masks.items():
        report["filter_policies"][policy] = {}
        for name, region in regions.items():
            use_error = name in ("strict_rods", "strict_background", "mixed_rod_center_diagnostic")
            report["filter_policies"][policy][name] = region_report(
                region, kept, z_error if use_error else None, point_error if use_error else None
            )
    for index, frame in enumerate(group["frames"]):
        frame_report = {"frame_id": frame["frame_id"], "policies": {}}
        for policy, kept in masks.items():
            frame_report["policies"][policy] = {"rods": {}, "strict_rods": {}}
            for rod in range(1, 7):
                region = ids[index] == rod
                frame_report["policies"][policy]["rods"][str(rod)] = region_report(
                    region, kept[index]
                )
                frame_report["policies"][policy]["strict_rods"][str(rod)] = region_report(
                    region & safe[index],
                    kept[index],
                    None if z_error is None else z_error[index],
                    None if point_error is None else point_error[index],
                )
        report["frames"].append(frame_report)
    write_json(output / "report.json", report)
    # 留下最直观的一张核对图：模型预测、GT、过滤分别画，别把过滤后的图冒充原预测。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = group["frame_order"].index("view_+00")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].imshow(images[index])
    axes[0, 0].set_title("Paired RGB")
    model_depth = data["depth"][index] * (scale if alignment["state"] == "fitted" else 1)
    limits = np.quantile(truth[index][np.isfinite(truth[index])], [0.02, 0.98])
    axes[0, 1].imshow(model_depth, vmin=limits[0], vmax=limits[1], cmap="viridis")
    axes[0, 1].set_title("Raw model Z, camera-fit scale")
    axes[0, 2].imshow(truth[index], vmin=limits[0], vmax=limits[1], cmap="viridis")
    axes[0, 2].set_title("True center-ray Z")
    axes[1, 0].imshow(ids[index], vmin=0, vmax=6, cmap="tab10")
    axes[1, 0].set_title("Visible rod centers (GT)")
    for ax, policy in zip(axes[1, 1:], ["default_p40", "fixed_1_05"]):
        display = np.zeros((h, w), np.uint8)
        display[ids[index] > 0] = 1
        display[(ids[index] > 0) & masks[policy][index]] = 2
        ax.imshow(display, vmin=0, vmax=2, cmap="viridis")
        ax.set_title(policy + ": teal removed / yellow kept")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle(entry["job_id"] + " | retention is not reconstruction accuracy")
    fig.savefig(output / "diagnostic.png", dpi=125)
    plt.close(fig)
    if sha256(prediction_path) != entry["prediction_sha256"]:
        raise ValueError("Evaluation modified native prediction")
    return report


def evaluate(run_id, evaluation_id=None):
    run = ROOT / ".runtime/experiments" / run_id
    record = read_json(run / "inference_manifest.json")
    if record["state"] not in ("complete", "complete_with_failures"):
        raise ValueError("Inference is not complete")
    protocol = read_json(run / "protocol.json")
    if sha256(run / "protocol.json") != record["protocol_sha256"]:
        raise ValueError("Protocol changed after inference started")
    manifest = read_json(run / "input_manifest.json")
    if sha256(run / "input_manifest.json") != record["input_manifest_sha256"]:
        raise ValueError("Input manifest mismatch")
    gt = ROOT / "data/eval_gt" / protocol["input_bundle"]
    status = read_json(gt / "status.json")
    if (
        status["state"] != "complete"
        or sha256(gt / "artifact_hashes.json") != status["artifact_hashes_sha256"]
    ):
        raise ValueError("GT bundle is incomplete or its hash index changed")
    hashes = read_json(gt / "artifact_hashes.json")
    summary = read_json(checked_gt(gt, hashes, Path("validation_summary.json")))
    if summary["input_manifest_sha256"] != record["input_manifest_sha256"]:
        raise ValueError("GT belongs to a different RGB input bundle")
    output = ROOT / "data/evaluation" / (evaluation_id or run_id)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run / "protocol.json", output / "protocol.json")
    result = {
        "run_id": run_id,
        "evaluation_id": evaluation_id or run_id,
        "scope": protocol["scope"],
        "state": "running",
        "jobs": [],
        "gt_artifact_hashes_sha256": status["artifact_hashes_sha256"],
        "inference_manifest_sha256": sha256(run / "inference_manifest.json"),
        "evaluation_sources": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [Path(__file__), ROOT / "experiments/src/creator_eval/native_diagnostics.py"]
        },
    }
    write_json(output / "summary.json", result)
    for entry in record["jobs"]:
        folder = output / entry["job_id"]
        folder.mkdir()
        if entry["state"] != "succeeded":
            result["jobs"].append({**entry, "evaluation_state": "inference_failed"})
            continue
        group = next(g for g in manifest["groups"] if g["case_id"] == entry["case_id"])
        try:
            report = evaluate_job(run, entry, group, protocol, gt, hashes, folder)
            report["evaluation_state"] = "succeeded"
        except Exception as error:
            report = {**entry, "evaluation_state": "failed", "error": str(error)}
            write_json(folder / "failure.json", report)
        result["jobs"].append(report)
        write_json(output / "summary.json", result)
        print("EVALUATED " + entry["job_id"] + " " + report["evaluation_state"], flush=True)
    result["state"] = (
        "complete"
        if all(j["evaluation_state"] == "succeeded" for j in result["jobs"])
        else "complete_with_failures"
    )
    write_json(output / "summary.json", result)
    print("EVALUATION_READY " + str(output), flush=True)
    if result["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["infer", "evaluate"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, help="Frozen input-only inference protocol")
    parser.add_argument(
        "--evaluation-id", help="New immutable output ID when re-evaluating saved predictions"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        parser.error("Invalid run ID")
    if args.evaluation_id and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.evaluation_id
    ):
        parser.error("Invalid evaluation ID")
    if args.stage == "infer":
        if args.evaluation_id:
            parser.error("--evaluation-id only applies to evaluation")
        infer(args.run_id, args.protocol)
    else:
        if args.protocol:
            parser.error("--protocol only applies to inference")
        evaluate(args.run_id, args.evaluation_id)
