"""Prepare, infer, and evaluate new paired Blender rod-identity layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_multiview_candidates import (  # noqa: E402
    apply_nested_band_guard,
    associate_multiview_lines,
    enumerate_image_lines,
)
from creator_eval.rod_observations import extract_rod_observations  # noqa: E402

CONFIG = ROOT / "configs/rod_identity_blender_v1.json"
DEFAULT_RUN_ID = "rod-identity-blender-v1-20260919r3"
SHARE = ROOT / "docs/experiments/results/2026-09-19-blender-identity-r3.json"
SOURCES = [
    "scripts/run_rod_identity_blender.py",
    "scripts/blender_rod_identity_pack.py",
    "scripts/blender_export_thin_pack.py",
    "experiments/src/creator_eval/rod_multiview_candidates.py",
    "experiments/src/creator_eval/rod_observations.py",
    "experiments/src/creator_eval/line_controls.py",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_run_id(run_id):
    if not run_id.replace("-", "").replace("_", "").isalnum() or len(run_id) > 96:
        raise ValueError("Invalid identity run ID")
    return run_id


def locations(run_id):
    run_id = validate_run_id(run_id)
    return (
        ROOT / ".runtime/experiments" / run_id,
        ROOT / "data/inputs" / run_id,
        ROOT / "data/eval_gt" / run_id,
        ROOT / "data/evaluation" / run_id,
    )


def raster_center_truth(rays, camera):
    width, height = camera["size_wh"]
    yy, xx = np.mgrid[:height, :width]
    uv_edge = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
    depth, surface, distance = rays.cast(camera, uv_edge)
    shape = (height, width)
    return {
        "depth_z": depth.reshape(shape),
        "surface_id": surface.reshape(shape),
        "ray_distance": distance.reshape(shape),
        "target_visible": (surface.reshape(shape) == 1),
    }


def validate_render_manifest(render, config, inputs, truth):
    """Require the entire frozen render plan before publishing prepared inputs."""
    planned_ids = [case["case_id"] for case in config["cases"]]
    rendered_ids = [case["case_id"] for case in render["cases"]]
    if not planned_ids or len(planned_ids) != len(set(planned_ids)):
        raise ValueError("Render protocol needs unique nonempty case IDs")
    if len(rendered_ids) != len(set(rendered_ids)) or set(rendered_ids) != set(planned_ids):
        raise ValueError("Rendered cases are missing, duplicate, or unexpected")
    angles = config["generator"]["angles_degrees"]
    if not angles or any(isinstance(angle, bool) or not isinstance(angle, int) for angle in angles):
        raise ValueError("Render protocol needs integer camera angles")
    expected_frames = {f"view_{angle:+03d}": angle for angle in angles}
    if len(expected_frames) != len(angles):
        raise ValueError("Render protocol needs unique camera angles")
    size = config["generator"]["size_wh"]

    def existing_artifact(root, relative):
        root = Path(root).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Rendered artifact leaves its assigned folder: {relative}")
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing or empty rendered artifact: {path}")
        return path

    for case in render["cases"]:
        case_id = case["case_id"]
        geometry_path = existing_artifact(truth, f"{case_id}/geometry.json")
        existing_artifact(truth, f"{case_id}/mesh.npz")
        if read_json(geometry_path) != case["geometry"]:
            raise ValueError(f"Rendered geometry record disagrees with its file: {case_id}")
        frame_ids = [frame["frame_id"] for frame in case["frames"]]
        if len(frame_ids) != len(set(frame_ids)) or set(frame_ids) != set(expected_frames):
            raise ValueError(f"Rendered frames are missing, duplicate, or unexpected: {case_id}")
        for frame in case["frames"]:
            frame_id = frame["frame_id"]
            if frame["angle_degrees"] != expected_frames[frame_id]:
                raise ValueError(f"Rendered frame angle disagrees with protocol: {case_id}/{frame_id}")
            # A surviving fourth view is enough for the algorithm, but not enough
            # for a five-view experiment. Also forbid reusing another frame's PNG.
            rgb_relative = f"{case_id}/{frame_id}.png"
            if frame["rgb"] != rgb_relative:
                raise ValueError(f"Rendered RGB does not belong to its case/frame: {case_id}/{frame_id}")
            rgb = existing_artifact(inputs, rgb_relative)
            with Image.open(rgb) as image:
                if list(image.size) != size:
                    raise ValueError(f"Rendered RGB dimensions disagree with protocol: {rgb_relative}")
                image.verify()
            camera_path = existing_artifact(truth, f"{case_id}/{frame_id}-camera.json")
            camera = read_json(camera_path)
            if camera != frame["camera"] or camera["size_wh"] != size:
                raise ValueError(f"Rendered camera record disagrees with its file/protocol: {case_id}/{frame_id}")


def prepare(config_path=CONFIG):
    # Keep Open3D and OpenCV out of scoring-only imports and lightweight CI.
    from thin_pack_gt import MeshRays, check_blender_rays

    config = read_json(config_path)
    run, inputs, truth, evaluation = locations(config["run_id"])
    for path in (run, inputs, truth, evaluation):
        if path.exists():
            raise FileExistsError(f"Keep existing output: {path}")
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    truth.mkdir(parents=True)
    (run / "source_snapshot").mkdir()
    shutil.copy2(config_path, run / "protocol.json")
    source_hashes = {}
    for relative in dict.fromkeys(SOURCES + config.get("additional_sources", [])):
        source_hashes[relative] = digest(ROOT / relative)
        shutil.copy2(ROOT / relative, run / "source_snapshot" / Path(relative).name)
    write_json(run / "method_config.json", config["method"])
    request = {
        "root": str(ROOT),
        "inputs": inputs.relative_to(ROOT).as_posix(),
        "truth": truth.relative_to(ROOT).as_posix(),
        "config": config,
    }
    write_json(run / "render_request.json", request)
    blender = Path(r"D:\CloudMusic\steam\steamapps\common\Blender\blender.exe")
    environment = os.environ.copy()
    for setting in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
        profile = ROOT / ".local/blender-profile" / setting.lower()
        profile.mkdir(parents=True, exist_ok=True)
        environment["BLENDER_USER_" + setting] = str(profile)
    # Long EEVEE batches exited inside Blender on this machine. Isolate each
    # case so the renderer releases its state between scenes; keep every log.
    requests = [(run / "render_request.json", "blender.log")]
    if config["generator"].get("render_case_isolation", False):
        requests = []
        for index, case in enumerate(config["cases"]):
            case_request = {
                **request, "config": {**config, "cases": [case]},
                "render_manifest_name": f"render_manifest-{index:03d}.json",
            }
            request_path = run / f"case-{index:03d}-render_request.json"
            write_json(request_path, case_request)
            requests.append((request_path, f"blender-{index:03d}.log"))
    for request_path, log_name in requests:
        command = [
            str(blender), "--background", "--factory-startup", "--disable-autoexec",
            "--python-exit-code", "1", "--python",
            str(ROOT / "scripts/blender_rod_identity_pack.py"), "--", str(request_path),
        ]
        with (run / log_name).open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode:
            raise RuntimeError(f"Blender identity render failed (exit {completed.returncode}); see preserved {log_name}")
        print("RENDER_BATCH", request_path.name, "complete", flush=True)
    if config["generator"].get("render_case_isolation", False):
        batches = [read_json(truth / f"render_manifest-{index:03d}.json")
                   for index in range(len(config["cases"]))]
        write_json(truth / "render_manifest.json", {
            **batches[0], "cases": [case for batch in batches for case in batch["cases"]],
            "render_case_isolation": True,
        })
    render = read_json(truth / "render_manifest.json")
    validate_render_manifest(render, config, inputs, truth)
    config_cases = {case["case_id"]: case for case in config["cases"]}
    input_cases, truth_cases = [], []
    for case in render["cases"]:
        case_id = case["case_id"]
        declared = config_cases[case_id]
        geometry_path = truth / case_id / "geometry.json"
        mesh_path = truth / case_id / "mesh.npz"
        # Loading the same triangle mesh for every view was needlessly slow.
        # One immutable ray caster per case is enough for all five cameras.
        rays = MeshRays(mesh_path)
        frames = []
        truth_frames = []
        for frame in case["frames"]:
            rgb_path = inputs / frame["rgb"]
            if list(Image.open(rgb_path).size) != config["generator"]["size_wh"]:
                raise ValueError("Unexpected render dimensions")
            native = truth / case_id / frame["frame_id"] / "native"
            native.mkdir(parents=True)
            arrays = raster_center_truth(rays, frame["camera"])
            ray_check = None
            if frame.get("blender_ray_probes") is not None:
                ray_check = check_blender_rays(rays, frame["camera"], frame["blender_ray_probes"], 0.0001)
            array_records = {}
            for name, array in arrays.items():
                path = native / f"{name}.npy"
                np.save(path, array, allow_pickle=False)
                array_records[name] = {
                    "path": path.relative_to(truth).as_posix(),
                    "sha256": digest(path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            frames.append(
                {
                    "view_id": frame["frame_id"],
                    "rgb": frame["rgb"],
                    "rgb_sha256": digest(rgb_path),
                    "K_index": frame["camera"]["K_index"],
                    "world_to_camera_cv": frame["camera"]["world_to_camera_cv"],
                    "guide_xyxy": frame["guide_xyxy"],
                    "guide_source": frame.get("guide_source", "synthetic_world_guide_projection"),
                    "camera_source": "oracle_camera",
                    "size_wh": frame["camera"]["size_wh"],
                }
            )
            truth_frames.append(
                {"view_id": frame["frame_id"], "camera_path": f"{frame['frame_id']}-camera.json",
                 "arrays": array_records, "target_visible_pixels": int(arrays["target_visible"].sum()),
                 "blender_ray_check": ray_check,
                 "blender_projection_error_px": frame.get("projection_check_max_px")}
            )
        input_cases.append({"case_id": case_id, "frames": frames})
        truth_record = {
            "case_id": case_id,
            "split": declared["split"],
            "label": declared["label"],
            "object_group_id": declared.get("object_group_id", "procedural-cylinder-family-identity-v1"),
            "target": declared["target"],
            "gap_segment": declared.get("gap_segment"),
            "occlusion_interval": declared.get("occlusion_interval"),
            "geometry_path": geometry_path.relative_to(truth).as_posix(),
            "geometry_sha256": digest(geometry_path),
            "mesh_path": mesh_path.relative_to(truth).as_posix(),
            "mesh_sha256": digest(mesh_path),
            "frames": truth_frames,
        }
        case_truth_path = truth / case_id / "truth.json"
        write_json(case_truth_path, truth_record)
        truth_cases.append(
            {"case_id": case_id, "path": case_truth_path.relative_to(truth).as_posix(),
             "sha256": digest(case_truth_path)}
        )
    write_json(
        inputs / "manifest.json",
        {"run_id": config["run_id"], "cases": input_cases,
         "method_inputs": "rendered RGB, exact supplied cameras, coarse guides",
         "truth_excluded": True},
    )
    write_json(
        truth / "manifest.json",
        {"run_id": config["run_id"], "cases": truth_cases,
         "render_manifest_sha256": digest(truth / "render_manifest.json")},
    )
    artifacts = {
        path.relative_to(truth).as_posix(): digest(path)
        for path in sorted(truth.rglob("*")) if path.is_file()
    }
    write_json(truth / "artifact_hashes.json", artifacts)
    write_json(
        run / "prepared.json",
        {
            "state": "prepared", "run_id": config["run_id"],
            "protocol_sha256": digest(run / "protocol.json"),
            "method_config_sha256": digest(run / "method_config.json"),
            "input_manifest_sha256": digest(inputs / "manifest.json"),
            "truth_manifest_sha256": digest(truth / "manifest.json"),
            "truth_artifacts_sha256": digest(truth / "artifact_hashes.json"),
            "source_sha256": source_hashes,
            "split_frozen_before_inference": True,
        },
    )
    print("PREPARED_IDENTITY", len(input_cases), "cases", sum(len(c["frames"]) for c in input_cases), "views")


def infer(run_id=DEFAULT_RUN_ID):
    run, inputs, _, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    if prepared["state"] != "prepared" or prepared["run_id"] != run_id:
        raise ValueError("Prepared identity mismatch")
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Identity inputs changed")
    method = read_json(run / "method_config.json")
    if digest(run / "method_config.json") != prepared["method_config_sha256"]:
        raise ValueError("Identity method config changed")
    for relative, expected in prepared["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen source changed; create a new run: {relative}")
    manifest = read_json(inputs / "manifest.json")
    case_results = []
    for case in manifest["cases"]:
        views = []
        for frame in case["frames"]:
            rgb_path = inputs / frame["rgb"]
            if digest(rgb_path) != frame["rgb_sha256"]:
                raise ValueError("Identity RGB changed")
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            observations = extract_rod_observations(
                rgb, frame["guide_xyxy"], method["observation"]
            )
            candidates = enumerate_image_lines(observations, method["image_hypotheses"])
            views.append(
                {
                    "view_id": frame["view_id"],
                    "K_index": frame["K_index"],
                    "world_to_camera_cv": frame["world_to_camera_cv"],
                    "y_range": [frame["guide_xyxy"][0][1], frame["guide_xyxy"][1][1]],
                    "candidates": candidates,
                    "row_state_counts": {
                        state: sum(row["status"] == state for row in observations["rows"])
                        for state in ("observed", "ambiguous", "absent", "unknown")
                    },
                }
            )
        association_cache = {}
        variants = []
        for variant in method["variants"]:
            minimum = variant["minimum_support_views"]
            if minimum not in association_cache:
                association_cache[minimum] = associate_multiview_lines(
                    views,
                    {**method["association_common"], "minimum_support_views": minimum},
                )
            result = association_cache[minimum]
            if variant["nested_guard"]:
                result = apply_nested_band_guard(result, views, method["nested_guard"])
            variants.append({"variant": variant["name"], "result": result})
        case_results.append({"case_id": case["case_id"], "views": views, "variants": variants})
        print(case["case_id"], [(row["variant"], row["result"]["state"]) for row in variants])
    write_json(
        run / "inference.json",
        {"state": "inferred", "run_id": run_id, "gt_read_during_inference": False,
         "input_manifest_sha256": prepared["input_manifest_sha256"],
         "method_config_sha256": prepared["method_config_sha256"], "cases": case_results},
    )


def line_metrics(selected, endpoints):
    model = selected["model"]
    anchor = np.asarray(model["anchor"], float)
    direction = np.asarray(model["direction"], float)
    direction /= np.linalg.norm(direction)
    endpoints = np.asarray(endpoints, float)
    truth_direction = endpoints[1] - endpoints[0]
    truth_direction /= np.linalg.norm(truth_direction)
    angle = np.degrees(np.arccos(np.clip(abs(direction @ truth_direction), 0, 1)))
    samples = endpoints[0] + np.linspace(0, 1, 101)[:, None] * (endpoints[1] - endpoints[0])
    distances = np.linalg.norm(np.cross(samples - anchor, direction), axis=1)
    return {
        "angle_error_deg": float(angle),
        "target_to_line_distance_median_m": float(np.median(distances)),
        "target_to_line_distance_p95_m": float(np.quantile(distances, 0.95)),
    }


def classify(result, truth, policy):
    target = truth["target"]
    metrics = None
    if result["state"] == "accepted" and target["present"]:
        metrics = line_metrics(result["selected"], target["endpoints"])
    if target.get("identity_ambiguous", False):
        label = "safe_identity_ambiguity" if result["state"] == "ambiguous" else (
            "unsupported_identity_accept" if result["state"] == "accepted" else "identity_refused"
        )
    elif not target["present"]:
        label = "false_accept_empty" if result["state"] == "accepted" else "safe_empty_refusal"
    elif result["state"] == "accepted":
        correct = (
            metrics["target_to_line_distance_p95_m"] <= policy["line_distance_tolerance_world_m"]
            and metrics["angle_error_deg"] <= policy["angle_tolerance_deg"]
        )
        label = "correct_accept" if correct else "wrong_line_accept"
    else:
        label = "target_refused_ambiguous" if result["state"] == "ambiguous" else "target_refused"
    return label, metrics


def evaluate(config_path=CONFIG, share_path=SHARE):
    config = read_json(config_path)
    run, inputs, truth, output = locations(config["run_id"])
    if output.exists() or share_path.exists():
        raise FileExistsError("Keep old identity evaluation/share output")
    prepared = read_json(run / "prepared.json")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Evaluation protocol changed after freezing")
    config = read_json(run / "protocol.json")
    inference_path = run / "inference.json"
    inference = read_json(inference_path)
    if inference["state"] != "inferred" or inference["gt_read_during_inference"]:
        raise ValueError("Identity inference boundary failed")
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Identity inputs changed after inference")
    if digest(truth / "manifest.json") != prepared["truth_manifest_sha256"]:
        raise ValueError("Identity truth manifest changed")
    if digest(truth / "artifact_hashes.json") != prepared["truth_artifacts_sha256"]:
        raise ValueError("Identity truth index changed")
    for relative, expected in read_json(truth / "artifact_hashes.json").items():
        if digest(truth / relative) != expected:
            raise ValueError(f"Identity truth artifact changed: {relative}")
    truth_manifest = read_json(truth / "manifest.json")
    truth_by_case = {}
    for record in truth_manifest["cases"]:
        path = truth / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("Identity case truth changed")
        truth_by_case[record["case_id"]] = read_json(path)
    rows = []
    for case in inference["cases"]:
        case_truth = truth_by_case[case["case_id"]]
        for variant in case["variants"]:
            result = variant["result"]
            label, metrics = classify(result, case_truth, config["evaluation"])
            rows.append(
                {
                    "case_id": case["case_id"], "label": case_truth["label"],
                    "split": case_truth["split"], "variant": variant["variant"],
                    "state": result["state"], "classification": label,
                    "support_view_count": result["selected"]["support_view_count"] if result["selected"] else 0,
                    "unique_hypothesis_count": result["unique_hypothesis_count"],
                    "nested_view_count": result.get("identity_guard", {}).get("nested_view_count"),
                    "target_visible_pixels_per_view": [frame["target_visible_pixels"] for frame in case_truth["frames"]],
                    "metrics": metrics,
                }
            )
    summaries = []
    for split in ("development", "evaluation"):
        for variant in [item["name"] for item in config["method"]["variants"]]:
            selected = [row for row in rows if row["split"] == split and row["variant"] == variant]
            labels = sorted({row["classification"] for row in selected})
            summaries.append(
                {"split": split, "variant": variant, "case_count": len(selected),
                 "classification_counts": {label: sum(row["classification"] == label for row in selected) for label in labels}}
            )
    result = {
        "state": "complete", "run_id": config["run_id"], "scope": config["scope"],
        "protocol_sha256": prepared["protocol_sha256"],
        "inference_sha256": digest(inference_path),
        "evaluation_policy": config["evaluation"], "summaries": summaries, "rows": rows,
        "limitations": [
            "One procedural Blender family with exact cameras is not cross-object or estimated-camera performance.",
            "The nested guard returns ambiguity only; it does not certify the widest pair as a silhouette.",
            "Infinite-line identity is scored here; endpoints, gaps, and patch composition remain unimplemented.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share_path, result)
    print("EVALUATED_IDENTITY", output / "summary.json")


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
