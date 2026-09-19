"""Prepare, infer, and evaluate frozen analytic multi-view candidate controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_multiview_candidates import (  # noqa: E402
    associate_multiview_lines,
    enumerate_image_lines,
)
from creator_eval.rod_observations import extract_rod_observations  # noqa: E402

CONFIG = ROOT / "configs/rod_multiview_candidates_v1.json"
SHARE = ROOT / "docs/experiments/results/2026-09-19-multiview-candidates.json"
DEFAULT_RUN_ID = "rod-multiview-candidates-v1-20260919r1"
SOURCES = [
    "scripts/run_rod_multiview_candidates.py",
    "experiments/src/creator_eval/rod_multiview_candidates.py",
    "experiments/src/creator_eval/rod_observations.py",
    "experiments/src/creator_eval/line_controls.py",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_ready(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def locations_from_run_id(run_id):
    if not run_id.replace("-", "").replace("_", "").isalnum() or len(run_id) > 96:
        raise ValueError("Invalid run ID")
    return (
        ROOT / ".runtime/experiments" / run_id,
        ROOT / "data/inputs" / run_id,
        ROOT / "data/eval_gt" / run_id,
        ROOT / "data/evaluation" / run_id,
    )


def locations(config):
    return locations_from_run_id(config["run_id"])


def camera(generator, camera_x):
    fx = generator["focal_px"]
    cx, cy = generator["principal_xy"]
    intrinsic = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
    extrinsic = np.c_[np.eye(3), [-camera_x, 0, 0]]
    return intrinsic, extrinsic


def world_endpoints(x_bottom, x_top, generator):
    bottom, top = generator["world_y_range"]
    z = generator["target_z"]
    return np.array([[x_bottom, bottom, z, 1.0], [x_top, top, z, 1.0]])


def project(points, intrinsic, extrinsic):
    values = np.asarray(points) @ (intrinsic @ extrinsic).T
    if np.any(values[:, 2] <= 0):
        raise ValueError("Analytic fixture is behind the camera")
    return values[:, :2] / values[:, 2, None]


def render_band(image, endpoints, width_px, value):
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    start, end = np.asarray(endpoints, float)
    delta = end - start
    length_square = float(delta @ delta)
    fraction = ((xx - start[0]) * delta[0] + (yy - start[1]) * delta[1]) / length_square
    closest_x = start[0] + fraction * delta[0]
    closest_y = start[1] + fraction * delta[1]
    distance = np.hypot(xx - closest_x, yy - closest_y)
    mask = (fraction >= 0) & (fraction <= 1) & (distance <= width_px / 2)
    image[mask] = value


def prepare(config_path=CONFIG):
    config = read_json(config_path)
    run, inputs, truth, evaluation = locations(config)
    for path in (run, inputs, truth, evaluation):
        if path.exists():
            raise FileExistsError(f"Keep existing output: {path}")
    run.mkdir(parents=True)
    inputs.mkdir(parents=True)
    truth.mkdir(parents=True)
    (run / "source_snapshot").mkdir()
    shutil.copy2(config_path, run / "protocol.json")
    source_hashes = {}
    for relative in SOURCES:
        source_hashes[relative] = digest(ROOT / relative)
        shutil.copy2(ROOT / relative, run / "source_snapshot" / Path(relative).name)
    write_json(run / "method_config.json", config["method"])
    generator = config["generator"]
    width, height = generator["size_wh"]
    input_cases, truth_cases = [], []
    for case in config["cases"]:
        case_dir = inputs / case["case_id"]
        case_dir.mkdir()
        frames = []
        target_points = world_endpoints(
            case["target"]["x_bottom"], case["target"]["x_top"], generator
        )
        for view_index, camera_x in enumerate(generator["camera_x"]):
            intrinsic, extrinsic = camera(generator, camera_x)
            target_pixels = project(target_points, intrinsic, extrinsic)
            image = np.full((height, width), generator["background_value"], dtype=np.uint8)
            for layer in case["layers"]:
                active = layer["views"] == "all" or view_index in layer["views"]
                if not active:
                    continue
                if layer["kind"] == "world_line":
                    points = world_endpoints(layer["x_bottom"], layer["x_top"], generator)
                    pixels = project(points, intrinsic, extrinsic)
                elif layer["kind"] == "image_offset":
                    pixels = target_pixels.copy()
                    pixels[:, 0] += layer["offset_px_by_view"][view_index]
                else:
                    raise ValueError(f"Unknown layer kind: {layer['kind']}")
                render_band(image, pixels, layer["width_px"], layer["value"])
            rgb = np.repeat(image[..., None], 3, axis=2)
            relative = Path(case["case_id"]) / f"view_{view_index}.png"
            Image.fromarray(rgb).save(inputs / relative)
            jitter = generator["guide_jitter_px"][view_index]
            guide = target_pixels.copy()
            guide[:, 0] += jitter
            frames.append(
                {
                    "view_id": f"view_{view_index}",
                    "rgb": relative.as_posix(),
                    "rgb_sha256": digest(inputs / relative),
                    "K_index": intrinsic,
                    "world_to_camera_cv": extrinsic,
                    "guide_xyxy": guide,
                    "size_wh": [width, height],
                }
            )
        input_cases.append({"case_id": case["case_id"], "frames": frames})
        case_truth = {
            "case_id": case["case_id"],
            "split": case["split"],
            "target": case["target"],
            "target_endpoints_world": target_points[:, :3],
        }
        case_truth_path = truth / f"{case['case_id']}.json"
        write_json(case_truth_path, case_truth)
        truth_cases.append(
            {"case_id": case["case_id"], "path": case_truth_path.name,
             "sha256": digest(case_truth_path)}
        )
    write_json(
        inputs / "manifest.json",
        {
            "run_id": config["run_id"],
            "cases": input_cases,
            "method_inputs": "RGB, calibrated cameras, and coarse per-view guides only",
            "truth_excluded": True,
        },
    )
    write_json(
        truth / "manifest.json",
        {"run_id": config["run_id"], "cases": truth_cases,
         "protocol_sha256": digest(run / "protocol.json")},
    )
    write_json(
        run / "prepared.json",
        {
            "state": "prepared",
            "run_id": config["run_id"],
            "protocol_sha256": digest(run / "protocol.json"),
            "method_config_sha256": digest(run / "method_config.json"),
            "input_manifest_sha256": digest(inputs / "manifest.json"),
            "truth_manifest_sha256": digest(truth / "manifest.json"),
            "source_sha256": source_hashes,
            "split_frozen_before_inference": True,
        },
    )
    print(f"PREPARED {len(input_cases)} cases; inputs and truth are separate")


def infer(run_id=DEFAULT_RUN_ID):
    # Do not open the full generation/evaluation protocol here. It contains the
    # fixture answers. The prepared record exposes only identity hashes, while
    # inference reads its own method settings and the input manifest.
    run, inputs, _, _ = locations_from_run_id(run_id)
    prepared = read_json(run / "prepared.json")
    if prepared["run_id"] != run_id or prepared["state"] != "prepared":
        raise ValueError("Prepared run identity mismatch")
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Input manifest changed")
    method = read_json(run / "method_config.json")
    if digest(run / "method_config.json") != prepared["method_config_sha256"]:
        raise ValueError("Method config changed")
    manifest = read_json(inputs / "manifest.json")
    results = []
    for case in manifest["cases"]:
        views = []
        for frame in case["frames"]:
            rgb_path = inputs / frame["rgb"]
            if digest(rgb_path) != frame["rgb_sha256"]:
                raise ValueError("RGB changed")
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            observations = extract_rod_observations(
                rgb, frame["guide_xyxy"], method["observation"]
            )
            lines = enumerate_image_lines(observations, method["image_hypotheses"])
            views.append(
                {
                    "view_id": frame["view_id"],
                    "K_index": frame["K_index"],
                    "world_to_camera_cv": frame["world_to_camera_cv"],
                    "y_range": [frame["guide_xyxy"][0][1], frame["guide_xyxy"][1][1]],
                    "candidates": lines,
                    "observation_state_counts": {
                        name: sum(row["status"] == name for row in observations["rows"])
                        for name in ("observed", "ambiguous", "absent", "unknown")
                    },
                }
            )
        variants = []
        for variant in method["variants"]:
            association_config = {
                **method["association_common"],
                "minimum_support_views": variant["minimum_support_views"],
            }
            variants.append(
                {"variant": variant["name"],
                 "result": associate_multiview_lines(views, association_config)}
            )
        results.append(
            {
                "case_id": case["case_id"],
                "views": views,
                "variants": variants,
            }
        )
        print(case["case_id"], [(row["variant"], row["result"]["state"]) for row in variants])
    output = run / "inference.json"
    write_json(
        output,
        {
            "state": "inferred",
            "run_id": run_id,
            "gt_read_during_inference": False,
            "input_manifest_sha256": prepared["input_manifest_sha256"],
            "method_config_sha256": prepared["method_config_sha256"],
            "cases": results,
        },
    )
    print("INFERRED", output)


def line_metrics(selected, endpoints):
    model = selected["model"]
    anchor = np.asarray(model["anchor"], float)
    direction = np.asarray(model["direction"], float)
    direction /= np.linalg.norm(direction)
    truth_direction = endpoints[1] - endpoints[0]
    truth_direction /= np.linalg.norm(truth_direction)
    angle = np.degrees(np.arccos(np.clip(abs(direction @ truth_direction), 0, 1)))
    samples = endpoints[0] + np.linspace(0, 1, 101)[:, None] * (endpoints[1] - endpoints[0])
    distances = np.linalg.norm(np.cross(samples - anchor, direction), axis=1)
    return {
        "angle_error_deg": float(angle),
        "target_to_line_distance_median": float(np.median(distances)),
        "target_to_line_distance_p95": float(np.quantile(distances, 0.95)),
    }


def classify(result, truth, evaluation):
    present = truth["target"]["present"]
    identity_ambiguous = truth["target"].get("identity_ambiguous", False)
    metrics = None
    if result["state"] == "accepted" and present:
        metrics = line_metrics(result["selected"], np.asarray(truth["target_endpoints_world"]))
    if identity_ambiguous:
        label = "safe_identity_ambiguity" if result["state"] == "ambiguous" else (
            "unsupported_identity_accept" if result["state"] == "accepted" else "identity_refused"
        )
    elif not present:
        label = "false_accept_empty" if result["state"] == "accepted" else "safe_empty_refusal"
    elif result["state"] == "accepted":
        correct = (
            metrics["target_to_line_distance_p95"] <= evaluation["line_distance_tolerance_world"]
            and metrics["angle_error_deg"] <= evaluation["angle_tolerance_deg"]
        )
        label = "correct_accept" if correct else "wrong_line_accept"
    else:
        label = "target_refused"
    return label, metrics


def evaluate(config_path=CONFIG, share_path=SHARE):
    config = read_json(config_path)
    run, inputs, truth, output = locations(config)
    if output.exists() or share_path.exists():
        raise FileExistsError("Keep existing evaluation/share output; use a new run and share path")
    prepared = read_json(run / "prepared.json")
    inference_path = run / "inference.json"
    inference = read_json(inference_path)
    if inference["state"] != "inferred" or inference["gt_read_during_inference"]:
        raise ValueError("Inference boundary failed")
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Inputs changed after inference")
    truth_manifest = read_json(truth / "manifest.json")
    if digest(truth / "manifest.json") != prepared["truth_manifest_sha256"]:
        raise ValueError("Truth manifest changed")
    truth_by_case = {}
    for record in truth_manifest["cases"]:
        path = truth / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("Truth case changed")
        truth_by_case[record["case_id"]] = read_json(path)
    rows = []
    for case in inference["cases"]:
        case_truth = truth_by_case[case["case_id"]]
        for variant in case["variants"]:
            label, metrics = classify(variant["result"], case_truth, config["evaluation"])
            rows.append(
                {
                    "case_id": case["case_id"],
                    "split": case_truth["split"],
                    "variant": variant["variant"],
                    "state": variant["result"]["state"],
                    "classification": label,
                    "support_view_count": (
                        variant["result"]["selected"]["support_view_count"]
                        if variant["result"]["selected"] is not None else 0
                    ),
                    "unique_hypothesis_count": variant["result"]["unique_hypothesis_count"],
                    "metrics": metrics,
                }
            )
    summaries = []
    for split in ("development", "evaluation"):
        for variant in [row["name"] for row in config["method"]["variants"]]:
            selected = [row for row in rows if row["split"] == split and row["variant"] == variant]
            labels = sorted({row["classification"] for row in selected})
            summaries.append(
                {"split": split, "variant": variant, "case_count": len(selected),
                 "classification_counts": {label: sum(row["classification"] == label for row in selected) for label in labels}}
            )
    result = {
        "state": "complete",
        "run_id": config["run_id"],
        "scope": config["scope"],
        "protocol_sha256": prepared["protocol_sha256"],
        "inference_sha256": digest(inference_path),
        "evaluation_policy": config["evaluation"],
        "summaries": summaries,
        "rows": rows,
        "limitations": [
            "Analytic images and exact supplied cameras are mechanism controls, not real-scene performance.",
            "The method estimates infinite line identity only; endpoint, gap, occlusion cause, and physical existence are not recovered.",
            "A multi-view-consistent surface stripe or separate line can satisfy geometry while remaining the wrong target.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share_path, result)
    print("EVALUATED", output / "summary.json")


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
