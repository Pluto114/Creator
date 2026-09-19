"""Independently verify hashes, arrays, cameras, and target-axis pixels in a rod pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "rod-identity-blender-v1-20260919r3"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def project_axis(camera, endpoints, sample_count=101):
    endpoints = np.asarray(endpoints, float)
    fractions = np.linspace(0, 1, sample_count)[:, None]
    points = endpoints[0] + fractions * (endpoints[1] - endpoints[0])
    homogeneous = np.c_[points, np.ones(len(points))]
    projection = np.asarray(camera["K_index"]) @ np.asarray(camera["world_to_camera_cv"])[:3]
    values = homogeneous @ projection.T
    return values[:, :2] / values[:, 2, None]


def verify(run_id, output_path):
    inputs = ROOT / "data/inputs" / run_id
    truth = ROOT / "data/eval_gt" / run_id
    input_manifest = read_json(inputs / "manifest.json")
    truth_manifest = read_json(truth / "manifest.json")
    artifacts = read_json(truth / "artifact_hashes.json")
    if not input_manifest["truth_excluded"]:
        raise ValueError("Input manifest does not declare truth exclusion")
    forbidden = [path for path in inputs.rglob("*") if path.is_file() and path.suffix != ".png"]
    if forbidden != [inputs / "manifest.json"]:
        raise ValueError("Input folder contains files beyond RGB and its manifest")
    for relative, expected in artifacts.items():
        if digest(truth / relative) != expected:
            raise ValueError(f"Truth artifact hash mismatch: {relative}")
    inputs_by_case = {case["case_id"]: case for case in input_manifest["cases"]}
    frame_rows = []
    axis_checks = []
    for case_record in truth_manifest["cases"]:
        case_path = truth / case_record["path"]
        if digest(case_path) != case_record["sha256"]:
            raise ValueError("Case truth hash mismatch")
        case = read_json(case_path)
        geometry = read_json(truth / case["geometry_path"])
        allowed_ids = {int(row["surface_id"]) for row in geometry["objects"]} | {0}
        input_frames = {
            frame["view_id"]: frame for frame in inputs_by_case[case["case_id"]]["frames"]
        }
        for frame in case["frames"]:
            source = input_frames[frame["view_id"]]
            rgb_path = inputs / source["rgb"]
            if digest(rgb_path) != source["rgb_sha256"]:
                raise ValueError("RGB hash mismatch")
            width, height = Image.open(rgb_path).size
            camera = read_json(truth / case["case_id"] / frame["camera_path"])
            if camera["size_wh"] != [width, height]:
                raise ValueError("Camera and RGB dimensions differ")
            if not np.allclose(camera["K_index"], source["K_index"], atol=0, rtol=0):
                raise ValueError("Input and truth camera intrinsics differ")
            if not np.allclose(
                camera["world_to_camera_cv"], source["world_to_camera_cv"], atol=0, rtol=0
            ):
                raise ValueError("Input and truth camera extrinsics differ")
            arrays = {
                name: np.load(truth / record["path"], allow_pickle=False)
                for name, record in frame["arrays"].items()
            }
            expected_shape = (height, width)
            if any(array.shape != expected_shape for array in arrays.values()):
                raise ValueError("Truth raster shape mismatch")
            surface = arrays["surface_id"]
            depth = arrays["depth_z"]
            distance = arrays["ray_distance"]
            target = arrays["target_visible"]
            if set(np.unique(surface)) - allowed_ids:
                raise ValueError("Raster contains an undeclared surface ID")
            if not np.array_equal(target, surface == 1):
                raise ValueError("Target mask and surface ID disagree")
            hit = surface != 0
            if not np.array_equal(np.isfinite(depth), hit):
                raise ValueError("Depth finite mask and ray hits disagree")
            if not np.array_equal(np.isfinite(distance), hit):
                raise ValueError("Ray distance finite mask and ray hits disagree")
            if np.any(distance[hit] + 1e-6 < depth[hit]):
                raise ValueError("Ray distance is shorter than camera-Z depth")
            frame_rows.append(
                {
                    "case_id": case["case_id"],
                    "view_id": frame["view_id"],
                    "hit_pixels": int(hit.sum()),
                    "target_visible_pixels": int(target.sum()),
                    "surface_ids": sorted(int(value) for value in np.unique(surface)),
                }
            )
            if case["target"]["present"]:
                pixels = project_axis(camera, case["target"]["endpoints"])
                indices = np.rint(pixels).astype(int)
                inside = (
                    (indices[:, 0] >= 0) & (indices[:, 0] < width)
                    & (indices[:, 1] >= 0) & (indices[:, 1] < height)
                )
                sampled_ids = surface[indices[inside, 1], indices[inside, 0]]
                supported = np.isin(sampled_ids, [1, 50])
                fraction = float(np.mean(supported)) if len(supported) else 0.0
                if fraction < 0.98:
                    raise ValueError("Projected target axis does not match target/occluder pixels")
                axis_checks.append(
                    {
                        "case_id": case["case_id"],
                        "view_id": frame["view_id"],
                        "sample_count": int(len(sampled_ids)),
                        "target_or_occluder_fraction": fraction,
                    }
                )
            elif target.any():
                raise ValueError("An empty target case has target-visible pixels")
    result = {
        "state": "verified",
        "run_id": run_id,
        "input_manifest_sha256": digest(inputs / "manifest.json"),
        "truth_manifest_sha256": digest(truth / "manifest.json"),
        "truth_artifact_index_sha256": digest(truth / "artifact_hashes.json"),
        "frame_count": len(frame_rows),
        "axis_check_count": len(axis_checks),
        "minimum_target_or_occluder_fraction": min(
            row["target_or_occluder_fraction"] for row in axis_checks
        ),
        "checks": [
            "input folder contains RGB plus manifest only",
            "all frozen RGB and truth artifact hashes match",
            "camera matrices match between allowed input and truth records",
            "surface IDs are declared by exported evaluated meshes",
            "finite depth/range pixels equal ray hits and range is not shorter than camera-Z",
            "target-visible mask equals surface_id == 1 exactly",
            "projected target-axis samples hit target or the declared occluder",
        ],
        "frames": frame_rows,
        "axis_checks": axis_checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print("VERIFIED_IDENTITY_PACK", len(frame_rows), "frames", len(axis_checks), "axis checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or (
        ROOT / "data/evaluation" / args.run_id / "pack_verification.json"
    )
    verify(args.run_id, destination)
