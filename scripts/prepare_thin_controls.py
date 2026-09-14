"""Freeze RGB crop inputs first; only the separate GT stage reads geometry."""

import argparse
import copy
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import (
    MeshRays,
    preprocess_rgb,
    pure_resize_support,
    read_json,
    resize_plan,
    save_arrays,
    sha256,
    stats,
    verify_projection_roundtrip,
    write_json,
)
from thin_pack_infer_worker import (
    input_bundle_path,
    rgb_input_path,
    safe_component,
    validate_frame_order,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/thin_pack_controls_v1.json"
SOURCE_ID = "thin-pack-v2-20260914r1"


def validate_crop_protocol(config, source_manifest):
    safe_component(config["input_bundle"], "input bundle ID")
    source_groups = source_manifest["groups"]
    source_ids = [safe_component(g["case_id"], "source case ID") for g in source_groups]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate or empty source groups")
    jobs = config["jobs"]
    job_ids = [safe_component(j["case_id"], "case ID") for j in jobs]
    if not job_ids or len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate or empty crop jobs")
    for job in jobs:
        safe_component(job["source_case"], "source case ID")
        if job["source_case"] not in source_ids:
            raise ValueError("Unknown crop source case")
        box = config["crops"][job["crop"]]
        if (
            len(box) != 4
            or any(type(v) is not int for v in box)
            or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])
        ):
            raise ValueError("Invalid integer crop box")
        original = source_groups[source_ids.index(job["source_case"])]
        validate_frame_order(original)
    return source_groups


def validate_crop_inputs(config, source, source_manifest, bundle, manifest):
    """Check the actual RGB derivation before any GT folder is created."""
    validate_crop_protocol(config, source_manifest)
    if manifest["bundle_id"] != config["input_bundle"] or manifest["known_cameras"]:
        raise ValueError("Crop bundle identity/camera mismatch")
    if sha256(source / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source RGB manifest changed")
    if sha256(bundle / "selection_protocol.json") != manifest["selection_protocol_sha256"]:
        raise ValueError("Selection protocol changed")
    if read_json(bundle / "selection_protocol.json") != config:
        raise ValueError("Config changed after crop inputs were frozen")
    # zip会悄悄吞掉漏组。先核对完整顺序，坏输入要在生成GT前报错。
    if [g["case_id"] for g in manifest["groups"]] != [j["case_id"] for j in config["jobs"]]:
        raise ValueError("Crop group set/order mismatch")
    source_groups = {g["case_id"]: g for g in source_manifest["groups"]}
    for job, group in zip(config["jobs"], manifest["groups"], strict=True):
        original = source_groups[job["source_case"]]
        if (
            group["source_case"] != job["source_case"]
            or group["crop_box_xyxy"] != config["crops"][job["crop"]]
        ):
            raise ValueError("Crop source/box mismatch")
        validate_frame_order(group)
        if group["frame_order"] != original["frame_order"]:
            raise ValueError("Crop/source frame order mismatch")
        box = group["crop_box_xyxy"]
        for frame, source_frame in zip(group["frames"], original["frames"], strict=True):
            if (
                frame["source_rgb"] != source_frame["rgb"]
                or frame["source_sha256"] != source_frame["sha256"]
            ):
                raise ValueError("Crop frame source identity mismatch")
            source_path = rgb_input_path(source, source_frame["rgb"])
            crop_path = rgb_input_path(bundle, frame["rgb"])
            if (
                sha256(source_path) != source_frame["sha256"]
                or sha256(crop_path) != frame["sha256"]
            ):
                raise ValueError("Source/crop RGB changed")
            with Image.open(source_path) as image:
                if box[2] > image.width or box[3] > image.height:
                    raise ValueError("Crop outside source RGB")
                expected = np.asarray(image.convert("RGB").crop(box))
            with Image.open(crop_path) as image:
                actual = np.asarray(image.convert("RGB"))
            if frame["size_wh"] != [box[2] - box[0], box[3] - box[1]] or not np.array_equal(
                actual, expected
            ):
                raise ValueError("Crop pixels do not match frozen source box")


def inputs(config_path=None):
    config = read_json(config_path or CONFIG)
    source = input_bundle_path(ROOT, SOURCE_ID)
    manifest = read_json(source / "manifest.json")
    validate_crop_protocol(config, manifest)
    output = input_bundle_path(ROOT, config["input_bundle"])
    output.mkdir(exist_ok=False)
    write_json(output / "selection_protocol.json", config)
    result = {
        "schema_version": "1.0.0",
        "document_type": "thin_pack_rgb_inputs",
        "bundle_id": config["input_bundle"],
        "known_cameras": [],
        "groups": [],
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "selection_protocol_sha256": sha256(output / "selection_protocol.json"),
        "gt_used_for_crop_selection": False,
    }
    for job in config["jobs"]:
        original = next(g for g in manifest["groups"] if g["case_id"] == job["source_case"])
        box = config["crops"][job["crop"]]
        folder = output / job["case_id"]
        folder.mkdir()
        group = {
            "case_id": job["case_id"],
            "source_case": job["source_case"],
            "crop_box_xyxy": box,
            "frame_order": original["frame_order"],
            "frames": [],
        }
        for frame in original["frames"]:
            path = rgb_input_path(source, frame["rgb"])
            if sha256(path) != frame["sha256"]:
                raise ValueError("Source RGB changed")
            image = Image.open(path).convert("RGB")
            if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                raise ValueError("Crop outside RGB")
            crop = image.crop(box)
            target = folder / (frame["frame_id"] + ".png")
            crop.save(target)
            # 这里只切图片，不接触相机或GT。人工粗框也要留下来源，不能装成自动定位。
            group["frames"].append(
                {
                    "frame_id": frame["frame_id"],
                    "rgb": str(target.relative_to(output)).replace("\\", "/"),
                    "sha256": sha256(target),
                    "source_rgb": frame["rgb"],
                    "source_sha256": frame["sha256"],
                    "size_wh": list(crop.size),
                }
            )
        result["groups"].append(group)
    write_json(output / "manifest.json", result)
    print("RGB_INPUTS_READY", output, flush=True)


def truth(config_path=None):
    from export_thin_pack import verify_da3_processor

    config = read_json(config_path or CONFIG)
    bundle = input_bundle_path(ROOT, config["input_bundle"])
    manifest = read_json(bundle / "manifest.json")
    source_rgb = input_bundle_path(ROOT, SOURCE_ID)
    validate_crop_inputs(
        config, source_rgb, read_json(source_rgb / "manifest.json"), bundle, manifest
    )
    source = ROOT / "data/eval_gt" / SOURCE_ID
    status = read_json(source / "status.json")
    if (
        status["state"] != "complete"
        or sha256(source / "artifact_hashes.json") != status["artifact_hashes_sha256"]
    ):
        raise ValueError("Original GT incomplete")
    hashes = read_json(source / "artifact_hashes.json")

    def checked(relative):
        path = source / relative
        if sha256(path) != hashes[Path(relative).as_posix()]["sha256"]:
            raise ValueError("Source GT hash mismatch: " + str(relative))
        return path

    original_summary = read_json(checked(Path("validation_summary.json")))
    if original_summary["input_manifest_sha256"] != manifest["source_manifest_sha256"]:
        raise ValueError("Source GT belongs to a different original RGB bundle")

    output = ROOT / "data/eval_gt" / config["input_bundle"]
    output.mkdir(exist_ok=False)
    write_json(output / "status.json", {"state": "running"})
    checks = []
    for job, group in zip(config["jobs"], manifest["groups"], strict=True):
        if job["case_id"] != group["case_id"]:
            raise ValueError("Group order mismatch")
        rays = MeshRays(checked(Path(job["source_case"]) / "mesh.npz"))
        left, top, right, bottom = group["crop_box_xyxy"]
        for frame in group["frames"]:
            prefix = Path(job["source_case"]) / frame["frame_id"]
            camera = read_json(checked(prefix / "camera.json"))
            cropped_camera = copy.deepcopy(camera)
            crop_matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1.0]])
            cropped_camera["K_edge"] = (crop_matrix @ camera["K_edge"]).tolist()
            cropped_camera["size_wh"] = [right - left, bottom - top]
            plan = resize_plan(right - left, bottom - top, job["process_res"])
            transformed = copy.deepcopy(cropped_camera)
            scale = np.array(plan["source_edge_to_target_edge"])
            transformed["K_edge"] = (scale @ cropped_camera["K_edge"]).tolist()
            k_index = np.array(transformed["K_edge"])
            k_index[:2, 2] -= 0.5
            transformed["K_index"] = k_index.tolist()
            transformed["size_wh"] = plan["target_size_wh"]
            rgb_path = bundle / frame["rgb"]
            if sha256(rgb_path) != frame["sha256"]:
                raise ValueError("Crop RGB changed")
            processed = preprocess_rgb(np.asarray(Image.open(rgb_path)), plan)
            arrays = rays.raster(transformed, 4)
            native_ids = np.load(checked(prefix / "native/surface_id.npy"))[top:bottom, left:right]
            native_safe = np.load(checked(prefix / "native/strict_depth_eval_valid.npy"))[
                top:bottom, left:right
            ]
            pure, identity = pure_resize_support(native_ids, native_safe, plan)
            arrays["strict_depth_eval_valid"] = (
                pure
                & (identity == arrays["surface_id"])
                & ~arrays["sample_mixed"]
                & arrays["hit_valid"]
            )
            folder = output / job["case_id"] / frame["frame_id"] / f"da3_{job['process_res']}"
            folder.mkdir(parents=True)
            save_arrays(folder, arrays)
            write_json(folder / "camera.json", transformed)
            write_json(
                folder / "crop_transform.json",
                {
                    "box_xyxy": group["crop_box_xyxy"],
                    "plan": plan,
                    "original_edge_to_processed_edge": (scale @ crop_matrix).tolist(),
                },
            )
            Image.fromarray(processed).save(folder / "processed_rgb.png")
            # 独立核对：同一批目标像素先映回原图再打射线，应当命中同一表面。
            w, h = transformed["size_wh"]
            y, x = np.mgrid[3:h:23, 3:w:23]
            uv = np.c_[x.ravel() + 0.5, y.ravel() + 0.5]
            native_uv = (np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(scale @ crop_matrix).T)[:, :2]
            z, ids, _ = rays.cast(camera, native_uv)
            z2, ids2, _ = rays.cast(transformed, uv)
            error = float(np.nanmax(np.abs(z - z2)))
            if error > 1e-4 or not np.array_equal(ids, ids2):
                raise ValueError("Crop rays do not correspond to original rays")
            check = {
                "case_id": job["case_id"],
                "frame_id": frame["frame_id"],
                **stats(arrays),
                "mapped_rays": len(z),
                "surface_id_mismatches": int((ids != ids2).sum()),
                "mapped_ray_max_z_error_m": error,
                "projection_max_px": verify_projection_roundtrip(transformed, arrays["depth_z"]),
                "processor": verify_da3_processor(
                    rgb_path, processed, job["process_res"], cropped_camera
                ),
            }
            checks.append(check)
            write_json(output / "validation_progress.json", checks)
            print("GT_CROP", job["case_id"], frame["frame_id"], flush=True)
    write_json(
        output / "validation_summary.json",
        {
            "state": "complete",
            "input_manifest_sha256": sha256(bundle / "manifest.json"),
            "source_gt_hash_index": status["artifact_hashes_sha256"],
            "source_script_sha256": sha256(Path(__file__)),
            "checks": checks,
        },
    )
    artifact_hashes = {
        p.relative_to(output).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "status.json"
    }
    write_json(output / "artifact_hashes.json", artifact_hashes)
    write_json(
        output / "status.json",
        {"state": "complete", "artifact_hashes_sha256": sha256(output / "artifact_hashes.json")},
    )
    print("CROP_GT_READY", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inputs", "truth"])
    parser.add_argument(
        "--config",
        type=Path,
        help="Separate frozen crop protocol; default keeps the original controls",
    )
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    (inputs if args.stage == "inputs" else truth)(args.config)
