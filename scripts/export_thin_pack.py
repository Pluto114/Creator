"""One-command, immutable paired export using the installed Blender and DA3 environments."""

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from thin_pack_gt import (
    MeshRays,
    check_blender_rays,
    depth_hypotheses,
    preprocess_rgb,
    pure_resize_support,
    read_blender_depth,
    read_json,
    resize_plan,
    safe_native,
    save_arrays,
    sha256,
    stats,
    verify_projection_roundtrip,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def verify_da3_processor(rgb_path, processed, resolution, camera):
    from depth_anything_3.utils.io.input_processor import InputProcessor

    processor = InputProcessor()
    tensor, _, k = processor(
        [str(rgb_path)],
        intrinsics=np.array([camera["K_edge"]], np.float32),
        process_res=resolution,
        sequential=True,
    )
    native = tensor.numpy().reshape(-1, 3, *processed.shape[:2])[0].transpose(1, 2, 0)
    expected = (
        processed.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)
    ) / np.array([0.229, 0.224, 0.225], np.float32)
    error = float(np.max(np.abs(native - expected)))
    assert error < 1e-6, error
    scale = np.diag(
        [processed.shape[1] / camera["size_wh"][0], processed.shape[0] / camera["size_wh"][1], 1]
    )
    k_error = float(np.max(np.abs(k.numpy()[0] - scale @ camera["K_edge"])))
    assert k_error < 1e-3
    return {
        "normalized_tensor_max_abs_error": error,
        "upstream_K_edge_scaling_max_abs_error": k_error,
    }


def gap_check(rays, camera, geometry):
    output = []
    for gap in geometry["gaps"]:
        a, b = np.array(gap["world_endpoints"])
        points = np.array([a * (1 - f) + b * f for f in np.linspace(0.1, 0.9, 9)])
        q = np.c_[points, np.ones(len(points))] @ np.array(camera["world_to_camera_cv"]).T
        pixels = q[:, :3] @ np.array(camera["K_edge"]).T
        uv = pixels[:, :2] / pixels[:, 2:]
        z, labels, _ = rays.cast(camera, uv)
        # These rays must not hit a segment from the intentionally broken rod.
        assert not np.isin(labels, gap["segment_surface_ids"]).any()
        output.append(
            {
                "rod_id": gap["rod_id"],
                "sample_count": len(points),
                "surface_ids_through_gap": labels.tolist(),
                "pixels_uv_edge": uv.tolist(),
                "unoccluded_gap_samples": int((~np.isfinite(z) | (z > q[:, 2])).sum()),
            }
        )
    return output


def plane_check(rays, camera, arrays):
    t = np.array(camera["world_to_camera_cv"])
    inverse = np.linalg.inv(t)
    h, w = arrays["depth_z"].shape
    y, x = np.mgrid[:h:17, :w:17]
    depth, ids = arrays["depth_z"][y, x], arrays["surface_id"][y, x]
    directions = (
        np.stack([x + 0.5, y + 0.5, np.ones_like(x)], axis=-1)
        @ np.linalg.inv(camera["K_edge"]).T
        @ inverse[:3, :3].T
    )
    result = {}
    for sid in (100, 101):
        face = rays.triangles[rays.labels == sid]
        if not len(face):
            continue
        a, b, c = rays.vertices[face[0]].astype(np.float64)
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        selected = ids == sid
        expected = (normal @ (a - inverse[:3, 3])) / (directions[selected] @ normal)
        error = float(np.max(np.abs(expected - depth[selected]))) if len(expected) else 0.0
        assert error < 1e-4, error
        result[str(sid)] = {"samples": len(expected), "max_z_error_m": error}
    return result


def make_preview(path, rgb, native, model, model_rgb, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("Paired RGB / native")
    axs[0, 1].imshow(native["depth_z"], cmap="viridis", vmin=7, vmax=22)
    axs[0, 1].set_title("Camera Z (m); no hit is blank")
    axs[0, 2].imshow(native["rod_id"], cmap="tab10", vmin=0, vmax=9)
    axs[0, 2].set_title("Visible rod ID from geometry")
    y, x = np.where(native["rod_id"] > 0)
    box = (
        max(0, x.min() - 25),
        min(rgb.shape[1], x.max() + 25),
        max(0, y.min() - 25),
        min(rgb.shape[0], y.max() + 25),
    )
    axs[1, 0].imshow(rgb)
    axs[1, 0].contour(native["rod_id"] > 0, levels=[0.5], colors=["red"], linewidths=0.45)
    axs[1, 0].set_xlim(box[:2])
    axs[1, 0].set_ylim(box[3], box[2])
    axs[1, 0].set_title("Native silhouette overlay; real gap retained")
    axs[1, 1].imshow(model_rgb)
    axs[1, 1].imshow(
        np.ma.masked_where(
            model["segment_coverage_counts"].sum(axis=0) == 0,
            model["segment_coverage_counts"].sum(axis=0) / 16,
        ),
        cmap="autumn",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    axs[1, 1].set_title("504 RGB + sampled visible rod coverage")
    display = np.zeros_like(model["rod_id"], np.uint8)
    display[model["segment_coverage_counts"].sum(axis=0) > 0] = 1
    display[(model["rod_id"] > 0) & model["strict_depth_eval_valid"]] = 2
    axs[1, 2].imshow(display, cmap="viridis", vmin=0, vmax=2)
    axs[1, 2].set_title("504: purple background; teal mixed; yellow strict rod")
    for ax in axs.flat:
        ax.set_axis_off()
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def process_bundle(gt, inputs, config):
    render = read_json(gt / "render_manifest.json")
    calibration = gt / "calibration"
    rays = MeshRays(calibration / "mesh.npz")
    camera = read_json(calibration / "view/camera.json")
    arrays = rays.raster(camera)
    safe = safe_native(arrays, 2)
    raw = read_blender_depth(calibration / "view/blender_depth_raw.exr")
    hypotheses = depth_hypotheses(raw, arrays, safe)
    kind = min(("camera_z", "ray_distance"), key=lambda x: hypotheses[x]["mean_abs_m"])
    alternative = "ray_distance" if kind == "camera_z" else "camera_z"
    assert hypotheses[kind]["max_abs_m"] < config["blender_depth_tolerance_m"], hypotheses
    assert hypotheses[alternative]["mean_abs_m"] > 0.01, hypotheses
    nohit = ~arrays["hit_valid"]
    # Check the raw background sentinel away from silhouette boundaries.
    interior_nohit = cv2.erode(nohit.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    assert interior_nohit.any()
    assert np.all((raw[interior_nohit] > 1e9) | ~np.isfinite(raw[interior_nohit]))
    calibration_result = {
        "depth_pass_kind_measured": kind,
        "hypotheses": hypotheses,
        "known_plane_z_m": 5,
        "known_occluder_z_m": 4,
        "raw_no_hit_min": float(np.min(raw[interior_nohit]))
        if np.isfinite(raw[interior_nohit]).all()
        else None,
        "no_hit_samples": int(interior_nohit.sum()),
        "ray_check": check_blender_rays(
            rays,
            camera,
            read_json(calibration / "view/blender_ray_probes.json"),
            config["blender_ray_tolerance_m"],
        ),
        "plane_check": plane_check(rays, camera, arrays),
    }
    save_arrays(calibration / "view/geometry_gt", arrays)
    write_json(calibration / "validation.json", calibration_result)
    print("DEPTH_CALIBRATION " + kind, flush=True)
    results = []
    input_manifest = {
        "schema_version": "1.0.0",
        "document_type": "thin_pack_rgb_inputs",
        "bundle_id": inputs.name,
        "known_cameras": [],
        "groups": [],
    }
    for case in render["cases"]:
        folder = gt / case["id"]
        rays = MeshRays(folder / "mesh.npz")
        geometry = read_json(folder / "geometry.json")
        group = {"case_id": case["id"], "frame_order": [], "frames": []}
        for view in case["views"]:
            frame = view["frame_id"]
            target = folder / frame
            camera = read_json(target / "camera.json")
            rgb_path = inputs / case["id"] / f"{frame}.png"
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            assert list(rgb.shape[1::-1]) == camera["size_wh"]
            started = time.monotonic()
            native = rays.raster(camera, config["coverage_samples_per_axis"])
            native["strict_depth_eval_valid"] = safe_native(
                native, config["native_edge_guard_pixels"]
            )
            checks = {
                "case_id": case["id"],
                "frame_id": frame,
                "rgb_sha256": sha256(rgb_path),
                "camera_projection_max_px": view["projection_check_max_px"],
                "blender_ray_check": check_blender_rays(
                    rays,
                    camera,
                    read_json(target / "blender_ray_probes.json"),
                    config["blender_ray_tolerance_m"],
                ),
                "plane_check": plane_check(rays, camera, native),
                "roundtrip_max_px": verify_projection_roundtrip(camera, native["depth_z"]),
                "gap_check": gap_check(rays, camera, geometry),
            }
            raw = read_blender_depth(target / "blender_depth_raw.exr")
            assert raw.shape == native["depth_z"].shape
            checks["blender_depth_check"] = depth_hypotheses(
                raw, native, native["strict_depth_eval_valid"]
            )
            expected_key = "depth_z" if kind == "camera_z" else "ray_distance"
            # A rasterizer and ray intersection can disagree at edges; retain and flag them.
            error = np.abs(raw - native[expected_key])
            agree = np.isfinite(error) & (error < config["blender_depth_tolerance_m"])
            # EEVEE Z is a single renderer sample and can be displaced within the
            # pixel. A slanted surface then has a legitimate Z gradient. Check the
            # same-pixel geometric depth envelope instead of relaxing a fixed
            # center-depth threshold until the scene passes.
            assert kind == "camera_z", (
                "Footprint check currently supports this verified EEVEE Z mode only"
            )
            slack = config["blender_depth_footprint_tolerance_m"]
            inside = (
                np.isfinite(raw)
                & (raw >= native["pixel_sample_depth_min"] - slack)
                & (raw <= native["pixel_sample_depth_max"] + slack)
            )
            fail = native["strict_depth_eval_valid"] & ~inside
            checks["interior_depth_disagreements"] = int(fail.sum())
            checks["interior_center_difference_above_5mm"] = int(
                (native["strict_depth_eval_valid"] & ~agree).sum()
            )
            checks["depth_validation_rule"] = (
                "EEVEE Z within same-surface 4x4 pixel depth envelope, plus 1mm float/raster allowance; exact center differences reported separately"
            )
            if fail.any():
                raise AssertionError(
                    f"Interior renderer/geometry mismatch: {case['id']} {frame}: {fail.sum()}"
                )
            native["blender_depth_agrees"] = agree
            native["blender_depth_within_sample_footprint"] = inside
            checks["native"] = stats(native)
            save_arrays(target / "native", native)
            checks["processed"] = {}
            for res in config["process_resolutions"]:
                plan = resize_plan(*camera["size_wh"], res)
                processed_rgb = preprocess_rgb(rgb, plan)
                model_camera = dict(camera)
                model_camera["size_wh"] = plan["target_size_wh"]
                k_edge = np.array(plan["source_edge_to_target_edge"]) @ np.array(camera["K_edge"])
                k_index = k_edge.copy()
                k_index[:2, 2] -= 0.5
                model_camera.update(K_edge=k_edge.tolist(), K_index=k_index.tolist())
                model = rays.raster(model_camera, config["coverage_samples_per_axis"])
                pure, ids = pure_resize_support(
                    native["surface_id"], native["strict_depth_eval_valid"], plan
                )
                model["strict_depth_eval_valid"] = (
                    pure
                    & (ids == model["surface_id"])
                    & ~model["sample_mixed"]
                    & model["hit_valid"]
                )
                checks["processed"][str(res)] = {
                    **stats(model),
                    "upstream_preprocessor": verify_da3_processor(
                        rgb_path, processed_rgb, res, camera
                    ),
                    "roundtrip_max_px": verify_projection_roundtrip(model_camera, model["depth_z"]),
                }
                if res == 504 and view["angle_degrees"] == 0:
                    finer = rays.raster(model_camera, 8)
                    four = model["segment_coverage_counts"]
                    eight = finer["segment_coverage_counts"]
                    delta = np.abs(four.astype(float) / 16 - eight.astype(float) / 64)
                    extra = (four.sum(0) == 0) & (eight.sum(0) > 0)
                    # Finite quadrature can miss tiny slivers. These must remain outside
                    # the conservative single-surface depth evaluation region.
                    assert not np.any(extra & model["strict_depth_eval_valid"])
                    checks["processed"][str(res)]["coverage_4x4_vs_8x8"] = {
                        "max_absolute_fraction_difference": float(delta.max()),
                        "mean_absolute_fraction_difference_all_segments_pixels": float(
                            delta.mean()
                        ),
                        "four_miss_eight_hit_pixels": int(extra.sum()),
                        "extra_pixels_in_strict_depth_eval": 0,
                        "meaning": "Finite sampled visibility, not analytic area or renderer antialias weights",
                    }
                    np.save(target / "coverage_check_504_8x8.npy", eight, allow_pickle=False)
                # Direct continuous rays at each target pixel center; never resize GT depth/IDs.
                target_folder = target / f"da3_{res}"
                save_arrays(target_folder, model)
                Image.fromarray(processed_rgb).save(target_folder / "processed_rgb.png")
                write_json(target_folder / "camera.json", model_camera)
                write_json(target_folder / "image_transform.json", plan)
                if res == 504 and (
                    view["angle_degrees"] == 0
                    or (case == render["cases"][0] and view == case["views"][0])
                ):
                    make_preview(
                        target / "verification.png",
                        rgb,
                        native,
                        model,
                        processed_rgb,
                        f"{case['id']} / {view['angle_degrees']} deg",
                    )
            old_path = (
                ROOT
                / ".runtime/inputs/test_pack"
                / case["id"]
                / f"{view['angle_degrees']}\N{DEGREE SIGN}.png"
            )
            old = np.asarray(Image.open(old_path).convert("RGB"))
            delta = np.abs(rgb.astype(np.int16) - old.astype(np.int16))
            checks["historical_rgb_comparison"] = {
                "original_sha256": sha256(old_path),
                "exact_rgb_pixel_fraction": float(np.all(delta == 0, axis=-1).mean()),
                "max_channel_difference": int(delta.max()),
                "mean_channel_difference": float(delta.mean()),
            }
            checks["processing_seconds"] = round(time.monotonic() - started, 3)
            write_json(target / "validation.json", checks)
            results.append(checks)
            group["frame_order"].append(frame)
            group["frames"].append(
                {
                    "frame_id": frame,
                    "rgb": f"{case['id']}/{frame}.png",
                    "size_wh": camera["size_wh"],
                    "sha256": sha256(rgb_path),
                }
            )
            print(f"GT_VALIDATED {case['id']} {frame} {checks['processing_seconds']}s", flush=True)
            write_json(gt / "validation_progress.json", results)
        input_manifest["groups"].append(group)
    write_json(inputs / "manifest.json", input_manifest)
    summary = {
        "status": "complete",
        "view_count": len(results),
        "calibration": calibration_result,
        "max_camera_projection_error_px": max(r["camera_projection_max_px"] for r in results),
        "max_blender_ray_z_error_m": max(r["blender_ray_check"]["max_z_error_m"] for r in results),
        "max_interior_blender_depth_error_m": max(
            r["blender_depth_check"][kind]["max_abs_m"] for r in results
        ),
        "input_manifest_sha256": sha256(inputs / "manifest.json"),
        "views": results,
    }
    write_json(gt / "validation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--angles", type=int, nargs="+")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.bundle_id):
        parser.error("Invalid bundle ID")
    if ROOT.drive.upper() != "D:":
        raise RuntimeError("This local exporter requires D: storage")
    config = read_json(ROOT / "configs/thin_pack_export.json")
    if args.cases:
        if set(args.cases) - {c["id"] for c in config["cases"]}:
            parser.error("Unknown case")
        config["cases"] = [c for c in config["cases"] if c["id"] in args.cases]
    if args.angles:
        if set(args.angles) - set(config["angles"]):
            parser.error("Unknown angle")
        config["angles"] = [a for a in config["angles"] if a in args.angles]
    scene_path = ROOT / config["source_scene"]
    assert sha256(scene_path) == config["source_sha256"]
    inputs, gt, work = (
        ROOT / p / args.bundle_id for p in ("data/inputs", "data/eval_gt", ".runtime/paired-export")
    )
    if any(p.exists() for p in (inputs, gt, work)):
        raise FileExistsError(
            "Bundle already exists. Choose a new ID; no overwrite or implicit resume."
        )
    for folder in (inputs, gt, work):
        folder.mkdir(parents=True)
    start = time.monotonic()
    write_json(gt / "status.json", {"state": "running"})
    try:
        frozen = gt / "producer_sources"
        frozen.mkdir()
        paths = [
            Path(__file__),
            ROOT / "scripts/blender_export_thin_pack.py",
            ROOT / "scripts/thin_pack_gt.py",
            ROOT / "scripts/Export-ThinPack.ps1",
            ROOT / "configs/thin_pack_export.json",
            ROOT / "tests/test_thin_pack_gt.py",
        ]
        for file in paths:
            shutil.copy2(file, frozen / file.name)
        shutil.copy2(ROOT / "docs/thin-pack-dataset.md", gt / "DATASET.md")
        upstream = ROOT / ".local/setup/Depth-Anything-3/src/depth_anything_3/utils"
        for file in (upstream / "io/input_processor.py", upstream / "geometry.py"):
            shutil.copy2(file, frozen / f"da3_{file.name}")
        request = {
            "root": str(ROOT),
            "inputs": inputs.relative_to(ROOT).as_posix(),
            "gt": gt.relative_to(ROOT).as_posix(),
            "config": config,
        }
        write_json(work / "request.json", request)
        write_json(gt / "export_config.json", config)
        cmd = [
            args.blender,
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(scene_path),
            "--python-exit-code",
            "1",
            "--python",
            str(ROOT / "scripts/blender_export_thin_pack.py"),
            "--",
            str(work / "request.json"),
        ]
        with (work / "blender.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        summary = process_bundle(gt, inputs, config)
        assert sha256(scene_path) == config["source_sha256"]
        # Model input publication contains only RGB files and a camera-free input manifest.
        assert all(
            p.suffix == ".png" or p.name == "manifest.json"
            for p in inputs.rglob("*")
            if p.is_file()
        )
        provenance = {
            "schema_version": "1.0.0",
            "document_type": "thin_pack_paired_export",
            "bundle_id": args.bundle_id,
            "source_scene": config["source_scene"],
            "source_sha256": config["source_sha256"],
            "source_scene_saved": False,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "git_head": subprocess.check_output(
                ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
            ).strip(),
            "git_dirty": subprocess.check_output(
                ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--short"],
                cwd=ROOT,
                text=True,
            ),
            "python": sys.version,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "depth_pass_kind_measured": summary["calibration"]["depth_pass_kind_measured"],
            "source_files": {p.name: sha256(p) for p in frozen.iterdir()},
            "storage_boundary": "RGB-only input bundle; camera/geometry/depth/masks and derived QA RGB restricted to eval_gt. No model inference performed.",
        }
        write_json(gt / "provenance.json", provenance)
        hashes = {
            p.relative_to(gt).as_posix(): {"sha256": sha256(p), "byte_size": p.stat().st_size}
            for p in gt.rglob("*")
            if p.is_file() and p.name not in ("status.json", "artifact_hashes.json")
        }
        write_json(gt / "artifact_hashes.json", hashes)
        write_json(
            gt / "status.json",
            {
                "state": "complete",
                "view_count": summary["view_count"],
                "artifact_hashes_sha256": sha256(gt / "artifact_hashes.json"),
            },
        )
        print(f"PAIRED_BUNDLE_COMPLETE {gt}", flush=True)
    except BaseException as error:
        write_json(gt / "status.json", {"state": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
