"""Export held-out camera angles to eval_gt only, with sparse visibility ground truth."""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from thin_pack_gt import MeshRays, check_blender_rays, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/thin_heldout_v1.json"


def visibility_samples(rays, camera, endpoints, segment_id, rod_id, spacing, is_gap=False):
    a, b = np.asarray(endpoints, dtype=np.float64)
    count = int(np.ceil(np.linalg.norm(b - a) / spacing)) + 1
    fraction = np.linspace(0, 1, count)
    points = a[None] * (1 - fraction[:, None]) + b[None] * fraction[:, None]
    q = np.c_[points, np.ones(count)] @ np.asarray(camera["world_to_camera_cv"]).T
    p = q[:, :3] @ np.asarray(camera["K_edge"]).T
    uv = p[:, :2] / p[:, 2, None]
    z, sid, _ = rays.cast(camera, uv)
    width, height = camera["size_wh"]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    clipped = (q[:, 2] >= camera["clip_start"]) & (q[:, 2] <= camera["clip_end"])
    # 中心线在实体内部：可见性看射线先击中哪根杆，不要求表面Z等于轴心Z。
    visible = inside & clipped & (sid == segment_id) if not is_gap else np.zeros(count, bool)
    gap_clear = (
        inside & clipped & (~np.isfinite(z) | (z > q[:, 2] + 1e-4))
        if is_gap
        else np.zeros(count, bool)
    )
    return {
        "world_points": points,
        "uv_edge": uv,
        "uv_index": uv - 0.5,
        "target_camera_z": q[:, 2],
        "segment_fraction": fraction,
        "segment_surface_id": np.full(count, segment_id, np.uint32),
        "rod_id": np.full(count, rod_id, np.uint32),
        "inside_image": inside,
        "within_clip": clipped,
        "first_hit_depth_z": z,
        "first_hit_surface_id": sid,
        "visible": visible,
        "intentional_gap": np.full(count, is_gap, bool),
        "gap_clear": gap_clear,
    }


def process(output, config):
    render = read_json(output / "render_manifest.json")
    result = {
        "schema_version": "1.0.0",
        "bundle_id": config["bundle_id"],
        "document_type": "thin_rod_heldout_evaluation_views",
        "eval_only": True,
        "scope": config["scope"],
        "groups": [],
        "validation": [],
    }
    for case in render["cases"]:
        folder = output / case["id"]
        geometry = read_json(folder / "geometry.json")
        rays = MeshRays(folder / "mesh.npz")
        group = {
            "case_id": case["id"],
            "mesh": f"{case['id']}/mesh.npz",
            "geometry": f"{case['id']}/geometry.json",
            "frames": [],
        }
        for view in case["views"]:
            prefix = Path(case["id"]) / view["frame_id"]
            target = output / prefix
            camera = read_json(target / "camera.json")
            with Image.open(target / "rgb.png") as rgb:
                assert list(rgb.size) == camera["size_wh"] and rgb.mode == "RGB"
            checks = {"case_id": case["id"], **view}
            checks["blender_grid_rays"] = check_blender_rays(
                rays,
                camera,
                read_json(target / "blender_ray_probes.json"),
                config["ray_tolerance_m"],
            )
            line_probes = read_json(target / "blender_centerline_probes.json")
            checks["blender_centerline_rays"] = check_blender_rays(
                rays, camera, line_probes, config["ray_tolerance_m"]
            )
            segments = [
                obj
                for obj in geometry["objects"]
                if "world_centerline_endpoints" in obj and obj["duplicate_geometry_of"] is None
            ]
            arrays = [
                visibility_samples(
                    rays,
                    camera,
                    obj["world_centerline_endpoints"],
                    obj["surface_id"],
                    obj["rod_id"],
                    config["sample_spacing_m"],
                )
                for obj in segments
            ]
            arrays += [
                visibility_samples(
                    rays,
                    camera,
                    gap["world_endpoints"],
                    0,
                    gap["rod_id"],
                    config["sample_spacing_m"],
                    True,
                )
                for gap in geometry["gaps"]
            ]
            samples = {name: np.concatenate([a[name] for a in arrays]) for name in arrays[0]}
            np.savez_compressed(target / "curve_visibility.npz", **samples)
            checks["curve_samples"] = len(samples["visible"])
            checks["visible_by_segment"] = {
                str(obj["surface_id"]): int(
                    np.count_nonzero(
                        samples["visible"] & (samples["segment_surface_id"] == obj["surface_id"])
                    )
                )
                for obj in segments
            }
            checks["gap_samples"] = int(samples["intentional_gap"].sum())
            checks["gap_clear_samples"] = int(samples["gap_clear"].sum())
            gap_inside = (
                samples["intentional_gap"] & samples["inside_image"] & samples["within_clip"]
            )
            checks["gap_interior_wrong_rod_hits"] = int(
                np.count_nonzero(
                    gap_inside
                    & (samples["segment_fraction"] > 0.01)
                    & (samples["segment_fraction"] < 0.99)
                    & np.isin(samples["first_hit_surface_id"], [6, 7])
                )
            )
            assert checks["gap_interior_wrong_rod_hits"] == 0
            frame = {
                "frame_id": view["frame_id"],
                "angle_degrees": view["angle_degrees"],
                "size_wh": camera["size_wh"],
            }
            for key, name in (
                ("rgb", "rgb.png"),
                ("camera", "camera.json"),
                ("curve_visibility", "curve_visibility.npz"),
            ):
                frame[key] = (prefix / name).as_posix()
                frame[key + "_sha256"] = sha256(target / name)
            group["frames"].append(frame)
            result["validation"].append(checks)
        result["groups"].append(group)
    result["view_count"] = sum(len(g["frames"]) for g in result["groups"])
    assert result["view_count"] == 6
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender", default=r"D:\CloudMusic\steam\steamapps\common\Blender\blender.exe"
    )
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    config = read_json(CONFIG)
    assert config["bundle_id"] == "thin-heldout-v1-20260916"
    assert config["heldout_angles_degrees"] == [-22.5, 22.5]
    assert {c["id"] for c in config["cases"]} == {
        "black-white-thinner",
        "brick-texture-thinner",
        "brick-texture-origin",
    }
    paired_config = read_json(ROOT / "configs/thin_pack_export.json")
    assert set(config["heldout_angles_degrees"]).isdisjoint(paired_config["angles"])
    assert all(case in paired_config["cases"] for case in config["cases"])
    scene_path = ROOT / config["source_scene"]
    assert sha256(scene_path) == config["source_sha256"]
    for name in ("verified_geometry", "paired_render_manifest", "source_input_manifest"):
        assert sha256(ROOT / config[name]) == config[name + "_sha256"]
    original_manifest = read_json(ROOT / config["source_input_manifest"])
    for case in config["cases"]:
        group = next(g for g in original_manifest["groups"] if g["case_id"] == case["id"])
        assert group["frame_order"] == ["view_-30", "view_-15", "view_+00", "view_+15", "view_+30"]
    output = ROOT / "data/eval_gt" / config["bundle_id"]
    assert not (ROOT / "data/inputs" / config["bundle_id"]).exists(), (
        "Evaluation-only bundle leaked into inputs"
    )
    output.mkdir(exist_ok=False)
    write_json(output / "status.json", {"state": "running", "eval_only": True})
    started = time.monotonic()
    try:
        frozen = output / "producer_sources"
        frozen.mkdir()
        source_files = [
            Path(__file__),
            ROOT / "scripts/blender_export_thin_heldout.py",
            ROOT / "scripts/blender_export_thin_pack.py",
            ROOT / "scripts/thin_pack_gt.py",
            CONFIG,
        ]
        before = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
        for source in source_files:
            shutil.copy2(source, frozen / source.name)
        write_json(output / "export_config.json", config)
        request = {
            "root": str(ROOT),
            "output": output.relative_to(ROOT).as_posix(),
            "config": config,
        }
        write_json(output / "render_request.json", request)
        environment = os.environ.copy()
        for setting in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
            profile = ROOT / ".local/blender-profile" / setting.lower()
            profile.mkdir(parents=True, exist_ok=True)
            environment["BLENDER_USER_" + setting] = str(profile)
        command = [
            args.blender,
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(scene_path),
            "--python-exit-code",
            "1",
            "--python",
            str(frozen / "blender_export_thin_heldout.py"),
            "--",
            str(output / "render_request.json"),
        ]
        with (output / "blender.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        summary = process(output, config)
        assert sha256(scene_path) == config["source_sha256"]
        assert before == {str(p.relative_to(ROOT)): sha256(p) for p in source_files}, (
            "Producer changed during export"
        )
        write_json(
            output / "provenance.json",
            {
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "source_files": before,
                "source_scene_sha256": config["source_sha256"],
                "source_scene_saved": False,
                "git_head": subprocess.check_output(
                    ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    text=True,
                ).strip(),
                "eval_only": True,
                "model_inference_performed": False,
                "storage_boundary": "Everything is inside eval_gt; these RGB/cameras/labels must not enter fitting or parameter selection.",
            },
        )
        (output / "README.md").write_text(
            "# Same-asset held-out camera diagnostic\n\n"
            "All files are evaluation-only. The six RGB frames at -22.5/+22.5 degrees must never be model inputs, "
            "ROI guides, fitting data, or parameter-selection data. This tests new views of one existing asset, not new-object generalization.\n\n"
            "manifest.json lists RGB, camera and curve_visibility.npz paths/hashes. Geometry contains the physical rod mapping: "
            "surface IDs 1..5 are whole rods; IDs 6 and 7 are the two segments of physical rod 6. Cylinder.003 is an unchanged "
            "render duplicate of Cylinder.002 and is deduplicated in the mesh/labels.\n\n"
            "Each curve sample stores world_points, uv_edge (pixel centers at i+0.5), uv_index (i), target_camera_z, "
            "segment_fraction, segment_surface_id, rod_id, inside_image, within_clip, first_hit_depth_z, first_hit_surface_id, "
            "visible, intentional_gap and gap_clear. Samples have at most 0.01 m spacing, including endpoints. "
            "Visible means the projected axis point is on-screen/in-clip and its ray first hits that segment; it does not mean "
            "a pure renderer pixel or resolved RGB line. Gap-clear means no surface before the sampled gap point. "
            "Invisible/occluded/out-of-frame samples remain in the arrays. Gap endpoints are boundary samples, not gap interior.\n\n"
            "blender_ray_probes.json and blender_centerline_probes.json contain independent Blender ray answers. "
            "The manifest records Open3D agreement and Blender world_to_camera_view projection errors. No full depth/masks are exported.\n",
            encoding="utf-8",
        )
        hashes = {
            p.relative_to(output).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name not in ("status.json", "artifact_hashes.json")
        }
        write_json(output / "artifact_hashes.json", hashes)
        write_json(
            output / "status.json",
            {
                "state": "complete",
                "eval_only": True,
                "view_count": summary["view_count"],
                "artifact_hashes_sha256": sha256(output / "artifact_hashes.json"),
            },
        )
        print("HELDOUT_COMPLETE " + str(output), flush=True)
    except BaseException as error:
        write_json(
            output / "status.json", {"state": "failed", "error": str(error), "eval_only": True}
        )
        raise


if __name__ == "__main__":
    main()
