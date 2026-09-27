"""Freeze declared CAD before rendering a metric-fixture acquisition."""
from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from prepare_rod_heldout_views import array_checks
from prepare_rod_reference import (
    ROOT,
    digest,
    locations,
    raster_center_truth,
    read_json,
    source_snapshot,
    validate_render_manifest,
    write_json,
)

CONFIG = ROOT / "configs/rod_fixture_v1.json"
CAD = ROOT / "configs/rod_fixture_cad_v1.json"
RUN_ID = "rod-fixture-scenes-v1-20260927"


def prepare():
    from thin_pack_gt import MeshRays, check_blender_rays

    config, cad = read_json(CONFIG), read_json(CAD)
    assert config["run_id"] == RUN_ID and digest(CAD) == config["declared_cad_sha256"]
    parent = ROOT / ".runtime/experiments/rod-heldout-view-scenes-v1-20260926"
    frozen = read_json(parent / "prepared.json")
    for name, sha in frozen["source_sha256"].items():
        assert digest(ROOT / name) == sha == digest(parent / "source_snapshot" / name)
    old = read_json(ROOT / ".runtime/experiments/rod-height-scenes-v1-20260923/protocol.json")
    assert config["cases"] == old["cases"] and config["generator"] == old["generator"]
    run, inputs, truth, evaluation = locations(RUN_ID)
    for folder in (run, inputs, truth, evaluation):
        if folder.exists():
            raise FileExistsError("Preserve prior attempt: " + str(folder))
    for folder in (run, inputs, truth):
        folder.mkdir(parents=True)
    staging, assets = run / "rendered-rgb", inputs / "fixture-artwork"
    staging.mkdir()
    assets.mkdir()
    write_json(run / "protocol.json", config)
    hashes = source_snapshot(run, config["sources"])
    write_json(run / "generation-freeze.json", dict(source_sha256=hashes,
        declared_cad_sha256=digest(CAD), protocol_sha256=digest(run / "protocol.json"),
        declaration_precedes_render=True, created_at_utc=datetime.now(timezone.utc).isoformat(), cad_source="New manufactured fixture design; not copied from eval_gt"))
    shutil.copy2(CAD, inputs / "declared_cad.json")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    artwork = {}
    for marker in cad["markers"]:
        image = cv2.aruco.generateImageMarker(dictionary, marker["marker_id"], 224, borderBits=1)
        path = assets / (str(marker["marker_id"]) + ".png")
        Image.fromarray(np.pad(image, 28, constant_values=255)).save(path)
        artwork[path.relative_to(inputs).as_posix()] = digest(path)
    environment = os.environ.copy()
    for key in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
        folder = ROOT / ".local/blender-profile" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        environment["BLENDER_USER_" + key] = str(folder)
    batches = []
    for i, case in enumerate(config["cases"]):
        generator = dict(config["generator"], reference_seed=case["reference_seed"], lens_by_view_mm=case["lens_by_view_mm"])
        request = dict(root=str(ROOT), inputs=staging.relative_to(ROOT).as_posix(), truth=truth.relative_to(ROOT).as_posix(),
            config=dict(config, generator=generator, cases=[case]), declared_cad=cad, marker_assets=str(assets),
            render_manifest_name=f"render-{i:02d}.json")
        request_path = run / f"case-{i:02d}-render_request.json"
        write_json(request_path, request)
        with (run / f"blender-{i:02d}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([config["blender"], "--background", "--factory-startup", "--disable-autoexec",
                "--python-exit-code", "1", "--python", str(ROOT / "scripts/blender_rod_fixture_pack.py"),
                "--", str(request_path)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError("Rendering failed; retain log and frozen attempt")
        batches.append(read_json(truth / f"render-{i:02d}.json"))
        print("FIXTURE_RENDER", case["case_id"], flush=True)
    render = dict(batches[0], cases=[case for batch in batches for case in batch["cases"]])
    write_json(truth / "render_manifest.json", render)
    validate_render_manifest(render, config, staging, truth)
    cases, truths, checks = [], [], []
    for rendered, declared in zip(render["cases"], config["cases"]):
        cid = declared["case_id"]
        assert rendered["case_id"] == cid and len(rendered["frames"]) == 5
        rays = MeshRays(truth / cid / "mesh.npz")
        frames, cameras, truth_frames = [], [], []
        for i, frame in enumerate(rendered["frames"]):
            vid, camera = f"view_{i:02d}", frame["camera"]
            arrays = raster_center_truth(rays, camera)
            native = truth / cid / vid
            native.mkdir()
            for name, array in arrays.items():
                np.save(native / (name + ".npy"), array, allow_pickle=False)
            probe = check_blender_rays(rays, camera, frame["blender_ray_probes"], .0001)
            checks.append(dict(case_id=cid, view_id=vid, ray_check=probe,
                projection_max_px=frame["projection_check_max_px"], **array_checks(arrays, camera)))
            rgb = inputs / cid / (vid + ".png")
            rgb.parent.mkdir(exist_ok=True)
            shutil.copy2(staging / frame["rgb"], rgb)
            with Image.open(rgb) as image:
                assert list(image.size) == camera["size_wh"]
                image.verify()
            assert digest(rgb) == digest(staging / frame["rgb"])
            assert frame["guide_xyxy"] == config["generator"]["guide_xyxy"]
            frames.append(dict(view_id=vid, rgb=rgb.relative_to(ROOT).as_posix(), rgb_sha256=digest(rgb),
                size_wh=camera["size_wh"], guide_xyxy=frame["guide_xyxy"], guide_source=frame["guide_source"]))
            cameras.append(dict(camera, view_id=vid))
            truth_frames.append(dict(view_id=vid, generation_frame_id=frame["frame_id"],
                rgb_sha256=digest(rgb), array_directory=f"{cid}/{vid}", ray_check=probe,
                target_visible_pixels=int(arrays["target_visible"].sum())))
        cases.append(dict(case_id=cid, frames=frames))
        truths.append(dict(case_id=cid, declared=declared, cameras=cameras, frames=truth_frames, mesh_path=f"{cid}/mesh.npz"))
    clean = dict(run_id=RUN_ID, cases=cases,
        declared_cad=dict(path=(inputs / "declared_cad.json").relative_to(ROOT).as_posix(), sha256=digest(CAD)),
        fixture_artwork_sha256=artwork,
        scope="Known manufactured metric fixture; no target geometry or camera truth in normal inputs")
    write_json(inputs / "manifest.json", clean)
    write_json(truth / "manifest.json", dict(cases=truths, input_sha256=digest(inputs / "manifest.json"),
        declared_cad_sha256=digest(CAD), camera_coordinate_system=cad["coordinate_system"]))
    artifacts = {p.relative_to(truth).as_posix(): digest(p) for p in sorted(truth.rglob("*")) if p.is_file()}
    write_json(truth / "artifact_hashes.json", artifacts)
    write_json(run / "generation-checks.json", dict(state="passed", frame_count=15, checks=checks,
        known_fixture_condition=True, target_and_camera_generation_equal_to_old_height_protocol=True))
    for name, sha in hashes.items():
        assert digest(ROOT / name) == sha == digest(run / "source_snapshot" / name)
    write_json(run / "prepared.json", dict(run_id=RUN_ID, source_sha256=hashes,
        input_sha256=digest(inputs / "manifest.json"), truth_sha256=digest(truth / "manifest.json"),
        truth_artifacts_sha256=digest(truth / "artifact_hashes.json"), protocol_sha256=digest(run / "protocol.json"),
        declared_cad_sha256=digest(CAD), generation_freeze_sha256=digest(run / "generation-freeze.json"),
        generation_checks_sha256=digest(run / "generation-checks.json"), frame_count=15,
        independent_ray_samples=sum(c["ray_check"]["samples"] for c in checks),
        surface_id_mismatches=sum(c["ray_check"]["surface_id_mismatches"] for c in checks), opencv_version=cv2.__version__))
    print("FIXTURE_PREPARED 15 strictly paired RGB frames; declared CAD frozen before rendering", flush=True)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    prepare()
