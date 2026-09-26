"""Prepare six genuinely unused RGB views of the frozen reference-board scenes.

Generation alone may read Blender geometry/cameras. The normal input manifest
contains anonymous RGB views, fixed screen guides and hash-only source receipts.
No observations, estimated camera poses, or rod decisions are made here.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess

import numpy as np
from PIL import Image
from prepare_rod_height import (
    ROOT,
    digest,
    locations,
    raster_center_truth,
    read_json,
    require_camera_free,
    source_snapshot,
    validate_render_manifest,
    write_json,
)

CONFIG = ROOT / "configs/rod_heldout_views_v1.json"
RUN_ID = "rod-heldout-view-scenes-v1-20260926"
OWN_SOURCES = {"scripts/prepare_rod_heldout_views.py", "configs/rod_heldout_views_v1.json"}


def assert_protocol(config):
    parent = ROOT / ".runtime/experiments" / config["source_parent_run_id"]
    assert config["run_id"] == RUN_ID
    assert digest(parent / "prepared.json") == config["source_parent_prepared_sha256"]
    assert digest(parent / "protocol.json") == config["source_parent_protocol_sha256"]
    frozen, old = read_json(parent / "prepared.json"), read_json(parent / "protocol.json")
    assert frozen["protocol_sha256"] == config["source_parent_protocol_sha256"]
    expected_sources = set(frozen["source_sha256"]) | OWN_SOURCES
    assert len(config["sources"]) == len(set(config["sources"])) and set(config["sources"]) == expected_sources
    for name, sha in frozen["source_sha256"].items():
        assert digest(ROOT / name) == sha == digest(parent / "source_snapshot" / name), name
    assert config["anonymous_view_ids"] == ["view_h00", "view_h01"]
    assert config["generator"]["angles_degrees"] == [-23, 27]
    assert config["generator"]["camera_heights_m"] == [1.05, -.8]
    same_generator = {key: value for key, value in config["generator"].items()
                      if key not in ("angles_degrees", "camera_heights_m")}
    old_generator = {key: value for key, value in old["generator"].items()
                     if key not in ("angles_degrees", "camera_heights_m")}
    assert same_generator == old_generator
    assert len(config["cases"]) == len(old["cases"]) == 3
    for current, previous in zip(config["cases"], old["cases"]):
        assert current["lens_by_view_mm"] == [49, 49]
        assert previous["lens_by_view_mm"] == [49] * 5
        assert {key: value for key, value in current.items() if key != "lens_by_view_mm"} == {
            key: value for key, value in previous.items() if key != "lens_by_view_mm"}
    assert config["tracks"] == old["tracks"]
    return parent, frozen


def array_checks(arrays, camera):
    width, height = camera["size_wh"]
    assert set(arrays) == {"depth_z", "surface_id", "ray_distance", "target_visible"}
    assert all(value.shape == (height, width) for value in arrays.values())
    sid, z, distance = arrays["surface_id"], arrays["depth_z"], arrays["ray_distance"]
    hit = sid != 0
    assert np.array_equal(arrays["target_visible"], sid == 1)
    assert np.array_equal(np.isfinite(z), hit) and np.array_equal(np.isfinite(distance), hit)
    assert np.all(z[hit] > 0) and np.all(distance[hit] > 0)
    # Integer pixel centers must produce the same ray length as edge-origin
    # coordinates plus half a pixel. Keep Z-depth and Euclidean range distinct.
    yy, xx = np.mgrid[:height, :width]
    directions = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(camera["K_index"]).T
    expected = z * np.linalg.norm(directions, axis=-1)
    np.testing.assert_allclose(distance[hit], expected[hit], rtol=3e-6, atol=3e-6)
    return dict(shape=[height, width], finite_depth_pixels=int(hit.sum()),
                target_visible_pixels=int(arrays["target_visible"].sum()),
                maximum_z_to_range_check_error_m=float(np.max(abs(distance[hit] - expected[hit]), initial=0.)))


def prepare(path=CONFIG):
    from thin_pack_gt import MeshRays, check_blender_rays

    config = read_json(path)
    parent, old_frozen = assert_protocol(config)
    run, inputs, truth, evaluation = locations(config["run_id"])
    for folder in (run, inputs, truth, evaluation):
        if folder.exists():
            raise FileExistsError("Preserve earlier attempt: " + str(folder))
    for folder in (run, inputs, truth):
        folder.mkdir(parents=True)
    staging = run / "rendered-rgb"
    staging.mkdir()
    write_json(run / "protocol.json", config)
    # No glob here: another agent can write a new module during rendering
    # without silently changing the scientific identity of this acquisition.
    hashes = source_snapshot(run, config["sources"])
    write_json(run / "generation-freeze.json", dict(source_sha256=hashes,
        protocol_sha256=digest(run / "protocol.json"), source_parent_prepared_sha256=digest(parent / "prepared.json"),
        scope="Explicit inherited source closure and this wrapper/config frozen before rendering"))
    environment = os.environ.copy()
    for key in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
        folder = ROOT / ".local/blender-profile" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        environment["BLENDER_USER_" + key] = str(folder)
    batches = []
    for index, case in enumerate(config["cases"]):
        generator = {**config["generator"], "reference_seed": case["reference_seed"],
                     "lens_by_view_mm": case["lens_by_view_mm"]}
        request = dict(root=str(ROOT), inputs=staging.relative_to(ROOT).as_posix(),
                       truth=truth.relative_to(ROOT).as_posix(),
                       config={**config, "generator": generator, "cases": [case]},
                       render_manifest_name=f"render-{index:02d}.json")
        request_path = run / f"case-{index:02d}-render_request.json"
        write_json(request_path, request)
        with (run / f"blender-{index:02d}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([config["blender"], "--background", "--factory-startup", "--disable-autoexec",
                "--python-exit-code", "1", "--python", str(ROOT / "scripts/blender_rod_height_pack.py"),
                "--", str(request_path)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Render failed; preserve blender-{index:02d}.log")
        batches.append(read_json(truth / f"render-{index:02d}.json"))
        print("HELDOUT_RENDER", case["case_id"], flush=True)
    render = {**batches[0], "cases": [case for batch in batches for case in batch["cases"]]}
    write_json(truth / "render_manifest.json", render)
    validate_render_manifest(render, config, staging, truth)
    parent_inputs = ROOT / "data/inputs" / config["source_parent_run_id"] / "manifest.json"
    parent_truth = ROOT / "data/eval_gt" / config["source_parent_run_id"]
    assert digest(parent_inputs) == old_frozen["input_sha256"]
    assert digest(parent_truth / "manifest.json") == old_frozen["truth_sha256"]
    assert digest(parent_truth / "artifact_hashes.json") == old_frozen["truth_artifacts_sha256"]
    parent_artifacts = read_json(parent_truth / "artifact_hashes.json")
    cases, truths, checks = [], [], []
    for rendered, declared in zip(render["cases"], config["cases"]):
        cid = declared["case_id"]
        assert rendered["case_id"] == cid and len(rendered["frames"]) == 2
        old_mesh, new_mesh = parent_truth / cid / "mesh.npz", truth / cid / "mesh.npz"
        assert digest(old_mesh) == parent_artifacts[f"{cid}/mesh.npz"]
        with np.load(old_mesh, allow_pickle=False) as old, np.load(new_mesh, allow_pickle=False) as new:
            assert set(old.files) == set(new.files)
            for key in old.files:
                assert np.array_equal(old[key], new[key]), (cid, key, "Scene geometry changed")
        rays = MeshRays(new_mesh)
        frames, cameras, truth_frames = [], [], []
        for index, frame in enumerate(rendered["frames"]):
            view_id, camera = config["anonymous_view_ids"][index], frame["camera"]
            assert frame["angle_degrees"] == config["generator"]["angles_degrees"][index]
            arrays = raster_center_truth(rays, camera)
            native = truth / cid / view_id
            native.mkdir()
            for name, array in arrays.items():
                np.save(native / (name + ".npy"), array, allow_pickle=False)
            ray_check = check_blender_rays(rays, camera, frame["blender_ray_probes"], .0001)
            checks.append(dict(case_id=cid, view_id=view_id, ray_check=ray_check,
                               projection_max_px=frame["projection_check_max_px"], **array_checks(arrays, camera)))
            rgb = inputs / cid / (view_id + ".png")
            rgb.parent.mkdir(exist_ok=True)
            shutil.copy2(staging / frame["rgb"], rgb)
            with Image.open(rgb) as image:
                assert list(image.size) == camera["size_wh"]
                image.verify()
            assert digest(rgb) == digest(staging / frame["rgb"])
            assert frame["guide_source"] == "fixed_screen_coordinates_independent_of_target_geometry"
            assert frame["guide_xyxy"] == config["generator"]["guide_xyxy"]
            frames.append(dict(view_id=view_id, rgb=rgb.relative_to(ROOT).as_posix(), rgb_sha256=digest(rgb),
                size_wh=camera["size_wh"], guide_xyxy=frame["guide_xyxy"], guide_source=frame["guide_source"]))
            cameras.append(dict(camera, view_id=view_id))
            truth_frames.append(dict(view_id=view_id, generation_frame_id=frame["frame_id"],
                rgb_sha256=digest(rgb), array_directory=f"{cid}/{view_id}", ray_check=ray_check,
                target_visible_pixels=int(arrays["target_visible"].sum()),
                projection_max_px=frame["projection_check_max_px"]))
        cases.append(dict(case_id=cid, frames=frames))
        truths.append(dict(case_id=cid, declared=declared, cameras=cameras, frames=truth_frames,
                           mesh_path=f"{cid}/mesh.npz", parent_mesh_sha256=digest(old_mesh), parent_mesh_arrays_equal=True))
    provenance = dict(parent_scene_run_id=config["source_parent_run_id"],
        parent_prepared_path=(parent / "prepared.json").relative_to(ROOT).as_posix(),
        parent_prepared_sha256=digest(parent / "prepared.json"),
        parent_input_manifest=parent_inputs.relative_to(ROOT).as_posix(), parent_input_sha256=digest(parent_inputs))
    clean = dict(cases=cases, method=config["tracks"], scene_source_provenance=provenance)
    require_camera_free(clean)
    assert len(cases) == 3 and sum(len(case["frames"]) for case in cases) == 6
    write_json(inputs / "manifest.json", clean)
    write_json(truth / "manifest.json", dict(cases=truths, input_sha256=digest(inputs / "manifest.json"),
        scene_source_provenance=provenance))
    artifact_hashes = {p.relative_to(truth).as_posix(): digest(p) for p in sorted(truth.rglob("*")) if p.is_file()}
    write_json(truth / "artifact_hashes.json", artifact_hashes)
    write_json(run / "generation-checks.json", dict(state="passed", frame_count=6, checks=checks,
        parent_mesh_arrays_equal=True, anonymous_rgb_copy_sha_equal=True))
    for name, sha in hashes.items():
        assert digest(ROOT / name) == sha == digest(run / "source_snapshot" / name), name
    assert_protocol(config)
    write_json(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=hashes,
        input_sha256=digest(inputs / "manifest.json"), truth_sha256=digest(truth / "manifest.json"),
        truth_artifacts_sha256=digest(truth / "artifact_hashes.json"), protocol_sha256=digest(run / "protocol.json"),
        generation_freeze_sha256=digest(run / "generation-freeze.json"),
        generation_checks_sha256=digest(run / "generation-checks.json"),
        source_parent_prepared_sha256=digest(parent / "prepared.json"), frame_count=6,
        independent_ray_samples=sum(check["ray_check"]["samples"] for check in checks),
        surface_id_mismatches=sum(check["ray_check"]["surface_id_mismatches"] for check in checks)))
    print("HELDOUT_PREPARED", 6, "strictly paired anonymous RGB frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    prepare(args.config)
