"""Freeze and render two synthetic assemblies, isolating all target geometry."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from blender_g1_object_pack import FIXED_GUIDE, GUIDE_SOURCE, validate_request
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

CONFIG = ROOT / "configs/g1_object_pilot_v1.json"
CAD = ROOT / "configs/rod_fixture_cad_v1.json"
PARENT_CONFIG = ROOT / "configs/rod_fixture_v1.json"
RUN_ID = "g1-object-scenes-v1-20261006"
CASE_IDS = ["chair01", "aframe01"]
FRAME_FIELDS = {"view_id", "rgb", "rgb_sha256", "size_wh", "guide_xyxy", "guide_source"}
OWN_SOURCES = {
    "scripts/blender_g1_object_pack.py",
    "scripts/prepare_g1_object_pack.py",
    "configs/g1_object_pilot_v1.json",
    "tests/test_g1_object_pack.py",
    # These imports are real generation dependencies absent from the old list.
    "scripts/run_rod_identity_blender.py",
    "reconstruction/src/creator_recon/__init__.py",
    "reconstruction/src/creator_recon/domain/__init__.py",
}


def validate_config(config, parent_config):
    if config["run_id"] != RUN_ID or [c["case_id"] for c in config["cases"]] != CASE_IDS:
        raise ValueError("Run or complete two-assembly case inventory differs")
    if config["source_parent_run_id"] != "rod-fixture-scenes-v1-20260927":
        raise ValueError("Unexpected source parent")
    if config["generator"] != parent_config["generator"] or config["blender"] != parent_config["blender"]:
        raise ValueError("Keep the declared fixture acquisition unchanged")
    if config["declared_cad_sha256"] != parent_config["declared_cad_sha256"]:
        raise ValueError("Manufactured fixture must remain unchanged")
    validate_request(dict(config=config))
    asset_ids = [case["source_asset_id"] for case in config["cases"]]
    group_ids = [case["object_group_id"] for case in config["cases"]]
    if len(set(asset_ids)) != 2 or len(set(group_ids)) != 2:
        raise ValueError("Two different procedural assembly identities required")
    for case in config["cases"]:
        if case["acquisition_kind"] != "synthetic_development" or case["independent_capture"] is not False:
            raise ValueError("Synthetic assemblies cannot claim independent real capture")
        if case["lens_by_view_mm"] != [49] * 5 or not case["distractors"]:
            raise ValueError("Keep five fixed-lens views and real assembly context")
    return len(config["cases"]) * len(config["generator"]["angles_degrees"])


def source_names(parent_frozen):
    return set(parent_frozen["source_sha256"]) | OWN_SOURCES


def verify_inherited_sources(config, frozen, parent):
    migrations = config.get("source_migrations", [])
    allowed_path = "experiments/src/creator_eval/rod_observations.py"
    if len(migrations) > 1 or any(m["path"] != allowed_path for m in migrations):
        raise ValueError("Only the declared historical observation-source migration is permitted")
    indexed = {item["path"]: item for item in migrations}
    if not set(indexed) <= set(frozen["source_sha256"]):
        raise ValueError("Migration must identify an inherited source")
    for name, sha in frozen["source_sha256"].items():
        if digest(parent / "source_snapshot" / name) != sha:
            raise ValueError("Original frozen parent snapshot changed: " + name)
        expected = sha
        if name in indexed:
            migration = indexed[name]
            if (migration["old_sha256"] != sha
                    or migration["source_run_id"] != "rod-fixture-narrow-v1-20260929"):
                raise ValueError("Historical migration identity differs")
            successor = ROOT / ".runtime/experiments" / migration["source_run_id"]
            if digest(successor / "prepared.json") != migration["prepared_sha256"]:
                raise ValueError("Historical successor receipt changed")
            newer = read_json(successor / "prepared.json")
            expected = migration["new_sha256"]
            if (newer["source_sha256"].get(name) != expected
                    or digest(successor / "source_snapshot" / name) != expected):
                raise ValueError("Historical successor source identity differs")
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen inherited source changed without authorized migration: " + name)
    return copy.deepcopy(migrations)


def require_new_paths(folders):
    for folder in folders:
        if folder.exists():
            raise FileExistsError("Preserve earlier attempt: " + str(folder))


def normal_manifest(cases, cad_path, cad_sha, artwork):
    """Construct, rather than redact, the normal input allowlist."""
    clean_cases = []
    if [case["case_id"] for case in cases] != CASE_IDS:
        raise ValueError("Normal manifest needs the complete fixed case order")
    for case in cases:
        frames = case["frames"]
        if [frame["view_id"] for frame in frames] != [f"view_{i:02d}" for i in range(5)]:
            raise ValueError("Normal manifest needs all five ordered views")
        clean_frames = []
        for frame in frames:
            if frame["guide_xyxy"] != FIXED_GUIDE or frame["guide_source"] != GUIDE_SOURCE:
                raise ValueError("Normal inputs require target-independent guides")
            clean_frames.append({key: copy.deepcopy(frame[key]) for key in sorted(FRAME_FIELDS)})
        clean_cases.append(dict(case_id=case["case_id"], frames=clean_frames))
    return dict(run_id=RUN_ID, cases=clean_cases, declared_cad=dict(path=cad_path, sha256=cad_sha),
                fixture_artwork_sha256=dict(artwork),
                scope="RGB and fixed screen guides with publicly supplied metric fixture CAD; target and camera truth excluded")


def prepare(path=CONFIG):
    from thin_pack_gt import MeshRays, check_blender_rays

    if Path(path).resolve() != CONFIG.resolve():
        raise ValueError("This frozen pilot accepts only its declared config path")
    config, parent_config, cad = read_json(path), read_json(PARENT_CONFIG), read_json(CAD)
    frame_count = validate_config(config, parent_config)
    if digest(PARENT_CONFIG) != config["source_parent_config_sha256"] or digest(CAD) != config["declared_cad_sha256"]:
        raise ValueError("Parent configuration or manufactured CAD changed")
    parent = ROOT / ".runtime/experiments" / config["source_parent_run_id"]
    if digest(parent / "prepared.json") != config["source_parent_prepared_sha256"]:
        raise ValueError("Frozen source parent receipt changed")
    frozen = read_json(parent / "prepared.json")
    migrations = verify_inherited_sources(config, frozen, parent)
    run, inputs, truth, evaluation = locations(RUN_ID)
    require_new_paths((run, inputs, truth, evaluation))
    for folder in (run, inputs, truth):
        folder.mkdir(parents=True)
    staging, assets = run / "rendered-rgb", inputs / "fixture-artwork"
    staging.mkdir()
    assets.mkdir()
    write_json(run / "protocol.json", config)
    hashes = source_snapshot(run, source_names(frozen))
    write_json(run / "generation-freeze.json", dict(source_sha256=hashes,
        declared_cad_sha256=digest(CAD), protocol_sha256=digest(run / "protocol.json"),
        declaration_precedes_render=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_parent_prepared_sha256=digest(parent / "prepared.json"),
        source_migrations=migrations,
        scope="New synthetic development assembly designs; no claim of real or independent capture"))
    shutil.copy2(CAD, inputs / "declared_cad.json")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    artwork = {}
    for marker in cad["markers"]:
        image = cv2.aruco.generateImageMarker(dictionary, marker["marker_id"], 224, borderBits=1)
        image_path = assets / (str(marker["marker_id"]) + ".png")
        Image.fromarray(np.pad(image, 28, constant_values=255)).save(image_path)
        artwork[image_path.relative_to(inputs).as_posix()] = digest(image_path)
    environment = os.environ.copy()
    for key in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
        folder = ROOT / ".local/blender-profile" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        environment["BLENDER_USER_" + key] = str(folder)
    batches = []
    for index, case in enumerate(config["cases"]):
        generator = dict(config["generator"], reference_seed=case["reference_seed"], lens_by_view_mm=case["lens_by_view_mm"])
        request = dict(root=str(ROOT), inputs=staging.relative_to(ROOT).as_posix(), truth=truth.relative_to(ROOT).as_posix(),
            config=dict(config, generator=generator, cases=[case]), declared_cad=cad, marker_assets=str(assets),
            render_manifest_name=f"render-{index:02d}.json")
        validate_request(request)
        request_path = run / f"case-{index:02d}-render_request.json"
        write_json(request_path, request)
        with (run / f"blender-{index:02d}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([config["blender"], "--background", "--factory-startup", "--disable-autoexec",
                "--python-exit-code", "1", "--python", str(ROOT / "scripts/blender_g1_object_pack.py"),
                "--", str(request_path)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError("Rendering failed; preserve all logs and frozen inputs")
        batches.append(read_json(truth / f"render-{index:02d}.json"))
        print("G1_OBJECT_RENDER", case["case_id"], flush=True)
    render = dict(batches[0], cases=[case for batch in batches for case in batch["cases"]])
    write_json(truth / "render_manifest.json", render)
    validate_render_manifest(render, config, staging, truth)
    cases, truths, checks = [], [], []
    for rendered, declared in zip(render["cases"], config["cases"]):
        cid = declared["case_id"]
        if (rendered["case_id"] != cid
                or [frame["angle_degrees"] for frame in rendered["frames"]] != config["generator"]["angles_degrees"]):
            raise ValueError("Rendered case/view inventory differs")
        rays = MeshRays(truth / cid / "mesh.npz")
        frames, cameras, truth_frames = [], [], []
        for index, frame in enumerate(rendered["frames"]):
            vid, camera = f"view_{index:02d}", frame["camera"]
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
                if list(image.size) != camera["size_wh"]:
                    raise ValueError("Published RGB size changed")
                image.verify()
            if digest(rgb) != digest(staging / frame["rgb"]):
                raise ValueError("Published RGB bytes changed")
            frames.append(dict(view_id=vid, rgb=rgb.relative_to(ROOT).as_posix(), rgb_sha256=digest(rgb),
                size_wh=camera["size_wh"], guide_xyxy=frame["guide_xyxy"], guide_source=frame["guide_source"]))
            cameras.append(dict(camera, view_id=vid))
            truth_frames.append(dict(view_id=vid, generation_frame_id=frame["frame_id"],
                rgb_sha256=digest(rgb), array_directory=f"{cid}/{vid}", ray_check=probe,
                target_visible_pixels=int(arrays["target_visible"].sum())))
        cases.append(dict(case_id=cid, frames=frames))
        truths.append(dict(case_id=cid, declared=declared, cameras=cameras, frames=truth_frames, mesh_path=f"{cid}/mesh.npz"))
    if len(checks) != frame_count:
        raise ValueError("Incomplete generated frame inventory")
    clean = normal_manifest(cases, (inputs / "declared_cad.json").relative_to(ROOT).as_posix(), digest(CAD), artwork)
    write_json(inputs / "manifest.json", clean)
    write_json(truth / "manifest.json", dict(cases=truths, input_sha256=digest(inputs / "manifest.json"),
        declared_cad_sha256=digest(CAD), camera_coordinate_system=cad["coordinate_system"]))
    artifacts = {p.relative_to(truth).as_posix(): digest(p) for p in sorted(truth.rglob("*")) if p.is_file()}
    write_json(truth / "artifact_hashes.json", artifacts)
    write_json(run / "generation-checks.json", dict(state="passed", frame_count=frame_count, checks=checks,
        known_fixture_condition=True, same_fixture_acquisition_as_parent=True,
        different_synthetic_assembly_count=2, independent_capture=False, gt_used_only_for_data_generation=True))
    for name, sha in hashes.items():
        if digest(ROOT / name) != sha or digest(run / "source_snapshot" / name) != sha:
            raise ValueError("Generation source changed during preparation: " + name)
    write_json(run / "prepared.json", dict(run_id=RUN_ID, source_sha256=hashes,
        input_sha256=digest(inputs / "manifest.json"), truth_sha256=digest(truth / "manifest.json"),
        truth_artifacts_sha256=digest(truth / "artifact_hashes.json"), protocol_sha256=digest(run / "protocol.json"),
        declared_cad_sha256=digest(CAD), generation_freeze_sha256=digest(run / "generation-freeze.json"),
        source_migrations=migrations,
        generation_checks_sha256=digest(run / "generation-checks.json"), frame_count=frame_count,
        independent_ray_samples=sum(c["ray_check"]["samples"] for c in checks),
        surface_id_mismatches=sum(c["ray_check"]["surface_id_mismatches"] for c in checks), opencv_version=cv2.__version__))
    print("G1_OBJECT_PREPARED", frame_count, "strictly paired RGB; two synthetic development assemblies", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    prepare(args.config)
