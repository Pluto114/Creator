"""RGB-only fixture calibration and unchanged narrow finite-line inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import PIL
import scipy
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.fixture_calibration import (  # noqa: E402
    POLICY,
    calibrate_case,
    detect_markers,
    validate_cad,
)
from creator_eval.rod_fixture_finite import (  # noqa: E402
    METHODS,
    canonical_hash,
    extract_fixture_narrow_evidence,
    json_ready,
    narrow_method_policy,
    reconstruct_fixture_narrow,
)

RUN_ID, SCENE_ID = "g1-object-rgb-v1-20261006", "g1-object-scenes-v1-20261006"
RUN, SCENE = ROOT / ".runtime/experiments" / RUN_ID, ROOT / ".runtime/experiments" / SCENE_ID
INPUTS, SCENE_INPUTS = ROOT / "data/inputs" / RUN_ID, ROOT / "data/inputs" / SCENE_ID
CONFIG = ROOT / "configs/g1_object_rgb_v1.json"
METHOD_PARENT = ROOT / ".runtime/experiments/rod-fixture-narrow-v1-20260929"
CALIBRATION_PARENT = ROOT / ".runtime/experiments/rod-fixture-calibration-v1-20260927"
FRAME_FIELDS = {"view_id", "rgb", "rgb_sha256", "size_wh", "guide_xyxy", "guide_source"}
ROOT_FIELDS = {"run_id", "cases", "declared_cad", "fixture_artwork_sha256", "scope"}
CALIBRATION_SOURCES = {"experiments/src/creator_eval/fixture_calibration.py", "tests/test_fixture_calibration.py"}
# Explicit additional transitive import: rod_cylinder_gate -> rod_radius_consistency.
EXTRA_SOURCES = CALIBRATION_SOURCES | {
    "experiments/src/creator_eval/rod_radius_consistency.py", "experiments/src/creator_eval/__init__.py",
    "scripts/environment_paths.py", "scripts/run_g1_object_rgb.py", "configs/g1_object_rgb_v1.json",
    "tests/test_g1_object_rgb.py"}
GENERATION_CONFIGS = {"g1_object_pilot_v1.json", "rod_fixture_v1.json", "rod_heldout_views_v1.json"}
_HASH_ONLY_PATH = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_ready(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(path):
    global _HASH_ONLY_PATH
    path = Path(path).resolve()
    previous = _HASH_ONLY_PATH
    try:
        # The Windows audit event normalizes text/binary open alike. Authorize
        # only this exact hash helper's synchronous read, never general rb reads.
        _HASH_ONLY_PATH = path if path.name in GENERATION_CONFIGS else None
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    finally:
        _HASH_ONLY_PATH = previous


def receipt(path, hashes, expected=None):
    path = Path(path).resolve()
    name, sha = path.relative_to(ROOT).as_posix(), digest(path)
    if (expected is not None and expected != sha) or (name in hashes and hashes[name] != sha):
        raise ValueError("Frozen receipt changed: " + name)
    hashes[name] = sha
    return sha


def block_truth(event, arguments):
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    forbidden = (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments/results")
    if (any(path.is_relative_to(folder) for folder in forbidden)
            or path.name in {"protocol.json", "generation-checks.json", "evaluation.json", "post.json"}
            or path.name.endswith("render_request.json") or "rendered-rgb" in path.parts or "staging" in path.parts):
        raise PermissionError("Normal RGB inference cannot read target/camera truth, staging or scores")
    if path.name in GENERATION_CONFIGS and path != _HASH_ONLY_PATH:
        raise PermissionError("Generation configuration may only be accessed by the byte-hash helper")


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                pillow=PIL.__version__, scipy=scipy.__version__)


def validated_config():
    config = read(CONFIG)
    expected = dict(run_id=RUN_ID, scene_run_id=SCENE_ID,
        scene_prepared_sha256="d437619e2fd66412343b7f92c92ef7d96714e7998df3b801d6e8f366f763b2ed",
        scene_input_sha256="bd56da4a289c78280d641d2893eb2e548b82e4758a46324a00b908a4dfaa722c",
        method_source_run_id=METHOD_PARENT.name, calibration_source_run_id=CALIBRATION_PARENT.name,
        method_sha256="b33ce471de3fbe6b7387d78494c7675108a68793fa9d68711a5eb6a1f4c3072e",
        case_ids=["chair01", "aframe01"], view_ids=[f"view_{i:02d}" for i in range(5)],
        methods=list(METHODS), method_overrides={}, calibration_overrides={})
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen object RGB protocol differs")
    return config


def validate_manifest(manifest, config):
    if set(manifest) != ROOT_FIELDS or manifest["run_id"] != SCENE_ID:
        raise ValueError("Anonymous normal root allowlist differs")
    if [case.get("case_id") for case in manifest["cases"]] != config["case_ids"]:
        raise ValueError("Complete ordered two-object inventory required")
    if (set(manifest["declared_cad"]) != {"path", "sha256"}
            or manifest["declared_cad"]["path"] != f"data/inputs/{SCENE_ID}/declared_cad.json"):
        raise ValueError("Only explicitly supplied metric fixture CAD is allowed")
    for case in manifest["cases"]:
        if set(case) != {"case_id", "frames"} or [f.get("view_id") for f in case["frames"]] != config["view_ids"]:
            raise ValueError("Five ordered anonymous RGB frames required")
        for frame in case["frames"]:
            if (set(frame) != FRAME_FIELDS or frame["rgb"] != f"data/inputs/{SCENE_ID}/{case['case_id']}/{frame['view_id']}.png"
                    or frame["size_wh"] != [640, 480] or frame["guide_xyxy"] != [[319.5, 20], [319.5, 459]]
                    or frame["guide_source"] != "fixed_screen_coordinates_independent_of_target_geometry"):
                raise ValueError("Anonymous RGB frame or target-independent guide differs")
            if not isinstance(frame["rgb_sha256"], str) or len(frame["rgb_sha256"]) != 64:
                raise ValueError("Every normal RGB needs a content SHA")
    return manifest


def source_inventory(parent_prepared):
    return sorted(set(parent_prepared["source_sha256"]) | EXTRA_SOURCES)


def verify_snapshot(folder, prepared, hashes, *, current=True, selected=None):
    sources = prepared["source_sha256"]
    for name in sorted(sources if selected is None else selected):
        receipt(folder / "source_snapshot" / name, hashes, sources[name])
        if current:
            receipt(ROOT / name, hashes, sources[name])


def checked():
    prepared, config = read(RUN / "prepared.json"), validated_config()
    if (prepared["run_id"] != RUN_ID or prepared["gt_read_during_prepare"] or prepared["runtime"] != runtime()
            or prepared["config"] != config or prepared["calibration_policy"] != POLICY):
        raise ValueError("Invalid object RGB freeze or runtime")
    for name, sha in prepared["receipts"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen object RGB receipt changed: " + name)
    for name, sha in prepared["source_sha256"].items():
        if digest(ROOT / name) != sha or digest(RUN / "source_snapshot" / name) != sha:
            raise ValueError("Frozen object RGB source changed: " + name)
    if (digest(INPUTS / "manifest.json") != prepared["input_sha256"]
            or digest(RUN / "method_config.json") != prepared["method_config_sha256"]):
        raise ValueError("Frozen normal manifest or method config changed")
    manifest, method_config = validate_manifest(read(INPUTS / "manifest.json"), config), read(RUN / "method_config.json")
    narrow_method_policy(method_config["method"])
    if method_config["method_sha256"] != config["method_sha256"] or canonical_hash(method_config["method"]) != config["method_sha256"]:
        raise ValueError("Unchanged narrow method SHA differs")
    return prepared, manifest, method_config


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or INPUTS.exists():
        raise FileExistsError("Preserve previous object RGB attempt")
    config, hashes = validated_config(), {}
    receipt(SCENE / "prepared.json", hashes, config["scene_prepared_sha256"])
    scene = read(SCENE / "prepared.json")
    if scene["run_id"] != SCENE_ID or scene["frame_count"] != 10:
        raise ValueError("Complete scene freeze required")
    receipt(SCENE_INPUTS / "manifest.json", hashes, config["scene_input_sha256"])
    if scene["input_sha256"] != config["scene_input_sha256"]:
        raise ValueError("Frozen anonymous input SHA differs")
    # Every generated source snapshot/current source is hash-checked, not parsed.
    verify_snapshot(SCENE, scene, hashes)
    manifest = validate_manifest(read(SCENE_INPUTS / "manifest.json"), config)
    cad_path = ROOT / manifest["declared_cad"]["path"]
    receipt(cad_path, hashes, manifest["declared_cad"]["sha256"])
    if scene["declared_cad_sha256"] != hashes[cad_path.relative_to(ROOT).as_posix()]:
        raise ValueError("Declared fixture CAD SHA differs")
    validate_cad(read(cad_path))
    for case in manifest["cases"]:
        for frame in case["frames"]:
            receipt(ROOT / frame["rgb"], hashes, frame["rgb_sha256"])
    for path, sha in manifest["fixture_artwork_sha256"].items():
        if not (SCENE_INPUTS / path).resolve().is_relative_to(SCENE_INPUTS / "fixture-artwork"):
            raise ValueError("Fixture artwork cannot escape its published directory")
        receipt(SCENE_INPUTS / path, hashes, sha)
    old, calibration = read(METHOD_PARENT / "prepared.json"), read(CALIBRATION_PARENT / "prepared.json")
    receipt(METHOD_PARENT / "prepared.json", hashes)
    receipt(CALIBRATION_PARENT / "prepared.json", hashes)
    verify_snapshot(METHOD_PARENT, old, hashes)
    verify_snapshot(CALIBRATION_PARENT, calibration, hashes, selected=CALIBRATION_SOURCES)
    receipt(METHOD_PARENT / "method_config.json", hashes, old["method_config_sha256"])
    method = read(METHOD_PARENT / "method_config.json")
    narrow_method_policy(method["method"])
    if method["method_sha256"] != config["method_sha256"]:
        raise ValueError("Original complete narrow method differs")
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    sources = {}
    for name in source_inventory(old):
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        sources[name] = receipt(ROOT / name, hashes)
        receipt(target, hashes, sources[name])
    shutil.copy2(SCENE_INPUTS / "manifest.json", INPUTS / "manifest.json")
    shutil.copy2(METHOD_PARENT / "method_config.json", RUN / "method_config.json")
    receipt(INPUTS / "manifest.json", hashes, config["scene_input_sha256"])
    receipt(RUN / "method_config.json", hashes, old["method_config_sha256"])
    write(RUN / "prepared.json", dict(run_id=RUN_ID, scene_run_id=SCENE_ID, config=config,
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, receipts=hashes,
        input_sha256=digest(INPUTS / "manifest.json"), method_config_sha256=digest(RUN / "method_config.json"),
        scene_prepared_sha256=config["scene_prepared_sha256"], runtime=runtime(), calibration_policy=POLICY,
        gt_read_during_prepare=False, independent_real_capture=False, g1_passed=False))
    checked()
    print("G1_OBJECT_RGB_PREPARED", len(sources), "sources", len(hashes), "receipts", flush=True)


def error_result(reason):
    return dict(state="error", reason=reason, camera_modified=False, target_geometry_used=False,
        methods=[dict(method=name, state="error", segments=[], segment_count=0, total_length_m=0.,
            endpoints=[], rejection_reasons=[reason], proposal=None, finite=None) for name in METHODS])


def unavailable_cameras(case, reason):
    return [dict(view_id=frame["view_id"], state="unavailable", reasons=[reason], K_index=None,
        world_to_camera_cv=None, training=None, validation=None, training_score=None, validation_score=None)
        for frame in case["frames"]]


def load_images(case):
    images = []
    for frame in case["frames"]:
        path = ROOT / frame["rgb"]
        if digest(path) != frame["rgb_sha256"]:
            raise ValueError("Normal RGB bytes changed before load")
        with Image.open(path) as image:
            if list(image.size) != frame["size_wh"]:
                raise ValueError("Normal RGB dimensions changed")
            images.append(np.asarray(image.convert("RGB")))
    return images


def infer_case(case, cad, method):
    """Retain every object and all five camera slots even after stage errors."""
    started, errors = time.perf_counter(), []
    evidence = [dict(frame, evidence_state="not_extracted") for frame in case["frames"]]
    cameras = unavailable_cameras(case, "calibration_not_completed")
    calibration = dict(state="error", cameras=cameras, frames=[], solver_error="calibration_not_completed")
    result, images = error_result("rgb_not_loaded"), None
    try:
        images = load_images(case)
    except Exception as exc:
        errors.append(dict(stage="rgb_load", reason=type(exc).__name__ + ": " + str(exc)))
    if images is not None:
        detections = []
        for frame, image in zip(case["frames"], images):
            row = dict(view_id=frame["view_id"], size_wh=frame["size_wh"], rgb_sha256=frame["rgb_sha256"])
            try:
                row.update(state="complete", detection=detect_markers(image, cad))
            except Exception as exc:
                reason = type(exc).__name__ + ": " + str(exc)
                row.update(state="error", detection=None, reason=reason)
                errors.append(dict(stage="marker_detection", view_id=frame["view_id"], reason=reason))
            detections.append(row)
        calibration["frames"] = detections
        if all(row["state"] == "complete" for row in detections):
            try:
                calibration = dict(calibrate_case(detections, cad), frames=detections)
                cameras = calibration["cameras"]
            except Exception as exc:
                reason = type(exc).__name__ + ": " + str(exc)
                calibration.update(state="error", solver_error=reason)
                errors.append(dict(stage="calibration", reason=reason))
        try:
            evidence = extract_fixture_narrow_evidence(case["frames"], images, method)
        except Exception as exc:
            reason = type(exc).__name__ + ": " + str(exc)
            result = error_result(reason)
            errors.append(dict(stage="narrow_evidence", reason=reason))
        else:
            try:
                # Withheld/unavailable cameras go through the original gate;
                # neither substitute generated cameras nor weaken validation.
                result = reconstruct_fixture_narrow(evidence, cameras, method)
            except Exception as exc:
                reason = type(exc).__name__ + ": " + str(exc)
                result = error_result(reason)
                errors.append(dict(stage="narrow_reconstruction", reason=reason))
    return dict(case_id=case["case_id"], state="error" if errors else "complete", errors=errors,
        input_case_sha256=canonical_hash(case), frames=evidence, cameras=cameras, calibration=calibration,
        camera_case_sha256=canonical_hash(calibration), camera_case_state=calibration["state"], result=result,
        method_sha256=canonical_hash(method), gt_read_during_inference=False,
        target_identity_verified=False, independent_real_capture=False, g1_passed=False,
        elapsed_seconds=time.perf_counter()-started,
        output_scope="Local finite curves from unchanged RGB rules in a new synthetic assembly; not whole-object repair")


def verified_inference():
    prepared, manifest, method = checked()
    normal = read(RUN / "inference.json")
    if (normal["run_id"] != RUN_ID or normal["state"] != "complete" or normal["gt_read"]
            or normal["prepared_sha256"] != digest(RUN / "prepared.json")
            or normal["input_sha256"] != prepared["input_sha256"]
            or normal["config_sha256"] != prepared["method_config_sha256"]
            or normal["source_sha256"] != prepared["source_sha256"]
            or [row["case_id"] for row in normal["records"]] != [case["case_id"] for case in manifest["cases"]]):
        raise ValueError("Complete two-object normal record inventory required")
    records = []
    for case, row in zip(manifest["cases"], normal["records"]):
        if row["path"] != f"records/{case['case_id']}.json" or digest(RUN / row["path"]) != row["sha256"]:
            raise ValueError("Object RGB normal record changed")
        record = read(RUN / row["path"])
        if (record["case_id"] != case["case_id"] or record["gt_read_during_inference"]
                or record["input_case_sha256"] != canonical_hash(case)
                or record["method_sha256"] != method["method_sha256"]
                or [frame["view_id"] for frame in record["frames"]] != [frame["view_id"] for frame in case["frames"]]
                or [camera["view_id"] for camera in record["cameras"]] != [frame["view_id"] for frame in case["frames"]]
                or [result["method"] for result in record["result"]["methods"]] != list(METHODS)):
            raise ValueError("Object RGB record identity or complete method inventory differs")
        records.append(record)
    return normal, records


def infer():
    sys.addaudithook(block_truth)
    prepared, manifest, method = checked()
    if (RUN / "records").exists() or (RUN / "inference.json").exists():
        raise FileExistsError("Preserve previous object RGB inference, including failures")
    (RUN / "records").mkdir()
    cad = read(ROOT / manifest["declared_cad"]["path"])
    validate_cad(cad)
    started, records = time.perf_counter(), []
    for case in manifest["cases"]:
        result = infer_case(case, cad, method["method"])
        path = RUN / "records" / f"{case['case_id']}.json"
        write(path, result)
        records.append(dict(case_id=case["case_id"], path=path.relative_to(RUN).as_posix(), sha256=digest(path),
            state=result["state"], calibration_state=result["camera_case_state"], error_count=len(result["errors"])))
        print("G1_OBJECT_RGB", case["case_id"], result["state"], result["camera_case_state"],
            [(row["method"], row["state"], row["segment_count"]) for row in result["result"]["methods"]], flush=True)
    checked()
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", records=records, gt_read=False,
        prepared_sha256=digest(RUN / "prepared.json"), input_sha256=prepared["input_sha256"],
        source_sha256=prepared["source_sha256"], config_sha256=prepared["method_config_sha256"],
        calibration_policy=POLICY, runtime=runtime(), elapsed_seconds=time.perf_counter()-started,
        error_object_count=sum(row["state"] == "error" for row in records),
        independent_real_capture=False, g1_passed=False,
        evaluation_status="Normal RGB inference only; no physical evaluation or qualification performed"))
    verified_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
