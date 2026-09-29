"""Replay the metric fixture with frozen unresolved-narrow RGB evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_fixture_finite import (  # noqa: E402
    canonical_hash,
    extract_fixture_narrow_evidence,
    json_ready,
    narrow_method_policy,
    reconstruct_fixture_narrow,
)

RUN_ID = "rod-fixture-narrow-v1-20260929"
SCENE_ID = "rod-fixture-scenes-v1-20260927"
CAMERA_ID = "rod-fixture-calibration-v1-20260927"
CONTROL_ID = "rod-narrow-controls-v1-20260929"
RUN = ROOT / ".runtime/experiments" / RUN_ID
INPUTS = ROOT / "data/inputs" / RUN_ID
SCENE = ROOT / ".runtime/experiments" / SCENE_ID
SCENE_INPUTS = ROOT / "data/inputs" / SCENE_ID
CAMERA = ROOT / ".runtime/experiments" / CAMERA_ID
CONTROL = ROOT / ".runtime/experiments" / CONTROL_ID
CONTROL_PUBLIC = ROOT / "docs/experiments/results/2026-09-29-rod-narrow-controls.json"
CONFIG = ROOT / "configs/rod_fixture_narrow_v1.json"
SOURCES = [
    "scripts/run_rod_fixture_narrow.py",
    "experiments/src/creator_eval/rod_fixture_finite.py",
    "experiments/src/creator_eval/rod_observations.py",
    "experiments/src/creator_eval/rod_candidate_association.py",
    "experiments/src/creator_eval/rod_candidate_pool.py",
    "experiments/src/creator_eval/rod_candidate_extent.py",
    "experiments/src/creator_eval/rod_cylinder_gate.py",
    "experiments/src/creator_eval/rod_cylinder_support.py",
    "experiments/src/creator_eval/rod_multiview_candidates.py",
    "experiments/src/creator_eval/rod_evidence.py",
    "experiments/src/creator_eval/line_controls.py",
    "experiments/src/creator_eval/native_diagnostics.py",
    "tests/test_rod_fixture_narrow.py",
    "configs/rod_fixture_narrow_v1.json",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_ready(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_truth(event, arguments):
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    if path == CONTROL_PUBLIC.resolve():
        return
    for forbidden in (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments/results"):
        if path == forbidden or forbidden in path.parents:
            raise PermissionError("Normal narrow fixture inference cannot read truth or evaluation")
    if path.name in {"protocol.json", "generation-checks.json"} or "rendered-rgb" in path.parts:
        raise PermissionError("Normal narrow fixture inference cannot read generation protocol")


def verify_parent_snapshot(folder, prepared):
    for name, sha in prepared["source_sha256"].items():
        if digest(folder / "source_snapshot" / name) != sha:
            raise ValueError("Parent source snapshot changed: " + name)


def checked():
    prepared = read(RUN / "prepared.json")
    if prepared["run_id"] != RUN_ID:
        raise ValueError("Wrong narrow fixture run")
    for name, sha in prepared["source_sha256"].items():
        if digest(ROOT / name) != sha or digest(RUN / "source_snapshot" / name) != sha:
            raise ValueError("Frozen narrow fixture source changed: " + name)
    if digest(INPUTS / "manifest.json") != prepared["input_sha256"]:
        raise ValueError("Narrow fixture input changed")
    if digest(RUN / "method_config.json") != prepared["method_config_sha256"]:
        raise ValueError("Narrow fixture method changed")
    config = read(RUN / "method_config.json")
    narrow_method_policy(config["method"])
    if config["method_sha256"] != canonical_hash(config["method"]):
        raise ValueError("Narrow fixture method digest differs")
    for name, sha in prepared["receipts"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Narrow fixture receipt changed: " + name)
    return prepared, read(INPUTS / "manifest.json"), config


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or INPUTS.exists():
        raise FileExistsError("Preserve prior narrow fixture attempt")
    config = read(CONFIG)
    narrow_method_policy(config["method"])
    if (
        config["run_id"] != RUN_ID
        or config["scene_run_id"] != SCENE_ID
        or config["camera_run_id"] != CAMERA_ID
        or config["control_run_id"] != CONTROL_ID
        or config["method_sha256"] != canonical_hash(config["method"])
    ):
        raise ValueError("Narrow fixture protocol identity differs")
    control = read(CONTROL_PUBLIC)
    if control["state"] != "passed" or control["acceptance_failures"]:
        raise ValueError("Frozen narrow controls did not pass")
    if control["inference_sha256"] != digest(CONTROL / "inference.json"):
        raise ValueError("Frozen narrow-control inference receipt differs")
    scene_prepared = read(SCENE / "prepared.json")
    camera_prepared = read(CAMERA / "prepared.json")
    verify_parent_snapshot(SCENE, scene_prepared)
    verify_parent_snapshot(CAMERA, camera_prepared)
    scene_manifest = read(SCENE_INPUTS / "manifest.json")
    camera_manifest = scene_manifest
    camera_inference = read(CAMERA / "inference.json")
    if (
        digest(SCENE_INPUTS / "manifest.json") != scene_prepared["input_sha256"]
        or digest(SCENE_INPUTS / "manifest.json") != camera_prepared["input_sha256"]
        or camera_inference["gt_read"] is not False
        or camera_inference["input_sha256"] != camera_prepared["input_sha256"]
        or camera_inference["source_sha256"] != camera_prepared["source_sha256"]
    ):
        raise ValueError("Parent scene/camera receipts differ")
    parent_method = read(ROOT / config["parent_method"]["path"])["method"]
    for key in config["parent_method"]["unchanged_components"]:
        if config["method"][key] != parent_method[key]:
            raise ValueError("Declared unchanged component differs: " + key)
    case_ids, view_ids = config["case_ids"], config["view_ids"]
    collections = (scene_manifest["cases"], camera_inference["cases"], camera_manifest["cases"])
    if any([case["case_id"] for case in rows] != case_ids for rows in collections):
        raise ValueError("Parent case order differs")
    cases = []
    receipts = {
        SCENE.relative_to(ROOT).as_posix() + "/prepared.json": digest(SCENE / "prepared.json"),
        SCENE_INPUTS.relative_to(ROOT).as_posix() + "/manifest.json": digest(SCENE_INPUTS / "manifest.json"),
        CAMERA.relative_to(ROOT).as_posix() + "/prepared.json": digest(CAMERA / "prepared.json"),
        CAMERA.relative_to(ROOT).as_posix() + "/inference.json": digest(CAMERA / "inference.json"),
        CONTROL.relative_to(ROOT).as_posix() + "/prepared.json": digest(CONTROL / "prepared.json"),
        CONTROL.relative_to(ROOT).as_posix() + "/inference.json": digest(CONTROL / "inference.json"),
        CONTROL_PUBLIC.relative_to(ROOT).as_posix(): digest(CONTROL_PUBLIC),
        config["parent_method"]["path"]: digest(ROOT / config["parent_method"]["path"]),
    }
    for scene_case, camera_case, camera_input in zip(*collections):
        if scene_case["frames"] != camera_input["frames"]:
            raise ValueError("Camera and scene RGB frames differ")
        if [frame["view_id"] for frame in scene_case["frames"]] != view_ids:
            raise ValueError("View order differs")
        if [camera["view_id"] for camera in camera_case["cameras"]] != view_ids:
            raise ValueError("Camera order differs")
        for frame in scene_case["frames"]:
            if digest(ROOT / frame["rgb"]) != frame["rgb_sha256"]:
                raise ValueError("Fixture RGB changed: " + frame["rgb"])
            receipts[frame["rgb"]] = frame["rgb_sha256"]
        cases.append(
            dict(
                case_id=scene_case["case_id"],
                frames=scene_case["frames"],
                cameras=camera_case["cameras"],
                camera_case_sha256=canonical_hash(camera_case),
                camera_case_state=camera_case["state"],
            )
        )
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    (RUN / "records").mkdir()
    sources = {}
    for name in SOURCES:
        source, target = ROOT / name, RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = digest(source)
        if digest(target) != sources[name]:
            raise ValueError("Narrow fixture source snapshot mismatch")
    write(
        INPUTS / "manifest.json",
        dict(
            run_id=RUN_ID,
            cases=cases,
            truth_excluded=True,
            coordinate_system="declared_metric_fixture_world",
            camera_inference_sha256=digest(CAMERA / "inference.json"),
            narrow_control_sha256=digest(CONTROL_PUBLIC),
        ),
    )
    write(RUN / "method_config.json", config)
    write(
        RUN / "source_freeze.json",
        dict(
            source_sha256=sources,
            input_sha256=digest(INPUTS / "manifest.json"),
            method_config_sha256=digest(RUN / "method_config.json"),
            before_inference=True,
        ),
    )
    write(
        RUN / "prepared.json",
        dict(
            run_id=RUN_ID,
            scene_run_id=SCENE_ID,
            camera_run_id=CAMERA_ID,
            control_run_id=CONTROL_ID,
            source_sha256=sources,
            input_sha256=digest(INPUTS / "manifest.json"),
            method_config_sha256=digest(RUN / "method_config.json"),
            source_freeze_sha256=digest(RUN / "source_freeze.json"),
            receipts=receipts,
            gt_read_during_prepare=False,
        ),
    )
    print("FIXTURE_NARROW_PREPARED 3 cases x 2 methods, 15 RGB", flush=True)


def infer():
    sys.addaudithook(block_truth)
    prepared, manifest, config = checked()
    if (RUN / "inference.json").exists() or any((RUN / "records").iterdir()):
        raise FileExistsError("Preserve prior narrow fixture inference")
    started = time.perf_counter()
    records = []
    for case in manifest["cases"]:
        images = []
        for frame in case["frames"]:
            with Image.open(ROOT / frame["rgb"]) as image:
                images.append(np.asarray(image.convert("RGB")))
        frames = extract_fixture_narrow_evidence(case["frames"], images, config["method"])
        result = reconstruct_fixture_narrow(frames, case["cameras"], config["method"])
        path = RUN / "records" / (case["case_id"] + ".json")
        write(
            path,
            dict(
                case_id=case["case_id"],
                input_case_sha256=canonical_hash(case),
                frames=frames,
                cameras=case["cameras"],
                result=result,
                gt_read_during_inference=False,
                target_identity_verified=False,
                output_scope="Finite curve with unresolved narrow RGB evidence; not dense/cloud/mesh repair",
            ),
        )
        records.append(
            dict(case_id=case["case_id"], path=path.relative_to(RUN).as_posix(), sha256=digest(path))
        )
        print(
            "FIXTURE_NARROW",
            case["case_id"],
            [(row["method"], row["state"], row["segment_count"]) for row in result["methods"]],
            flush=True,
        )
    checked()
    write(
        RUN / "inference.json",
        dict(
            state="complete",
            run_id=RUN_ID,
            records=records,
            input_sha256=prepared["input_sha256"],
            source_sha256=prepared["source_sha256"],
            config_sha256=prepared["method_config_sha256"],
            gt_read=False,
            elapsed_seconds=time.perf_counter() - started,
            evaluation_status="Await independent pre-frozen physical evaluation",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
