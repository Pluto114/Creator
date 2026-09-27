"""Freeze and infer RGB-only cameras from an explicitly supplied metric fixture."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from prepare_rod_reference import ROOT, digest, read_json, source_snapshot, write_json

# isort: split
from creator_eval.fixture_calibration import POLICY, calibrate_case, detect_markers, validate_cad

RUN_ID = "rod-fixture-calibration-v1-20260927"
SCENE_ID = "rod-fixture-scenes-v1-20260927"
RUN = ROOT / ".runtime/experiments" / RUN_ID
SCENE = ROOT / ".runtime/experiments" / SCENE_ID
INPUT = ROOT / "data/inputs" / SCENE_ID / "manifest.json"
CONFIG = ROOT / "configs/fixture_calibration_v1.json"
SOURCES = {"scripts/run_fixture_calibration.py", "experiments/src/creator_eval/fixture_calibration.py",
    "configs/fixture_calibration_v1.json", "tests/test_fixture_calibration.py"}


def block_truth(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    forbidden_roots = (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments/results")
    if any(path.is_relative_to(folder) for folder in forbidden_roots) or path.name == "protocol.json" or path.name.endswith("render_request.json") or path.name == "generation-checks.json" or path.is_relative_to(SCENE / "rendered-rgb"):
        raise PermissionError("Calibration may read RGB and declared CAD, not generation truth or physical scores")


def checked():
    prepared = read_json(RUN / "prepared.json")
    assert prepared["run_id"] == RUN_ID
    for name, sha in prepared["source_sha256"].items():
        assert digest(ROOT / name) == digest(RUN / "source_snapshot" / name) == sha, name
    assert digest(SCENE / "prepared.json") == prepared["scene_prepared_sha256"]
    assert digest(INPUT) == prepared["input_sha256"] == read_json(SCENE / "prepared.json")["input_sha256"]
    manifest = read_json(INPUT)
    assert digest(ROOT / manifest["declared_cad"]["path"]) == manifest["declared_cad"]["sha256"] == prepared["cad_sha256"]
    assert read_json(CONFIG)["policy"] == POLICY
    return prepared, manifest


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists():
        raise FileExistsError("Preserve earlier calibration")
    scene, config, manifest = read_json(SCENE / "prepared.json"), read_json(CONFIG), read_json(INPUT)
    assert config["run_id"] == RUN_ID and config["policy"] == POLICY
    assert digest(INPUT) == scene["input_sha256"]
    cad_path = ROOT / manifest["declared_cad"]["path"]
    assert digest(cad_path) == manifest["declared_cad"]["sha256"] == scene["declared_cad_sha256"]
    validate_cad(read_json(cad_path))
    for name, sha in scene["source_sha256"].items():
        assert digest(ROOT / name) == digest(SCENE / "source_snapshot" / name) == sha
    RUN.mkdir(parents=True)
    hashes = source_snapshot(RUN, set(scene["source_sha256"]) | SOURCES)
    write_json(RUN / "prepared.json", dict(run_id=RUN_ID, scene_run_id=SCENE_ID,
        scene_prepared_sha256=digest(SCENE / "prepared.json"), input_sha256=digest(INPUT),
        cad_sha256=digest(cad_path), source_sha256=hashes, policy=POLICY,
        created_at_utc=datetime.now(timezone.utc).isoformat(), gt_read=False,
        calibration_unit="Each case separately: five RGB views sharing one unknown K"))
    checked()
    print("FIXTURE_CALIBRATION_PREPARED", len(hashes), "sources", flush=True)


def infer():
    sys.addaudithook(block_truth)
    prepared, manifest = checked()
    if (RUN / "inference.json").exists():
        raise FileExistsError("Preserve completed calibration")
    cad = read_json(ROOT / manifest["declared_cad"]["path"])
    cases = []
    started = perf_counter()
    for case in manifest["cases"]:
        case_started = perf_counter()
        frames = []
        for frame in case["frames"]:
            path = ROOT / frame["rgb"]
            assert digest(path) == frame["rgb_sha256"]
            with Image.open(path) as image:
                assert list(image.size) == frame["size_wh"]
                rgb = np.asarray(image.convert("RGB"))
            frames.append(dict(view_id=frame["view_id"], size_wh=frame["size_wh"],
                rgb_sha256=frame["rgb_sha256"], detection=detect_markers(rgb, cad)))
        result = calibrate_case(frames, cad)
        cases.append(dict(case_id=case["case_id"], frames=frames, elapsed_seconds=perf_counter()-case_started, **result))
        print("FIXTURE_CALIBRATED", case["case_id"], result["state"],
            [len(c["training"]["marker_ids"]) for c in result["cameras"]],
            [c["validation_score"]["p95_px"] if c["validation_score"] else None for c in result["cameras"]], flush=True)
    checked()
    write_json(RUN / "inference.json", dict(run_id=RUN_ID, cases=cases, input_sha256=prepared["input_sha256"],
        cad_sha256=prepared["cad_sha256"], source_sha256=prepared["source_sha256"],
        prepared_sha256=digest(RUN / "prepared.json"), gt_read=False, policy=POLICY,
        opencv_version=cv2.__version__, numpy_version=np.__version__, elapsed_seconds=perf_counter()-started,
        scope="Known metric fixture and ideal zero-distortion model; not arbitrary-photo calibration"))
    print("FIXTURE_CALIBRATION_INFERRED", len(cases), "cases / 15 fixed views", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer"))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    globals()[args.stage]()
