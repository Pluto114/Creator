"""Freeze, infer, and separately evaluate paired one-pixel RGB controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_narrow_controls import (  # noqa: E402
    acceptance_failures,
    evaluate_narrow_control,
    render_narrow_fixture,
    run_narrow_control_methods,
)

RUN_ID = "rod-narrow-controls-v1-20260929"
CONFIG = ROOT / "configs/rod_narrow_controls_v1.json"
INPUTS = ROOT / "data/inputs" / RUN_ID
TRUTH = ROOT / "data/eval_gt" / RUN_ID
RUN = ROOT / ".runtime/experiments" / RUN_ID
EVALUATION = ROOT / "data/evaluation" / RUN_ID
PUBLIC = ROOT / "docs/experiments/results/2026-09-29-rod-narrow-controls.json"
SOURCES = [
    "scripts/run_rod_narrow_controls.py",
    "experiments/src/creator_eval/rod_narrow_controls.py",
    "experiments/src/creator_eval/rod_observations.py",
    "experiments/src/creator_eval/rod_profile_controls.py",
    "tests/test_rod_narrow_controls.py",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def block_truth(event, arguments):
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    for forbidden in (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments/results"):
        if path == forbidden or forbidden in path.parents:
            raise PermissionError("Narrow-control inference cannot read truth or evaluation")


def verify_prepared():
    prepared = read(RUN / "prepared.json")
    if prepared["run_id"] != RUN_ID:
        raise ValueError("Wrong narrow-control run")
    for name, sha in prepared["source_sha256"].items():
        if digest(ROOT / name) != sha or digest(RUN / "source_snapshot" / name) != sha:
            raise ValueError("Frozen control source changed: " + name)
    if digest(INPUTS / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Frozen control inputs changed")
    if digest(RUN / "method_config.json") != prepared["method_config_sha256"]:
        raise ValueError("Frozen control method changed")
    return prepared


def prepare():
    for destination in (INPUTS, TRUTH, RUN, EVALUATION):
        if destination.exists():
            raise FileExistsError("Preserve prior narrow-control run: " + str(destination))
    if PUBLIC.exists():
        raise FileExistsError("Preserve prior public narrow-control result")
    config = read(CONFIG)
    if config["run_id"] != RUN_ID:
        raise ValueError("Unexpected narrow-control protocol")
    case_ids = [case["case_id"] for case in config["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Control case IDs must be unique")
    for destination in (INPUTS, TRUTH, RUN):
        destination.mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        source, target = ROOT / name, RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = digest(source)
        if digest(target) != sources[name]:
            raise ValueError("Control source snapshot mismatch")
    inputs, truths = [], []
    for ordinal, case in enumerate(config["cases"]):
        image, truth = render_narrow_fixture(
            case, config["generator"], seed=config["generator"]["seed"] + ordinal
        )
        image_path = INPUTS / (case["case_id"] + ".png")
        Image.fromarray(image).save(image_path)
        truth_path = TRUTH / (case["case_id"] + ".json")
        write(truth_path, truth)
        inputs.append(
            dict(
                case_id=case["case_id"],
                image=image_path.name,
                image_sha256=digest(image_path),
                size_wh=list(image.shape[1::-1]),
                guide_xyxy=config["guide_xyxy"],
            )
        )
        truths.append(
            dict(case_id=case["case_id"], truth=truth_path.name, truth_sha256=digest(truth_path))
        )
    write(INPUTS / "manifest.json", dict(run_id=RUN_ID, cases=inputs))
    write(TRUTH / "protocol.json", config)
    write(
        TRUTH / "manifest.json",
        dict(
            run_id=RUN_ID,
            cases=truths,
            config_sha256=digest(CONFIG),
            protocol_sha256=digest(TRUTH / "protocol.json"),
            input_manifest_sha256=digest(INPUTS / "manifest.json"),
        ),
    )
    write(RUN / "method_config.json", config["method"])
    write(
        RUN / "prepared.json",
        dict(
            run_id=RUN_ID,
            state="prepared",
            created_at_utc=now(),
            source_sha256=sources,
            config_sha256=digest(CONFIG),
            input_manifest_sha256=digest(INPUTS / "manifest.json"),
            truth_manifest_sha256=digest(TRUTH / "manifest.json"),
            method_config_sha256=digest(RUN / "method_config.json"),
            gt_read_during_prepare=True,
            inference_boundary="RGB, common guide, and frozen method only; no fixture labels or truth geometry",
        ),
    )
    print("NARROW_CONTROLS_PREPARED", len(inputs), flush=True)


def infer():
    sys.addaudithook(block_truth)
    prepared = verify_prepared()
    predictions = RUN / "predictions"
    if predictions.exists() or (RUN / "inference.json").exists():
        raise FileExistsError("Preserve prior narrow-control inference")
    predictions.mkdir()
    method = read(RUN / "method_config.json")
    outcomes = []
    for entry in read(INPUTS / "manifest.json")["cases"]:
        image_path = INPUTS / entry["image"]
        if digest(image_path) != entry["image_sha256"]:
            raise ValueError("Control RGB changed")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        raw, methods = run_narrow_control_methods(rgb, entry["guide_xyxy"], method)
        path = predictions / (entry["case_id"] + ".json")
        write(path, dict(case_id=entry["case_id"], raw=raw, methods=methods))
        outcomes.append(dict(case_id=entry["case_id"], path=path.name, sha256=digest(path)))
    write(
        RUN / "inference.json",
        dict(
            run_id=RUN_ID,
            state="complete",
            created_at_utc=now(),
            prepared_sha256=digest(RUN / "prepared.json"),
            input_sha256=prepared["input_manifest_sha256"],
            source_sha256=prepared["source_sha256"],
            cases=outcomes,
            gt_read=False,
        ),
    )
    print("NARROW_CONTROLS_INFERRED", len(outcomes), flush=True)


def evaluate():
    prepared = verify_prepared()
    if EVALUATION.exists() or PUBLIC.exists():
        raise FileExistsError("Preserve prior narrow-control evaluation")
    inference = read(RUN / "inference.json")
    if inference["state"] != "complete" or inference["gt_read"] is not False:
        raise ValueError("Incomplete normal control inference")
    if digest(RUN / "prepared.json") != inference["prepared_sha256"]:
        raise ValueError("Prepared control receipt changed")
    truth_manifest = read(TRUTH / "manifest.json")
    if digest(TRUTH / "manifest.json") != prepared["truth_manifest_sha256"]:
        raise ValueError("Control truth manifest changed")
    protocol = read(TRUTH / "protocol.json")
    if digest(TRUTH / "protocol.json") != truth_manifest["protocol_sha256"]:
        raise ValueError("Control protocol changed")
    truth_index = {item["case_id"]: item for item in truth_manifest["cases"]}
    rows = []
    for entry in inference["cases"]:
        if digest(RUN / "predictions" / entry["path"]) != entry["sha256"]:
            raise ValueError("Control prediction changed")
        prediction = read(RUN / "predictions" / entry["path"])
        truth_entry = truth_index[entry["case_id"]]
        truth_path = TRUTH / truth_entry["truth"]
        if digest(truth_path) != truth_entry["truth_sha256"]:
            raise ValueError("Control truth changed")
        truth = read(truth_path)
        for method, value in prediction["methods"].items():
            score, _ = evaluate_narrow_control(value, truth, protocol["evaluation"])
            score["method"] = method
            rows.append(score)
    failures = acceptance_failures(rows, protocol["acceptance_contract"])
    result = dict(
        schema_version="1.0.0",
        run_id=RUN_ID,
        state="passed" if not failures else "failed",
        created_at_utc=now(),
        case_count=len(inference["cases"]),
        method_case_count=len(rows),
        prepared_sha256=digest(RUN / "prepared.json"),
        inference_sha256=digest(RUN / "inference.json"),
        protocol_sha256=digest(TRUTH / "protocol.json"),
        acceptance_contract=protocol["acceptance_contract"],
        acceptance_failures=failures,
        rows=rows,
        limitations=[
            "Analytic quantized RGB profiles, not a real lens or 3D reconstruction benchmark.",
            "One-pixel bilateral evidence does not resolve physical width.",
            "A painted one-pixel texture is intentionally indistinguishable in one view and remains a measured false positive.",
            "Uniform occlusion and low contrast remain unknown; neither is certified absence.",
        ],
    )
    EVALUATION.mkdir(parents=True)
    write(EVALUATION / "summary.json", result)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    write(PUBLIC, result)
    print("NARROW_CONTROLS_EVALUATED", result["state"], failures, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
