"""Freeze, infer and audit paired fixture-calibration interventions."""

from __future__ import annotations

import argparse
import copy
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
from creator_eval.fixture_calibration import POLICY as CAMERA_POLICY  # noqa: E402
from creator_eval.fixture_calibration import calibrate_case, detect_markers  # noqa: E402
from creator_eval.fixture_challenges import (  # noqa: E402
    distort_rgb,
    perturb_cad,
    perturb_detections,
)
from creator_eval.fixture_physical_metrics import POLICY, score_finite_structure  # noqa: E402
from creator_eval.rod_fixture_finite import (  # noqa: E402
    canonical_hash,
    extract_fixture_narrow_evidence,
    json_ready,
    narrow_method_policy,
    reconstruct_fixture_narrow,
)

RUN_ID = "rod-fixture-calibration-challenges-v1-20260929"
SCENE_ID = "rod-fixture-scenes-v1-20260927"
RUN = ROOT / ".runtime/experiments" / RUN_ID
INPUTS = ROOT / "data/inputs" / RUN_ID
SCENE = ROOT / ".runtime/experiments" / SCENE_ID
SCENE_INPUT = ROOT / "data/inputs" / SCENE_ID / "manifest.json"
CONFIG = ROOT / "configs/fixture_calibration_challenges_v1.json"
TRUTH = ROOT / "data/eval_gt" / SCENE_ID / "manifest.json"
PUBLIC = ROOT / "docs/experiments/results/2026-09-29-fixture-calibration-challenges.json"
PUBLIC_AUDIT = ROOT / "docs/experiments/results/2026-09-29-fixture-calibration-challenges-audit.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_ready(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def block_truth(event, arguments):
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    roots = (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments/results")
    if (any(path.is_relative_to(folder) for folder in roots)
            or path.name in {"protocol.json", "generation-checks.json"}
            or path.name.endswith("render_request.json") or "rendered-rgb" in path.parts):
        raise PermissionError("Challenge inference cannot read truth, scores or generation metadata")


def receipt(path, hashes, expected=None):
    path = Path(path).resolve()
    sha = digest(path)
    name = path.relative_to(ROOT).as_posix()
    if (expected is not None and sha != expected) or (name in hashes and hashes[name] != sha):
        raise ValueError("Receipt changed: " + name)
    hashes[name] = sha
    return sha


def checked():
    before = read(RUN / "pre.json")
    if before["run_id"] != RUN_ID or before["gt_read"] or before["inference_existed"]:
        raise ValueError("Invalid pre-inference receipt")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source/input changed: " + name)
    return before, read(INPUTS / "manifest.json"), read(CONFIG)


def prepare():
    sys.addaudithook(block_truth)
    if RUN.exists() or INPUTS.exists():
        raise FileExistsError("Preserve previous challenge attempt; use a new run ID")
    config, manifest, scene = read(CONFIG), read(SCENE_INPUT), read(SCENE / "prepared.json")
    if config["run_id"] != RUN_ID or config["scene_run_id"] != SCENE_ID:
        raise ValueError("Protocol identity differs")
    if [c["case_id"] for c in manifest["cases"]] != config["case_ids"]:
        raise ValueError("Case order differs")
    rod = read(ROOT / config["rod_method_config"])
    narrow_method_policy(rod["method"])
    if canonical_hash(rod["method"]) != rod["method_sha256"]:
        raise ValueError("Rod method digest differs")
    if read(ROOT / config["camera_method_config"])["policy"] != CAMERA_POLICY:
        raise ValueError("Camera policy differs")
    if read(ROOT / config["physical_evaluation_config"])["physical_policy"] != POLICY:
        raise ValueError("Physical policy differs")
    ids = [s["id"] for s in config["scenarios"]]
    if len(set(ids)) != 10 or ids[0] != "c00_nominal":
        raise ValueError("Ten ordered, unique challenge scenarios required")
    hashes = {}
    receipt(SCENE / "prepared.json", hashes)
    receipt(SCENE_INPUT, hashes, scene["input_sha256"])
    cad_path = ROOT / manifest["declared_cad"]["path"]
    receipt(cad_path, hashes, scene["declared_cad_sha256"])
    if hashes[cad_path.relative_to(ROOT).as_posix()] != manifest["declared_cad"]["sha256"]:
        raise ValueError("Declared CAD receipt differs")
    cad = read(cad_path)
    for case in manifest["cases"]:
        if [f["view_id"] for f in case["frames"]] != config["view_ids"]:
            raise ValueError("View order differs")
        for frame in case["frames"]:
            receipt(ROOT / frame["rgb"], hashes, frame["rgb_sha256"])
    RUN.mkdir(parents=True)
    INPUTS.mkdir(parents=True)
    (RUN / "records").mkdir()
    (RUN / "evidence").mkdir()
    sources = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments/src/creator_eval").glob("*.py")}
    sources.update({
        "scripts/run_fixture_challenges.py", "scripts/environment_paths.py",
        "tests/test_fixture_challenges.py", "tests/test_fixture_calibration.py",
        "tests/test_fixture_physical_metrics.py", CONFIG.relative_to(ROOT).as_posix(),
        config["rod_method_config"], config["camera_method_config"], config["physical_evaluation_config"],
    })
    source_hashes = {}
    for name in sorted(sources):
        target = RUN / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        sha = receipt(ROOT / name, hashes)
        receipt(target, hashes, sha)
        source_hashes[name] = sha
    variants = []
    for scenario in config["scenarios"]:
        cp = INPUTS / (scenario["id"] + "-cad.json")
        write(cp, perturb_cad(cad, scenario))
        receipt(cp, hashes)
        cases = copy.deepcopy(manifest["cases"])
        if scenario["kind"] == "radial_rgb":
            for case in cases:
                directory = INPUTS / scenario["id"] / case["case_id"]
                directory.mkdir(parents=True)
                for frame in case["frames"]:
                    parent_rgb, parent_sha = frame["rgb"], frame["rgb_sha256"]
                    with Image.open(ROOT / parent_rgb) as image:
                        rgb = distort_rgb(np.asarray(image.convert("RGB")), scenario["k"])
                    path = directory / (frame["view_id"] + ".png")
                    with path.open("xb") as stream:
                        Image.fromarray(rgb).save(stream, format="PNG")
                    frame.update(rgb=path.relative_to(ROOT).as_posix(), rgb_sha256=receipt(path, hashes),
                                 parent_rgb=parent_rgb, parent_rgb_sha256=parent_sha,
                                 radial_intervention=copy.deepcopy(scenario))
        variants.append(dict(scenario=scenario, cad=cp.relative_to(ROOT).as_posix(), cases=cases))
    write(INPUTS / "manifest.json", dict(run_id=RUN_ID, variants=variants, truth_excluded=True))
    receipt(INPUTS / "manifest.json", hashes)
    write(RUN / "evaluation-freeze.json", dict(
        created_at_utc=now(), gt_read=False, physical_policy=POLICY,
        evaluation_policy=config["evaluation_policy"], expected_invariants=config["expected_invariants"],
        coordinate_alignment_performed=False, source_sha256=source_hashes,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                     scipy=scipy.__version__, pillow=PIL.__version__),
    ))
    receipt(RUN / "evaluation-freeze.json", hashes)
    write(RUN / "pre.json", dict(run_id=RUN_ID, created_at_utc=now(), hashes=hashes,
          gt_read=False, inference_existed=False, source_count=len(sources)))
    checked()
    print("FIXTURE_CHALLENGES_PREPARED", len(hashes), "receipts; 30 cases / 60 method rows", flush=True)


def infer():
    sys.addaudithook(block_truth)
    before, manifest, config = checked()
    if (RUN / "inference.json").exists() or any((RUN / "records").iterdir()) or any((RUN / "evidence").iterdir()):
        raise FileExistsError("Preserve prior challenge inference")
    started = time.perf_counter()
    method = read(ROOT / config["rod_method_config"])["method"]
    cache, evidence_entries, entries = {}, [], []
    for variant in manifest["variants"]:
        scenario = variant["scenario"]
        cad = read(ROOT / variant["cad"])
        for case in variant["cases"]:
            case_started = time.perf_counter()
            key = (scenario["id"] if scenario["kind"] == "radial_rgb" else "nominal", case["case_id"])
            if key not in cache:
                images, detected = [], []
                for frame in case["frames"]:
                    with Image.open(ROOT / frame["rgb"]) as image:
                        images.append(np.asarray(image.convert("RGB")))
                    detected.append(dict(view_id=frame["view_id"], size_wh=frame["size_wh"],
                        rgb_sha256=frame["rgb_sha256"], detection=detect_markers(images[-1], cad)))
                evidence = extract_fixture_narrow_evidence(case["frames"], images, method)
                path = RUN / "evidence" / ("-".join(key) + ".json")
                write(path, dict(frames=evidence, detections=detected, gt_read=False))
                entry = dict(path=path.relative_to(RUN).as_posix(), sha256=digest(path))
                evidence_entries.append(entry)
                cache[key] = (evidence, detected, entry)
            evidence, detected, evidence_entry = cache[key]
            observations = perturb_detections(detected, scenario)
            calibration = calibrate_case(observations, cad)
            result = reconstruct_fixture_narrow(evidence, calibration["cameras"], method)
            path = RUN / "records" / (scenario["id"] + "-" + case["case_id"] + ".json")
            write(path, dict(scenario_id=scenario["id"], case_id=case["case_id"],
                input_case_sha256=canonical_hash(case), cad_sha256=digest(ROOT / variant["cad"]),
                observations=observations, evidence=evidence_entry, calibration=calibration, result=result,
                gt_read=False, elapsed_seconds=time.perf_counter()-case_started))
            entries.append(dict(scenario_id=scenario["id"], case_id=case["case_id"],
                                path=path.relative_to(RUN).as_posix(), sha256=digest(path)))
            print("FIXTURE_CHALLENGE", scenario["id"], case["case_id"], calibration["state"],
                  [(m["method"], m["state"], m["segment_count"]) for m in result["methods"]], flush=True)
    checked()
    write(RUN / "inference.json", dict(run_id=RUN_ID, state="complete", gt_read=False,
          completed_at_utc=now(), pre_sha256=digest(RUN / "pre.json"), records=entries,
          evidence=evidence_entries, input_sha256=digest(INPUTS / "manifest.json"),
          receipt_count=len(before["hashes"]), elapsed_seconds=time.perf_counter()-started))


def physical_summary(prediction, truth, gaps):
    p = score_finite_structure(prediction, truth, gaps)
    return dict(segment_count=p["prediction_segment_count"], total_length_m=p["length"]["prediction_union_m"],
        truth_length_m=p["length"]["truth_union_m"], signed_length_error_m=p["length"]["signed_error_m"],
        boundary_count=p["predicted_endpoints"]["boundary_endpoint_count"],
        boundary_bijection_state=p["boundary_bijection"]["state"],
        boundary_max_error_m=p["boundary_bijection"]["maximum_error_m"],
        recovery_fraction=p["curves"]["recovery_fraction"], precision_fraction=p["curves"]["precision_fraction"],
        truth_to_prediction_p95_m=p["curves"]["truth_to_prediction"]["distance_p95"],
        prediction_to_truth_p95_m=p["curves"]["prediction_to_truth"]["distance_p95"],
        guarded_gap_false_length_m=p["guarded_gap"]["false_proximity_length_m"],
        guarded_gap_length_m=p["guarded_gap"]["interior_length_m"])


def camera_summary(calibration, truth):
    rows = []
    for predicted, gt in zip(calibration["cameras"], truth):
        if predicted["view_id"] != gt["view_id"]:
            raise ValueError("Camera/truth view pairing differs")
        center_error, angle = None, None
        if predicted["world_to_camera_cv"] is not None:
            pe = np.asarray(predicted["world_to_camera_cv"])[:3]
            te = np.asarray(gt["world_to_camera_cv"])[:3]
            pc, tc = -pe[:, :3].T @ pe[:, 3], -te[:, :3].T @ te[:, 3]
            center_error = float(np.linalg.norm(pc-tc))
            angle = float(np.degrees(np.arccos(np.clip((np.trace(pe[:, :3] @ te[:, :3].T)-1)/2, -1, 1))))
        rows.append(dict(view_id=predicted["view_id"], state=predicted["state"], reasons=predicted["reasons"],
            center_error_m=center_error, rotation_error_degrees=angle,
            training_p95_px=(predicted["training_score"] or {}).get("p95_px"),
            validation_p95_px=(predicted["validation_score"] or {}).get("p95_px")))
    return rows


def evaluation_payload():
    before, manifest, config = checked()
    inference = read(RUN / "inference.json")
    if (inference["state"] != "complete" or inference["gt_read"]
            or inference["pre_sha256"] != digest(RUN / "pre.json")
            or inference["input_sha256"] != digest(INPUTS / "manifest.json")):
        raise ValueError("Normal inference is incomplete or changed")
    expected = [(s["id"], c) for s in config["scenarios"] for c in config["case_ids"]]
    if [(r["scenario_id"], r["case_id"]) for r in inference["records"]] != expected:
        raise ValueError("Incomplete ordered prediction inventory")
    for entry in inference["evidence"] + inference["records"]:
        if digest(RUN / entry["path"]) != entry["sha256"]:
            raise ValueError("Normal output receipt changed")
    scene = read(SCENE / "prepared.json")
    if digest(TRUTH) != scene["truth_sha256"]:
        raise ValueError("Truth manifest changed")
    truth = read(TRUTH)
    if truth["input_sha256"] != digest(SCENE_INPUT) or [c["case_id"] for c in truth["cases"]] != config["case_ids"]:
        raise ValueError("Truth/input pairing differs")
    gt_by_case = {c["case_id"]: c for c in truth["cases"]}
    variant_map = {v["scenario"]["id"]: v for v in manifest["variants"]}
    rods, cameras, baseline, failures = [], [], {}, []
    for entry in inference["records"]:
        record = read(RUN / entry["path"])
        sid, cid = entry["scenario_id"], entry["case_id"]
        variant = variant_map[sid]
        case = next(c for c in variant["cases"] if c["case_id"] == cid)
        if record["gt_read"] or record["input_case_sha256"] != canonical_hash(case):
            raise ValueError("Record input pairing differs")
        evidence = read(RUN / record["evidence"]["path"])
        if (digest(RUN / record["evidence"]["path"]) != record["evidence"]["sha256"]
                or record["result"]["evidence_input_sha256"] != canonical_hash(evidence["frames"])
                or record["result"]["camera_input_sha256"] != canonical_hash(record["calibration"]["cameras"])
                or record["cad_sha256"] != digest(ROOT / variant["cad"])):
            raise ValueError("Record camera/evidence/CAD pairing differs")
        gt = gt_by_case[cid]
        source_case = next(c for c in read(SCENE_INPUT)["cases"] if c["case_id"] == cid)
        if [f["rgb_sha256"] for f in gt["frames"]] != [f["rgb_sha256"] for f in source_case["frames"]]:
            raise ValueError("Truth RGB pairing differs")
        cameras.append(dict(scenario_id=sid, case_id=cid, state=record["calibration"]["state"],
            views=camera_summary(record["calibration"], gt["cameras"])))
        if sid == "c00_nominal":
            baseline[cid] = record
        if sid == "c05_missing_train_plane" and record["calibration"]["state"] != "unavailable":
            failures.append(sid + ":" + cid + ":training_loss_not_unavailable")
        if sid == "c06_missing_validation":
            if record["calibration"]["state"] != "withheld":
                failures.append(sid + ":" + cid + ":validation_loss_not_withheld")
            for a, b in zip(record["calibration"]["cameras"], baseline[cid]["calibration"]["cameras"]):
                if not (np.allclose(a["K_index"], b["K_index"], atol=1e-8, rtol=0)
                        and np.allclose(a["world_to_camera_cv"], b["world_to_camera_cv"], atol=1e-10, rtol=0)):
                    failures.append(sid + ":" + cid + ":validation_changed_fit")
        target = gt["declared"]["target"]
        target_segments = target.get("segments", [target["endpoints"]]) if target["present"] else []
        gaps = [gt["declared"]["gap_segment"]] if "gap_segment" in gt["declared"] else []
        if [m["method"] for m in record["result"]["methods"]] != ["baseline", "cylinder_support"]:
            raise ValueError("Missing method row")
        for method in record["result"]["methods"]:
            if record["calibration"]["state"] != "validated" and method["segments"]:
                failures.append(sid + ":" + cid + ":nonvalidated_camera_emitted_rod")
            summary = physical_summary(method["segments"], target_segments, gaps)
            nominal = next(m for m in baseline[cid]["result"]["methods"] if m["method"] == method["method"])
            rods.append(dict(scenario_id=sid, case_id=cid, method=method["method"], normal_state=method["state"],
                normal_record_sha256=entry["sha256"], prediction_segments=method["segments"],
                physical=summary, delta_length_from_nominal_m=summary["total_length_m"]-nominal["total_length_m"],
                rejection_reasons=method["rejection_reasons"]))
    return dict(run_id=RUN_ID, state="evaluated", integrity_state="passed" if not failures else "failed",
        integrity_failures=failures, pre_sha256=digest(RUN / "pre.json"), inference_sha256=digest(RUN / "inference.json"),
        truth_sha256=digest(TRUTH), unchanged_pre_receipts=len(before["hashes"]), physical_policy=POLICY,
        elapsed_seconds=inference["elapsed_seconds"], camera_cases=cameras, rod_rows=rods,
        coordinate_alignment_performed=False, g1_passed=False, scope=config["development_scope"],
        radial_scope=config["radial_policy"], accuracy_acceptance_threshold=None)


def evaluate():
    if PUBLIC.exists() or (RUN / "evaluation.json").exists():
        raise FileExistsError("Preserve previous evaluation")
    result = evaluation_payload()
    write(RUN / "evaluation.json", result)
    write(PUBLIC, result)
    print("FIXTURE_CHALLENGES_EVALUATED", result["integrity_state"], result["integrity_failures"], flush=True)


def post():
    public = read(PUBLIC)
    if PUBLIC.read_bytes() != (RUN / "evaluation.json").read_bytes() or public != evaluation_payload():
        raise ValueError("Physical evaluation replay differs")
    result = dict(run_id=RUN_ID, state=public["integrity_state"], created_at_utc=now(),
        unchanged_pre_receipts=public["unchanged_pre_receipts"], public_sha256=digest(PUBLIC),
        normal_inference_sha256=public["inference_sha256"], exact_evaluation_replay=True,
        camera_case_count=len(public["camera_cases"]), rod_row_count=len(public["rod_rows"]),
        limitations=["Application read guards, not OS isolation", "Same known-answer-tested metric implementation",
                    "Three old procedural objects; 10 interventions are not 30 independent objects",
                    "No real-lens, manufacturing, foreground-identity or dense-cloud qualification"])
    write(RUN / "post.json", result)
    write(PUBLIC_AUDIT, result)
    print("FIXTURE_CHALLENGES_POST", result["state"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
