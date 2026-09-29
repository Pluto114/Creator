"""Pre-receipt and physical evaluation for the frozen narrow fixture replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.fixture_physical_metrics import POLICY, score_finite_structure  # noqa: E402

AUDIT_ID = "rod-fixture-narrow-audit-v3-20260929"
SCENE_ID = "rod-fixture-scenes-v1-20260927"
FINITE_ID = "rod-fixture-narrow-v1-20260929"
RUN = ROOT / ".runtime/experiments" / AUDIT_ID
SCENE = ROOT / ".runtime/experiments" / SCENE_ID
FINITE = ROOT / ".runtime/experiments" / FINITE_ID
TRUTH = ROOT / "data/eval_gt" / SCENE_ID
EVALUATION = ROOT / "data/evaluation" / AUDIT_ID
PUBLIC = ROOT / "docs/experiments/results/2026-09-29-rod-fixture-narrow-physical.json"
PUBLIC_AUDIT = ROOT / "docs/experiments/results/2026-09-29-rod-fixture-narrow-audit-v3.json"
OLD_PUBLIC = ROOT / "docs/experiments/results/2026-09-27-rod-fixture-physical.json"
CONTROL_PUBLIC = ROOT / "docs/experiments/results/2026-09-29-rod-narrow-controls.json"
POLICY_PATH = ROOT / "configs/rod_fixture_evaluation_v1.json"
CASE_IDS = ["r01", "r02", "r03"]
SOURCES = [
    "scripts/audit_rod_fixture_narrow.py",
    "experiments/src/creator_eval/fixture_physical_metrics.py",
    "tests/test_fixture_physical_metrics.py",
    "configs/rod_fixture_evaluation_v1.json",
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
    if path == CONTROL_PUBLIC.resolve():
        return
    for forbidden in (TRUTH, ROOT / "data/evaluation", ROOT / "docs/experiments/results"):
        if path == forbidden or forbidden in path.parents:
            raise PermissionError("Narrow fixture pre-audit cannot read truth or evaluation")


def receipt(path, hashes, expected=None):
    path = Path(path).resolve()
    sha = digest(path)
    if expected is not None and sha != expected:
        raise ValueError("Receipt changed: " + str(path))
    name = path.relative_to(ROOT).as_posix()
    if name in hashes and hashes[name] != sha:
        raise ValueError("Conflicting receipt: " + name)
    hashes[name] = sha
    return sha


def freeze():
    sys.addaudithook(block_truth)
    if RUN.exists() or EVALUATION.exists() or PUBLIC.exists() or PUBLIC_AUDIT.exists():
        raise FileExistsError("Preserve prior narrow fixture audit")
    if (FINITE / "inference.json").exists():
        raise FileExistsError("Evaluation definition must precede narrow inference")
    protocol = read(POLICY_PATH)
    if protocol["physical_policy"] != POLICY:
        raise ValueError("Physical policy source differs from declared protocol")
    RUN.mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        source, target = ROOT / name, RUN / "evaluation_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = digest(source)
        if digest(target) != sources[name]:
            raise ValueError("Evaluation source snapshot mismatch")
    write(
        RUN / "evaluation-freeze.json",
        dict(
            audit_run_id=AUDIT_ID,
            finite_run_id=FINITE_ID,
            created_at_utc=now(),
            gt_read=False,
            source_sha256=sources,
            physical_policy=POLICY,
            comparison_policy=(
                "Compare the same six case/method rows to the frozen 2026-09-27 physical result; "
                "report segment count, total-length error, boundary count/error, and guarded true-gap filling."
            ),
            acceptance_policy=(
                "No global pass claim. A useful r03 change requires one segment, two boundaries, "
                "smaller absolute length error, and no r02 guarded-gap regression in both methods."
            ),
        ),
    )
    write(
        RUN / "prepared.json",
        dict(
            audit_run_id=AUDIT_ID,
            finite_run_id=FINITE_ID,
            source_sha256=sources,
            evaluation_freeze_sha256=digest(RUN / "evaluation-freeze.json"),
            physical_policy_sha256=digest(POLICY_PATH),
            gt_read=False,
        ),
    )
    print("NARROW_FIXTURE_AUDIT_FROZEN", len(sources), flush=True)


def verify_freeze():
    prepared = read(RUN / "prepared.json")
    if digest(RUN / "evaluation-freeze.json") != prepared["evaluation_freeze_sha256"]:
        raise ValueError("Evaluation freeze changed")
    if digest(POLICY_PATH) != prepared["physical_policy_sha256"]:
        raise ValueError("Physical policy changed")
    frozen = read(RUN / "evaluation-freeze.json")
    if frozen["gt_read"] is not False or frozen["physical_policy"] != POLICY:
        raise ValueError("Invalid evaluation freeze")
    for name, sha in prepared["source_sha256"].items():
        if digest(ROOT / name) != sha or digest(RUN / "evaluation_source_snapshot" / name) != sha:
            raise ValueError("Frozen evaluation source changed: " + name)
    return prepared


def pre():
    sys.addaudithook(block_truth)
    prepared = verify_freeze()
    if (RUN / "pre.json").exists() or EVALUATION.exists() or (FINITE / "inference.json").exists():
        raise FileExistsError("A genuine pre must precede narrow inference and evaluation")
    finite = read(FINITE / "prepared.json")
    hashes = {}
    receipt(RUN / "prepared.json", hashes)
    receipt(RUN / "evaluation-freeze.json", hashes, prepared["evaluation_freeze_sha256"])
    receipt(POLICY_PATH, hashes, prepared["physical_policy_sha256"])
    for name, sha in prepared["source_sha256"].items():
        receipt(ROOT / name, hashes, sha)
        receipt(RUN / "evaluation_source_snapshot" / name, hashes, sha)
    receipt(FINITE / "prepared.json", hashes)
    receipt(FINITE / "source_freeze.json", hashes, finite["source_freeze_sha256"])
    receipt(FINITE / "method_config.json", hashes, finite["method_config_sha256"])
    receipt(ROOT / "data/inputs" / FINITE_ID / "manifest.json", hashes, finite["input_sha256"])
    for name, sha in finite["source_sha256"].items():
        receipt(ROOT / name, hashes, sha)
        receipt(FINITE / "source_snapshot" / name, hashes, sha)
    for name, sha in finite["receipts"].items():
        receipt(ROOT / name, hashes, sha)
    if any((FINITE / "records").iterdir()):
        raise FileExistsError("Narrow finite records already exist before pre")
    write(
        RUN / "pre.json",
        dict(
            state="passed",
            audit_run_id=AUDIT_ID,
            created_at_utc=now(),
            hashes=hashes,
            receipt_count=len(hashes),
            inference_existed=False,
            evaluation_existed=False,
            gt_read=False,
        ),
    )
    print("NARROW_FIXTURE_PRE_PASSED", len(hashes), "receipts", flush=True)


def check_pre():
    before = read(RUN / "pre.json")
    if before["state"] != "passed" or before["gt_read"] is not False:
        raise ValueError("Invalid narrow fixture pre")
    for name, sha in before["hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Pre receipt changed: " + name)
    return before


def _summary(physical):
    return dict(
        segment_count=physical["prediction_segment_count"],
        total_length_m=physical["length"]["prediction_union_m"],
        signed_length_error_m=physical["length"]["signed_error_m"],
        absolute_length_error_m=physical["length"]["absolute_error_m"],
        boundary_count=physical["predicted_endpoints"]["boundary_endpoint_count"],
        boundary_bijection_state=physical["boundary_bijection"]["state"],
        boundary_max_error_m=physical["boundary_bijection"]["maximum_error_m"],
        guarded_gap_false_length_m=physical["guarded_gap"]["false_proximity_length_m"],
        recovery_fraction=physical["curves"]["recovery_fraction"],
        precision_fraction=physical["curves"]["precision_fraction"],
    )


def evaluate_payload():
    before = check_pre()
    inference = read(FINITE / "inference.json")
    if inference["state"] != "complete" or inference["gt_read"] is not False:
        raise ValueError("Incomplete normal narrow inference")
    truth = read(TRUTH / "manifest.json")
    scene_prepared = read(SCENE / "prepared.json")
    if digest(TRUTH / "manifest.json") != scene_prepared["truth_sha256"]:
        raise ValueError("Fixture truth manifest changed")
    if [case["case_id"] for case in truth["cases"]] != CASE_IDS:
        raise ValueError("Truth case order differs")
    if [case["case_id"] for case in inference["records"]] != CASE_IDS:
        raise ValueError("Prediction case order differs")
    old = read(OLD_PUBLIC)
    old_rows = {(row["case_id"], row["method"]): row for row in old["rod_rows"]}
    rows = []
    for gt, entry in zip(truth["cases"], inference["records"]):
        if digest(FINITE / entry["path"]) != entry["sha256"]:
            raise ValueError("Narrow prediction record changed")
        record = read(FINITE / entry["path"])
        target = gt["declared"]["target"]
        truth_segments = target.get("segments", [target["endpoints"]]) if target["present"] else []
        gaps = [gt["declared"]["gap_segment"]] if "gap_segment" in gt["declared"] else []
        for method in record["result"]["methods"]:
            physical = score_finite_structure(method["segments"], truth_segments, gaps)
            previous = old_rows[(entry["case_id"], method["method"])]["physical"]
            rows.append(
                dict(
                    case_id=entry["case_id"],
                    method=method["method"],
                    normal_state=method["state"],
                    normal_record_sha256=entry["sha256"],
                    prediction_segments=method["segments"],
                    truth_segments=truth_segments,
                    declared_gap_segments=gaps,
                    physical=physical,
                    comparison=dict(previous=_summary(previous), current=_summary(physical)),
                )
            )
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    acceptance_failures = []
    for method in ("baseline", "cylinder_support"):
        r03 = by_key[("r03", method)]["comparison"]
        if r03["current"]["segment_count"] != 1:
            acceptance_failures.append(method + ":r03_not_one_segment")
        if r03["current"]["boundary_count"] != 2:
            acceptance_failures.append(method + ":r03_not_two_boundaries")
        if not r03["current"]["absolute_length_error_m"] < r03["previous"]["absolute_length_error_m"]:
            acceptance_failures.append(method + ":r03_length_not_improved")
        r02 = by_key[("r02", method)]["comparison"]
        if r02["current"]["guarded_gap_false_length_m"] > r02["previous"]["guarded_gap_false_length_m"]:
            acceptance_failures.append(method + ":r02_gap_regressed")
    return dict(
        audit_run_id=AUDIT_ID,
        state="passed" if not acceptance_failures else "failed",
        pre_sha256=digest(RUN / "pre.json"),
        evaluation_definition_sha256=digest(RUN / "evaluation-freeze.json"),
        normal_inference_sha256=digest(FINITE / "inference.json"),
        old_physical_sha256=digest(OLD_PUBLIC),
        truth_manifest_sha256=digest(TRUTH / "manifest.json"),
        normal_receipt_count=len(before["hashes"]),
        policy=POLICY,
        acceptance_failures=acceptance_failures,
        rod_rows=rows,
        alignment_performed=False,
        scope=(
            "Same old procedural fixture RGB and cameras; isolates the unresolved-narrow observation "
            "branch. No independent objects, real photos, dense point-cloud repair, or identity proof."
        ),
    )


def evaluate():
    if EVALUATION.exists() or PUBLIC.exists():
        raise FileExistsError("Preserve prior narrow fixture physical evaluation")
    result = evaluate_payload()
    EVALUATION.mkdir(parents=True)
    write(EVALUATION / "summary.json", result)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    write(PUBLIC, result)
    print("NARROW_FIXTURE_EVALUATED", result["state"], result["acceptance_failures"], flush=True)


def post():
    before = check_pre()
    public = read(PUBLIC)
    if PUBLIC.read_bytes() != (EVALUATION / "summary.json").read_bytes():
        raise ValueError("Public and private evaluation bytes differ")
    if public != evaluate_payload():
        raise ValueError("Physical evaluation replay differs")
    result = dict(
        state="passed",
        audit_run_id=AUDIT_ID,
        created_at_utc=now(),
        pre_sha256=digest(RUN / "pre.json"),
        normal_receipts_unchanged=len(before["hashes"]),
        physical_sha256=digest(PUBLIC),
        physical_rows_recomputed=True,
        coordinate_alignment_performed=False,
        limitations=[
            "Application read guards and receipts are not OS isolation.",
            "Physical replay uses the same frozen known-answer-tested metric implementation.",
            "The old procedural objects, RGB, and calibrated cameras are development material, not new external validation.",
        ],
    )
    write(RUN / "post.json", result)
    write(PUBLIC_AUDIT, result)
    print("NARROW_FIXTURE_POST_PASSED", len(before["hashes"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pre", "evaluate", "post"))
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    globals()[args.stage]()
