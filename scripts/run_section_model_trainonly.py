"""Freeze train-only preprocessing on the same 64 development section inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.section_model_trainonly import DEFAULTS, diagnose  # noqa: E402

RUN_ID = "section-model-trainonly-v1-20260926r1"
RUN = ROOT / ".runtime/experiments" / RUN_ID
INPUTS = ROOT / "data/inputs" / RUN_ID
GT = ROOT / "data/eval_gt" / RUN_ID
CONFIG = ROOT / "configs/section_model_trainonly_v1.json"
SHARE = ROOT / "docs/experiments/results/2026-09-26-section-model-trainonly-r1.json"
PARENT_ID = "section-model-diagnostic-v1-20260926"
SOURCES = ["scripts/run_section_model_trainonly.py", "experiments/src/creator_eval/section_model_trainonly.py",
    "experiments/src/creator_eval/section_model_diagnostic.py", "configs/section_model_trainonly_v1.json",
    "tests/test_section_model_trainonly.py", "scripts/check_section_model_trainonly.py",
    "experiments/src/creator_eval/section_model_controls.py", "configs/section_model_diagnostic_v1.json"]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def reject_truth_open(event, arguments):
    if event == "open" and arguments and isinstance(arguments[0], (str, bytes)):
        path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).resolve()
        for forbidden in (ROOT / "data/eval_gt", ROOT / "data/evaluation", ROOT / "docs/experiments/results"):
            if path == forbidden or forbidden in path.parents:
                raise PermissionError("GT/evaluation access blocked during diagnostic inference")


def checked():
    frozen = read(RUN / "prepared.json")
    for name, sha in frozen["source_sha256"].items():
        assert digest(ROOT / name) == digest(RUN / "source_snapshot" / name) == sha, name
    assert digest(INPUTS / "manifest.json") == frozen["input_sha256"]
    assert digest(RUN / "protocol.json") == frozen["protocol_sha256"]
    assert digest(RUN / "source_freeze.json") == frozen["source_freeze_sha256"]
    return frozen


def prepare():
    for folder in (RUN, INPUTS, GT):
        assert not folder.exists(), "Keep all earlier runs"
    command = ["git", "-c", "safe.directory="+ROOT.as_posix(), "hash-object"]
    for name in SOURCES:
        raw = (ROOT/name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n"), name
        assert subprocess.check_output(command+["--no-filters", name], cwd=ROOT) == subprocess.check_output(command+["--path="+name, name], cwd=ROOT)
    protocol = read(CONFIG)
    assert protocol["run_id"] == RUN_ID and protocol["method"] == DEFAULTS
    parent = ROOT / ".runtime/experiments" / PARENT_ID
    parent_inputs = ROOT / "data/inputs" / PARENT_ID
    parent_truth = ROOT / "data/eval_gt" / PARENT_ID / "manifest.json"
    prior = read(parent / "prepared.json")
    baseline = protocol["previous_shared_result"]
    assert digest(ROOT/baseline["path"]) == baseline["sha256"]
    assert digest(parent_inputs/"manifest.json") == prior["input_sha256"]
    assert digest(parent_truth) == prior["truth_sha256"]
    RUN.mkdir(parents=True)
    hashes = {}
    for name in SOURCES:
        destination = RUN / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
        hashes[name] = digest(destination)
    assert hashes["experiments/src/creator_eval/section_model_diagnostic.py"] == prior["source_sha256"]["experiments/src/creator_eval/section_model_diagnostic.py"]
    write(RUN/"protocol.json", protocol)
    write(RUN/"source_freeze.json", dict(source_sha256=hashes, protocol_sha256=digest(RUN/"protocol.json"),
        previous_shared_result=baseline, note="Frozen before new diagnostic fits; previous shared scores joined only at evaluate"))
    INPUTS.mkdir(parents=True)
    GT.mkdir(parents=True)
    inputs = read(parent_inputs/"manifest.json")["cases"]
    for case in inputs:
        source = parent_inputs/case["points_file"]
        assert digest(source) == case["points_sha256"]
        shutil.copy2(source, INPUTS/case["points_file"])
    write(INPUTS/"manifest.json", dict(run_id=RUN_ID, parent_run_id=PARENT_ID, method=DEFAULTS,
        parent_input_sha256=prior["input_sha256"], cases=inputs, truth_excluded=True))
    write(GT/"manifest.json", dict(run_id=RUN_ID, input_sha256=digest(INPUTS/"manifest.json"),
        parent_truth_sha256=prior["truth_sha256"], cases=read(parent_truth)["cases"]))
    write(RUN/"prepared.json", dict(run_id=RUN_ID, source_sha256=hashes,
        protocol_sha256=digest(RUN/"protocol.json"), source_freeze_sha256=digest(RUN/"source_freeze.json"),
        input_sha256=digest(INPUTS/"manifest.json"), truth_sha256=digest(GT/"manifest.json"),
        parent_run_id=PARENT_ID, parent_prepared_sha256=digest(parent/"prepared.json"),
        parent_input_sha256=prior["input_sha256"], parent_truth_sha256=prior["truth_sha256"],
        previous_shared_result=baseline, previous_source_sha256=prior["source_sha256"],
        engineering_revision=protocol["engineering_revision"], new_git_filter_bytes_unchanged=True))
    print("PREPARED_SECTION_TRAINONLY", len(inputs), flush=True)


def infer():
    frozen = checked()
    manifest = read(INPUTS / "manifest.json")
    assert manifest["method"] == DEFAULTS
    sys.addaudithook(reject_truth_open)
    start, records = time.perf_counter(), []
    folder = RUN / "records"
    folder.mkdir(exist_ok=False)
    for case in manifest["cases"]:
        assert set(case) == {"case_id", "points_file", "points_sha256", "voxel_size"}
        path = INPUTS / case["points_file"]
        assert digest(path) == case["points_sha256"]
        begin = time.perf_counter()
        output = diagnose(np.load(path, allow_pickle=False), case["voxel_size"], manifest["method"])
        target = folder/(case["case_id"]+".json")
        write(target, output)
        records.append(dict(case_id=case["case_id"], path=target.relative_to(RUN).as_posix(), sha256=digest(target),
            input_sha256=case["points_sha256"], elapsed_seconds=time.perf_counter()-begin))
        print("SECTION_TRAINONLY", case["case_id"], f"{records[-1]['elapsed_seconds']:.2f}s", flush=True)
    checked()
    write(RUN / "inference.json", dict(state="complete", run_id=RUN_ID, records=records,
        gt_read_during_inference=False, truth_read_tripwire_enabled=True, source_sha256=frozen["source_sha256"],
        input_sha256=frozen["input_sha256"], elapsed_seconds=time.perf_counter()-start,
        numpy_version=np.__version__, scipy_version=scipy.__version__, python_version=sys.version))


def evaluate():
    frozen = checked()
    assert digest(GT / "manifest.json") == frozen["truth_sha256"]
    truth = read(GT / "manifest.json")
    assert truth["input_sha256"] == frozen["input_sha256"]
    labels = {c["case_id"]: c for c in truth["cases"]}
    manifest = read(INPUTS / "manifest.json")
    inputs = {c["case_id"]: c for c in manifest["cases"]}
    inference = read(RUN / "inference.json")
    assert inference["state"] == "complete" and inference["gt_read_during_inference"] is False
    assert inference["input_sha256"] == frozen["input_sha256"] and inference["source_sha256"] == frozen["source_sha256"]
    assert len(inference["records"]) == len(labels) == len(inputs) == 64
    assert {r["case_id"] for r in inference["records"]} == set(labels) == set(inputs)
    baseline = frozen["previous_shared_result"]
    assert digest(ROOT/baseline["path"]) == baseline["sha256"]
    previous = read(ROOT/baseline["path"])
    assert previous["source_sha256"] == frozen["previous_source_sha256"]
    old_rows = {(r["case_id"], r["representation"], r["train_slices"][0], r["model"]):r for r in previous["rows"]}
    assert len(old_rows) == 768
    rows, comparisons = [], []
    for entry in inference["records"]:
        assert digest(RUN / entry["path"]) == entry["sha256"]
        assert entry["input_sha256"] == inputs[entry["case_id"]]["points_sha256"]
        result = read(RUN / entry["path"])
        assert result["state"] == "complete_diagnostic" and result["emits_axis"] is False
        assert result["policy"] == DEFAULTS
        assert [r["name"] for r in result["representations"]] == ["voxel_centers", "equal_voxel_raw_points"]
        for representation in result["representations"]:
            assert len(representation["rows"]) == 6
            scores = {}
            for r in representation["rows"]:
                assert not set(r["train_slices"]) & set(r["test_slices"])
                assert sorted(r["train_slices"]+r["test_slices"]) == list(range(6))
                complete = r["fit"]["state"] == "fitted" and len(r["heldout"]) == 3 and all("rmse_voxels" in x for x in r["heldout"])
                rmse = None
                if complete:
                    rmse = float(np.sqrt(sum(x["rmse_voxels"]**2*x["occupied_weight"] for x in r["heldout"])/sum(x["occupied_weight"] for x in r["heldout"])))
                row = dict(**labels[entry["case_id"]], representation=representation["name"], **r,
                    heldout_rmse_voxels=rmse, accepted_model=False)
                old = old_rows[(entry["case_id"], representation["name"], r["fold"], r["model"])]
                row["training_preprocessing"] = result["folds"][r["fold"]]
                row["previous_shared"] = dict(heldout_rmse_voxels=old["heldout_rmse_voxels"], fit_state=old["fit"]["state"],
                    converged=old["fit"].get("converged"), at_parameter_boundary=old["fit"].get("at_parameter_boundary"))
                row["rmse_change_from_shared_voxels"] = None if rmse is None or old["heldout_rmse_voxels"] is None else rmse-old["heldout_rmse_voxels"]
                rows.append(row)
                scores[(r["train_slices"][0], r["model"])] = row
            for parity in (0, 1):
                ell, dual, line = [scores[(parity, name)] for name in ("ellipse", "two_circles", "line")]
                a, b, c = [r["heldout_rmse_voxels"] for r in (ell, dual, line)]
                comparisons.append(dict(**labels[entry["case_id"]], representation=representation["name"], train_parity=parity,
                    ellipse_minus_two_circles_rmse_voxels=None if a is None or b is None else a-b,
                    ellipse_minus_line_rmse_voxels=None if a is None or c is None else a-c,
                    models_converged={name:r["fit"].get("converged", False) for name, r in zip(("ellipse", "two_circles", "line"), (ell, dual, line))},
                    parameter_boundary={name:r["fit"].get("at_parameter_boundary") for name, r in zip(("ellipse", "two_circles", "line"), (ell, dual, line))},
                    selected_model=None, scope="Raw residual differences only; unequal model flexibility and degenerate fits preclude automatic physical classification"))
    assert len(rows) == 768 and len(comparisons) == 256
    output = dict(run_id=RUN_ID, state="complete_diagnostic", condition_count=64, representation_count=128,
        model_fold_row_count=len(rows), comparisons=comparisons, rows=rows,
        parent_run_id=PARENT_ID, previous_shared_result=frozen["previous_shared_result"],
        source_sha256=frozen["source_sha256"], input_sha256=frozen["input_sha256"], truth_sha256=frozen["truth_sha256"],
        source_freeze_sha256=frozen["source_freeze_sha256"], protocol_sha256=frozen["protocol_sha256"],
        inference_sha256=digest(RUN / "inference.json"), elapsed_seconds=inference["elapsed_seconds"], gt_read_during_inference=False,
        fit_states=dict(Counter(r["fit"]["state"] for r in rows)),
        model_degrees_of_freedom=dict(ellipse=5, two_circles=6, line=2),
        qualification=None, emits_axis=False, interpretation_limits=read(RUN / "protocol.json")["interpretation_limits"])
    evaluation = ROOT / "data/evaluation" / RUN_ID
    evaluation.mkdir(exist_ok=False)
    write(evaluation / "summary.json", output)
    write(SHARE, output)
    print("EVALUATED_SECTION_TRAINONLY", len(rows), "model-fold rows; no model accepted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    {"prepare": prepare, "infer": infer, "evaluate": evaluate}[args.stage]()
