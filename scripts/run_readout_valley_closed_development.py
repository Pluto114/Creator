"""Replay v4/v5 on all 234 now-known development conditions without retuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout_sections import DEFAULTS as SECTION_DEFAULTS  # noqa: E402
from creator_eval.common_readout_sections import readout as section_readout  # noqa: E402
from creator_eval.common_readout_valley_closed import DEFAULTS as VALLEY_DEFAULTS  # noqa: E402
from creator_eval.common_readout_valley_closed import readout as valley_readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402

CONFIG = ROOT / "configs/readout_valley_closed_development_v1.json"
RUN_ID = "readout-valley-closed-development-v1-20260924"
READERS = {"robust_sections": section_readout, "persistent_valley_closed": valley_readout}
POLICIES = {"robust_sections": SECTION_DEFAULTS, "persistent_valley_closed": VALLEY_DEFAULTS}
SOURCES = ["scripts/run_readout_valley_closed_development.py", "experiments/src/creator_eval/readout_background_controls.py",
           "tests/test_common_readout_valley_closed.py", "experiments/src/creator_eval/common_readout.py",
           "experiments/src/creator_eval/common_readout_components.py", "experiments/src/creator_eval/common_readout_split.py",
           "experiments/src/creator_eval/common_readout_sections.py", "experiments/src/creator_eval/line_controls.py",
           "experiments/src/creator_eval/common_readout_valley_closed.py", "experiments/src/creator_eval/common_readout_valley.py"]
NEW_FILES = [SOURCES[0], SOURCES[2], "experiments/src/creator_eval/common_readout_valley_closed.py", "configs/readout_valley_closed_development_v1.json"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def locations(run_id):
    if not run_id or Path(run_id).name != run_id or ":" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("Run ID must be one directory component")
    return ROOT / ".runtime/experiments" / run_id, ROOT / "data/inputs" / run_id, ROOT / "data/eval_gt" / run_id


def reject_truth_open(event, arguments):
    if event == "open" and arguments and isinstance(arguments[0], (str, bytes)):
        candidate = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).resolve()
        for forbidden in (ROOT / "data/eval_gt", ROOT / "data/evaluation"):
            if candidate == forbidden or forbidden in candidate.parents:
                raise PermissionError("Truth/evaluation file access is forbidden during readout inference")


def input_record(case_id, path, voxel):
    return dict(case_id=case_id, points_file=path.name, points_sha256=digest(path), voxel_size=voxel)


def checked(run_id):
    run, inputs, _ = locations(run_id)
    receipt = read(run / "prepared.json")
    for name, sha in receipt["source_sha256"].items():
        frozen = run / "source_snapshot" / name
        assert digest(frozen) == sha, name
        assert hashlib.sha256(frozen.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == receipt["source_lf_sha256"][name], name
        if digest(ROOT / name) not in (sha, receipt["source_lf_sha256"][name]):
            raise ValueError("Frozen source changed beyond declared legacy CRLF mapping: " + name)
    if digest(inputs / "manifest.json") != receipt["input_sha256"]:
        raise ValueError("Frozen input manifest changed")
    return run, inputs, receipt


def prepare(config_path):
    config = read(config_path)
    if config["readers"] != list(READERS) or config["readout"] != {}:
        raise ValueError("Use the frozen v4 and v5 defaults without case-specific overrides")
    for name in NEW_FILES:
        raw = (ROOT / name).read_bytes()
        if b"\r" in raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ValueError("New files must be LF with one trailing newline: " + name)
        command = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "hash-object"]
        assert subprocess.check_output(command + ["--no-filters", name], cwd=ROOT) == subprocess.check_output(command + ["--path=" + name, name], cwd=ROOT), name
    previous = ROOT / ".runtime/experiments" / config["reader_reference_run_id"]
    for name in ("experiments/src/creator_eval/common_readout.py", "experiments/src/creator_eval/common_readout_components.py",
                 "experiments/src/creator_eval/common_readout_split.py", "experiments/src/creator_eval/common_readout_sections.py", "experiments/src/creator_eval/line_controls.py"):
        assert (ROOT / name).read_bytes().replace(b"\r\n", b"\n") == (previous / "source_snapshot" / name).read_bytes().replace(b"\r\n", b"\n"), name
    run, inputs, truth_root = locations(config["run_id"])
    if any(path.exists() for path in (run, inputs, truth_root)):
        raise FileExistsError("Preserve every old development run")
    for path in (run, inputs, truth_root):
        path.mkdir(parents=True)
    hashes, lf_hashes = {}, {}
    for name in SOURCES:
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = digest(destination)
        lf_hashes[name] = hashlib.sha256(destination.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    cases, targets, parents = [], [], []
    for dataset in config["datasets"]:
        old_run, old_inputs, old_truth = locations(dataset["run_id"])
        receipt = read(old_run / "prepared.json")
        assert digest(old_inputs / "manifest.json") == receipt["input_sha256"]
        assert digest(old_truth / "truth.json") == receipt["gt_sha256"]
        manifest = read(old_inputs / "manifest.json")
        truth = {item["case_id"]: item for item in read(old_truth / "truth.json")}
        assert len(manifest["cases"]) == dataset["expected_cases"]
        for item in manifest["cases"]:
            old_file = old_inputs / item["points_file"]
            assert digest(old_file) == item["points_sha256"]
            case_id = f"development-{len(cases):04d}"
            path = inputs / (case_id + ".npy")
            shutil.copy2(old_file, path)
            cases.append(input_record(case_id, path, item["voxel_size"]))
            original = truth[item["case_id"]]
            labels = {**original["labels"], "source_dataset": dataset["name"], "original_case_id": item["case_id"],
                "family": original["labels"].get("family") or original["labels"].get("layout"),
                "ambiguity_pair": original["labels"].get("ambiguity_pair")}
            targets.append(dict(case_id=case_id, truth=original["truth"], gap=original["gap"], labels=labels))
        parents.append(dict(name=dataset["name"], run_id=dataset["run_id"], input_sha256=receipt["input_sha256"], gt_sha256=receipt["gt_sha256"]))
    assert len(cases) == config["expected_case_count"] == 234
    write(inputs / "manifest.json", dict(run_id=config["run_id"], cases=cases, readers=config["readers"], reader_defaults=POLICIES, readout={}, gt_excluded=True))
    write(truth_root / "truth.json", targets)
    write(run / "protocol.json", config)
    write(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=hashes, source_lf_sha256=lf_hashes,
        source_byte_policy="new sources LF; legacy CRLF snapshots preserved with explicit LF hash mapping only",
        input_sha256=digest(inputs / "manifest.json"), gt_sha256=digest(truth_root / "truth.json"), protocol_sha256=digest(run / "protocol.json"),
        new_git_filter_bytes_unchanged=True, reader_reference_run_id=config["reader_reference_run_id"],
        reader_parameters_fixed_after_freeze=True, parent_datasets=parents))
    print("PREPARED_VALLEY_DEVELOPMENT", len(cases), flush=True)


def infer(run_id):
    run, inputs, receipt = checked(run_id)
    manifest = read(inputs / "manifest.json")
    if manifest["reader_defaults"] != POLICIES or manifest["readout"] != {}:
        raise ValueError("Frozen reader defaults changed")
    sys.addaudithook(reject_truth_open)
    records, started = [], time.perf_counter()
    for case in manifest["cases"]:
        assert set(case) == {"case_id", "points_file", "points_sha256", "voxel_size"}
        path = inputs / case["points_file"]
        if digest(path) != case["points_sha256"]:
            raise ValueError("Anonymous point input changed")
        points = np.load(path, allow_pickle=False)
        for name in manifest["readers"]:
            start = time.perf_counter()
            result = READERS[name](points, np.empty((0, 2, 3)), {"voxel_size": case["voxel_size"]})
            relative = case["case_id"] + "-" + name + ".json"
            write(run / relative, result)
            records.append(dict(case_id=case["case_id"], reader=name, path=relative, sha256=digest(run / relative), elapsed_seconds=time.perf_counter() - start))
        if len(records) % 12 == 0:
            print("BACKGROUND_CHALLENGE_PROGRESS", len(records), flush=True)
    write(run / "inference.json", dict(state="complete", records=records, source_sha256=receipt["source_sha256"],
        source_lf_sha256=receipt["source_lf_sha256"], input_sha256=receipt["input_sha256"], gt_read_during_inference=False,
        numpy_version=np.__version__, python_version=sys.version, elapsed_seconds=time.perf_counter() - started))


def evaluate(run_id, output):
    run, inputs, receipt = checked(run_id)
    _, _, truth_root = locations(run_id)
    if digest(truth_root / "truth.json") != receipt["gt_sha256"] or digest(run / "protocol.json") != receipt["protocol_sha256"]:
        raise ValueError("Frozen evaluation changed")
    config, manifest, inference = read(run / "protocol.json"), read(inputs / "manifest.json"), read(run / "inference.json")
    truth = {item["case_id"]: item for item in read(truth_root / "truth.json")}
    expected = {(case["case_id"], name) for case in manifest["cases"] for name in manifest["readers"]}
    actual = [(item["case_id"], item["reader"]) for item in inference["records"]]
    assert len(actual) == len(set(actual)) and set(actual) == expected and inference["state"] == "complete"
    assert inference["source_sha256"] == receipt["source_sha256"] and inference["input_sha256"] == receipt["input_sha256"]
    rows, results = [], {}
    for record in inference["records"]:
        assert digest(run / record["path"]) == record["sha256"]
        result, target = read(run / record["path"]), truth[record["case_id"]]
        results[(record["case_id"], record["reader"])] = result
        labels, limits = target["labels"], config["qualification"]
        metrics = curve_metrics(result["segments"], target["truth"], tolerance=config["tolerance_m"], spacing=config["spacing_m"])
        gap = gap_coverage(result["segments"], target["gap"], tolerance=config["tolerance_m"], spacing=config["spacing_m"]) if target["gap"] else None
        length = metrics["prediction_to_truth"]["source_length"]
        qualified = None
        if labels["expectation"] == "positive":
            qualified = result["state"] == "complete" and metrics["recovery_fraction"] >= limits["positive_minimum_recovery"] and metrics["precision_fraction"] is not None and metrics["precision_fraction"] >= limits["positive_minimum_precision"]
        elif labels["expectation"] == "negative":
            qualified = result["state"] == "complete" and length <= limits["negative_maximum_predicted_length_m"]
        rows.append(dict(case_id=record["case_id"], reader=record["reader"], **labels, state=result["state"], segments=result["segments"],
            metrics=metrics, gap=gap, predicted_length=length, diagnostic_qualified=qualified,
            rejected_nonlinear_components=result.get("rejected_nonlinear_components"), rejected_side_branches=result.get("rejected_side_branches"),
            rejected_unsupported_sections=result.get("rejected_unsupported_sections"), split_decisions=result.get("split_decisions"),
            valley_checks=result.get("split_valley_checks")))
    inputs_by_id = {case["case_id"]: case for case in manifest["cases"]}
    ambiguity = []
    for pair in sorted({target["labels"]["ambiguity_pair"] for target in truth.values()} - {None}):
        members = [target for target in truth.values() if target["labels"]["ambiguity_pair"] == pair]
        assert len(members) == 2
        ids = [member["case_id"] for member in members]
        assert inputs_by_id[ids[0]]["points_sha256"] == inputs_by_id[ids[1]]["points_sha256"]
        for reader in manifest["readers"]:
            assert results[(ids[0], reader)] == results[(ids[1], reader)], (pair, reader)
        ambiguity.append(dict(pair=pair, case_ids=ids, input_arrays_identical=True, both_reader_outputs_identical=True,
            interpretations=[member["labels"]["interpretation"] for member in members], qualified=None))
    baseline_parity = []
    for dataset in config["datasets"]:
        previous = read(ROOT / dataset["source_results_file"])
        prior = {row["case_id"]: row for row in previous["rows"] if row["reader"] == "robust_sections"}
        baseline = [row for row in rows if row["reader"] == "robust_sections" and row["source_dataset"] == dataset["name"]]
        assert len(baseline) == dataset["expected_cases"]
        for row in baseline:
            old = prior[row["original_case_id"]]
            for key in ("state", "segments", "metrics", "gap", "diagnostic_qualified"):
                assert row[key] == old[key], (dataset["name"], row["original_case_id"], key)
        baseline_parity.append(dict(dataset=dataset["name"], identical_v4_rows=len(baseline), source_results_sha256=digest(ROOT / dataset["source_results_file"])))
    report = dict(run_id=run_id, state="complete", scope=config["scope"], qualification=config["qualification"], baseline_parity=baseline_parity,
        source_sha256=receipt["source_sha256"], source_lf_sha256=receipt["source_lf_sha256"],
        input_sha256=receipt["input_sha256"], gt_sha256=receipt["gt_sha256"], protocol_sha256=receipt["protocol_sha256"],
        inference_sha256=digest(run / "inference.json"), gt_read_during_inference=False, reader_parameters_fixed_after_freeze=True,
        new_git_filter_bytes_unchanged=receipt["new_git_filter_bytes_unchanged"], elapsed_seconds=inference["elapsed_seconds"],
        condition_count=len(manifest["cases"]), unique_point_arrays=len({case["points_sha256"] for case in manifest["cases"]}),
        ambiguity_pairs=ambiguity, rows=rows, interpretation_limits=config["interpretation_limits"])
    evaluation = ROOT / "data/evaluation" / run_id
    evaluation.mkdir(parents=True, exist_ok=False)
    write(evaluation / "summary.json", report)
    write(output, report)
    for expectation in ("positive", "negative", "unidentifiable"):
        for reader in manifest["readers"]:
            selected = [row for row in rows if row["expectation"] == expectation and row["reader"] == reader]
            if expectation == "unidentifiable":
                print(expectation, reader, "UNSCORED", len(selected), flush=True)
            else:
                print(expectation, reader, sum(row["diagnostic_qualified"] is True for row in selected), "/", len(selected), flush=True)
    print("IDENTICAL_INPUT_AMBIGUITY_PAIRS", len(ambiguity), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-24-readout-valley-closed-development.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.run_id, args.output)
