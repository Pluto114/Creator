"""Freeze new background/identity challenges, then replay unchanged v3 and v4 readers."""
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
from creator_eval.common_readout_split import DEFAULTS as SPLIT_DEFAULTS  # noqa: E402
from creator_eval.common_readout_split import readout as split_readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402

CONFIG = ROOT / "configs/readout_background_challenges_v1.json"
RUN_ID = "readout-background-challenges-v1-20260924"
READERS = {"transverse_split": split_readout, "robust_sections": section_readout}
POLICIES = {"transverse_split": SPLIT_DEFAULTS, "robust_sections": SECTION_DEFAULTS}
SOURCES = ["scripts/run_readout_background_challenges.py", "experiments/src/creator_eval/readout_background_controls.py",
           "tests/test_common_readout_background.py", "experiments/src/creator_eval/common_readout.py",
           "experiments/src/creator_eval/common_readout_components.py", "experiments/src/creator_eval/common_readout_split.py",
           "experiments/src/creator_eval/common_readout_sections.py", "experiments/src/creator_eval/line_controls.py"]
NEW_FILES = SOURCES[:3] + ["configs/readout_background_challenges_v1.json"]


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
    from creator_eval.readout_background_controls import generate
    config = read(config_path)
    if config["readers"] != list(READERS) or config["readout"] != {}:
        raise ValueError("This challenge fixes both readers at their existing defaults")
    for name in NEW_FILES:
        raw = (ROOT / name).read_bytes()
        if b"\r" in raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ValueError("New files must be LF with one trailing newline: " + name)
        command = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "hash-object"]
        exact = subprocess.check_output(command + ["--no-filters", name], cwd=ROOT, text=True).strip()
        filtered = subprocess.check_output(command + ["--path=" + name, name], cwd=ROOT, text=True).strip()
        if exact != filtered:
            raise ValueError("Git would change newly frozen bytes: " + name)
    previous = read(ROOT / ".runtime/experiments" / config["reader_reference_run_id"] / "prepared.json")
    for name in SOURCES[3:]:
        if digest(ROOT / name) != previous["source_sha256"][name]:
            frozen = ROOT / ".runtime/experiments" / config["reader_reference_run_id"] / "source_snapshot" / name
            if (ROOT / name).read_bytes().replace(b"\r\n", b"\n") != frozen.read_bytes().replace(b"\r\n", b"\n"):
                raise ValueError("Reader or scoring code differs from yesterday's frozen version: " + name)
    run, inputs, truth_root = locations(config["run_id"])
    if any(path.exists() for path in (run, inputs, truth_root)):
        raise FileExistsError("Keep all previous challenge runs")
    for path in (run, inputs, truth_root):
        path.mkdir(parents=True)
    hashes, lf_hashes = {}, {}
    for name in SOURCES:
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = digest(destination)
        lf_hashes[name] = hashlib.sha256(destination.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    cases, truth = [], []
    for number, (points, target, gap, labels) in enumerate(generate(config["generation"])):
        case_id = f"background-{number:04d}"
        path = inputs / (case_id + ".npy")
        with path.open("xb") as stream:
            np.save(stream, points, allow_pickle=False)
        cases.append(input_record(case_id, path, labels["voxel_size"]))
        truth.append(dict(case_id=case_id, truth=target, gap=gap, labels=labels))
    assert len(cases) == config["expected_case_count"] == 36
    write(inputs / "manifest.json", dict(run_id=config["run_id"], cases=cases, readers=config["readers"],
        reader_defaults=POLICIES, readout=config["readout"], gt_excluded=True))
    write(truth_root / "truth.json", truth)
    write(run / "protocol.json", config)
    write(run / "prepared.json", dict(run_id=config["run_id"], source_sha256=hashes, source_lf_sha256=lf_hashes,
        source_byte_policy="new sources LF; legacy CRLF snapshots preserved with explicit LF hash mapping only",
        input_sha256=digest(inputs / "manifest.json"), gt_sha256=digest(truth_root / "truth.json"),
        protocol_sha256=digest(run / "protocol.json"), new_git_filter_bytes_unchanged=True,
        reader_reference_run_id=config["reader_reference_run_id"], reader_parameters_unchanged=True))
    print("PREPARED_BACKGROUND_CHALLENGES", len(cases), flush=True)


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
            rejected_unsupported_sections=result.get("rejected_unsupported_sections"), split_decisions=result.get("split_decisions")))
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
    report = dict(run_id=run_id, state="complete", scope=config["scope"], qualification=config["qualification"],
        source_sha256=receipt["source_sha256"], source_lf_sha256=receipt["source_lf_sha256"],
        input_sha256=receipt["input_sha256"], gt_sha256=receipt["gt_sha256"], protocol_sha256=receipt["protocol_sha256"],
        inference_sha256=digest(run / "inference.json"), gt_read_during_inference=False, reader_parameters_unchanged=True,
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
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-24-readout-background-challenges.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.run_id, args.output)
