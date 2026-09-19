"""Replay a frozen coarse-guide identity guard over prior no-truth inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from creator_eval.rod_multiview_candidates import (  # noqa: E402
    apply_guide_identity_guard,
    image_line_from_endpoints,
)
from run_rod_identity_blender import classify as classify_blender  # noqa: E402
from run_rod_multiview_candidates import classify as classify_analytic  # noqa: E402

CONFIG = ROOT / "configs/rod_guide_identity_replay_v1.json"
SHARE = ROOT / "docs/experiments/results/2026-09-19-guide-identity-replay.json"
SOURCES = [
    "scripts/run_rod_guide_identity_replay.py",
    "scripts/run_rod_identity_blender.py",
    "scripts/run_rod_multiview_candidates.py",
    "experiments/src/creator_eval/rod_multiview_candidates.py",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_run_id(run_id):
    if not run_id.replace("-", "").replace("_", "").isalnum() or len(run_id) > 96:
        raise ValueError("Invalid guide replay run ID")
    return run_id


def locations(run_id):
    run_id = validate_run_id(run_id)
    return ROOT / ".runtime/experiments" / run_id, ROOT / "data/evaluation" / run_id


def source_paths(source):
    run_id = validate_run_id(source["run_id"])
    return {
        "run": ROOT / ".runtime/experiments" / run_id,
        "inputs": ROOT / "data/inputs" / run_id,
        "truth": ROOT / "data/eval_gt" / run_id,
    }


def prepare(config_path=CONFIG):
    config = read_json(config_path)
    run, output = locations(config["run_id"])
    if run.exists() or output.exists() or SHARE.exists():
        raise FileExistsError("Keep existing guide replay output")
    run.mkdir(parents=True)
    (run / "source_snapshot").mkdir()
    shutil.copy2(config_path, run / "protocol.json")
    source_hashes = {}
    for relative in SOURCES:
        source_hashes[relative] = digest(ROOT / relative)
        shutil.copy2(ROOT / relative, run / "source_snapshot" / Path(relative).name)
    write_json(run / "method_config.json", config["method"])
    index = []
    for source in config["sources"]:
        paths = source_paths(source)
        inference = paths["run"] / "inference.json"
        manifest = paths["inputs"] / "manifest.json"
        if not inference.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"Missing frozen source run: {source['run_id']}")
        index.append(
            {
                **source,
                "inference": inference.relative_to(ROOT).as_posix(),
                "inference_sha256": digest(inference),
                "input_manifest": manifest.relative_to(ROOT).as_posix(),
                "input_manifest_sha256": digest(manifest),
            }
        )
    write_json(run / "source_index.json", {"sources": index})
    write_json(
        run / "prepared.json",
        {
            "state": "prepared",
            "run_id": config["run_id"],
            "protocol_sha256": digest(run / "protocol.json"),
            "method_config_sha256": digest(run / "method_config.json"),
            "source_index_sha256": digest(run / "source_index.json"),
            "source_sha256": source_hashes,
            "gt_read_during_prepare": False,
            "evaluation_reuse_declared_before_inference": True,
        },
    )
    print("PREPARED_GUIDE_REPLAY", len(index), "frozen source runs")


def _views_with_guides(case, input_case):
    frames = {frame["view_id"]: frame for frame in input_case["frames"]}
    views = []
    for view in case["views"]:
        frame = frames[view["view_id"]]
        views.append(
            {
                **view,
                "guide_line": image_line_from_endpoints(frame["guide_xyxy"]),
                "y_range": [frame["guide_xyxy"][0][1], frame["guide_xyxy"][1][1]],
            }
        )
    return views


def infer(run_id="rod-guide-identity-replay-v1-20260919"):
    run, _ = locations(run_id)
    prepared = read_json(run / "prepared.json")
    method = read_json(run / "method_config.json")
    index = read_json(run / "source_index.json")
    for relative, expected in prepared["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    if digest(run / "method_config.json") != prepared["method_config_sha256"]:
        raise ValueError("Guide replay method changed")
    if digest(run / "source_index.json") != prepared["source_index_sha256"]:
        raise ValueError("Guide replay source index changed")
    source_results = []
    for source in index["sources"]:
        inference_path = ROOT / source["inference"]
        manifest_path = ROOT / source["input_manifest"]
        if digest(inference_path) != source["inference_sha256"]:
            raise ValueError("Frozen source inference changed")
        if digest(manifest_path) != source["input_manifest_sha256"]:
            raise ValueError("Frozen source input manifest changed")
        prior = read_json(inference_path)
        inputs = read_json(manifest_path)
        inputs_by_case = {case["case_id"]: case for case in inputs["cases"]}
        cases = []
        threshold = (
            source["scan_half_width_px"]
            * method["maximum_residual_fraction_of_scan_half_width"]
        )
        guard_config = {
            "maximum_median_residual_px": threshold,
            "minimum_views": method["minimum_views"],
            "minimum_fraction": method["minimum_fraction"],
        }
        for case in prior["cases"]:
            views = _views_with_guides(case, inputs_by_case[case["case_id"]])
            variants = []
            for variant in case["variants"]:
                if variant["variant"] not in source["variants"]:
                    continue
                guarded = apply_guide_identity_guard(variant["result"], views, guard_config)
                variants.append(
                    {
                        "variant": variant["variant"],
                        "baseline_result": variant["result"],
                        "guarded_result": guarded,
                    }
                )
            cases.append({"case_id": case["case_id"], "variants": variants})
        source_results.append(
            {
                "source_key": source["source_key"],
                "kind": source["kind"],
                "source_run_id": source["run_id"],
                "guard_config": guard_config,
                "cases": cases,
            }
        )
    write_json(
        run / "inference.json",
        {
            "state": "inferred",
            "run_id": run_id,
            "gt_read_during_inference": False,
            "evaluation_reuse": True,
            "sources": source_results,
        },
    )
    print("INFERRED_GUIDE_REPLAY", sum(len(row["cases"]) for row in source_results), "cases")


def _truth_by_case(source_run_id):
    root = ROOT / "data/eval_gt" / source_run_id
    manifest = read_json(root / "manifest.json")
    output = {}
    for record in manifest["cases"]:
        path = root / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Source truth changed: {record['case_id']}")
        output[record["case_id"]] = read_json(path)
    return output


def evaluate(config_path=CONFIG, share_path=SHARE):
    config = read_json(config_path)
    run, output = locations(config["run_id"])
    if output.exists() or share_path.exists():
        raise FileExistsError("Keep old guide replay evaluation")
    prepared = read_json(run / "prepared.json")
    if digest(config_path) != prepared["protocol_sha256"] or digest(run / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Guide replay protocol changed after freezing")
    config = read_json(run / "protocol.json")
    inference_path = run / "inference.json"
    inference = read_json(inference_path)
    if inference["state"] != "inferred" or not inference["evaluation_reuse"]:
        raise ValueError("Guide replay inference boundary failed")
    rows = []
    source_config = {source["source_key"]: source for source in config["sources"]}
    for source in inference["sources"]:
        declared = source_config[source["source_key"]]
        truth = _truth_by_case(source["source_run_id"])
        source_protocol = read_json(
            ROOT / ".runtime/experiments" / source["source_run_id"] / "protocol.json"
        )
        policy = source_protocol["evaluation"]
        classifier = classify_analytic if declared["kind"] == "analytic" else classify_blender
        for case in source["cases"]:
            case_truth = truth[case["case_id"]]
            for variant in case["variants"]:
                before, _ = classifier(variant["baseline_result"], case_truth, policy)
                after, _ = classifier(variant["guarded_result"], case_truth, policy)
                diagnostics = variant["guarded_result"].get("guide_identity_guard", {})
                rows.append(
                    {
                        "source_key": source["source_key"],
                        "case_id": case["case_id"],
                        "split": case_truth["split"],
                        "variant": variant["variant"],
                        "baseline_state": variant["baseline_result"]["state"],
                        "guarded_state": variant["guarded_result"]["state"],
                        "baseline_classification": before,
                        "guarded_classification": after,
                        "guard_reason": diagnostics.get("reason"),
                        "agreeing_view_count": diagnostics.get("agreeing_view_count"),
                        "residual_px_per_view": diagnostics.get("residual_px_per_view"),
                    }
                )
    summaries = []
    for source_key in sorted({row["source_key"] for row in rows}):
        for variant in sorted(
            {row["variant"] for row in rows if row["source_key"] == source_key}
        ):
            selected = [
                row for row in rows
                if row["source_key"] == source_key and row["variant"] == variant
            ]
            summaries.append(
                {
                    "source_key": source_key,
                    "variant": variant,
                    "case_count": len(selected),
                    "baseline_classification_counts": dict(
                        sorted(Counter(row["baseline_classification"] for row in selected).items())
                    ),
                    "guarded_classification_counts": dict(
                        sorted(Counter(row["guarded_classification"] for row in selected).items())
                    ),
                    "transitions": dict(
                        sorted(Counter(
                            f"{row['baseline_classification']} -> {row['guarded_classification']}"
                            for row in selected
                        ).items())
                    ),
                }
            )
    result = {
        "state": "complete",
        "run_id": config["run_id"],
        "scope": config["scope"],
        "protocol_sha256": prepared["protocol_sha256"],
        "inference_sha256": digest(inference_path),
        "method": config["method"],
        "summaries": summaries,
        "rows": rows,
        "decision_boundary": [
            "All cases were inspected before this guard was designed; this is regression evidence only.",
            "The guide is an allowed user input and a target-identity prior, not geometric ground truth.",
            "A nearby surface stripe can remain inside the guide band and still be the wrong physical axis.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    write_json(share_path, result)
    print("EVALUATED_GUIDE_REPLAY", output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", default="rod-guide-identity-replay-v1-20260919")
    parser.add_argument("--share-json", type=Path, default=SHARE)
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    if args.stage == "prepare":
        prepare(args.config)
    elif args.stage == "infer":
        infer(args.run_id)
    else:
        evaluate(args.config, args.share_json)
