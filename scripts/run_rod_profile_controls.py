"""Prepare, infer, then evaluate the frozen analytic profile controls.

Run the stages separately. Inference reads RGB and coarse guides from inputs;
geometry and fixture labels are opened only by prepare/evaluate.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from environment_paths import require_project_environment
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_profile_controls import (  # noqa: E402 - repository-local source root
    evaluate_profile_method,
    render_profile_fixture,
    run_profile_methods,
)

RUN_ID = "rod-profile-controls-v1-20260917"
CONFIG = ROOT / "configs/rod_profile_controls_v1.json"
INPUTS = ROOT / "data/inputs" / RUN_ID
TRUTH = ROOT / "data/eval_gt" / RUN_ID
RUN = ROOT / ".runtime/experiments" / RUN_ID
EVALUATION = ROOT / "data/evaluation" / RUN_ID
SOURCES = [
    "scripts/run_rod_profile_controls.py",
    "experiments/src/creator_eval/rod_profile_controls.py",
    "experiments/src/creator_eval/rod_observations.py",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    # x模式是故意的：一轮出错也留下记录，不能悄悄把不顺眼的输出覆盖掉。
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require_hash(path, expected):
    if digest(path) != expected:
        raise RuntimeError(f"Frozen artifact changed: {path}")


def verify_sources(manifest):
    for relative, expected in manifest["source_sha256"].items():
        require_hash(ROOT / relative, expected)


def prepare():
    for destination in (INPUTS, TRUTH, RUN, EVALUATION):
        if destination.exists():
            raise FileExistsError(f"Keep existing outputs: {destination}")
    config = read_json(CONFIG)
    if config["run_id"] != RUN_ID:
        raise ValueError("Unexpected frozen protocol identity")
    case_ids = [case["case_id"] for case in config["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique")
    for destination in (INPUTS, TRUTH, RUN):
        destination.mkdir(parents=True)
    (RUN / "source_snapshot").mkdir()
    source_hashes = {}
    for relative in SOURCES:
        source_hashes[relative] = digest(ROOT / relative)
        target = RUN / "source_snapshot" / Path(relative).name
        with target.open("xb") as stream:
            stream.write((ROOT / relative).read_bytes())
    inputs, truths = [], []
    for ordinal, fixture in enumerate(config["cases"]):
        case_id = fixture["case_id"]
        image, truth = render_profile_fixture(
            fixture, config["generator"], seed=config["generator"]["seed"] + ordinal
        )
        image_path = INPUTS / f"{case_id}.png"
        Image.fromarray(image).save(image_path)
        truth_path = TRUTH / f"{case_id}.json"
        write_json(truth_path, truth)
        # 输入清单只给不带语义的ID与同一粗guide；轮廓位置、病例标签只存在评价侧。
        inputs.append(
            {
                "case_id": case_id,
                "image": image_path.name,
                "image_sha256": digest(image_path),
                "size_wh": list(image.shape[1::-1]),
                "guide_xyxy": config["guide_xyxy"],
            }
        )
        truths.append(
            {"case_id": case_id, "truth": truth_path.name, "truth_sha256": digest(truth_path)}
        )
    write_json(
        INPUTS / "manifest.json",
        {
            "run_id": RUN_ID,
            "cases": inputs,
            "pixel_convention": "original RGB integer pixel centers",
        },
    )
    write_json(TRUTH / "protocol.json", config)
    write_json(
        TRUTH / "manifest.json",
        {
            "run_id": RUN_ID,
            "cases": truths,
            "config_sha256": digest(CONFIG),
            "protocol_copy_sha256": digest(TRUTH / "protocol.json"),
            "input_manifest_sha256": digest(INPUTS / "manifest.json"),
        },
    )
    write_json(RUN / "method_config.json", config["method"])
    write_json(
        RUN / "prepared.json",
        {
            "run_id": RUN_ID,
            "state": "prepared",
            "prepared_utc": utc_now(),
            "case_count": len(inputs),
            "config_sha256": digest(CONFIG),
            "source_sha256": source_hashes,
            "input_manifest_sha256": digest(INPUTS / "manifest.json"),
            "method_config_sha256": digest(RUN / "method_config.json"),
            "truth_manifest_sha256": digest(TRUTH / "manifest.json"),
            "boundary": "Infer must not open eval_gt or full protocol; only method settings, RGB, and common coarse guides.",
        },
    )
    print(
        f"Prepared {len(inputs)} fixed fixtures. Inputs, truth, and method settings are separate."
    )


def infer():
    prepared = read_json(RUN / "prepared.json")
    verify_sources(prepared)
    require_hash(INPUTS / "manifest.json", prepared["input_manifest_sha256"])
    require_hash(RUN / "method_config.json", prepared["method_config_sha256"])
    output = RUN / "predictions"
    output.mkdir()  # 存在就停止，禁止重跑覆盖；这一阶段没有任何读取真值的操作。
    method_config = read_json(RUN / "method_config.json")
    input_manifest = read_json(INPUTS / "manifest.json")
    outcomes = []
    for entry in input_manifest["cases"]:
        case_id = entry["case_id"]
        image_path = INPUTS / entry["image"]
        require_hash(image_path, entry["image_sha256"])
        result_path = output / f"{case_id}.json"
        raw_path = output / f"{case_id}-raw.json"
        try:
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            raw, methods = run_profile_methods(rgb, entry["guide_xyxy"], method_config)
            write_json(raw_path, raw)
            write_json(result_path, {"case_id": case_id, "methods": methods})
            outcome = {
                "case_id": case_id,
                "state": "complete",
                "prediction": result_path.name,
                "prediction_sha256": digest(result_path),
                "raw_observations": raw_path.name,
                "raw_sha256": digest(raw_path),
            }
        except Exception as error:
            # 病例失败也进总表，不用仅保留成功样例给自己壮胆。
            outcome = {
                "case_id": case_id,
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            write_json(output / f"{case_id}-failure.json", outcome)
        outcomes.append(outcome)
        print(case_id, outcome["state"], flush=True)
    write_json(
        RUN / "inference.json",
        {
            "run_id": RUN_ID,
            "state": "complete"
            if all(item["state"] == "complete" for item in outcomes)
            else "complete_with_errors",
            "finished_utc": utc_now(),
            "prepared_sha256": digest(RUN / "prepared.json"),
            "cases": outcomes,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "truth_access": "none; inference reads no fixture configuration or eval_gt file",
        },
    )


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def number(value):
    return "—" if value is None else f"{value:.2f}"


def format_table(rows):
    lines = [
        "# Frozen analytic profile controls",
        "",
        "These are single-image mechanism tests in pixels, not real-data or 3D reconstruction scores. All 22 cases remain in the table.",
        "Coverage is accepted unique-target rows / all physical target rows. Accurate coverage additionally requires center error <= 1 px. Center/boundary errors are conditional on accepted rows; empty predictions have null errors, not zero errors.",
        "",
        "B = existing paired-edge + robust line. C = reject rows whose reliable pair centers span > 1 px, then the same line fitter. No true boundary is used to choose an edge pair.",
        "",
        "| Case | Method | Fit | Target coverage | Accurate <=1px | Center med/p95 px | Boundary max-error med px | Gap false accept | Empty-row false accept | False absence | Rejected disagreement rows |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        method = "B" if row["method"] == "paired_robust" else "C"
        if row.get("inference_error"):
            lines.append(
                f"| {row['case_id']} {row['label']} | {method} | ERROR | — | — | — | — | — | — | — | — |"
            )
            continue
        center = row["selected_center_error_px"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"] + " " + row["label"],
                    method,
                    row["fit_state"],
                    pct(row["target_row_coverage"]),
                    pct(row["accurate_target_row_coverage"]["1"]),
                    f"{number(center['median'])}/{number(center['p95'])}",
                    number(row["selected_maximum_boundary_error_px"]["median"]),
                    pct(row["gap_false_acceptance_fraction"]),
                    pct(row["false_positive_empty_row_fraction"]),
                    pct(row["false_absence_fraction"]),
                    str(row["center_disagreement_rejected_rows"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Background stripes are image texture without foreground rod geometry. Two equal objects have no declared unique identity, so acceptance would count as an unsupported identity choice; center error is not computed against the nearest object.",
        "",
        "A fitted infinite line is only scored at the rows actually selected by the fitter. No bridge or finite extent is inferred across missing observations. Gap false acceptance here is therefore row evidence, not topology.",
        "",
    ]
    return "\n".join(lines)


def evaluate():
    prepared = read_json(RUN / "prepared.json")
    inference = read_json(RUN / "inference.json")
    if inference["state"] not in ("complete", "complete_with_errors"):
        raise RuntimeError("Freeze all inference results before evaluation")
    verify_sources(prepared)
    require_hash(RUN / "prepared.json", inference["prepared_sha256"])
    require_hash(TRUTH / "manifest.json", prepared["truth_manifest_sha256"])
    truth_manifest = read_json(TRUTH / "manifest.json")
    require_hash(TRUTH / "protocol.json", truth_manifest["protocol_copy_sha256"])
    require_hash(INPUTS / "manifest.json", truth_manifest["input_manifest_sha256"])
    protocol = read_json(TRUTH / "protocol.json")
    truth_by_id = {entry["case_id"]: entry for entry in truth_manifest["cases"]}
    if set(truth_by_id) != {entry["case_id"] for entry in inference["cases"]}:
        raise ValueError("Truth and inference case identities differ")
    EVALUATION.mkdir(parents=True)  # Always a fresh result directory.
    (EVALUATION / "details").mkdir()
    rows = []
    for outcome in inference["cases"]:
        case_id = outcome["case_id"]
        truth_entry = truth_by_id[case_id]
        truth_path = TRUTH / truth_entry["truth"]
        require_hash(truth_path, truth_entry["truth_sha256"])
        truth = read_json(truth_path)
        if outcome["state"] == "failed":
            for method in protocol["method"]["variants"]:
                rows.append(
                    {
                        "case_id": case_id,
                        "label": truth["label"],
                        "method": method,
                        "fit_state": "inference_failed",
                        "inference_error": outcome["error"],
                    }
                )
            continue
        prediction_path = RUN / "predictions" / outcome["prediction"]
        require_hash(prediction_path, outcome["prediction_sha256"])
        require_hash(RUN / "predictions" / outcome["raw_observations"], outcome["raw_sha256"])
        prediction = read_json(prediction_path)
        if set(prediction["methods"]) != set(protocol["method"]["variants"]):
            raise ValueError("Missing or unexpected method variant")
        for method, result in prediction["methods"].items():
            score, details = evaluate_profile_method(result, truth, protocol["evaluation"])
            score["method"] = method
            rows.append(score)
            write_json(
                EVALUATION / "details" / f"{case_id}-{method}.json",
                {"summary": score, "rows": details},
            )
    summary = {
        "schema_version": "1.0.0",
        "run_id": RUN_ID,
        "state": inference["state"],
        "evaluated_utc": utc_now(),
        "case_count": len(inference["cases"]),
        "inference_failure_count": sum(case["state"] == "failed" for case in inference["cases"]),
        "source_sha256": prepared["source_sha256"],
        "config_sha256": prepared["config_sha256"],
        "prepared_sha256": digest(RUN / "prepared.json"),
        "inference_sha256": digest(RUN / "inference.json"),
        "truth_manifest_sha256": digest(TRUTH / "manifest.json"),
        "evaluation_config": protocol["evaluation"],
        "rows": rows,
        "limitations": [
            "Synthetic 1D illumination and horizontal PSF, no claim of real cylindrical lighting or camera optics.",
            "Single-image row evidence, not 3D centerline recovery or topology.",
            "Reliable opposite photometric edges do not certify geometric silhouette edges.",
            "Rejecting pair disagreement cannot detect a unique internal highlight or painted background stripe.",
            "Flat-background-like absence can also be a low-contrast physical rod; absence is not certified.",
        ],
    }
    write_json(EVALUATION / "summary.json", summary)
    with (EVALUATION / "summary.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(format_table(rows))
    print(f"Evaluated {len(rows)} method/case outcomes. Table: {EVALUATION / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "evaluate"])
    args = parser.parse_args()
    require_project_environment(ROOT)
    {"prepare": prepare, "infer": infer, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
