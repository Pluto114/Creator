"""Collect all planned controls without dropping failed or empty candidates."""
import json
from pathlib import Path

from thin_pack_gt import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "thin-pack-baseline-v1-20260914", "thin-controls-v1-20260914",
    "thin-centered-controls-v1-20260914", "thin-oracle-controls-v1-20260914",
]


def main():
    result = {
        "scope": "G0 single development asset; all native diagnostics separate from manual-target curve diagnostics",
        "new_model_jobs": 9, "new_crop_jobs": 6, "new_oracle_jobs": 3,
        "runs": [], "native": [], "line_controls": [],
        "limitations": [
            "One source asset and manually corresponding RGB strips, not automatic cross-object recovery.",
            "Crop changes sampling, context and estimated cameras; native strict regions differ.",
            "Oracle is extra information. Returned cameras are overwritten from input; not estimated accuracy.",
            "Curves are native fits for two targets, not a common point-cloud readout or whole candidate.",
            "Three-view occupancy has no ray-line separation gate and is not validated correspondence.",
            "All five views are inputs; there is no held-out-view score.",
        ],
    }
    prediction_checks = 0
    for run_id in RUNS:
        run = ROOT / ".runtime/experiments" / run_id
        manifest = read_json(run / "inference_manifest.json")
        evaluation = ROOT / "data/evaluation" / run_id / "summary.json"
        summary = read_json(evaluation)
        if manifest["state"] != "complete" or summary["state"] != "complete":
            raise ValueError("A planned native run has failures; do not silently omit it")
        result["runs"].append({
            "run_id": run_id, "job_count": len(manifest["jobs"]), "state": summary["state"],
            "inference_manifest_sha256": sha256(run / "inference_manifest.json"),
            "evaluation_summary_sha256": sha256(evaluation),
            "source_hashes": manifest.get("source_hashes", {}),
            "evaluation_source_hashes": summary.get("evaluation_sources", {}),
        })
        for entry, job in zip(manifest["jobs"], summary["jobs"]):
            if entry["job_id"] != job["job_id"]:
                raise ValueError("Job order changed")
            folder = run / entry["job_id"]
            if sha256(folder / "prediction.npz") != entry["prediction_sha256"] or sha256(folder / "report.json") != entry["report_sha256"]:
                raise ValueError("Prediction/report was modified")
            prediction_checks += 1
            policies = job["filter_policies"]
            result["native"].append({
                "run_id": run_id, "job_id": job["job_id"], "case_id": job["case_id"],
                "oracle": "oracle" in run_id,
                "prediction_sha256": entry["prediction_sha256"],
                "report_sha256": entry["report_sha256"],
                "input_sha256": job["input_sha256"],
                "camera_alignment": job.get("alignment", job.get("raw_predicted_camera_audit")),
                "default_confidence_threshold": job["default_confidence_threshold"],
                "cost": job["cost"],
                "regions_by_policy": {
                    policy: {region: data[region] for region in [
                        "rod_center_all", "strict_rods", "strict_background", "mixed_rod_center_diagnostic"
                    ]} for policy, data in policies.items()
                },
            })
    line_path = ROOT / "data/evaluation/thin-line-controls-v2-20260914/summary.json"
    lines = read_json(line_path)
    fit_root = ROOT / ".runtime/experiments/thin-line-controls-v2-20260914"
    fit_record = read_json(fit_root / "manifest.json")
    for job in fit_record["jobs"]:
        if job["fit_state"] == "succeeded":
            folder = fit_root / job["line_job_id"]
            for name in ("curves", "observations"):
                if sha256(folder / (name + ".json")) != job[name + "_sha256"]:
                    raise ValueError("Line artifact changed")
    for row in lines["rows"]:
        m = row["metrics"]
        output = {key: row[key] for key in [
            "line_job_id", "kind", "case_id", "process_res", "target", "variant", "mode",
            "tolerance_m", "state", "failure_reason", "segment_count",
        ]}
        output.update(
            recovery_fraction=m["recovery_fraction"], precision_fraction=m["precision_fraction"],
            false_predicted_length_m=m["false_predicted_length"],
            truth_to_prediction=m["truth_to_prediction"], prediction_to_truth=m["prediction_to_truth"],
        )
        if "gap" in row:
            output["gap_coverage"] = row["gap"]
        result["line_controls"].append(output)
    previous = read_json(ROOT / "data/evaluation/thin-line-controls-v1-20260914/summary.json")
    keys = ("line_job_id", "target", "variant", "mode", "tolerance_m")
    later = {tuple(r[k] for k in keys): r for r in lines["rows"]}
    for row in previous["rows"]:
        other = later[tuple(row[k] for k in keys)]
        if row["metrics"] != other["metrics"] or row.get("gap") != other.get("gap"):
            raise ValueError("Shared v1/v2 metrics changed")
    original = ROOT / "data/inputs/thin-pack-v2-20260914r1"
    originals = read_json(original / "manifest.json")
    rgb_checks = 0
    for group in originals["groups"]:
        for frame in group["frames"]:
            if sha256(original / frame["rgb"]) != frame["sha256"]:
                raise ValueError("Original RGB modified")
            rgb_checks += 1
    result["integrity"] = {
        "native_predictions_and_reports_unchanged": prediction_checks,
        "original_paired_rgb_unchanged": rgb_checks,
        "shared_v1_v2_metric_rows_exactly_equal": len(previous["rows"]),
        "line_prediction_sets": len(fit_record["jobs"]),
        "line_metric_rows": len(lines["rows"]),
        "failed_metric_rows": lines["failed_metric_rows"],
        "empty_metric_rows": lines["empty_curve_metric_rows"],
        "empty_unique_curve_outputs": lines["empty_curve_metric_rows"]//len(read_json(fit_root/"protocol.json")["tolerances_m"]),
        "line_summary_sha256": sha256(line_path),
        "line_fit_manifest_sha256": sha256(fit_root / "manifest.json"),
        "line_source_hashes": fit_record["source_hashes"],
        "line_evaluation_source_hashes": lines["evaluation_sources"],
    }
    target = ROOT / "docs/experiments/2026-09-14-controls-summary.json"
    # 数值表一条记录一行，避免自动生成的缩进淹没真正值得review的代码。
    metadata = {k: v for k, v in result.items() if k not in ("native", "line_controls")}
    serialized = json.dumps(metadata, indent=2)[:-2]
    for key in ("native", "line_controls"):
        serialized += ',\n  "' + key + '": [\n'
        serialized += ",\n".join("    " + json.dumps(row, allow_nan=False) for row in result[key])
        serialized += "\n  ]"
    serialized += "\n}\n"
    if json.loads(serialized) != result:
        raise ValueError("Compact summary changed values")
    target.write_text(serialized, encoding="utf-8")
    print(json.dumps(result["integrity"], indent=2))
    print("SHAREABLE_SUMMARY", target, target.stat().st_size)


if __name__ == "__main__":
    main()
