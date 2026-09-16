"""Publish small, complete metric tables without copying model arrays into Git."""
import json
from pathlib import Path

from thin_pack_gt import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["rod-evidence-v1-20260916", "rod-evidence-dense-v1-20260916"]


def compact_tables(value, level=0, table=False):
    """One metric record per line keeps the complete tables reviewable in Git."""
    prefix, child = "  " * level, "  " * (level + 1)
    if isinstance(value, dict):
        return "{\n" + ",\n".join(child + json.dumps(key) + ": " + compact_tables(item, level + 1, key in (
            "curve_rows", "heldout_rows", "proposals", "centering")) for key, item in value.items()) + "\n" + prefix + "}"
    if isinstance(value, list) and not table:
        return "[\n" + ",\n".join(child + compact_tables(item, level + 1) for item in value) + "\n" + prefix + "]"
    if isinstance(value, list):
        return "[\n" + ",\n".join(child + json.dumps(item, allow_nan=False, separators=(",", ":")) for item in value) + "\n" + prefix + "]"
    return json.dumps(value, allow_nan=False)


def main():
    result = {"scope": "Same-asset pilot; estimated and privileged oracle cameras separate; all tolerances and refused outputs retained", "runs": []}
    for run_id in RUNS:
        source = ROOT / "data/evaluation" / run_id / "summary.json"
        data = read_json(source)
        manifest = read_json(ROOT / ".runtime/experiments" / run_id / "manifest.json")
        if data["state"] != "complete":
            raise ValueError("Incomplete evaluation")
        run = {"run_id": run_id, "full_summary_sha256": sha256(source),
               "fit_manifest_sha256": data["fit_manifest_sha256"], "protocol_sha256": manifest["protocol_sha256"],
               "input_manifest_sha256": manifest["input_manifest_sha256"], "producer_sources": manifest["source_hashes"],
               "evaluation_sources": data["evaluation_sources"], "heldout_manifest_sha256": data["heldout_manifest_sha256"],
               "accepted_proposals": data["accepted_proposals"], "refused_proposals": data["refused_proposals"],
               "proposals": data["proposals"], "centering": data["centering"], "alignments": data["alignments"],
               "background_audits": data["background_audits"], "curve_rows": [], "heldout_rows": []}
        for row in data["rows"]:
            metrics = row["metrics"]
            run["curve_rows"].append({
                **{k: row[k] for k in ("candidate_job_id", "target", "variant", "state", "tolerance_m", "segment_count")},
                "recovery": metrics["recovery_fraction"], "precision": metrics["precision_fraction"],
                "truth_to_prediction_median_m": metrics["truth_to_prediction"]["distance_median"],
                "truth_to_prediction_p95_m": metrics["truth_to_prediction"]["distance_p95"],
                "predicted_length_m": metrics["prediction_to_truth"]["source_length"],
                "false_predicted_length_m": metrics["false_predicted_length"], "missing_truth_length_m": row["missing_truth_length_m"],
                "guarded_gap_coverage": row.get("gap", {}).get("guarded_interior", {}).get("covered_fraction"),
            })
        for row in data["heldout_rows"]:
            metrics = row["metrics"]
            run["heldout_rows"].append({
                **{k: row[k] for k in ("candidate_job_id", "target", "variant", "state", "frame_id", "tolerance_px")},
                "visible_recall": metrics["visible_gt_to_candidate"]["recall_fraction"],
                "projected_precision": metrics["candidate_to_visible_gt"]["precision_fraction"],
                "visible_truth_distance_median_px": metrics["visible_gt_to_candidate"]["distance_median_px"],
                "visible_truth_distance_p95_px": metrics["visible_gt_to_candidate"]["distance_p95_px"],
                "guarded_gap_false_coverage": metrics["gap_clear_guarded_interior_to_candidate"]["false_coverage_fraction"],
                "candidate_projection": metrics["candidate_projection"], "eligible_for_success_claim": metrics["eligible_for_success_claim"],
            })
        result["runs"].append(run)
    destination = ROOT / "docs/experiments/results/2026-09-16-rod-evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = compact_tables(result) + "\n"
    if json.loads(encoded) != result:
        raise ValueError("Compact table serialization changed the data")
    destination.write_text(encoded, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    main()
