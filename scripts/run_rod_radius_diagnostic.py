"""Measure edge-ray radius consistency without changing any frozen predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))

from creator_eval.rod_radius_consistency import diagnose_edge_radii, summary  # noqa: E402
from run_rod_candidate_ablation import (  # noqa: E402
    canonical_hash,
    checked_artifact,
    inference_view,
)
from run_rod_identity_blender import digest, locations, read_json, write_json  # noqa: E402
from run_rod_identity_stress import reject_truth_open, validate_inference_identity  # noqa: E402
from run_rod_support_recheck import checked_run  # noqa: E402


def run(run_id, output):
    if output.exists():
        raise FileExistsError("Keep existing radius diagnostics")
    source_paths = ["scripts/run_rod_radius_diagnostic.py", "experiments/src/creator_eval/rod_radius_consistency.py"]
    hashes = {path: digest(ROOT / path) for path in source_paths}
    _, _, prepared, _, _, _ = checked_run(run_id)
    # Predicted axes are the subjects of this audit. Neither masks nor target
    # geometry are loaded, and this script has no route to change predictions.
    sys.addaudithook(reject_truth_open)
    rows, references = [], []
    for source_id in (prepared["baseline_run_id"], run_id):
        folder, _, _, _ = locations(source_id)
        inference_path = folder / "inference.json"
        if source_id == prepared["baseline_run_id"] and digest(inference_path) != prepared["baseline_inference_sha256"]:
            raise ValueError("Baseline inference changed")
        inference = read_json(inference_path)
        validate_inference_identity(inference, read_json(folder / "prepared.json"))
        pools = {}
        for entry in inference["pools"]:
            pool = checked_artifact(folder, entry)
            pools[(pool["case_id"], pool["search_offset_px"])] = (pool, entry["sha256"])
        selected_count = 0
        for entry in inference["associations"]:
            record = checked_artifact(folder, entry)
            if record["cap"] != 8:
                continue
            selected_count += 1
            association = record["association"]
            if canonical_hash(association) != record["association_sha256"]:
                raise ValueError("Association canonical hash mismatch")
            key = (record["case_id"], record["search_offset_px"])
            pool, pool_hash = pools[key]
            if pool_hash != record["pool_sha256"]:
                raise ValueError("Association/pool mismatch")
            frames = pool["frames"]
            views = [inference_view(f, f["pool"]["candidates"][:8], 0) for f in frames]
            hypotheses = ([association["selected"]] + association["alternatives"]) if association["selected"] else []
            for ordinal, hypothesis in enumerate(hypotheses):
                diagnostic = diagnose_edge_radii(hypothesis, views, [f["observations"] for f in frames])
                radii = [v["combined"]["median"] for v in diagnostic["views"] if v["combined"]["median"] is not None]
                imbalances = [v["relative_side_imbalance"] for v in diagnostic["views"] if v["relative_side_imbalance"] is not None]
                rows.append({"source_run_id": source_id, "case_id": key[0], "search_offset_px": key[1], "cap": 8,
                             "association_state": association["state"], "association_sha256": record["association_sha256"],
                             "hypothesis_ordinal": ordinal, "role": ("accepted_selection" if association["state"] == "accepted" else "unaccepted_best") if ordinal == 0 else "competing_alternative",
                             "view_median_radii": summary(radii), "view_side_imbalances": summary(imbalances), **diagnostic})
        if selected_count != 10:
            raise ValueError("The diagnostic requires every cap8 condition")
        references.append({"run_id": source_id, "inference_sha256": digest(inference_path)})
    if any(digest(ROOT / path) != sha for path, sha in hashes.items()):
        raise ValueError("Diagnostic source changed during execution")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"state": "diagnostic_complete", "source_sha256": hashes, "input_references": references,
                        "scope": "all_cap8_best_and_competing_alternatives_of_both_development_runs",
                        "gt_read": False, "truth_read_tripwire_enabled": True, "predictions_changed": False,
                        "radius_gate_applied": False, "rows": rows,
                        "limitations": ["Conditional circular-cylinder tangent geometry; an image edge need not be a silhouette.",
                                        "Small distance dispersion is not physical or user-target identity proof.",
                                        "Raw rows/views are correlated; these are diagnostics, not independent accuracy samples."]})
    print("RADIUS_DIAGNOSTIC", len(rows), "hypotheses; no prediction changed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="rod-refit-support-v1-20260920")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-20-edge-radius-diagnostic.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    run(args.run_id, args.output)
