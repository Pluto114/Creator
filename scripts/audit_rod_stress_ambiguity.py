"""Inspect guide compatibility of saved stress hypotheses without reading truth.

This is an after-the-run diagnostic. It never changes the selected line, original
state, or finite output. Saved alternatives are only a filtered subset of the
unique hypotheses, so their count cannot establish a unique physical target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_multiview_candidates import (  # noqa: E402
    apply_guide_identity_guard,
    image_line_from_endpoints,
)
from run_rod_identity_blender import (  # noqa: E402
    digest,
    locations,
    read_json,
    write_json,
)
from run_rod_identity_stress import (  # noqa: E402
    frozen_source_check,
    reject_truth_open,
    validate_inference_identity,
)


def audit_saved_candidates(record, frames, method):
    """Hypothetically apply the same guide test to each saved top-support line."""
    association = record["association"]
    selected = association["selected"]
    alternatives = association["alternatives"]
    views = []
    for frame in frames:
        guide = np.asarray(frame["guide_xyxy"], float).copy()
        guide[:, 0] += record["guide_offset_px"]
        views.append({
            "view_id": frame["view_id"], "K_index": frame["K_index"],
            "world_to_camera_cv": frame["world_to_camera_cv"],
            "y_range": guide[:, 1].tolist(),
            "guide_line": image_line_from_endpoints(guide),
        })
    saved = []
    lower_support_count = 0
    if selected is not None:
        saved.append(("selected", selected, 0.0))
        for index, candidate in enumerate(alternatives):
            if candidate["support_view_count"] != selected["support_view_count"]:
                lower_support_count += 1
                continue
            saved.append((f"alternative_{index}", candidate,
                          candidate["separation_from_best_px"]))
    candidates = []
    for label, candidate, separation in saved:
        # The ordinary guard intentionally skips ambiguity. For diagnosis only,
        # ask how each already-saved line would fare if considered on its own.
        # This temporary wrapper is never written back as a new method result.
        hypothetical = {"state": "accepted", "selected": candidate}
        checked = apply_guide_identity_guard(hypothetical, views, method["guide_guard"])
        diagnostics = checked["guide_identity_guard"]
        residuals = diagnostics["residual_px_per_view"]
        candidates.append({
            "saved_label": label,
            "support_view_count": candidate["support_view_count"],
            "supporting_view_ids": [views[index]["view_id"] for index in candidate["supporting_views"]],
            "separation_from_original_selected_px": float(separation),
            "association_residual_median_px": candidate["residual_median_px"],
            "guide_compatible": checked["state"] == "accepted",
            "guide_residual_median_across_views_px": float(np.median(residuals)),
            "guide_residual_max_across_views_px": float(max(residuals)),
            "guide_check": diagnostics,
        })
    return {
        "case_id": record["case_id"], "guide_offset_px": record["guide_offset_px"],
        "original_association_state": association["state"],
        "original_association_reason": association["reason"],
        "original_guarded_state": record["guarded"]["state"],
        "original_finite_state": record["finite"]["state"],
        "search_complete": association["search_complete"],
        "attempted_combination_count": association["attempted_combination_count"],
        "total_combination_count": association["total_combination_count"],
        "original_unique_hypothesis_count": association["unique_hypothesis_count"],
        "image_candidate_counts": record["image_candidate_counts"],
        "image_candidate_limit_reached_views": record["image_candidate_limit_reached_views"],
        "saved_alternative_count": len(alternatives),
        "saved_lower_support_alternatives_skipped": lower_support_count,
        "audited_top_support_candidate_count": len(candidates),
        "guide_compatible_saved_candidate_count": sum(row["guide_compatible"] for row in candidates),
        "candidates": candidates,
    }


def audit(run_id, output_path=None):
    run, inputs, _, _ = locations(run_id)
    destination = Path(output_path) if output_path else run / "saved_ambiguity_guide_audit.json"
    if destination.exists():
        raise FileExistsError(f"Keep previous ambiguity audit: {destination}")
    inference_path = run / "inference.json"
    if not inference_path.is_file():
        raise FileNotFoundError("Wait for the frozen inference.json to finish before auditing")
    # Use the same accidental-truth-read tripwire. It is not an OS sandbox.
    sys.addaudithook(reject_truth_open)
    prepared = read_json(run / "prepared.json")
    if prepared["run_id"] != run_id or prepared["state"] != "prepared":
        raise ValueError("Prepared run identity mismatch")
    frozen_source_check(run, prepared)
    if digest(inputs / "manifest.json") != prepared["input_manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    method = read_json(run / "method_config.json")
    manifest = read_json(inputs / "manifest.json")
    inference = read_json(inference_path)
    validate_inference_identity(inference, prepared)
    frames_by_case = {case["case_id"]: case["frames"] for case in manifest["cases"]}
    keys = [(row["case_id"], row["guide_offset_px"]) for row in inference["rows"]]
    expected = {(case_id, offset) for case_id in frames_by_case for offset in method["guide_offsets_px"]}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete, duplicate, or unexpected inference rows")
    rows = [audit_saved_candidates(record, frames_by_case[record["case_id"]], method)
            for record in inference["rows"]]
    result = {
        "state": "diagnostic_complete", "run_id": run_id,
        "gt_read_during_audit": False, "truth_read_tripwire_enabled": True,
        "inference_sha256": digest(inference_path),
        "input_manifest_sha256": prepared["input_manifest_sha256"],
        "method_config_sha256": prepared["method_config_sha256"],
        "audit_source_sha256": digest(Path(__file__)),
        "candidate_scope": "Original selected line plus saved alternatives with exactly the same support count. Alternatives already passed the association support and separation filters; this is not the full unique-hypothesis list.",
        "separation_definition": "Saved median horizontal pixel separation across input views and nine y samples per view, measured against the original selected line.",
        "guide_rule": method["guide_guard"],
        "limitations": [
            "No new winner, rod segment, method state, or physical-identity claim is produced.",
            "Even one guide-compatible saved line does not prove uniqueness: nearby unique lines, lower-support hypotheses, and image candidates omitted by RANSAC or its cap are not audited.",
            "Search completion describes the retained image hypotheses only.",
            "Guide incompatibility does not identify a line as a brick seam; confirming that interpretation requires inspecting allowed RGB images.",
            "This post-inference diagnostic cannot be promoted to a frozen algorithm result or blind-test score.",
        ],
        "rows": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    print(f"AUDITED_SAVED_AMBIGUITY {len(rows)} rows: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from environment_paths import require_project_environment

    require_project_environment(ROOT)
    audit(args.run_id, args.output)
