"""Frozen analytic controls for unresolved one-pixel photometric line evidence."""

from __future__ import annotations

import copy

from .rod_observations import extract_rod_observations, fit_robust_image_line
from .rod_profile_controls import evaluate_profile_method, render_profile_fixture


def render_narrow_fixture(case, generator, *, seed):
    """Render geometry first, then optional opaque image-space occluders."""
    image, truth = render_profile_fixture(case, generator, seed=seed)
    image = image.copy()
    occluded_rows = set()
    for occluder in case.get("occluders", []):
        x0, x1 = map(int, occluder["x_interval"])
        y0, y1 = map(int, occluder["y_interval"])
        if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
            raise ValueError("Occluder is outside the analytic image")
        image[y0:y1, x0:x1] = int(occluder["value"])
        occluded_rows.update(range(y0, y1))
    truth["occluded_rows"] = sorted(occluded_rows)
    truth["occlusion_semantics"] = (
        "Physical rod geometry remains present; the opaque image overlay is not negative rod evidence."
    )
    return image, truth


def _compact(observations):
    return [
        {
            "y": row["y"],
            "status": row["status"],
            "reason": row["reason"],
            "pair_center_control": None,
            "candidates": [
                {
                    "center_x": candidate["center_x"],
                    "width": candidate["width"],
                    "left_edge_x": candidate["left_edge"]["x"],
                    "right_edge_x": candidate["right_edge"]["x"],
                    "polarity": candidate["polarity"],
                    "contrast": candidate["contrast"],
                    "candidate_kind": candidate.get("candidate_kind", "resolved_double_edge"),
                    "physical_width_resolved": candidate.get("physical_width_resolved", True),
                }
                for candidate in row["candidates"]
            ],
        }
        for row in observations["rows"]
    ]


def run_narrow_control_methods(rgb, guide_xyxy, method):
    """Run the legacy and opt-in detectors without fixture/truth inputs."""
    if method["variants"] != ["resolved_pairs", "unresolved_narrow_pairs"]:
        raise ValueError("Unexpected narrow-control variants")
    baseline = extract_rod_observations(rgb, guide_xyxy, method["observation"])
    narrow = extract_rod_observations(
        rgb,
        guide_xyxy,
        method["observation"],
        narrow_config=method["narrow_observation"],
    )
    outputs = {}
    for name, observations in zip(method["variants"], (baseline, narrow)):
        outputs[name] = {
            "rows": _compact(observations),
            "fitted": fit_robust_image_line(observations, method["line_fit"]),
        }
    return {"resolved_pairs": baseline, "unresolved_narrow_pairs": narrow}, outputs


def evaluate_narrow_control(prediction, truth, evaluation):
    summary, rows = evaluate_profile_method(prediction, truth, evaluation)
    occluded = set(truth.get("occluded_rows", []))
    summary["occluded_rows"] = len(occluded)
    summary["selected_occluded_rows"] = sum(
        row["y"] in occluded and row["selected"] is not None for row in rows
    )
    summary["unresolved_narrow_selected_rows"] = sum(
        row["selected"] is not None
        and row["selected"].get("candidate_kind") == "unresolved_narrow_photometric_pair"
        for row in rows
    )
    return summary, rows


def acceptance_failures(rows, contract):
    """Evaluate the predeclared mechanism contract without hiding failed cases."""
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    narrow = "unresolved_narrow_pairs"
    failures = []
    for case_id in contract["must_recover"]:
        row = by_key[(case_id, narrow)]
        if not row["usable"] or row["accurate_target_row_coverage"]["1"] != 1:
            failures.append(case_id + ":narrow_not_fully_recovered")
    for case_id in contract["must_not_fill_gap"]:
        if by_key[(case_id, narrow)]["gap_false_acceptance_fraction"] != 0:
            failures.append(case_id + ":gap_filled")
    for case_id in contract["must_abstain_or_remain_unselected"]:
        row = by_key[(case_id, narrow)]
        if case_id == "n04" and row["selected_rows"] != 0:
            failures.append(case_id + ":low_contrast_selected")
        if case_id == "n05" and row["fit_state"] != "ambiguous":
            failures.append(case_id + ":neighbor_identity_not_ambiguous")
        if case_id == "n06" and row["selected_occluded_rows"] != 0:
            failures.append(case_id + ":occlusion_selected")
    false_positive = by_key[(contract["known_indistinguishable_false_positive"][0], narrow)]
    if false_positive["false_positive_empty_row_fraction"] != 1:
        failures.append("n07:indistinguishable_limit_not_exposed")
    resolved = contract["resolved_control_must_not_change"][0]
    old, new = by_key[(resolved, "resolved_pairs")], by_key[(resolved, narrow)]
    for key in ("fit_state", "selected_rows", "target_row_coverage", "selected_center_error_px"):
        if old[key] != new[key]:
            failures.append(resolved + ":resolved_control_changed:" + key)
    return failures


def copy_without_truth(value):
    """Deep-copy helper used by boundary tests to prove inference input isolation."""
    return copy.deepcopy(value)
