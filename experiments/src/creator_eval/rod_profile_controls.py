"""Analytic profile fixtures and an explicitly RGB-only conservative control.

Fixture geometry is an evaluator resource. ``run_profile_methods`` accepts only
an RGB image, the coarse guide, and method settings; it never receives a fixture.
"""

import copy
from collections import Counter

import numpy as np

from .rod_observations import extract_rod_observations, fit_robust_image_line


def _profile_values(normalized_x, profile):
    if profile["kind"] == "flat":
        return np.full_like(normalized_x, float(profile["value"]))
    if profile["kind"] == "linear_knots":
        return np.interp(normalized_x, profile["positions"], profile["values"])
    if profile["kind"] == "bands":
        values = np.full_like(normalized_x, float(profile["base"]))
        for band in profile["bands"]:
            low, high = band["interval"]
            values[(normalized_x >= low) & (normalized_x < high)] = band["value"]
        return values
    raise ValueError("Unknown analytic profile: " + profile["kind"])


def render_profile_fixture(fixture, generator, *, seed):
    """Return a quantized analytic RGB and a separate per-row geometry answer.

    Horizontal pixel-area integration uses fixed midpoint supersampling. Blur is
    the declared discrete horizontal PSF, not a claim about real-camera optics.
    """
    width, height = int(generator["width"]), int(generator["height"])
    samples = int(generator["horizontal_area_samples"])
    if min(width, height, samples) <= 0:
        raise ValueError("Positive image dimensions and area samples required")
    x = (np.arange(width * samples) + 0.5) / samples - 0.5
    background = fixture["background"]
    subpixels = np.full((height, len(x)), float(background["value"]))
    for stripe in background["stripes"]:
        inside = np.abs(x - stripe["center_x"]) < stripe["width"] / 2
        subpixels[:, inside] = stripe["value"]
    truth_rows = [{"y": y, "objects": []} for y in range(height)]
    for rod in fixture["rods"]:
        for y in range(height):
            if not any(low <= y < high for low, high in rod["y_intervals"]):
                continue
            center = rod["center_x_at_reference_y"] + rod["slope"] * (y - rod["reference_y"])
            left, right = center - rod["width"] / 2, center + rod["width"] / 2
            inside = (x >= left) & (x < right)
            normalized = (x[inside] - left) / rod["width"]
            subpixels[y, inside] = _profile_values(normalized, rod["profile"])
            truth_rows[y]["objects"].append(
                {
                    "object_id": rod["object_id"],
                    "center_x": center,
                    "left_x": left,
                    "right_x": right,
                    "width": rod["width"],
                }
            )
    gray = subpixels.reshape(height, width, samples).mean(axis=2)
    sigma = float(fixture["blur_sigma_px"])
    if sigma > 0:
        radius = int(np.ceil(4 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(gray, ((0, 0), (radius, radius)), mode="edge")
        gray = np.array([np.convolve(row, kernel, mode="valid") for row in padded])
    rng = np.random.default_rng(seed)
    gray += rng.normal(0, fixture["noise_sigma"], size=gray.shape)
    image = np.repeat(np.rint(np.clip(gray, 0, 255)).astype(np.uint8)[..., None], 3, axis=2)
    truth = {
        "case_id": fixture["case_id"],
        "label": fixture["label"],
        "target_identity": fixture["target_identity"],
        "gap_intervals": fixture.get("gap_intervals", []),
        "size_wh": [width, height],
        "rows": truth_rows,
        "seed": seed,
        "geometry_semantics": "physical interval before illumination, blur, and quantization",
        "background_semantics": "painted vertical stripes are not foreground rod geometry",
    }
    return image, truth


def reject_disagreeing_pair_centers(observations, maximum_center_spread_px):
    """Reject conflicting reliable pairs without choosing a preferred boundary.

    A unique reliable pair can still be an internal highlight. This check cannot
    certify a silhouette; all original candidates remain in the audit record.
    """
    if not np.isfinite(maximum_center_spread_px) or maximum_center_spread_px < 0:
        raise ValueError("Center-spread threshold must be finite and nonnegative")
    result = copy.deepcopy(observations)
    for row in result["rows"]:
        centers = [candidate["center_x"] for candidate in row["candidates"]]
        spread = float(np.ptp(centers)) if centers else None
        rejected = spread is not None and spread > maximum_center_spread_px
        row["pair_center_control"] = {
            "spread_px": spread,
            "threshold_px": maximum_center_spread_px,
            "rejected": rejected,
        }
        if rejected:
            # 最外边那一对不一定是轮廓。这里宁可把整行退回unknown，不能偷挑答案。
            row["candidates_before_control"] = row["candidates"]
            row["candidates"] = []
            row["status"] = "unknown"
            row["reason"] = "reliable_pairs_disagree_on_center"
    return result


def _compact_rows(observations):
    return [
        {
            "y": row["y"],
            "status": row["status"],
            "reason": row["reason"],
            "pair_center_control": row.get("pair_center_control"),
            "raw_candidate_count": len(row.get("candidates_before_control", row["candidates"])),
            "candidates": [
                {
                    "center_x": candidate["center_x"],
                    "width": candidate["width"],
                    "left_edge_x": candidate["left_edge"]["x"],
                    "right_edge_x": candidate["right_edge"]["x"],
                    "polarity": candidate["polarity"],
                    "contrast": candidate["contrast"],
                }
                for candidate in row["candidates"]
            ],
        }
        for row in observations["rows"]
    ]


def run_profile_methods(rgb, guide_xyxy, method_config):
    """Method boundary: no fixture specification, truth center, or case label."""
    expected_variants = ["paired_robust", "pair_center_consensus_reject"]
    if method_config["variants"] != expected_variants:
        raise ValueError("Unsupported method protocol")
    raw = extract_rod_observations(rgb, guide_xyxy, method_config["observation"])
    controlled = reject_disagreeing_pair_centers(raw, method_config["maximum_center_spread_px"])
    outputs = {}
    for name, observations in zip(expected_variants, [raw, controlled]):
        outputs[name] = {
            "rows": _compact_rows(observations),
            "fitted": fit_robust_image_line(observations, method_config["line_fit"]),
        }
    return raw, outputs


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _distribution(values, reason="no_accepted_unique_target_rows"):
    values = np.asarray(values, float)
    return {
        "count": len(values),
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.quantile(values, 0.95)) if len(values) else None,
        "max": float(np.max(values)) if len(values) else None,
        "reason": None if len(values) else reason,
    }


def evaluate_profile_method(prediction, truth, evaluation_config):
    """Score the chosen centers, never use truth to choose among candidate pairs."""
    truth_rows = {row["y"]: row for row in truth["rows"]}
    fitted = prediction["fitted"]
    matches = (
        {item["row_index"]: item["candidate_index"] for item in fitted["row_matches"]}
        if fitted["usable"]
        else {}
    )
    center_errors, boundary_errors, width_errors, details = [], [], [], []
    selected_count = target_rows = physical_rows = empty_rows = gap_rows = 0
    selected_target = false_positive = gap_selected = false_absent = ambiguous_choices = 0
    accurate_counts = {str(value): 0 for value in evaluation_config["center_tolerances_px"]}
    counts = Counter(row["status"] for row in prediction["rows"])
    rejected_rows = 0
    for index, row in enumerate(prediction["rows"]):
        actual = truth_rows[row["y"]]["objects"]
        unique = truth["target_identity"] == "unique" and len(actual) == 1
        physical_rows += bool(actual)
        empty_rows += not actual
        target_rows += unique
        in_gap = any(low <= row["y"] < high for low, high in truth["gap_intervals"])
        gap_rows += in_gap
        selected = row["candidates"][matches[index]] if index in matches else None
        selected_count += selected is not None
        false_positive += selected is not None and not actual
        gap_selected += selected is not None and in_gap
        false_absent += bool(actual) and row["status"] == "absent"
        ambiguous_choices += selected is not None and truth["target_identity"] == "ambiguous"
        rejected_rows += bool((row["pair_center_control"] or {}).get("rejected"))
        detail = {
            "y": row["y"],
            "detector_status": row["status"],
            "selected": selected,
            "physical_objects": actual,
            "in_gap": bool(in_gap),
            "center_error_px": None,
        }
        if selected is not None and unique:
            selected_target += 1
            center_error = abs(selected["center_x"] - actual[0]["center_x"])
            boundary_error = max(
                abs(selected["left_edge_x"] - actual[0]["left_x"]),
                abs(selected["right_edge_x"] - actual[0]["right_x"]),
            )
            center_errors.append(center_error)
            boundary_errors.append(boundary_error)
            width_errors.append(abs(selected["width"] - actual[0]["width"]))
            for tolerance in evaluation_config["center_tolerances_px"]:
                accurate_counts[str(tolerance)] += center_error <= tolerance
            detail.update(center_error_px=center_error, maximum_boundary_error_px=boundary_error)
        details.append(detail)
    no_center_reason = (
        "no_accepted_unique_target_rows"
        if truth["target_identity"] == "unique"
        else "no_unique_target_identity"
    )
    summary = {
        "case_id": truth["case_id"],
        "label": truth["label"],
        "target_identity": truth["target_identity"],
        "fit_state": fitted["state"],
        "fit_reason": fitted["reason"],
        "usable": fitted["usable"],
        "sampled_rows": len(prediction["rows"]),
        "detector_row_states": dict(counts),
        "selected_rows": selected_count,
        "physical_rod_rows": physical_rows,
        "unique_target_rows": target_rows,
        "selected_unique_target_rows": selected_target,
        "target_row_coverage": _fraction(selected_target, target_rows),
        "accurate_target_row_coverage": {
            key: _fraction(value, target_rows) for key, value in accurate_counts.items()
        },
        "selected_center_error_px": _distribution(center_errors, no_center_reason),
        "selected_maximum_boundary_error_px": _distribution(boundary_errors, no_center_reason),
        "selected_width_error_px": _distribution(width_errors, no_center_reason),
        "empty_truth_rows": empty_rows,
        "false_selected_empty_rows": false_positive,
        "false_positive_empty_row_fraction": _fraction(false_positive, empty_rows),
        "gap_rows": gap_rows,
        "false_selected_gap_rows": gap_selected,
        "gap_false_acceptance_fraction": _fraction(gap_selected, gap_rows),
        "false_absent_physical_rows": false_absent,
        "false_absence_fraction": _fraction(false_absent, physical_rows),
        "ambiguous_identity_accepted_rows": ambiguous_choices,
        "unsupported_identity_choice_fraction": _fraction(
            ambiguous_choices, len(prediction["rows"])
        )
        if truth["target_identity"] == "ambiguous"
        else None,
        "center_disagreement_rejected_rows": rejected_rows,
        "scope": "single-image row diagnostics; not 3D recovery, physical-width estimation, or silhouette certification",
    }
    return summary, details
