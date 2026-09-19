"""Associate multiple RGB line hypotheses across calibrated views without truth."""

from __future__ import annotations

import itertools

import numpy as np

from creator_eval.line_controls import LineFitDegenerate, fit_multiview_line


def _as_line(slope, intercept):
    line = np.array([1.0, -float(slope), -float(intercept)])
    return line / np.linalg.norm(line[:2])


def _line_x(line, y):
    line = np.asarray(line, float)
    if abs(line[0]) <= 1e-12:
        raise ValueError("Near-horizontal image lines are outside this pilot")
    return -(line[1] * np.asarray(y, float) + line[2]) / line[0]


def _support(rows, slope, intercept, threshold):
    matches = []
    divisor = float(np.hypot(1.0, slope))
    for row_index, row in rows:
        centers = np.array([candidate["center_x"] for candidate in row["candidates"]])
        residuals = np.abs(centers - (slope * row["y"] + intercept)) / divisor
        candidate = int(np.argmin(residuals))
        if residuals[candidate] <= threshold:
            matches.append((row_index, candidate, float(residuals[candidate])))
    return matches


def enumerate_image_lines(observations, config):
    """Return distinct supported image lines while retaining competing explanations."""
    rows = [(index, row) for index, row in enumerate(observations["rows"]) if row["candidates"]]
    if len(rows) < config["minimum_rows"]:
        return []
    rng = np.random.default_rng(int(config["seed"]))
    hypotheses = []
    for _ in range(int(config["trials"])):
        first, second = rng.choice(len(rows), 2, replace=False)
        a, b = rows[first][1], rows[second][1]
        if abs(b["y"] - a["y"]) < 20:
            continue
        xa = a["candidates"][rng.integers(len(a["candidates"]))]["center_x"]
        xb = b["candidates"][rng.integers(len(b["candidates"]))]["center_x"]
        slope = float((xb - xa) / (b["y"] - a["y"]))
        intercept = float(xa - slope * a["y"])
        matches = _support(rows, slope, intercept, config["inlier_distance_px"])
        if len(matches) < config["minimum_rows"]:
            continue
        ys = np.array([observations["rows"][i]["y"] for i, _, _ in matches], float)
        if np.ptp(ys) < config["minimum_y_span"]:
            continue
        xs = np.array(
            [observations["rows"][i]["candidates"][j]["center_x"] for i, j, _ in matches]
        )
        slope, intercept = np.linalg.lstsq(
            np.c_[ys, np.ones(len(ys))], xs, rcond=None
        )[0]
        matches = _support(rows, float(slope), float(intercept), config["inlier_distance_px"])
        residuals = np.array([value for _, _, value in matches])
        hypotheses.append(
            {
                "line": _as_line(slope, intercept),
                "slope": float(slope),
                "intercept": float(intercept),
                "support_rows": len(matches),
                "residual_median_px": float(np.median(residuals)),
                "row_matches": matches,
            }
        )
    hypotheses.sort(key=lambda row: (-row["support_rows"], row["residual_median_px"]))
    kept = []
    ys = np.array([row["y"] for _, row in rows], float)
    for item in hypotheses:
        if any(
            np.median(np.abs(_line_x(item["line"], ys) - _line_x(old["line"], ys)))
            < config["deduplicate_separation_px"]
            for old in kept
        ):
            continue
        kept.append(item)
        if len(kept) >= int(config["maximum_models"]):
            break
    return kept


def _project_world_line(model, view):
    anchor, direction = np.asarray(model["anchor"]), np.asarray(model["direction"])
    points = np.c_[np.stack((anchor - direction, anchor + direction)), np.ones(2)]
    projection = np.asarray(view["K_index"]) @ np.asarray(view["world_to_camera_cv"])
    pixels = points @ projection.T
    if np.any(pixels[:, 2] <= 1e-9):
        raise LineFitDegenerate("line_projection_behind_camera")
    line = np.cross(pixels[0], pixels[1])
    if np.linalg.norm(line[:2]) <= 1e-12 or abs(line[0]) <= 1e-12:
        raise LineFitDegenerate("unsupported_projected_line_orientation")
    return line / np.linalg.norm(line[:2])


def _view_match(model, view):
    projected = _project_world_line(model, view)
    ys = np.linspace(view["y_range"][0], view["y_range"][1], 9)
    options = []
    for index, candidate in enumerate(view["candidates"]):
        distances = np.abs(_line_x(projected, ys) - _line_x(candidate["line"], ys))
        options.append((float(np.median(distances)), index))
    if not options:
        return {"candidate_index": None, "residual_px": None, "projected_line": projected}
    residual, index = min(options)
    return {"candidate_index": index, "residual_px": residual, "projected_line": projected}


def _signature_separation(first, second, views):
    distances = []
    for view in views:
        ys = np.linspace(view["y_range"][0], view["y_range"][1], 9)
        a, b = _project_world_line(first, view), _project_world_line(second, view)
        distances.extend(np.abs(_line_x(a, ys) - _line_x(b, ys)))
    return float(np.median(distances))


def associate_multiview_lines(views, config):
    """Enumerate calibrated 3D line explanations and keep uncertainty explicit."""
    if len({view["view_id"] for view in views}) != len(views):
        raise ValueError("View IDs must be unique")
    fit_count = int(config["fit_view_count"])
    generated = []
    for selected_views in itertools.combinations(range(len(views)), fit_count):
        candidate_ranges = [range(len(views[index]["candidates"])) for index in selected_views]
        for selected_candidates in itertools.product(*candidate_ranges):
            if len(generated) >= int(config["maximum_hypotheses"]):
                break
            try:
                model = fit_multiview_line(
                    [views[v]["candidates"][c]["line"] for v, c in zip(selected_views, selected_candidates)],
                    [views[v]["K_index"] for v in selected_views],
                    [views[v]["world_to_camera_cv"] for v in selected_views],
                )
                matches = [_view_match(model, view) for view in views]
            except (LineFitDegenerate, ValueError, np.linalg.LinAlgError):
                continue
            supporting = [
                index for index, match in enumerate(matches)
                if match["residual_px"] is not None
                and match["residual_px"] <= config["reprojection_threshold_px"]
            ]
            if len(supporting) < int(config["minimum_support_views"]):
                continue
            residuals = [matches[index]["residual_px"] for index in supporting]
            generated.append(
                {
                    "model": model,
                    "matches": matches,
                    "supporting_views": supporting,
                    "support_view_count": len(supporting),
                    "residual_median_px": float(np.median(residuals)),
                    "source_fit_views": list(selected_views),
                    "source_candidate_indices": list(selected_candidates),
                }
            )
    generated.sort(key=lambda row: (-row["support_view_count"], row["residual_median_px"]))
    unique = []
    for hypothesis in generated:
        if any(
            _signature_separation(hypothesis["model"], old["model"], views)
            < config["deduplicate_separation_px"]
            for old in unique
        ):
            continue
        unique.append(hypothesis)
    result = {
        "state": "rejected",
        "reason": "no_3d_line_with_required_view_support",
        "selected": None,
        "alternatives": [],
        "unique_hypothesis_count": len(unique),
        "generated_hypothesis_count": len(generated),
        "scope": "infinite_3d_line_identity_only_no_extent_gap_or_physical_existence_proof",
    }
    if not unique:
        return result
    best = unique[0]
    alternatives = []
    for candidate in unique[1:]:
        if candidate["support_view_count"] < (
            config["ambiguity_support_fraction"] * best["support_view_count"]
        ):
            continue
        separation = _signature_separation(best["model"], candidate["model"], views)
        if separation >= config["ambiguity_separation_px"]:
            alternatives.append({**candidate, "separation_from_best_px": separation})
    result.update(
        state="ambiguous" if alternatives else "accepted",
        reason="competing_multiview_lines" if alternatives else "unique_multiview_line",
        selected=best,
        alternatives=alternatives,
    )
    return result
