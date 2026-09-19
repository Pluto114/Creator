"""Associate multiple RGB line hypotheses across calibrated views without truth."""

from __future__ import annotations

import copy
import itertools
import math

import numpy as np

from creator_eval.line_controls import LineFitDegenerate, fit_multiview_line


def _as_line(slope, intercept):
    line = np.array([1.0, -float(slope), -float(intercept)])
    return line / np.linalg.norm(line[:2])


def image_line_from_endpoints(points):
    """Return one normalized homogeneous image line from two finite pixel points."""
    points = np.asarray(points, float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Expected two finite image points")
    line = np.cross(np.c_[points, np.ones(2)][0], np.c_[points, np.ones(2)][1])
    length = np.linalg.norm(line[:2])
    if length <= 1e-12 or abs(line[0]) <= 1e-12:
        raise ValueError("Coincident or near-horizontal guide is outside this pilot")
    return line / length


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
        widths, enclosed = [], []
        for row_index, candidate_index, _ in matches:
            candidates = observations["rows"][row_index]["candidates"]
            chosen = candidates[candidate_index]
            widths.append(chosen["width"])
            enclosed.append(
                any(
                    other["width"] > chosen["width"] + 1
                    and other["left_edge"]["x"] <= chosen["left_edge"]["x"]
                    and other["right_edge"]["x"] >= chosen["right_edge"]["x"]
                    for other_index, other in enumerate(candidates)
                    if other_index != candidate_index
                )
            )
        hypotheses.append(
            {
                "line": _as_line(slope, intercept),
                "slope": float(slope),
                "intercept": float(intercept),
                "support_rows": len(matches),
                "residual_median_px": float(np.median(residuals)),
                "width_median_px": float(np.median(widths)),
                "width_p10_px": float(np.quantile(widths, 0.1)),
                "width_p90_px": float(np.quantile(widths, 0.9)),
                "enclosing_wider_row_fraction": float(np.mean(enclosed)),
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
    extrinsic = np.asarray(view["world_to_camera_cv"])
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :]
    if extrinsic.shape != (3, 4):
        raise ValueError("Expected a 3x4 or homogeneous 4x4 world-to-camera transform")
    projection = np.asarray(view["K_index"]) @ extrinsic
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


def _validated_views(views):
    # A malformed camera is a pipeline error, not a scene with no rod. Validate
    # before candidate enumeration so a catch-all cannot quietly eat it again.
    if len({view["view_id"] for view in views}) != len(views):
        raise ValueError("View IDs must be unique")
    for view in views:
        intrinsic = np.asarray(view["K_index"], float)
        extrinsic = np.asarray(view["world_to_camera_cv"], float)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("Expected finite 3x3 intrinsics")
        if abs(np.linalg.det(intrinsic)) <= 1e-12 or not np.allclose(intrinsic[2], [0, 0, 1]):
            raise ValueError("Expected invertible pinhole intrinsics")
        if extrinsic.shape not in ((3, 4), (4, 4)) or not np.isfinite(extrinsic).all():
            raise ValueError("Expected finite 3x4 or 4x4 extrinsics")
        if extrinsic.shape == (4, 4) and not np.allclose(extrinsic[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous camera row")
        rotation = extrinsic[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Camera rotation must be proper and orthonormal")
        ys = np.asarray(view["y_range"], float)
        if ys.shape != (2,) or not np.isfinite(ys).all() or ys[1] <= ys[0]:
            raise ValueError("Expected increasing finite image y range")
        for candidate in view["candidates"]:
            line = np.asarray(candidate["line"], float)
            if line.shape != (3,) or not np.isfinite(line).all() or abs(line[0]) <= 1e-12:
                raise ValueError("Expected finite non-horizontal image lines")


def associate_multiview_lines(views, config):
    """Associate lines with explicit search completion; truncation cannot prove uniqueness."""
    _validated_views(views)
    fit_count = int(config["fit_view_count"])
    maximum = int(config["maximum_hypotheses"])
    max_attempts = int(config.get("maximum_fit_attempts", 100000))
    if fit_count < 3 or maximum < 1 or max_attempts < 1:
        raise ValueError("Invalid candidate search budget")
    combinations = list(itertools.combinations(range(len(views)), fit_count))
    total = sum(math.prod(len(views[i]["candidates"]) for i in indices) for indices in combinations)
    generated = []
    attempted = 0
    stop = False
    for selected_views in combinations:
        candidate_ranges = [range(len(views[index]["candidates"])) for index in selected_views]
        for selected_candidates in itertools.product(*candidate_ranges):
            if len(generated) >= maximum or attempted >= max_attempts:
                stop = True
                break
            attempted += 1
            try:
                model = fit_multiview_line(
                    [views[v]["candidates"][c]["line"] for v, c in zip(selected_views, selected_candidates)],
                    [views[v]["K_index"] for v in selected_views],
                    [views[v]["world_to_camera_cv"] for v in selected_views],
                )
                matches = [_view_match(model, view) for view in views]
            except (LineFitDegenerate, np.linalg.LinAlgError):
                continue
            supporting = [
                index for index, match in enumerate(matches)
                if match["residual_px"] is not None
                and match["residual_px"] <= config["reprojection_threshold_px"]
            ]
            if len(supporting) < int(config["minimum_support_views"]):
                continue
            residuals = [matches[index]["residual_px"] for index in supporting]
            # These projections were just calculated. Re-projecting every pair
            # during deduplication cost minutes without adding any information.
            signature = np.concatenate([
                _line_x(match["projected_line"], np.linspace(*view["y_range"], 9))
                for match, view in zip(matches, views)
            ])
            generated.append({
                "model": model, "matches": matches, "supporting_views": supporting,
                "support_view_count": len(supporting),
                "residual_median_px": float(np.median(residuals)),
                "source_fit_views": list(selected_views),
                "source_candidate_indices": list(selected_candidates),
                "_signature": signature,
            })
        if stop:
            break
    generated.sort(key=lambda row: (-row["support_view_count"], row["residual_median_px"]))
    unique, signatures = [], []
    for hypothesis in generated:
        signature = hypothesis.pop("_signature")
        if signatures and np.any(np.median(np.abs(np.asarray(signatures) - signature), axis=1) < config["deduplicate_separation_px"]):
            continue
        unique.append(hypothesis)
        signatures.append(signature)
    complete = attempted == total
    result = {
        "state": "rejected" if complete else "ambiguous",
        "reason": "no_3d_line_with_required_view_support" if complete else "search_budget_exhausted",
        "selected": None, "alternatives": [],
        "unique_hypothesis_count": len(unique),
        "generated_hypothesis_count": len(generated),
        "search_complete": complete,
        "attempted_combination_count": attempted,
        "total_combination_count": total,
        "scope": "infinite_3d_line_identity_only_no_extent_gap_or_physical_existence_proof",
    }
    if not unique:
        return result
    best = unique[0]
    alternatives = []
    for index, candidate in enumerate(unique[1:], 1):
        if candidate["support_view_count"] < config["ambiguity_support_fraction"] * best["support_view_count"]:
            continue
        separation = float(np.median(np.abs(signatures[0] - signatures[index])))
        if separation >= config["ambiguity_separation_px"]:
            alternatives.append({**candidate, "separation_from_best_px": separation})
    result.update(
        state="ambiguous" if alternatives or not complete else "accepted",
        reason=("search_budget_exhausted" if not complete else
                "competing_multiview_lines" if alternatives else "unique_multiview_line"),
        selected=best, alternatives=alternatives,
    )
    return result


def apply_nested_band_guard(result, views, config):
    """Turn a unique geometric result into ambiguity when wider bands enclose it.

    This guard never replaces the selected line with the widest candidate. A wider
    pair may be a silhouette, another object, or background texture; it is only
    evidence that the physical-axis identity is unresolved.
    """
    output = copy.deepcopy(result)
    if output["state"] != "accepted":
        output["identity_guard"] = {
            "applied": False,
            "reason": "geometric_result_not_uniquely_accepted",
            "nested_view_count": 0,
        }
        return output
    nested = []
    for view_index in output["selected"]["supporting_views"]:
        match = output["selected"]["matches"][view_index]
        candidate = views[view_index]["candidates"][match["candidate_index"]]
        if candidate["enclosing_wider_row_fraction"] >= config["minimum_nested_row_fraction"]:
            nested.append(view_index)
    passed = len(nested) >= int(config["minimum_nested_views"])
    output["identity_guard"] = {
        "applied": True,
        "reason": "nested_supported_band_identity_unresolved" if passed else "no_repeated_nested_band_warning",
        "nested_view_count": len(nested),
        "nested_views": nested,
        "minimum_nested_views": int(config["minimum_nested_views"]),
        "minimum_nested_row_fraction": config["minimum_nested_row_fraction"],
    }
    if passed:
        output["state"] = "ambiguous"
        output["reason"] = "nested_supported_band_identity_unresolved"
    return output


def apply_guide_identity_guard(result, views, config):
    """Reject a physical line that does not stay near the supplied target guide.

    Geometry can prove that a line exists, but not that it is the line requested by
    the user. This check treats the coarse per-view guide as part of target identity.
    The tolerance is intentionally wider than the geometric reprojection threshold.
    """
    output = copy.deepcopy(result)
    if output["state"] != "accepted":
        output["guide_identity_guard"] = {
            "applied": False,
            "reason": "geometric_result_not_uniquely_accepted",
        }
        return output
    residuals = []
    for view in views:
        projected = _project_world_line(output["selected"]["model"], view)
        guide = np.asarray(view["guide_line"], float)
        ys = np.linspace(view["y_range"][0], view["y_range"][1], 9)
        residuals.append(float(np.median(np.abs(_line_x(projected, ys) - _line_x(guide, ys)))))
    threshold = float(config["maximum_median_residual_px"])
    agreeing = [index for index, value in enumerate(residuals) if value <= threshold]
    minimum_views = int(config["minimum_views"])
    minimum_fraction = float(config["minimum_fraction"])
    passed = len(agreeing) >= minimum_views and len(agreeing) / len(views) >= minimum_fraction
    output["guide_identity_guard"] = {
        "applied": True,
        "reason": "guide_identity_supported" if passed else "selected_line_outside_guide_identity_band",
        "residual_px_per_view": residuals,
        "agreeing_views": agreeing,
        "agreeing_view_count": len(agreeing),
        "view_count": len(views),
        "maximum_median_residual_px": threshold,
        "minimum_views": minimum_views,
        "minimum_fraction": minimum_fraction,
    }
    if not passed:
        output["state"] = "rejected"
        output["reason"] = "selected_line_outside_guide_identity_band"
    return output
