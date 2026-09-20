"""Conditional cylinder-silhouette screening in original image pixels.

The radius is a nuisance parameter, not a delivered surface estimate. A matching
pair may still be a stripe, or a different real object. This filter cannot supply
identity, unseen candidate completeness, or camera accuracy.
"""
from __future__ import annotations

import copy

import numpy as np

from .line_controls import LineFitDegenerate, fit_multiview_line
from .rod_multiview_candidates import _validated_views
from .rod_radius_consistency import axis_ray_distances

DEFAULT_POLICY = {"edge_tolerance_px": 1.0, "minimum_side_fraction": .9, "minimum_views": 4}


def policy_values(config=None):
    policy = {**DEFAULT_POLICY, **(config or {})}
    if set(policy) != set(DEFAULT_POLICY):
        raise ValueError("Unknown cylinder gate option")
    if not np.isfinite(policy["edge_tolerance_px"]) or policy["edge_tolerance_px"] <= 0:
        raise ValueError("Positive finite pixel tolerance required")
    if not 0 < policy["minimum_side_fraction"] <= 1:
        raise ValueError("Side fraction must lie in (0, 1]")
    if isinstance(policy["minimum_views"], bool) or not isinstance(policy["minimum_views"], int) or policy["minimum_views"] < 4:
        raise ValueError("At least four views required")
    return policy


def silhouette_x(model, view, radius, ys):
    """Project the two tangent planes of an infinite circular cylinder.

    Each plane passes through the camera. Its unit normal has dot(n, u)=r/d,
    where u points from the camera towards the closest axis point. Intersecting
    the projected lines with an image row gives the predicted horizontal edges.
    """
    a, v = np.asarray(model["anchor"], float), np.asarray(model["direction"], float)
    if a.shape != (3,) or v.shape != (3,) or not np.isfinite([a, v]).all() or np.linalg.norm(v) <= 1e-12:
        raise ValueError("Finite nonzero 3D axis required")
    v = v / np.linalg.norm(v)
    camera = np.asarray(view["world_to_camera_cv"], float)[:3]
    rotation, translation = camera[:, :3], camera[:, 3]
    center = -rotation.T @ translation
    delta = a - center
    q = delta - np.dot(delta, v) * v
    distance = np.linalg.norm(q)
    if not np.isfinite(radius) or not 0 < radius < distance:
        raise LineFitDegenerate("camera_inside_or_on_cylinder_or_nonpositive_radius")
    u = q / distance
    w = np.cross(v, u)
    ratio = radius / distance
    normals = np.stack([ratio * u + sign * np.sqrt(1 - ratio * ratio) * w for sign in (-1, 1)])
    # 世界平面法向转进相机，再用K的逆转置变成图像线。别把点的投影公式硬套给线。
    lines = np.linalg.solve(np.asarray(view["K_index"]).T, (normals @ rotation.T).T).T
    if not np.isfinite(lines).all() or np.any(np.abs(lines[:, 0]) <= 1e-12 * np.linalg.norm(lines[:, :2], axis=1)):
        raise LineFitDegenerate("near_horizontal_cylinder_silhouette")
    xs = -(lines[:, 1, None] * np.asarray(ys)[None, :] + lines[:, 2, None]) / lines[:, 0, None]
    return np.sort(xs, axis=0)


def _matched_edges(hypothesis, views, observations):
    _validated_views(views)
    if len(views) != len(observations) or len(hypothesis["matches"]) != len(views):
        raise ValueError("Views, raw observations and matches must align")
    supporting = hypothesis["supporting_views"]
    if len(set(supporting)) != len(supporting):
        raise ValueError("Duplicate supporting view")
    result = []
    for vi in supporting:
        if isinstance(vi, bool) or not isinstance(vi, (int, np.integer)) or not 0 <= vi < len(views):
            raise ValueError("Invalid supporting view index")
        view, raw = views[vi], observations[vi]
        ci = hypothesis["matches"][vi]["candidate_index"]
        if isinstance(ci, bool) or not isinstance(ci, (int, np.integer)) or not 0 <= ci < len(view["candidates"]):
            raise ValueError("Invalid image candidate index")
        rows, seen = [], set()
        for ri, pi, _ in view["candidates"][ci]["row_matches"]:
            if (isinstance(ri, bool) or isinstance(pi, bool) or not isinstance(ri, (int, np.integer))
                    or not isinstance(pi, (int, np.integer)) or not 0 <= ri < len(raw["rows"]) or ri in seen):
                raise ValueError("Invalid or duplicate raw row")
            seen.add(ri)
            row = raw["rows"][ri]
            if not 0 <= pi < len(row["candidates"]):
                raise ValueError("Invalid raw edge pair")
            pair = row["candidates"][pi]
            rows.append([row["y"], pair["left_edge"]["x"], pair["right_edge"]["x"]])
        points = np.asarray(rows, float).reshape(-1, 3)
        if not np.isfinite(points).all() or np.any(points[:, 1] >= points[:, 2]):
            raise ValueError("Finite, ordered left/right edges required")
        result.append((vi, points))
    return result


def screen_cylinder(hypothesis, views, observations, config=None):
    """Fit one shared radius, then require both sides in every supporting view.

    Medians are balanced by view and side, so 400 rows cannot drown out a camera
    with 60 rows. This is a robust point estimate, not interval feasibility or a
    calibrated confidence bound. The one-pixel assumption needs RGB validation.
    """
    policy = policy_values(config)
    data = _matched_edges(hypothesis, views, observations)
    model = hypothesis["model"]
    medians, cached = [], []
    for vi, points in data:
        view = views[vi]
        camera = np.asarray(view["world_to_camera_cv"], float)[:3]
        rotation = camera[:, :3]
        origin = -rotation.T @ camera[:, 3]
        masks = []
        for side in (1, 2):
            rays = np.linalg.solve(np.asarray(view["K_index"]), np.c_[points[:, side], points[:, 0], np.ones(len(points))].T).T @ rotation
            distances, valid = axis_ray_distances(model["anchor"], model["direction"], origin, rays)
            masks.append(valid)
            medians.append(float(np.median(distances[valid])) if valid.any() else np.nan)
        cached.append((vi, points, masks))
    output = {"passed": False, "radius": None, "policy": policy, "views": [], "reasons": [],
              "scope": "conditional_circular_silhouette_not_target_identity", "radius_estimator": "median_of_per_view_per_side_medians"}
    if len(data) < policy["minimum_views"]:
        output["reasons"].append("too_few_supporting_views")
    if not medians or not np.isfinite(medians).all():
        output["reasons"].append("missing_forward_edge_rays")
        return output
    radius = float(np.median(medians))
    output["radius"] = radius
    for vi, points, masks in cached:
        try:
            predicted = silhouette_x(model, views[vi], radius, points[:, 0])
        except LineFitDegenerate as error:
            output["reasons"].append(str(error))
            return output
        sides = {}
        for side, name in enumerate(("left", "right")):
            residual = np.abs(predicted[side] - points[:, side + 1])
            valid = masks[side] & np.isfinite(residual)
            count = int(np.count_nonzero(valid & (residual <= policy["edge_tolerance_px"] + 1e-9)))
            fraction = count / len(points) if len(points) else 0.0
            sides[name] = {"count": len(points), "within_tolerance_count": count, "within_tolerance_fraction": fraction,
                           "invalid_ray_count": int(np.count_nonzero(~valid)),
                           "median_residual_px": float(np.median(residual)) if len(residual) else None,
                           "p90_residual_px": float(np.quantile(residual, .9)) if len(residual) else None}
        passed = all(item["within_tolerance_fraction"] >= policy["minimum_side_fraction"] for item in sides.values())
        output["views"].append({"view_id": views[vi]["view_id"], **sides, "passed": passed})
        if not passed:
            output["reasons"].append("edge_pixel_mismatch:" + views[vi]["view_id"])
    output["passed"] = not output["reasons"]
    return output


def refit_hypothesis(hypothesis, views, extent_config):
    """Use exactly the same fixed-camera line fit as the later finite adapter."""
    indices = hypothesis["supporting_views"]
    model = fit_multiview_line(
        [views[i]["candidates"][hypothesis["matches"][i]["candidate_index"]]["line"] for i in indices],
        [views[i]["K_index"] for i in indices], [views[i]["world_to_camera_cv"] for i in indices],
        min_second_to_first_ratio=extent_config["minimum_second_to_first_plane_ratio"],
        min_direction_gap_ratio=extent_config["minimum_direction_gap_ratio"],
    )
    return {**copy.deepcopy(hypothesis), "model": model}


def select_cylinder_hypothesis(association, views, observations, extent_config, config=None):
    """Screen only the stored best and competing alternatives; preserve ambiguity."""
    output = copy.deepcopy(association)
    hypotheses = ([association["selected"]] if association.get("selected") is not None else []) + association["alternatives"]
    survivors, audits, seen = [], [], {}
    for ordinal, hypothesis in enumerate(hypotheses):
        signature = tuple((i, hypothesis["matches"][i]["candidate_index"]) for i in sorted(hypothesis["supporting_views"]))
        if signature in seen:
            audits.append({"ordinal": ordinal, "duplicate_assignment_of": seen[signature]})
            continue
        seen[signature] = ordinal
        try:
            fitted = refit_hypothesis(hypothesis, views, extent_config)
            diagnostic = screen_cylinder(fitted, views, observations, config)
        except LineFitDegenerate as error:
            diagnostic = {"passed": False, "reasons": ["refit_degenerate:" + str(error)]}
        audits.append({"ordinal": ordinal, "diagnostic": diagnostic})
        if diagnostic["passed"]:
            survivors.append((ordinal, fitted))
    output["cylinder_screen"] = {"hypotheses": audits, "input_count": len(hypotheses), "distinct_assignment_count": len(seen),
                                  "surviving_ordinals": [i for i, _ in survivors], "survivor_count": len(survivors),
                                  "scope": "stored_best_and_support_competitors_only_not_all_unique_lines", "axis_policy": "refit_all_selected_supporting_views_before_screen"}
    output["selected"] = survivors[0][1] if survivors else None
    output["alternatives"] = [h for _, h in survivors[1:]]
    if not association["search_complete"]:
        output.update(state="ambiguous", reason="search_budget_exhausted")
    elif len(survivors) == 1:
        output.update(state="accepted", reason="one_stored_assignment_passes_cylinder_screen")
    elif survivors:
        output.update(state="ambiguous", reason="competing_cylinder_consistent_assignments")
    else:
        output.update(state="rejected", reason="no_stored_assignment_passes_cylinder_screen")
    output["scope"] = "conditional_cylinder_screen_of_retained_associations_no_identity_proof"
    return output


def recheck_finite_cylinder(finite, selected, views, observations, config=None):
    output = copy.deepcopy(finite)
    if output["state"] != "accepted":
        output["final_cylinder_screen"] = {"applied": False, "reason": "finite_geometry_not_accepted"}
        return output
    diagnostic = screen_cylinder({**selected, "model": output["line"]}, views, observations, config)
    output["final_cylinder_screen"] = {"applied": True, **diagnostic}
    if not diagnostic["passed"]:
        # 最后交出去的是重拟合后的轴；不能只验上游那根，再把变化后的结果蒙混过关。
        output.update(state="rejected", segments=np.empty((0, 2, 3)))
        output["rejection_reasons"].append("final_axis_fails_cylinder_screen")
    return output
