"""Conservative rod proposals from fixed cameras and explicit per-view RGB evidence.

This module never reads GT, refits cameras, or calls a depth model. ``absent`` is
an upstream background-like hypothesis under a declared visibility assumption;
RGB silence alone does not establish that an unoccluded rod is absent.
"""

from __future__ import annotations

import numpy as np

from .line_controls import (
    LineFitDegenerate,
    closest_ray_line_parameters,
    fit_multiview_line,
)
from .native_diagnostics import camera_centers, homogeneous

DEFAULT_CONFIG = {
    "minimum_fit_views": 4,
    "minimum_positive_views": 3,
    "minimum_absent_views": 2,
    "row_tolerance_px": 1.0,
    "match_base_px": 2.0,
    "maximum_half_width_px": 4.0,
    "step_camera_span_fraction": 0.001,
    "minimum_run_samples": 3,
    "maximum_reprojection_error_px": 2.0,
    "maximum_leave_one_out_angle_degrees": 1.0,
    "maximum_leave_one_out_offset_camera_span_fraction": 0.002,
    "minimum_second_to_first_plane_ratio": 1e-5,
    "minimum_direction_gap_ratio": 1e-8,
    "maximum_sample_count": 100000,
}
STATE_CODES = {"unknown": 0, "positive": 1, "absent": -1, "ambiguous": 2, "occluded": 3, "out_of_frame": 4}
_ALLOWED_OBSERVATIONS = {"accepted", "absent", "ambiguous", "unknown", "occluded"}


def _config(value):
    result = {**DEFAULT_CONFIG, **(value or {})}
    if set(result) != set(DEFAULT_CONFIG):
        raise ValueError("Unknown rod-evidence configuration key")
    integer_minima = {"minimum_fit_views": 4, "minimum_positive_views": 1,
                      "minimum_absent_views": 1, "minimum_run_samples": 2,
                      "maximum_sample_count": 2}
    for key, minimum in integer_minima.items():
        if not isinstance(result[key], (int, np.integer)) or result[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in set(result) - set(integer_minima):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if result["minimum_second_to_first_plane_ratio"] >= 1 or result["minimum_direction_gap_ratio"] >= 1:
        raise ValueError("Plane degeneracy ratios must be less than one")
    return result


def _views(values):
    result = []
    for value in values:
        intrinsic = np.asarray(value["K_index"], dtype=float).copy()
        ext = homogeneous(value["world_to_camera_cv"]).copy()
        size = np.asarray(value["size_wh"], dtype=float)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or abs(np.linalg.det(intrinsic)) <= 1e-12:
            raise ValueError("K_index must be a finite invertible 3x3 matrix")
        if not np.allclose(intrinsic[2], [0, 0, 1]) or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
            raise ValueError("K_index must be a forward pinhole calibration")
        if not np.isfinite(ext).all() or not np.allclose(ext[3], [0, 0, 0, 1]):
            raise ValueError("Expected finite affine OpenCV world-to-camera extrinsics")
        rotation = ext[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Extrinsics must contain a proper camera rotation")
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size != np.floor(size)):
            raise ValueError("size_wh must contain two positive integer pixel dimensions")
        line = value.get("image_line")
        if line is not None:
            line = np.asarray(line, dtype=float).copy()
            if line.shape != (3,) or not np.isfinite(line).all() or np.linalg.norm(line[:2]) <= 1e-12:
                raise ValueError("image_line must be a finite homogeneous 2D line or None")
            line /= np.linalg.norm(line[:2])
        observations = []
        for observation in value.get("observations", []):
            status = observation["state"]
            xy = np.asarray(observation["xy"], dtype=float).copy()
            width = float(observation.get("width_px", 0.0))
            if status not in _ALLOWED_OBSERVATIONS or xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError("Invalid observation state or xy")
            if not np.isfinite(width) or width < 0:
                raise ValueError("width_px must be finite and nonnegative")
            window = observation.get("negative_window_x")
            if window is not None:
                window = np.asarray(window, dtype=float).copy()
                if window.shape != (2,) or not np.isfinite(window).all() or window[0] > window[1]:
                    raise ValueError("negative_window_x must be a finite ordered [low, high]")
            observations.append({"xy": xy, "state": status, "width_px": width, "negative_window_x": window})
        observations.sort(key=lambda item: item["xy"][1])
        result.append({"view_id": str(value["view_id"]), "K_index": intrinsic,
                       "world_to_camera_cv": ext, "size_wh": size, "image_line": line,
                       "observations": observations})
    return result


def _project(points, view):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    ext = view["world_to_camera_cv"]
    camera = points @ ext[:3, :3].T + ext[:3, 3]
    homogeneous_pixels = camera @ view["K_index"].T
    valid = np.isfinite(homogeneous_pixels).all(axis=1) & (camera[:, 2] > 1e-10)
    pixels = np.full((len(points), 2), np.nan)
    pixels[valid] = homogeneous_pixels[valid, :2] / homogeneous_pixels[valid, 2, None]
    return pixels, valid


def _projected_line(anchor, direction, view):
    projection = view["K_index"] @ view["world_to_camera_cv"][:3]
    first = projection @ np.r_[anchor, 1.0]
    delta = projection @ np.r_[direction, 0.0]
    line = np.cross(first, delta)
    normal = np.linalg.norm(line[:2])
    if not np.isfinite(line).all() or normal <= 1e-12:
        raise LineFitDegenerate("line_projects_to_a_point")
    return line / normal


def _fit(views, config):
    return fit_multiview_line(
        [view["image_line"] for view in views],
        [view["K_index"] for view in views],
        [view["world_to_camera_cv"] for view in views],
        min_second_to_first_ratio=config["minimum_second_to_first_plane_ratio"],
        min_direction_gap_ratio=config["minimum_direction_gap_ratio"],
    )


def _match_tolerance(observation, config):
    return config["match_base_px"] + min(observation["width_px"] / 2, config["maximum_half_width_px"])


def _accepted_extent(fitted, views, config):
    anchor, direction = fitted["anchor"], fitted["direction"]
    extents, diagnostics = [], []
    for view in views:
        accepted = [item for item in view["observations"] if item["state"] == "accepted"]
        diagnostic = {"view_id": view["view_id"], "raw_accepted_count": len(accepted), "eligible_unique_count": 0}
        if not accepted:
            diagnostics.append(diagnostic)
            continue
        xy = np.array([item["xy"] for item in accepted])
        pixels = np.c_[xy, np.ones(len(xy))]
        try:
            line = _projected_line(anchor, direction, view)
        except LineFitDegenerate as error:
            diagnostic["reason"] = str(error)
            diagnostics.append(diagnostic)
            continue
        residual = np.abs(pixels @ line)
        tolerance = np.array([_match_tolerance(item, config) for item in accepted])
        inverse = np.linalg.inv(view["world_to_camera_cv"])
        rays = (pixels @ np.linalg.inv(view["K_index"]).T) @ inverse[:3, :3].T
        origins = np.broadcast_to(inverse[:3, 3], rays.shape)
        closest = closest_ray_line_parameters(anchor, direction, origins, rays)
        inside = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] <= view["size_wh"][0] - 1) & (xy[:, 1] <= view["size_wh"][1] - 1)
        good = closest["valid"] & inside & (residual <= tolerance)
        parameters = np.unique(closest["line_t"][good])
        diagnostic.update(eligible_unique_count=len(parameters),
                          accepted_reprojection_median_px=float(np.median(residual)),
                          accepted_reprojection_max_px=float(np.max(residual)))
        if len(parameters) >= 2 and parameters[-1] > parameters[0]:
            extent = [float(parameters[0]), float(parameters[-1])]
            extents.append(extent)
            diagnostic["eligible_parameter_extent"] = extent
        diagnostics.append(diagnostic)
    if len(extents) < config["minimum_positive_views"]:
        return None, diagnostics
    # The search envelope requires endpoint ranges in at least the minimum number
    # of views. One stray far-away pixel cannot make a huge candidate by itself.
    count = config["minimum_positive_views"]
    lower = float(np.sort(np.array(extents)[:, 0])[count - 1])
    upper = float(np.sort(np.array(extents)[:, 1])[-count])
    return (np.array([lower, upper]) if upper > lower else None), diagnostics


def _geometry_diagnostics(fitted, views, extent, span, config):
    anchor, direction = fitted["anchor"], fitted["direction"]
    probes = anchor + np.array([extent[0], np.mean(extent), extent[1]])[:, None] * direction
    per_view, leave_out, failures = [], [], []
    for index, view in enumerate(views):
        pixels, valid = _project(probes, view)
        error = np.abs(np.c_[pixels, np.ones(len(pixels))] @ view["image_line"])
        maximum = float(error.max()) if valid.all() else None
        per_view.append({"view_id": view["view_id"], "probe_reprojection_errors_px": error,
                         "maximum_reprojection_error_px": maximum, "all_probes_in_front": bool(valid.all())})
        if maximum is None or maximum > config["maximum_reprojection_error_px"]:
            failures.append("view_reprojection:" + view["view_id"])
        try:
            fit = _fit([other for j, other in enumerate(views) if j != index], config)
            angle = float(np.degrees(np.arccos(np.clip(abs(direction @ fit["direction"]), 0, 1))))
            offsets = probes - fit["anchor"]
            distances = np.linalg.norm(offsets - (offsets @ fit["direction"])[:, None] * fit["direction"], axis=1)
            offset_ratio = float(distances.max() / span)
            held_out_line = _projected_line(fit["anchor"], fit["direction"], view)
            accepted = [item for item in view["observations"] if item["state"] == "accepted"]
            held_out_residual = [abs(np.r_[item["xy"], 1] @ held_out_line) for item in accepted]
            # Compare the held-out candidate to its supplied 2D line at the same
            # image rows; raw detector outliers are reported separately, not hidden.
            fitted_probe_line = _projected_line(anchor, direction, view)
            line_denominator = view["image_line"][0]
            if abs(line_denominator) > 1e-10 and valid.all():
                y = pixels[:, 1]
                x = -(view["image_line"][1] * y + view["image_line"][2]) / line_denominator
                held_out_error = float(np.max(np.abs(np.c_[x, y, np.ones(3)] @ held_out_line)))
            else:
                held_out_error = float(np.max(np.abs(np.c_[pixels, np.ones(3)] @ held_out_line))) if valid.all() else None
            leave_out.append({"omitted_view_id": view["view_id"], "state": "fitted",
                              "direction_angle_degrees": angle, "maximum_offset_camera_span_fraction": offset_ratio,
                              "held_out_line_reprojection_max_px": held_out_error,
                              "held_out_raw_accepted_median_px": float(np.median(held_out_residual)) if held_out_residual else None,
                              "full_projected_line": fitted_probe_line})
            if angle > config["maximum_leave_one_out_angle_degrees"]:
                failures.append("leave_one_out_angle:" + view["view_id"])
            if offset_ratio > config["maximum_leave_one_out_offset_camera_span_fraction"]:
                failures.append("leave_one_out_offset:" + view["view_id"])
            if held_out_error is None or held_out_error > config["maximum_reprojection_error_px"]:
                failures.append("leave_one_out_reprojection:" + view["view_id"])
        except LineFitDegenerate as error:
            leave_out.append({"omitted_view_id": view["view_id"], "state": "degenerate", "reason": str(error)})
            failures.append("leave_one_out_degenerate:" + view["view_id"])
    return {"per_view": per_view, "leave_one_out": leave_out, "failures": failures, "passed": not failures}


def _observation_applies(observation, x, config, *, negative=False):
    if negative and observation["negative_window_x"] is not None:
        low, high = observation["negative_window_x"]
        return bool(low <= x <= high)
    return abs(x - observation["xy"][0]) <= _match_tolerance(observation, config)


def _view_evidence(points, view, config):
    pixels, in_front = _project(points, view)
    width, height = view["size_wh"]
    in_frame = in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1)
    states = np.full(len(points), STATE_CODES["unknown"], dtype=np.int8)
    states[~in_frame] = STATE_CODES["out_of_frame"]
    observations = view["observations"]
    rows = np.array([item["xy"][1] for item in observations])
    for index in np.flatnonzero(in_frame):
        x, y = pixels[index]
        begin = np.searchsorted(rows, y - config["row_tolerance_px"], side="left")
        end = np.searchsorted(rows, y + config["row_tolerance_px"], side="right")
        if begin == end:
            continue
        # We inspect the closest measured row(s), never interpolate an unobserved
        # negative or dilate support. Duplicate records still make only one vote.
        differences = np.abs(rows[begin:end] - y)
        nearest = differences.min()
        candidates = [observations[begin + j] for j in np.flatnonzero(np.isclose(differences, nearest, atol=1e-9, rtol=0))]
        positive = any(item["state"] == "accepted" and _observation_applies(item, x, config) for item in candidates)
        negative = any(item["state"] == "absent" and _observation_applies(item, x, config, negative=True) for item in candidates)
        occluded = any(item["state"] == "occluded" and _observation_applies(item, x, config, negative=True) for item in candidates)
        ambiguous = any(item["state"] == "ambiguous" and _observation_applies(item, x, config, negative=True) for item in candidates)
        if occluded:
            states[index] = STATE_CODES["occluded"]
        elif ambiguous or (positive and negative):
            states[index] = STATE_CODES["ambiguous"]
        elif positive:
            states[index] = STATE_CODES["positive"]
        elif negative:
            states[index] = STATE_CODES["absent"]
    return states, pixels


def _finite_segments(parameters, kept, anchor, direction, minimum_run_samples):
    accepted = np.flatnonzero(kept)
    if not len(accepted):
        return np.empty((0, 2, 3)), np.empty((0, 2))
    runs = np.split(accepted, np.flatnonzero(np.diff(accepted) > 1) + 1)
    intervals = np.array([[parameters[run[0]], parameters[run[-1]]] for run in runs if len(run) >= minimum_run_samples]).reshape(-1, 2)
    return anchor + intervals[..., None] * direction, intervals


def build_rod_candidate(views, config=None, *, geometric_gate=True, negative_veto=True):
    """Fit an infinite candidate, audit fixed cameras, then conservatively bound it.

    Each view supplies ``view_id``, ``K_index``, ``world_to_camera_cv``, ``size_wh``,
    ``image_line`` (homogeneous abc or None), and ``observations``. An observation
    is ``{xy, width_px, state}``; states are accepted, absent, ambiguous, unknown,
    occluded. ``absent`` may include ``negative_window_x=[low,high]``: only a
    projected point inside that measured strip can receive its negative vote.

    Geometric ablation skips only global/leave-one-view-out rejection. Positive
    pixel-distance matching remains active. Negative ablation ignores the veto,
    never converts negatives to positives. Failed geometry retains shadow_segments
    for diagnosis, while segments are the actual publishable proposal.
    """
    policy = _config(config)
    values = _views(views)
    output = {"state": "rejected", "segments": np.empty((0, 2, 3)),
              "shadow_segments": np.empty((0, 2, 3)), "rejection_reasons": [],
              "config": policy, "ablations": {"geometric_gate": bool(geometric_gate), "negative_veto": bool(negative_veto), "positive_pixel_distance_gate": True},
              "camera_policy": "fixed input cameras; no refinement or GT",
              "negative_semantics": "upstream background-like hypothesis under an explicit unoccluded-visibility assumption; missing responses remain unknown",
              "endpoint_policy": "first/last supported grid samples; no dilation, interpolation, gap filling, or singleton output",
              "resolution_limit": "Unsampled gaps narrower than the 3D grid or measured row spacing can remain undetected; this is not continuous visibility proof."}
    if len({view["view_id"] for view in values}) != len(values):
        output["rejection_reasons"] = ["duplicate_view_id"]
        return output
    fit_views = [view for view in values if view["image_line"] is not None]
    if len(fit_views) < policy["minimum_fit_views"]:
        output["rejection_reasons"] = ["too_few_fit_views_for_leave_one_out"]
        return output
    centers = camera_centers([view["world_to_camera_cv"] for view in values])
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    span = float(distances.max())
    output["camera_span"] = span
    if span <= 1e-12:
        output["rejection_reasons"] = ["zero_camera_span"]
        return output
    if np.any(distances[np.triu_indices(len(values), 1)] <= max(1.0, span) * 1e-9):
        output["rejection_reasons"] = ["duplicate_camera_center_not_independent_evidence"]
        return output
    try:
        fitted = _fit(fit_views, policy)
    except LineFitDegenerate as error:
        output["rejection_reasons"] = ["candidate_fit_degenerate:" + str(error)]
        return output
    output["line"] = fitted
    extent, extent_diagnostics = _accepted_extent(fitted, values, policy)
    output["extent_diagnostics"] = extent_diagnostics
    if extent is None:
        output["rejection_reasons"] = ["insufficient_distinct_view_accepted_extent"]
        return output
    geometry = _geometry_diagnostics(fitted, fit_views, extent, span, policy)
    output["geometry"] = geometry
    step = span * policy["step_camera_span_fraction"]
    first_index, last_index = int(np.ceil(extent[0] / step)), int(np.floor(extent[1] / step))
    count = last_index - first_index + 1
    if count < policy["minimum_run_samples"] or count > policy["maximum_sample_count"]:
        output["rejection_reasons"] = ["candidate_grid_extent_too_small_or_large"]
        return output
    parameters = np.arange(first_index, last_index + 1, dtype=float) * step
    points = fitted["anchor"] + parameters[:, None] * fitted["direction"]
    evidence = [_view_evidence(points, view, policy) for view in values]
    states = np.stack([item[0] for item in evidence])
    positive = np.sum(states == STATE_CODES["positive"], axis=0)
    absent = np.sum(states == STATE_CODES["absent"], axis=0)
    enough_positive = positive >= policy["minimum_positive_views"]
    vetoed = absent >= policy["minimum_absent_views"]
    kept = enough_positive & (~vetoed if negative_veto else True)
    segments, intervals = _finite_segments(parameters, kept, fitted["anchor"], fitted["direction"], policy["minimum_run_samples"])
    output.update(shadow_segments=segments, shadow_parameter_intervals=intervals,
                  evidence={"view_ids": [view["view_id"] for view in values], "state_codes": STATE_CODES.copy(),
                            "view_states": states, "projected_xy": np.stack([item[1] for item in evidence]),
                            "parameters": parameters, "step": step, "positive_view_counts": positive,
                            "absent_view_counts": absent, "enough_positive": enough_positive,
                            "negative_veto_condition": vetoed, "kept_samples": kept,
                            "supported_sample_count": int(kept.sum()),
                            "vetoed_despite_positive_sample_count": int(np.sum(enough_positive & vetoed))})
    if geometric_gate and not geometry["passed"]:
        output["rejection_reasons"].extend(geometry["failures"])
    if not len(segments):
        output["rejection_reasons"].append("no_consecutive_supported_segment")
    if not output["rejection_reasons"]:
        output.update(state="accepted", segments=segments.copy())
    return output
