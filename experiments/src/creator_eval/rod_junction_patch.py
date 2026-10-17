"""Infer a terminal connector midline, never a last-visible-rod boundary.

This research arm assumes a rod terminates at the centerline of a transverse
member visible on both flanks. Occlusion alone does not imply termination.
Unresolved or inconsistent junctions leave the stable native boundary intact.
"""

import hashlib

import numpy as np

from . import rod_boundary_patch as previous
from .rod_evidence import DEFAULT_CONFIG
from .rod_fixture_finite import canonical_hash
from .rod_selected_boundary_patch import build_selected_context

POLICY = dict(previous.POLICY, method="baseline_junction", version="junction-patch-1",
    endpoint_prior="terminal_transverse_member_centerline_connection_not_visibility_boundary",
    search_half_span_fraction=.25, central_reference_fraction=[.25, .75],
    flank_width_multipliers=[-3., -2., -1., 1., 2., 3.], flank_clearance_px=2.,
    minimum_reference_rows=3, minimum_body_contrast=2./255.,
    minimum_connector_contrast=8./255., noise_mad_multiplier=4.,
    profile_step_px=.5, profile_level_fraction=.5, minimum_profile_samples=2,
    minimum_flanks_per_side=2, maximum_transverse_slope=1.,
    maximum_image_residual_px=DEFAULT_CONFIG["maximum_reprojection_error_px"],
    minimum_junction_views=2, minimum_native_fit_views=DEFAULT_CONFIG["minimum_fit_views"],
    view_consensus="all_detected_views_agree_no_best_subset",
    unresolved_policy="retain_stable_endpoint", endpoint_extension=False,
    outside_guard="three_rows_center_and_near_flanks_match_body_background_not_continuing_X",
    outside_guard_width_multipliers=[1., 2., 3.],
    body_width_use="flank_sampling_distance_only_no_support_filter")


def project(points, view):
    e, k = view["world_to_camera_cv"], view["K_index"]
    h = (np.asarray(points) @ e[:, :3].T+e[:, 3]) @ k.T
    if not np.isfinite(h).all() or np.any(h[..., 2] <= 1e-10):
        raise ValueError("Forward finite endpoint projections required")
    return h[..., :2]/h[..., 2, None]


def sample_image(image, xy):
    """Bilinear original-pixel samples; outside pixels remain unknown."""
    xy = np.asarray(xy, float)
    valid = (xy[..., 0] >= 0) & (xy[..., 0] < image.shape[1]-1)
    valid &= (xy[..., 1] >= 0) & (xy[..., 1] < image.shape[0]-1)
    values = np.full(xy.shape[:-1], np.nan)
    x, y = xy[valid].T
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x-ix, y-iy
    values[valid] = ((1-dy)*((1-dx)*image[iy, ix]+dx*image[iy, ix+1])
                    + dy*((1-dx)*image[iy+1, ix]+dx*image[iy+1, ix+1]))
    return values


def bounded_profile_center(samples, positions, background, sign, noise):
    signal = sign*(samples-background)
    if not np.isfinite(signal).all():
        return None
    peak = float(signal.max())
    if peak < max(POLICY["minimum_connector_contrast"], POLICY["noise_mad_multiplier"]*noise):
        return None
    mask = signal >= POLICY["profile_level_fraction"]*peak
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    stops = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    if len(starts) != 1:
        return None
    a, b = int(starts[0]), int(stops[0])
    if a == 0 or b == len(mask)-1 or b-a+1 < POLICY["minimum_profile_samples"]:
        return None
    level = POLICY["profile_level_fraction"]*peak
    left = positions[a-1]+(positions[a]-positions[a-1])*(level-signal[a-1])/(signal[a]-signal[a-1])
    right = positions[b]+(positions[b+1]-positions[b])*(signal[b]-level)/(signal[b]-signal[b+1])
    return dict(center_px=float((left+right)/2), bounds_px=[float(left), float(right)], peak=peak)


def detect_view_junctions(image, view, endpoints, reference_width):
    xy = project(endpoints, view)
    delta = xy[1]-xy[0]
    length = float(np.linalg.norm(delta))
    if length <= 4:
        return dict(state="unknown", reason="short_projected_axis", endpoints=[None, None])
    along = delta/length
    normal = np.array([-along[1], along[0]])
    offsets = np.array(POLICY["flank_width_multipliers"])*(reference_width+POLICY["flank_clearance_px"])
    body = np.linspace(.25*length, .75*length, 31)
    flank_values = sample_image(image, xy[0]+body[:, None, None]*along+offsets[None, :, None]*normal)
    center = sample_image(image, xy[0]+body[:, None]*along)
    if not np.isfinite(flank_values).all() or not np.isfinite(center).all():
        return dict(state="unknown", reason="body_out_of_image", endpoints=[None, None])
    background = float(np.median(flank_values))
    noise = float(np.median(np.abs(flank_values-background)))
    contrast = float(np.median(center)-background)
    if abs(contrast) < POLICY["minimum_body_contrast"]:
        return dict(state="unknown", reason="unresolved_body_polarity", endpoints=[None, None])
    sign = float(np.sign(contrast))
    detections = []
    for side, endpoint in enumerate((0., length)):
        radius = POLICY["search_half_span_fraction"]*length
        positions = np.arange(endpoint-radius, endpoint+radius+POLICY["profile_step_px"]*.5,
                              POLICY["profile_step_px"])
        samples = sample_image(image, xy[0]+positions[:, None, None]*along+offsets[None, :, None]*normal)
        profiles = []
        for offset, values in zip(offsets, samples.T):
            found = bounded_profile_center(values, positions, background, sign, noise)
            if found is not None:
                profiles.append(dict(offset_px=float(offset), **found))
        v = np.array([p["offset_px"] for p in profiles])
        minimum = POLICY["minimum_flanks_per_side"]
        if np.count_nonzero(v < 0) < minimum or np.count_nonzero(v > 0) < minimum:
            detections.append(None)
            continue
        centers = np.array([p["center_px"] for p in profiles])
        fit = np.linalg.lstsq(np.c_[np.ones(len(v)), v], centers, rcond=None)[0]
        residuals = centers-(fit[0]+fit[1]*v)
        if (abs(fit[1]) > POLICY["maximum_transverse_slope"]
                or np.max(np.abs(residuals)) > POLICY["maximum_image_residual_px"]):
            detections.append(None)
            continue
        # A bounded crossbar alone is an X/T ambiguity, not termination. Require
        # observed background beyond its outward edge as an additional cue.
        outside = min(p["bounds_px"][0] for p in profiles) if side == 0 else max(p["bounds_px"][1] for p in profiles)
        guard_u = outside+(-1 if side == 0 else 1)*np.array(POLICY["outside_guard_width_multipliers"])*max(2., reference_width)
        guard_v = np.array([-1., 0., 1.])*(reference_width+POLICY["flank_clearance_px"])
        guard = sample_image(image, xy[0]+guard_u[:, None, None]*along+guard_v[None, :, None]*normal)
        tolerance = max(POLICY["minimum_connector_contrast"], POLICY["noise_mad_multiplier"]*noise)
        if not np.isfinite(guard).all() or np.max(np.abs(guard-background)) > tolerance:
            detections.append(None)
            continue
        detections.append(dict(center_px=float(fit[0]), pixel_xy=xy[0]+fit[0]*along,
            transverse_slope=float(fit[1]), flank_residuals_px=residuals, flanks=profiles,
            outside_guard_u_px=guard_u, outside_guard_values=guard,
            outside_background_tolerance=tolerance))
    return dict(state="measured", reason=None, background=background, background_mad=noise,
                body_contrast=contrast, reference_width_px=reference_width, endpoints=detections)


def build_junction_context(frames, cameras, baseline, images):
    selected = build_selected_context(frames, cameras, baseline)
    if len(images) != len(frames):
        raise ValueError("One original RGB array per native camera required")
    if len(baseline["proposal"]["selected"]["supporting_views"]) < POLICY["minimum_native_fit_views"]:
        raise ValueError("Original four-view native axis required")
    segments = np.asarray(baseline["segments"], float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all() or not len(segments):
        raise ValueError("Nonempty finite native segments required")
    anchor = segments[0, 0]
    direction = segments[0, 1]-anchor
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("Nondegenerate native axis required")
    direction = direction/length
    if np.max(np.linalg.norm(np.cross(segments-anchor, direction), axis=2)) > 1e-9:
        raise ValueError("One native axis required")
    bounds = (segments-anchor) @ direction
    endpoints = anchor+np.array([bounds.min(), bounds.max()])[:, None]*direction
    diagnostics, measurements = [], [[], []]
    for image, view, members in zip(images, selected["views"], selected["row_members"]):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape != (view["size_wh"][1], view["size_wh"][0], 3):
            raise ValueError("Unmodified uint8 RGB image with original dimensions required")
        gray = np.mean(image.astype(float), axis=2)/255.
        ys = project(endpoints, view)[:, 1]
        low, high = np.sort(ys)
        rows, slots = np.nonzero(members[..., 0])
        middle = (rows >= low+.25*(high-low)) & (rows <= low+.75*(high-low))
        if np.count_nonzero(middle) < POLICY["minimum_reference_rows"]:
            diagnostics.append(dict(view_id=view["view_id"], state="unknown",
                                    reason="no_measured_body_reference", endpoints=[None, None]))
            continue
        widths = view["right"][rows[middle], slots[middle]]-view["left"][rows[middle], slots[middle]]
        diagnostic = dict(view_id=view["view_id"], **detect_view_junctions(gray, view, endpoints, float(np.median(widths))))
        diagnostics.append(diagnostic)
        projection = view["K_index"] @ view["world_to_camera_cv"]
        h = projection @ np.r_[anchor, 1.]
        d = projection[:, :3] @ direction
        for side, detection in enumerate(diagnostic["endpoints"]):
            if detection is None:
                continue
            pixel = detection["pixel_xy"]
            a = d[:2]-pixel*d[2]
            denominator = float(a @ a)
            if denominator <= 1e-12:
                continue
            parameter = float(a @ (pixel*h[2]-h[:2])/denominator)
            measurements[side].append(dict(view_id=view["view_id"], parameter_m=parameter,
                                           pixel_xy=pixel))
    identity = dict(policy=POLICY, selected_context_sha256=selected["sha256"],
        image_array_sha256=[hashlib.sha256(np.asarray(image).tobytes()).hexdigest() for image in images],
        diagnostics=diagnostics, measurements=measurements)
    return dict(views=selected["views"], anchor=anchor, direction=direction,
                diagnostics=diagnostics, measurements=measurements, sha256=canonical_hash(identity))


def refine_boundary(segments, shared_context, junction_context):
    parent = previous.refine_boundary(segments, shared_context)
    stable = np.asarray(parent["segments"], float).reshape(-1, 2, 3)
    result = dict(parent, method=POLICY["method"], policy=dict(POLICY), method_sha256=canonical_hash(POLICY),
        previous_boundary_segments_sha256=canonical_hash(stable),
        previous_boundary_removed_length_m=parent["boundary_removed_length_m"],
        junction_context_sha256=junction_context["sha256"], junction_diagnostics=junction_context["diagnostics"],
        junction_removed_length_m=0., junction_decisions=[], physical_endpoint_observed=False,
        structural_connection_prior_used=True)
    if not len(stable):
        return result
    anchor, direction = junction_context["anchor"], junction_context["direction"]
    coordinates = (stable-anchor) @ direction
    lower, upper = float(coordinates.min()), float(coordinates.max())
    limits = [lower, upper]
    for side, measurements in enumerate(junction_context["measurements"]):
        decision = dict(side=side, state="unchanged", reason="insufficient_junction_views", measurements=measurements)
        names = [m["view_id"] for m in measurements]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate junction cameras cannot create consensus")
        if len(measurements) >= POLICY["minimum_junction_views"]:
            parameter = float(np.median([m["parameter_m"] for m in measurements]))
            point = anchor+parameter*direction
            residuals = []
            for measurement in measurements:
                view = next(v for v in junction_context["views"] if v["view_id"] == measurement["view_id"])
                residuals.append(float(np.linalg.norm(project(point, view)-measurement["pixel_xy"])))
            decision.update(parameter_m=parameter, reprojection_errors_px=residuals)
            if max(residuals) > POLICY["maximum_image_residual_px"]:
                decision["reason"] = "inconsistent_junction_views"
            elif not lower < parameter < upper:
                decision["reason"] = "junction_outside_stable_extent_no_extension"
            else:
                limits[side] = parameter
                decision.update(state="applied", reason="consistent_terminal_connector_center")
        result["junction_decisions"].append(decision)
    if limits[0] >= limits[1]:
        for decision in result["junction_decisions"]:
            decision.update(state="unchanged", reason="conflicting_terminal_connectors")
        return result
    kept = []
    for segment, parameters in zip(stable, coordinates):
        ordered = np.sort(parameters)
        left, right = max(ordered[0], limits[0]), min(ordered[1], limits[1])
        if right-left <= 1e-10:
            continue
        pair = segment.copy() if parameters[0] <= parameters[1] else segment[::-1].copy()
        if left > ordered[0]:
            pair[0] = anchor+left*direction
        if right < ordered[1]:
            pair[1] = anchor+right*direction
        kept.append(pair if parameters[0] <= parameters[1] else pair[::-1])
    output = np.asarray(kept, float).reshape(-1, 2, 3)
    removed = float(np.linalg.norm(stable[:, 1]-stable[:, 0], axis=1).sum()
                    - np.linalg.norm(output[:, 1]-output[:, 0], axis=1).sum())
    result.update(segments=output, junction_removed_length_m=removed,
                  boundary_removed_length_m=parent["boundary_removed_length_m"]+removed,
                  state="accepted" if len(output) else "rejected",
                  rejection_reasons=[] if len(output) else ["empty_terminal_connector_intersection"])
    return result
