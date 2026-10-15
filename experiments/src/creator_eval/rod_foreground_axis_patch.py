"""Refit a native patch axis from measured foreground bands, not reader outputs.

Two or more marked camera planes determine an infinite line. Measured band
midpoints are an explicit centerline assumption; uncertain click coordinates
only identify the bands. Original finite endpoints and gaps are projected onto
that line, then intersected with the fixed four-view positive outer envelope.
"""

import numpy as np

from .consensus_boundary_readout import trim_consensus_envelope
from .line_controls import LineFitDegenerate
from .rod_evidence import DEFAULT_CONFIG
from .rod_fixture_finite import canonical_hash
from .stroke_selected_axis_readout import foreground_bands, projected_band_score

POLICY = dict(method="baseline_foreground_axis", version="foreground-axis-patch-1",
    axis_policy="measured_foreground_band_center_image_planes_fixed_cameras",
    band_center_assumption="measured_edge_midpoint_is_centerline_not_click_center",
    image_fit="unique_measured_bands_centered_y_least_squares_x_equal_weight",
    plane_fit="unit_normal_equal_weight_svd_two_or_more_distinct_marked_views",
    minimum_fit_views=2,
    minimum_second_to_first_plane_ratio=DEFAULT_CONFIG["minimum_second_to_first_plane_ratio"],
    minimum_direction_gap_ratio=DEFAULT_CONFIG["minimum_direction_gap_ratio"],
    minimum_boundary_views=4, horizontal_boundary_padding_px=0.,
    extent_policy="project_original_native_endpoints_then_outer_intersection",
    interior_policy="preserve_projected_original_components_and_gaps",
    common_reader_changed=False, base_point_suppression=False)


def fit_marked_axis(context):
    bands = foreground_bands(context)
    unique = {(b["view_id"], b["row"], b["left"], b["right"]): b for b in bands}
    bands = [unique[key] for key in sorted(unique)]
    planes, image_fits = [], []
    for view_id in sorted({b["view_id"] for b in bands}):
        measured = [b for b in bands if b["view_id"] == view_id]
        y = np.array([b["row"] for b in measured], float)
        x = np.array([(b["left"]+b["right"])/2 for b in measured])
        if len(np.unique(y)) < 2:
            raise LineFitDegenerate("marked_view_needs_distinct_measured_rows")
        centered = y-y.mean()
        slope = float(centered @ (x-x.mean()) / (centered @ centered))
        intercept = float(x.mean()-slope*y.mean())
        line = np.array([1., -slope, -intercept])
        residual = x-(slope*y+intercept)
        plane = line @ measured[0]["projection"]
        norm = np.linalg.norm(plane[:3])
        if not np.isfinite(plane).all() or norm <= 1e-12:
            raise LineFitDegenerate("invalid_marked_backprojection_plane")
        planes.append(plane/norm)
        image_fits.append(dict(view_id=view_id, measured_rows=y, image_line=line,
            center_x=x, residual_x_px=residual, center_rmse_px=float(np.sqrt(np.mean(residual**2)))))
    planes = np.asarray(planes)
    if len(planes) < POLICY["minimum_fit_views"]:
        raise LineFitDegenerate("too_few_distinct_marked_views")
    normals = planes[:, :3]
    # full_matrices retains the null direction when there are exactly two real
    # views. No duplicated view is inserted to satisfy the old 3-view fitter.
    _, singular, basis = np.linalg.svd(normals, full_matrices=True)
    ratio = float(singular[1]/singular[0])
    if ratio < POLICY["minimum_second_to_first_plane_ratio"]:
        raise LineFitDegenerate("coincident_or_nearly_parallel_marked_planes")
    third = singular[2] if len(singular) > 2 else 0.
    gap = float((singular[1]-third)/singular[1])
    if gap <= POLICY["minimum_direction_gap_ratio"]:
        raise LineFitDegenerate("ambiguous_marked_axis_direction")
    direction = basis[-1].copy()
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    perpendicular = basis[:2].T
    coordinates = np.linalg.lstsq(normals @ perpendicular, -planes[:, 3], rcond=None)[0]
    anchor = perpendicular @ coordinates
    model = dict(anchor=anchor, centroid=anchor, direction=direction,
        fit_view_count=len(planes), singular_values=singular,
        second_to_first_plane_ratio=ratio, direction_gap_ratio=gap,
        plane_residuals=normals @ anchor+planes[:, 3], image_fits=image_fits)
    return model, bands


def refine_boundary(segments, context):
    segments = np.asarray(segments, float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Finite Mx2x3 native baseline segments required")
    if context["association"]["fallback_to_raw_union"]:
        raise ValueError("Complete original foreground chain context required")
    result = dict(method=POLICY["method"], policy=dict(POLICY),
        method_sha256=canonical_hash(POLICY), parent_native_sha256=canonical_hash(segments),
        context_sha256=context["sha256"], gt_read=False, extra_annotations=0,
        native_axis_refitted=False, image_constructed_geometry=True,
        reader_selected_model_used=False, base_points_used_for_fit=False,
        original_baseline_preserved_as_control=True, common_reader_changed=False,
        state="rejected", segments=np.empty((0, 2, 3)), rejection_reasons=[],
        boundary_removed_length_m=0.)
    if not len(segments):
        return dict(result, rejection_reasons=["empty_native_baseline"])
    lengths = np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Nondegenerate original native components required")
    original_direction = (segments[0, 1]-segments[0, 0])/lengths[0]
    tolerance = 64*np.finfo(float).eps*max(1., float(np.abs(segments).max()))
    if np.linalg.norm(np.cross(segments-segments[0, 0], original_direction), axis=2).max() > tolerance:
        raise ValueError("One original native axis required")
    try:
        model, bands = fit_marked_axis(context)
    except (LineFitDegenerate, ValueError) as exc:
        return dict(result, rejection_reasons=[str(exc)])
    anchor, direction = model["anchor"], model["direction"]
    parameters = (segments-anchor) @ direction
    projected = anchor+parameters[..., None]*direction
    if np.any(np.linalg.norm(projected[:, 1]-projected[:, 0], axis=1) <= 1e-10):
        return dict(result, rejection_reasons=["native_extent_collapsed_on_marked_axis"])
    clipped = trim_consensus_envelope(projected, context)
    result.update(clipped, model=model, projected_native_segments=projected,
        native_axis_refitted=True,
        measured_bands=[{k: v for k, v in b.items() if k != "projection"} for b in bands],
        original_band_score=projected_band_score(dict(centroid=segments[0, 0], direction=original_direction), bands),
        refined_band_score=projected_band_score(model, bands),
        axis_change_degrees=float(np.degrees(np.arccos(np.clip(abs(original_direction @ direction), 0., 1.)))),
        maximum_endpoint_displacement_m=float(np.linalg.norm(projected-segments, axis=2).max()),
        projected_native_length_m=float(np.linalg.norm(projected[:, 1]-projected[:, 0], axis=1).sum()))
    result.update(state="accepted" if len(result["segments"]) else "rejected",
        rejection_reasons=[] if len(result["segments"]) else ["no_four_view_foreground_axis_boundary_support"])
    return result
