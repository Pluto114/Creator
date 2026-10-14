"""Bound an unchanged actual axis using unpadded, foreground-view evidence.

Input admission and interior occupancy remain the bracketed reader's policy.
Only the overall outer limits are intersected with the positive envelope of
the same selected chain, requiring every explicitly marked view. Missing RGB
is not negative evidence and no geometry is extended or fitted from images.
"""

import numpy as np

from . import bracketed_occupancy_readout as previous
from . import rgb_segment_cells as cells
from .chain_partition_readout import support_groups
from .rgb_candidate_readout import POLICY
from .rgb_required_view_support import required_stroke_support_mask


def measured_boundary_context(context):
    """Cancel the admission pad locally; never mutate the frozen measurements."""
    pad = POLICY["horizontal_padding_px"]
    views = [dict(view, left=view["left"]+pad, right=view["right"]-pad)
             for view in context["views"]]
    return dict(context, views=views)


def trim_outer_envelope(segments, context, *, support_mask=required_stroke_support_mask):
    """Trim only outer tails; preserve every existing interior gap and endpoint."""
    segments = np.asarray(segments, float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Finite Mx2x3 segments required")
    output = dict(segments=segments.copy(), boundary_removed_length_m=0.,
                  boundary_positive_cells=0, boundary_cell_count=0)
    if not len(segments):
        return output
    lengths = np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Nondegenerate collinear segments required")
    anchor = segments[0, 0]
    direction = (segments[0, 1]-anchor)/lengths[0]
    tolerance = 64*np.finfo(float).eps*max(1., float(np.abs(segments).max()))
    if np.linalg.norm(np.cross(segments-anchor, direction), axis=2).max() > tolerance:
        raise ValueError("One unchanged actual axis required")
    projected = (segments-anchor) @ direction
    low, high = float(projected.min()), float(projected.max())
    span = anchor+np.array([low, high])[:, None]*direction
    strict = measured_boundary_context(context)
    events = np.unique(np.r_[cells.segment_events(span, strict),
                             ((projected-low)/(high-low)).ravel()])
    count = len(events)-1
    checks = (2*count+1)*len(context["association"]["assignments"])
    if count > cells.DEFAULTS["maximum_cells"] or checks > cells.DEFAULTS["maximum_membership_checks"]:
        raise OverflowError("boundary_cell_budget_exceeded")
    delta = span[1]-span[0]
    middles = (events[:-1]+events[1:])/2
    positive = support_mask(span[0]+middles[:, None]*delta, strict)
    # Evidence inside an existing geometric gap must not qualify an outer tail.
    parameters = low+middles*(high-low)
    bounds = np.sort(projected, axis=1)
    inside = np.zeros(len(parameters), bool)
    for left, right in bounds:
        inside |= (parameters >= left) & (parameters <= right)
    indices = np.flatnonzero(positive & inside)
    output.update(boundary_cell_count=count, boundary_positive_cells=len(indices))
    if not len(indices):
        return dict(output, segments=np.empty((0, 2, 3)),
                    boundary_removed_length_m=float(lengths.sum()))
    first, last = indices[0], indices[-1]
    limits = [float(events[first]), float(events[last+1])]
    for side, target in ((0, middles[first]), (1, middles[last])):
        value = limits[side]
        if not support_mask((span[0]+value*delta)[None], strict)[0]:
            for _ in range(8):
                value = float(np.nextafter(value, target))
            if not support_mask((span[0]+value*delta)[None], strict)[0]:
                value = limits[side]+(target-limits[side])*1e-9
            if not support_mask((span[0]+value*delta)[None], strict)[0]:
                raise OverflowError("boundary_endpoint_numerically_unresolved")
            limits[side] = value
    lower, upper = low+np.array(limits)*(high-low)
    kept = []
    for segment, coordinates in zip(segments, projected):
        reverse = coordinates[0] > coordinates[1]
        pair = segment[::-1].copy() if reverse else segment.copy()
        left, right = np.sort(coordinates)
        clipped_left, clipped_right = max(left, lower), min(right, upper)
        if clipped_right-clipped_left <= 1e-10:
            continue
        if clipped_left > left:
            pair[0] = anchor+clipped_left*direction
        if clipped_right < right:
            pair[1] = anchor+clipped_right*direction
        kept.append(pair[::-1] if reverse else pair)
    result = np.asarray(kept, float).reshape(-1, 2, 3)
    retained = np.linalg.norm(result[:, 1]-result[:, 0], axis=1).sum()
    return dict(output, segments=result, boundary_removed_length_m=float(lengths.sum()-retained))


def selected_context(values, result, context):
    """Recover the unchanged winner's same-chain assignments, not a new winner."""
    selected = result["readouts"][result["selected_candidate"]]
    local = context
    if selected["support_set_sha256"] is not None:
        groups = support_groups(np.unique(np.asarray(values, float), axis=0), context)
        group = next(g for g in groups if g["support_set_sha256"] == selected["support_set_sha256"])
        local = dict(context, association=dict(context["association"],
                     assignments=group["assignments"], chain_count=len(group["assignments"])))
    return local


def readout(values, voxel_size, origin, context):
    result = previous.readout(values, voxel_size, origin, context)
    result.update(output_support_policy="bracketed_interior_with_unpadded_required_view_outer_limits",
        rgb_outer_endpoints_preserved=False, boundary_requires_all_marked_views=True,
        boundary_horizontal_padding_px=0., boundary_only_intervention=True,
        boundary_removed_length_m=0., boundary_positive_cells=0, boundary_cell_count=0)
    index = result["selected_candidate"]
    if result["state"] != "complete" or index is None or not len(result["segments"]):
        return result
    result.update(trim_outer_envelope(result["segments"], selected_context(values, result, context)))
    result["components"] = len(result["segments"])
    if not result["components"]:
        result["reason"] = "no_unpadded_required_view_boundary_support"
    return result
