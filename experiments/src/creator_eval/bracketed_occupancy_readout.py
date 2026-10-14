"""Close only short, two-sided RGB holes inside one actual occupied interval.

The selected TLS axis and all outer RGB endpoints are unchanged. A hole must
be no wider than the existing 1v geometric support radius AND both sides
must belong to the same exact actual-point occupancy component. No image
endpoint extension, gap crossing, new axis fit, or unbounded interpolation.
"""

import numpy as np

from . import occupied_axis_readout as previous


def bridge_internal_holes(segments, occupied, voxel_size):
    segments, occupied = np.asarray(segments, float), np.asarray(occupied, float)
    if any(a.ndim != 3 or a.shape[1:] != (2, 3) or not np.isfinite(a).all()
           for a in (segments, occupied)):
        raise ValueError("Finite Mx2x3 clipped and occupied segments required")
    if isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Positive finite voxel size required")
    result = dict(segments=segments.copy(), bridged_gap_count=0, bridged_length_m=0.)
    if not len(segments):
        return result
    if not len(occupied) or any(np.any(np.linalg.norm(a[:, 1]-a[:, 0], axis=1) <= 1e-12)
                                for a in (segments, occupied)):
        raise ValueError("Nondegenerate actual occupancy required")
    anchor = occupied[0, 0]
    direction = occupied[0, 1]-anchor
    direction /= np.linalg.norm(direction)
    tolerance = 64*np.finfo(float).eps*max(1., float(np.abs(occupied).max()), float(np.abs(segments).max()))
    for array in (segments, occupied):
        if np.linalg.norm(np.cross(array-anchor, direction), axis=2).max() > tolerance:
            raise ValueError("Segments must lie on the same selected actual axis")
    bounds = np.sort((occupied-anchor) @ direction, axis=1)
    projected = (segments-anchor) @ direction
    pairs = segments.copy()
    reverse = projected[:, 0] > projected[:, 1]
    pairs[reverse] = pairs[reverse, ::-1]
    projected = np.sort(projected, axis=1)
    order = np.argsort(projected[:, 0], kind="stable")
    output, owners, extents = [], [], []
    for index in order:
        left, right = projected[index]
        inside = np.flatnonzero((bounds[:, 0] <= left+tolerance) & (bounds[:, 1] >= right-tolerance))
        if len(inside) != 1:
            raise ValueError("Each RGB interval must belong to one actual occupied component")
        owner = int(inside[0])
        gap = left-extents[-1][1] if output else np.inf
        if output and gap < -tolerance:
            raise ValueError("Clipped intervals must not overlap")
        if output and owners[-1] == owner and gap <= voxel_size:
            output[-1][1] = pairs[index, 1]
            extents[-1][1] = right
            result["bridged_gap_count"] += int(gap > 0)
            result["bridged_length_m"] += max(0., float(gap))
        else:
            output.append(pairs[index].copy())
            owners.append(owner)
            extents.append([left, right])
    result["segments"] = np.asarray(output, float).reshape(-1, 2, 3)
    return result


def readout(values, voxel_size, origin, context):
    result = previous.readout(values, voxel_size, origin, context)
    result.update(output_support_policy="short_rgb_holes_bracketed_within_one_actual_occupancy_component",
        maximum_internal_rgb_hole_voxels=1., actual_occupied_component_required=True,
        rgb_outer_endpoints_preserved=True, continuous_rgb_support_required=False,
        axis_refitted=False, image_constructed_geometry=False,
        previous_rgb_clipped_components=result["components"],
        bridged_gap_count=0, bridged_length_m=0.)
    selected = result["selected_candidate"]
    if result["state"] != "complete" or selected is None:
        return result
    occupied = result["readouts"][selected]["graph"]["segments"]
    result.update(bridge_internal_holes(result["segments"], occupied, voxel_size))
    result["components"] = len(result["segments"])
    return result
