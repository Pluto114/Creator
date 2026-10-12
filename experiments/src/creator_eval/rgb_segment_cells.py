"""Clip finite actual fitted segments at every projective membership boundary.

For a fixed 3D segment, camera homogeneous coordinates are affine in t.
Positive depth, rounded image rows, and measured horizontal bands can change
only at the enumerated linear roots. Same-chain voting is constant inside
each resulting open cell. This addresses gaps missed by uniform sampling.
"""

import numpy as np

from .rgb_candidate_readout import POLICY
from .rgb_chain_support import chain_support_mask

DEFAULTS = dict(maximum_cells=2000000, maximum_membership_checks=200000000)


def _roots(origin, delta, component, boundaries):
    boundaries = np.asarray(boundaries, float)
    numerator = origin[component]-boundaries*origin[2]
    denominator = delta[component]-boundaries*delta[2]
    selected = denominator != 0
    roots = -numerator[selected]/denominator[selected]
    return roots[np.isfinite(roots) & (roots > 0) & (roots < 1)]


def segment_events(segment, context):
    """All float64 event roots, including bands belonging to other image rows."""
    parts = [np.array([0., 1.])]
    for view in context["views"]:
        e, k = view["world_to_camera_cv"], view["K_index"]
        projected = (segment @ e[:, :3].T+e[:, 3]) @ k.T
        origin, delta = projected[0], projected[1]-projected[0]
        if delta[2] != 0:
            root = (1e-10-origin[2])/delta[2]
            if 0 < root < 1:
                parts.append(np.array([root]))
        parts.append(_roots(origin, delta, 1, np.arange(len(view["left"])+1)-.5))
        pad = POLICY["horizontal_padding_px"]
        for field, margin in (("left", -pad), ("right", pad)):
            values = view[field]
            boundaries = np.unique(values[np.isfinite(values)]+margin)
            parts.append(_roots(origin, delta, 0, boundaries))
    return np.unique(np.concatenate(parts))


def clip_segments(segments, context, minimum_length):
    segments = np.asarray(segments, float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Finite Mx2x3 segments required")
    if np.any(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1) <= 1e-12):
        raise ValueError("Nondegenerate finite segments required")
    if (isinstance(minimum_length, (bool, np.bool_)) or not np.isfinite(minimum_length)
            or minimum_length < 0):
        raise ValueError("Finite nonnegative minimum length required")
    if context["association"]["fallback_to_raw_union"]:
        raise ValueError("Complete same-chain association required")
    output, count, work, discarded, trims = [], 0, 0, 0, 0
    def query(points):
        nonlocal work
        work += len(points)*len(context["association"]["assignments"])
        if work > DEFAULTS["maximum_membership_checks"]:
            raise OverflowError("rgb_cell_budget_exceeded")
        return chain_support_mask(points, context)
    for segment in segments:
        events = segment_events(segment, context)
        count += len(events)-1
        if count > DEFAULTS["maximum_cells"]:
            raise OverflowError("rgb_cell_budget_exceeded")
        delta = segment[1]-segment[0]
        middles = (events[:-1]+events[1:])/2
        positive = query(segment[0]+middles[:, None]*delta)
        boundary = query(segment[0]+events[:, None]*delta)
        index = 0
        while index < len(positive):
            if not positive[index]:
                index += 1
                continue
            first = index
            while index+1 < len(positive) and positive[index+1] and boundary[index+1]:
                index += 1
            left, right = float(events[first]), float(events[index+1])
            pair = [left, right]
            for side, target, supported in ((0, float(middles[first]), boundary[first]),
                                           (1, float(middles[index]), boundary[index+1])):
                if supported:
                    continue
                # Open row/depth boundaries cannot be emitted as unsupported
                # closed endpoints. Trim inside the first/last positive cell.
                value = pair[side]
                for _ in range(8):
                    value = float(np.nextafter(value, target))
                if not query((segment[0]+value*delta)[None])[0]:
                    value = pair[side]+(target-pair[side])*1e-9
                if not query((segment[0]+value*delta)[None])[0]:
                    raise OverflowError("rgb_cell_endpoint_numerically_unresolved")
                pair[side] = value
                trims += 1
            clipped = segment[0]+np.asarray(pair)[:, None]*delta
            if pair[1] <= pair[0] or np.linalg.norm(clipped[1]-clipped[0]) < minimum_length:
                discarded += 1
            else:
                output.append(clipped)
            index += 1
    return dict(segments=np.asarray(output, float).reshape(-1, 2, 3), cell_count=count,
        input_segments=segments.copy(), algorithm="all_projective_membership_cells_float64",
        discarded_short_runs=discarded, trimmed_open_endpoints=trims,
        membership_checks=work, defaults=dict(DEFAULTS))
