"""Actual-coordinate support operations; no image-axis snapping or truth input."""

import numpy as np

from . import single_axis_readout as axis
from .line_controls import LineFitDegenerate, fit_line_tls, point_to_segments_distance

DEFAULTS = dict(seed_radius_voxels=2., explained_radius_voxels=2.)


def _points(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all():
        raise ValueError("Finite Nx3 actual coordinates required")
    return values


def fixed_seed_expansion(values, seed_indices, voxel_size, origin):
    """Keep every seed and add points inside one ORIGINAL actual TLS tube.

    The caller supplies one same-chain positive pool, not arbitrary scene data.
    Returned indices refer to unchanged input coordinates. No iterative growth.
    Invalid/underconstrained seeds cannot borrow support to bootstrap an axis.
    """
    values, origin = _points(values), np.asarray(origin, float)
    indices = np.asarray(seed_indices)
    if (origin.shape != (3,) or not np.isfinite(origin).all()
            or isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Fixed finite origin and positive voxel size required")
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= len(values)) or len(np.unique(indices)) != len(indices)):
        raise ValueError("Unique in-range integer seed indices required")
    indices = np.sort(indices)
    result = dict(state="complete", reason=None, indices=indices,
        seed_model=None, seed_point_count=len(indices), seed_radius_voxels=DEFAULTS["seed_radius_voxels"],
        original_seed_preserved=True, iterative_expansion=False)
    try:
        _, representatives = axis.voxel_representatives(values[indices], voxel_size, origin)
        model = fit_line_tls(representatives, min_points=axis.DEFAULTS["minimum_voxels"])
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    except LineFitDegenerate as exc:
        return {**result, "reason": str(exc)}
    result["seed_model"] = model
    if 1-model["eigen_gap_ratio"] > axis.DEFAULTS["maximum_second_eigen_ratio"]:
        return {**result, "reason": "not_a_unique_dominant_seed_axis"}
    centered = values-model["centroid"]
    residual = centered-(centered @ model["direction"])[:, None]*model["direction"]
    selected = np.linalg.norm(residual, axis=1) <= DEFAULTS["seed_radius_voxels"]*voxel_size
    selected[indices] = True
    result.update(indices=np.flatnonzero(selected), added_point_count=int(selected.sum())-len(indices))
    return result


def novel_support_mask(points, segments, radius):
    """Points outside the radius of a FINITE segment union, endpoints included."""
    points = _points(points)
    segments = np.asarray(segments, float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Finite Mx2x3 segments required")
    if isinstance(radius, (bool, np.bool_)) or not np.isfinite(radius) or radius < 0:
        raise ValueError("Nonnegative finite distance required")
    if not len(segments):
        return np.ones(len(points), bool)
    return point_to_segments_distance(points, segments) > radius
