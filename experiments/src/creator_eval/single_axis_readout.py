"""Shared geometry-only readout for a predeclared local single-axis task.

This is not a rod detector, surface classifier or general skeletonizer. The
same voxel-balanced fit consumes complete base plus sampled added geometry.
No primitive identity, truth, fitted-patch endpoints or best-model selection.
"""

from __future__ import annotations

import numpy as np

from .line_controls import LineFitDegenerate, fit_line_tls

DEFAULTS = dict(minimum_voxels=6, maximum_second_eigen_ratio=.2,
    minimum_length_voxels=4., projection_support_radius_voxels=1.,
    axis_sampling_step_voxels=.5, maximum_voxels=80000, maximum_axis_samples=2000000)
SCOPE = "predeclared_local_single_axis_readout_not_target_identity_surface_or_general_topology"


def voxel_representatives(values, voxel_size, origin):
    """One equal-weight mean of UNIQUE coordinates per fixed world voxel.

    Multiplicity and input order do not change the fit. Different within-voxel
    sampling locations still can: this is recorded, not claimed density-free.
    """
    unique = np.unique(values, axis=0)
    scaled = (unique-origin)/voxel_size
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**52):
        raise ValueError("Coordinates exceed exact voxel-index precision")
    grid, inverse, counts = np.unique(np.floor(scaled).astype(np.int64),
        axis=0, return_inverse=True, return_counts=True)
    if len(grid) > DEFAULTS["maximum_voxels"]:
        raise OverflowError("occupied_voxel_budget_exceeded")
    totals = np.zeros((len(grid), 3))
    np.add.at(totals, inverse, unique)
    return unique, totals/counts[:, None]


def projection_support(samples, observed, radius):
    """An unobserved axial interval is never filled merely because RGB votes."""
    observed = np.sort(observed)
    if not len(observed):
        return np.zeros(len(samples), bool)
    indices = np.searchsorted(observed, samples)
    left = np.abs(samples-observed[np.clip(indices-1, 0, len(observed)-1)])
    right = np.abs(samples-observed[np.clip(indices, 0, len(observed)-1)])
    return np.minimum(left, right) <= radius


def supported_runs(samples, mask, minimum_length):
    """Keep only consecutive supported samples; no morphological gap closing."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1
    segments, discarded = [], 0
    for start, stop in zip(starts, stops):
        if start == stop or np.linalg.norm(samples[stop]-samples[start]) < minimum_length:
            discarded += 1
        else:
            segments.append([samples[start], samples[stop]])
    return np.asarray(segments, float).reshape(-1, 2, 3), discarded


def readout(values, voxel_size, origin, support_fn):
    """Read one axis with geometry occupancy AND a shared normal RGB callback."""
    values, origin = np.asarray(values, float), np.asarray(origin, float)
    if (values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all()
            or origin.shape != (3,) or not np.isfinite(origin).all()
            or isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Finite Nx3 geometry, fixed origin and positive voxel size required")
    result = dict(state="complete", resolution_state="unresolved", reason=None,
        segments=np.empty((0, 2, 3)), components=0, input_count=len(values), scope=SCOPE,
        defaults=dict(DEFAULTS), voxel_size=float(voxel_size), origin=origin.tolist(),
        weighting="equal_per_occupied_voxel_mean_of_unique_coordinates",
        within_voxel_sampling_invariant=False, multiplicity_invariant=True,
        unresolved_axial_gap_scale_m=2*voxel_size,
        endpoint_policy="observed_projection_min_max_then_shared_geometry_and_RGB_supported_sample_runs")
    if not len(values):
        return {**result, "reason": "empty_input", "occupied_voxels": 0, "unique_count": 0}
    try:
        unique, representatives = voxel_representatives(values, voxel_size, origin)
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    result.update(unique_count=len(unique), occupied_voxels=len(representatives))
    try:
        model = fit_line_tls(representatives, min_points=DEFAULTS["minimum_voxels"])
    except LineFitDegenerate as exc:
        return {**result, "reason": str(exc)}
    second_ratio = 1-model["eigen_gap_ratio"]
    result.update(model=model, second_eigen_ratio=second_ratio)
    if second_ratio > DEFAULTS["maximum_second_eigen_ratio"]:
        return {**result, "reason": "not_a_unique_dominant_axis"}
    center, direction = model["centroid"], model["direction"]
    observed = (unique-center) @ direction
    begin, end = float(observed.min()), float(observed.max())
    result["observed_projection_extent_m"] = end-begin
    if end-begin < DEFAULTS["minimum_length_voxels"]*voxel_size:
        return {**result, "reason": "below_minimum_axis_length"}
    count = int(np.ceil((end-begin)/(DEFAULTS["axis_sampling_step_voxels"]*voxel_size)))+1
    if count > DEFAULTS["maximum_axis_samples"]:
        return {**result, "state": "unmeasurable", "reason": "axis_sample_budget_exceeded"}
    coordinates = np.linspace(begin, end, count)
    samples = center+coordinates[:, None]*direction
    # Use occupied representatives for support, not the number of raw points or
    # the proposed infinite TLS axis. Unknown geometry/RGB is never interpolated.
    geometric = projection_support(coordinates, (representatives-center) @ direction,
        DEFAULTS["projection_support_radius_voxels"]*voxel_size)
    rgb = np.asarray(support_fn(samples))
    if rgb.dtype != np.dtype(bool) or rgb.shape != (count,):
        raise ValueError("Normal support callback must return one boolean per axis sample")
    segments, discarded = supported_runs(samples, geometric & rgb,
        DEFAULTS["minimum_length_voxels"]*voxel_size)
    result.update(segments=segments, components=len(segments), axis_sample_count=count,
        geometry_supported_samples=int(geometric.sum()), rgb_supported_samples=int(rgb.sum()),
        jointly_supported_samples=int((geometric & rgb).sum()), discarded_short_runs=discarded,
        resolution_state="resolved" if len(segments) else "unresolved",
        reason=None if len(segments) else "no_jointly_supported_finite_run")
    return result
