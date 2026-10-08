"""Raw-point coverage with symmetric independent support for competing axes.

Voxels balance geometry fitting, but never replace their points with a mean.
Fixed-budget hypotheses are not an exhaustive rod detector or skeletonizer.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np
from scipy.spatial import cKDTree

from . import single_axis_readout as previous
from .line_controls import LineFitDegenerate, fit_line_tls

DEFAULTS = {**previous.DEFAULTS, "consensus_radius_voxels": .5,
    "minimum_pair_extent_voxels": 4., "ransac_trials": 512, "ransac_seed": 0,
    "competing_support_fraction": .8, "competing_separation_radii": 2.,
    "maximum_unique_points": 2000000, "local_anchor_budget": 512, "local_neighbors": 4,
    "minimum_local_pair_voxels": .5, "maximum_point_hypothesis_checks": 200000000, "minimum_exclusive_coverage_fraction": .5}
SCOPE = "predeclared_local_single_axis_independent_support_coverage_not_semantic_identity_or_general_topology"


def raw_pool(values, size, origin):
    unique = np.unique(values, axis=0)
    if len(unique) > DEFAULTS["maximum_unique_points"]:
        raise OverflowError("unique_point_budget_exceeded")
    scaled = (unique-origin)/size
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**52):
        raise ValueError("Coordinates exceed exact voxel-index precision")
    grid, inverse = np.unique(np.floor(scaled).astype(np.int64), axis=0, return_inverse=True)
    if len(grid) > DEFAULTS["maximum_voxels"]:
        raise OverflowError("occupied_voxel_budget_exceeded")
    return unique, inverse, len(grid)


def axis_support(points, inverse, anchor, direction, size):
    """Support uses original coordinates; adding off-axis points cannot erase it.

    That property is for a FIXED hypothesis, not for selecting a winning axis:
    adding a genuinely competing structure may correctly make readout unresolved.
    """
    offset = points-anchor
    axial = offset @ direction
    distance2 = np.sum((offset-axial[:, None]*direction)**2, axis=1)
    indices = np.flatnonzero(distance2 <= (DEFAULTS["consensus_radius_voxels"]*size)**2)
    if not len(indices):
        return dict(indices=indices, representatives=indices, axial=np.empty(0), coverage=0., longest_run=0.)
    # Stable original-coordinate order breaks exact distance ties, never density.
    order = np.lexsort((indices, distance2[indices], inverse[indices]))
    ordered = indices[order]
    representatives = ordered[np.r_[True, np.diff(inverse[ordered]) != 0]]
    observed = np.sort(axial[indices])
    starts = np.r_[0, np.flatnonzero(np.diff(observed) > 2*size)+1]
    stops = np.r_[starts[1:]-1, len(observed)-1]
    # Union of support intervals, clipped to actual overall observed extrema.
    left = np.maximum(observed[starts]-size, observed[0])
    right = np.minimum(observed[stops]+size, observed[-1])
    lengths = right-left
    lengths = lengths[lengths >= DEFAULTS["minimum_length_voxels"]*size]
    return dict(indices=indices, representatives=representatives, axial=observed,
                coverage=float(lengths.sum()), longest_run=float(lengths.max()) if len(lengths) else 0.)


def proposals(points, inverse, size):
    """Local neighbors find sparse chains; global/extreme/random seeds stay too."""
    try:
        global_fit = fit_line_tls(points, min_points=DEFAULTS["minimum_voxels"])
    except LineFitDegenerate:
        pass
    else:
        yield global_fit["centroid"], global_fit["direction"]
    anchors = np.unique(np.linspace(0, len(points)-1,
        min(len(points), DEFAULTS["local_anchor_budget"]), dtype=int))
    neighbors = cKDTree(points).query(points[anchors], k=min(len(points), DEFAULTS["local_neighbors"]+1))[1]
    for a, nearby in zip(anchors, neighbors):
        for b in nearby:
            direction = points[b]-points[a]
            length = float(np.linalg.norm(direction))
            if length >= DEFAULTS["minimum_local_pair_voxels"]*size:
                yield points[a], direction/length
    extrema = np.unique(np.r_[np.argmin(points, axis=0), np.argmax(points, axis=0)])
    pairs = [(int(a), int(b)) for a in extrema for b in extrema if a < b]
    order = np.argsort(inverse, kind="stable")
    _, starts, counts = np.unique(inverse[order], return_index=True, return_counts=True)
    rng = np.random.default_rng(DEFAULTS["ransac_seed"])
    for _ in range(DEFAULTS["ransac_trials"]):
        groups = rng.choice(len(starts), 2, replace=False)
        pairs.append(tuple(int(order[starts[g]+rng.integers(counts[g])]) for g in groups))
    for a, b in pairs:
        direction = points[b]-points[a]
        length = float(np.linalg.norm(direction))
        if length >= DEFAULTS["minimum_pair_extent_voxels"]*size:
            yield points[a], direction/length


def refine(points, inverse, anchor, direction, size, seen=None):
    support = axis_support(points, inverse, anchor, direction, size)
    if len(support["representatives"]) < DEFAULTS["minimum_voxels"]:
        return None
    signature = sha256(support["representatives"].tobytes()).digest()
    if seen is not None:
        if signature in seen:
            return None
        seen.add(signature)
    try:
        model = fit_line_tls(points[support["representatives"]], min_points=DEFAULTS["minimum_voxels"])
    except LineFitDegenerate:
        return None
    if 1-model["eigen_gap_ratio"] > DEFAULTS["maximum_second_eigen_ratio"]:
        return None
    # All hypotheses are rescored against the SAME full raw pool after fitting.
    support = axis_support(points, inverse, model["centroid"], model["direction"], size)
    if len(support["representatives"]) < DEFAULTS["minimum_voxels"] or support["coverage"] <= 0:
        return None
    model.update(support_coverage_m=support["coverage"], longest_supported_run_m=support["longest_run"],
                 support_voxels=len(support["representatives"]), support_point_count=len(support["indices"]),
                 weighting="nearest_original_supported_point_per_world_voxel")
    # Never retain per-point arrays per candidate: bounded scalar/model cache.
    return model


def separated(first, second, size):
    offset = second["segment"]-first["centroid"]
    residual = offset-(offset @ first["direction"])[:, None]*first["direction"]
    return float(np.linalg.norm(residual, axis=1).max()) > (
        DEFAULTS["competing_separation_radii"]*DEFAULTS["consensus_radius_voxels"]*size)



def independent_support(points, inverse, first, second, size):
    """Both axes must have independent evidence, measured in the same full pool.

    Shared points may support alternative estimates of ONE noisy structure.
    This is a symmetric qualification of competition, not residual-only fitting.
    """
    a = axis_support(points, inverse, first["centroid"], first["direction"], size)
    b = axis_support(points, inverse, second["centroid"], second["direction"], size)
    ratios, coverages = [], []
    for model, own, other in ((first, a, b), (second, b, a)):
        exclusive = np.setdiff1d(own["indices"], other["indices"], assume_unique=True)
        support = axis_support(points[exclusive], inverse[exclusive],
            model["centroid"], model["direction"], size)
        # Independent fragments must themselves meet the unchanged 4v run gate.
        coverage = support["coverage"] if len(support["representatives"]) >= DEFAULTS["minimum_voxels"] else 0.
        coverages.append(coverage)
        ratios.append(coverage/own["coverage"] if own["coverage"] > 0 else 0.)
    return dict(exclusive_coverage_m=coverages, exclusive_coverage_fractions=ratios,
        independent=bool(min(ratios) >= DEFAULTS["minimum_exclusive_coverage_fraction"]))


def readout(values, voxel_size, origin, support_fn):
    values, origin = np.asarray(values, float), np.asarray(origin, float)
    if (values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all()
            or origin.shape != (3,) or not np.isfinite(origin).all()
            or isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Finite Nx3 geometry, fixed origin and positive voxel size required")
    result = dict(state="complete", resolution_state="unresolved", reason=None,
        segments=np.empty((0, 2, 3)), components=0, input_count=len(values), scope=SCOPE,
        defaults=dict(DEFAULTS), voxel_size=float(voxel_size), origin=origin.tolist(),
        weighting="nearest_original_supported_point_per_world_voxel",
        hypothesis_ranking="observed_axial_interval_union_length_not_point_count",
        within_voxel_sampling_invariant=False, multiplicity_invariant=True,
        fixed_axis_support_set_monotone=True, exhaustive_axis_search=False,
        unresolved_axial_gap_scale_m=2*voxel_size,
        endpoint_policy="raw_supported_projection_extrema_then_geometry_AND_RGB_runs")
    if not len(values):
        return {**result, "reason": "empty_input", "unique_count": 0, "occupied_voxels": 0}
    try:
        points, inverse, count = raw_pool(values, voxel_size, origin)
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    result.update(unique_count=len(points), occupied_voxels=count)
    if count < DEFAULTS["minimum_voxels"]:
        return {**result, "reason": "too_few_occupied_voxels"}
    candidates, signatures, proposed = [], set(), 0
    for anchor, direction in proposals(points, inverse, voxel_size):
        proposed += 1
        if 2*proposed*len(points) > DEFAULTS["maximum_point_hypothesis_checks"]:
            return {**result, "state": "unmeasurable", "reason": "point_hypothesis_work_budget_exceeded",
                    "proposed_axes": proposed-1}
        model = refine(points, inverse, anchor, direction, voxel_size, signatures)
        if model is not None:
            candidates.append(model)
    result.update(proposed_axes=proposed, valid_axes=len(candidates))
    if not candidates:
        return {**result, "reason": "no_supported_axis_hypothesis"}
    candidates.sort(key=lambda item: (-item["support_coverage_m"], -item["longest_supported_run_m"],
        item["perpendicular_rmse"], tuple(item["direction"]), tuple(item["centroid"])))
    model = candidates[0]
    result.update(model=model, consensus_voxels=model["support_voxels"],
                  excluded_voxels=count-model["support_voxels"])
    shared_alternatives = 0
    point_checks = 2*proposed*len(points)
    for other in candidates[1:]:
        if other["support_coverage_m"] < DEFAULTS["competing_support_fraction"]*model["support_coverage_m"]:
            break
        if separated(model, other, voxel_size) or separated(other, model, voxel_size):
            point_checks += 4*len(points)
            if point_checks > DEFAULTS["maximum_point_hypothesis_checks"]:
                return {**result, "state": "unmeasurable", "reason": "point_hypothesis_work_budget_exceeded"}
            independence = independent_support(points, inverse, model, other, voxel_size)
            if independence["independent"]:
                result.update(competitor_model=other, competitor_independence=independence,
                              shared_support_alternatives=shared_alternatives)
                return {**result, "reason": "strong_competing_axis"}
            shared_alternatives += 1
    result["shared_support_alternatives"] = shared_alternatives
    observed = axis_support(points, inverse, model["centroid"], model["direction"], voxel_size)
    begin, end = float(observed["axial"][0]), float(observed["axial"][-1])
    sample_count = int(np.ceil((end-begin)/(DEFAULTS["axis_sampling_step_voxels"]*voxel_size)))+1
    if sample_count > DEFAULTS["maximum_axis_samples"]:
        return {**result, "state": "unmeasurable", "reason": "axis_sample_budget_exceeded"}
    coordinates = np.linspace(begin, end, sample_count)
    samples = model["centroid"]+coordinates[:, None]*model["direction"]
    geometry = previous.projection_support(coordinates, observed["axial"], voxel_size)
    rgb = np.asarray(support_fn(samples))
    if rgb.dtype != np.dtype(bool) or rgb.shape != (sample_count,):
        raise ValueError("Normal support callback must return one boolean per axis sample")
    segments, discarded = previous.supported_runs(samples, geometry & rgb,
        DEFAULTS["minimum_length_voxels"]*voxel_size)
    result.update(segments=segments, components=len(segments), axis_sample_count=sample_count,
        observed_projection_extent_m=end-begin, geometry_supported_samples=int(geometry.sum()),
        rgb_supported_samples=int(rgb.sum()), jointly_supported_samples=int((geometry & rgb).sum()),
        discarded_short_runs=discarded, resolution_state="resolved" if len(segments) else "unresolved",
        reason=None if len(segments) else "no_jointly_supported_finite_run")
    return result
