"""Fixed geometric consensus, not one TLS over an entire noisy local volume.

This development reader is not a semantic target selector or a skeletonizer.
Every primitive is discarded before the same voxel-balanced consensus search.
"""

import numpy as np

from . import single_axis_readout as previous
from .line_controls import LineFitDegenerate, deterministic_ransac_line

DEFAULTS = {**previous.DEFAULTS, "consensus_radius_voxels": .5,
    "minimum_pair_extent_voxels": 4., "ransac_trials": 512, "ransac_seed": 0,
    "competing_support_fraction": .8, "competing_separation_radii": 2.}
SCOPE = "predeclared_local_single_axis_geometric_consensus_not_semantic_identity_or_general_topology"


def fit_consensus(points, size):
    # These identities deliberately mean ONE geometric pool, not source views.
    # RGB distinct-view evidence remains in the shared callback/adapter.
    return deterministic_ransac_line(points, np.zeros(len(points), dtype=np.int8),
        distance_threshold=DEFAULTS["consensus_radius_voxels"]*size,
        min_pair_extent=DEFAULTS["minimum_pair_extent_voxels"]*size,
        min_inliers=DEFAULTS["minimum_voxels"], min_views=1,
        trials=DEFAULTS["ransac_trials"], seed=DEFAULTS["ransac_seed"])


def full_axis_support(model, representatives, radius):
    offsets = representatives-model["centroid"]
    residuals = offsets-(offsets @ model["direction"])[:, None]*model["direction"]
    return int(np.count_nonzero(np.sum(residuals**2, axis=1) <= radius**2))


def competing_axis(model, alternative, size, representatives):
    points = alternative["segment"]-model["centroid"]
    residuals = points-(points @ model["direction"])[:, None]*model["direction"]
    separated = float(np.linalg.norm(residuals, axis=1).max()) > (
        DEFAULTS["competing_separation_radii"]*DEFAULTS["consensus_radius_voxels"]*size)
    # Residual points propose the second axis; BOTH axes are then measured on
    # the complete same pool. Otherwise the first axis consumes shared support
    # at shallow crossings and unfairly hides an equally supported competitor.
    radius = DEFAULTS["consensus_radius_voxels"]*size
    first = full_axis_support(model, representatives, radius)
    second = full_axis_support(alternative, representatives, radius)
    model["full_pool_comparison_support"] = first
    alternative["full_pool_comparison_support"] = second
    strong = second >= DEFAULTS["competing_support_fraction"]*first
    return bool(separated and strong)


def readout(values, voxel_size, origin, support_fn):
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
        consensus_view_ids_are_geometric_pool_not_source_views=True,
        unresolved_axial_gap_scale_m=2*voxel_size,
        endpoint_policy="winning_voxels_observed_projection_then_geometry_and_RGB_supported_runs")
    if not len(values):
        return {**result, "reason": "empty_input", "occupied_voxels": 0, "unique_count": 0}
    try:
        unique, representatives = previous.voxel_representatives(values, voxel_size, origin)
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    result.update(unique_count=len(unique), occupied_voxels=len(representatives))
    if len(representatives) > DEFAULTS["maximum_voxels"]:
        return {**result, "state": "unmeasurable", "reason": "occupied_voxel_budget_exceeded"}
    try:
        model = fit_consensus(representatives, voxel_size)
    except LineFitDegenerate as exc:
        return {**result, "reason": str(exc)}
    selected = model["inlier_mask"]
    result.update(model=model, consensus_voxels=int(selected.sum()), excluded_voxels=int((~selected).sum()))
    if 1-model["eigen_gap_ratio"] > DEFAULTS["maximum_second_eigen_ratio"]:
        return {**result, "reason": "consensus_has_no_unique_dominant_axis"}
    if (~selected).sum() >= DEFAULTS["minimum_voxels"]:
        try:
            alternative = fit_consensus(representatives[~selected], voxel_size)
        except LineFitDegenerate as exc:
            result["competitor_reason"] = str(exc)
        else:
            result["competitor_model"] = alternative
            if competing_axis(model, alternative, voxel_size, representatives):
                return {**result, "reason": "strong_competing_axis"}
    # Preserve the observed extrema in winning voxels, rather than replacing
    # them with voxel centers or incorporating rejected distant geometry.
    _, inverse = np.unique(np.floor((unique-origin)/voxel_size).astype(np.int64), axis=0, return_inverse=True)
    observed_values = unique[selected[inverse]]
    center, direction = model["centroid"], model["direction"]
    observed = (observed_values-center) @ direction
    begin, end = float(observed.min()), float(observed.max())
    result["observed_projection_extent_m"] = end-begin
    if end-begin < DEFAULTS["minimum_length_voxels"]*voxel_size:
        return {**result, "reason": "below_minimum_axis_length"}
    count = int(np.ceil((end-begin)/(DEFAULTS["axis_sampling_step_voxels"]*voxel_size)))+1
    if count > DEFAULTS["maximum_axis_samples"]:
        return {**result, "state": "unmeasurable", "reason": "axis_sample_budget_exceeded"}
    coordinates = np.linspace(begin, end, count)
    samples = center+coordinates[:, None]*direction
    geometry = previous.projection_support(coordinates, (representatives[selected]-center) @ direction,
        DEFAULTS["projection_support_radius_voxels"]*voxel_size)
    rgb = np.asarray(support_fn(samples))
    if rgb.dtype != np.dtype(bool) or rgb.shape != (count,):
        raise ValueError("Normal support callback must return one boolean per axis sample")
    segments, discarded = previous.supported_runs(samples, geometry & rgb,
        DEFAULTS["minimum_length_voxels"]*voxel_size)
    result.update(segments=segments, components=len(segments), axis_sample_count=count,
        geometry_supported_samples=int(geometry.sum()), rgb_supported_samples=int(rgb.sum()),
        jointly_supported_samples=int((geometry & rgb).sum()), discarded_short_runs=discarded,
        resolution_state="resolved" if len(segments) else "unresolved",
        reason=None if len(segments) else "no_jointly_supported_finite_run")
    return result
