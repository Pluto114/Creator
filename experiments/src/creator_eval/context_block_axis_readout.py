"""Keep isolated transverse ambiguity local; never alter the evaluation domain."""

import numpy as np

from . import guided_block_axis_readout as guided
from . import single_axis_readout as previous
from .guided_block_axis_readout import canonical, guide_direction, transverse_evidence
from .line_controls import LineFitDegenerate, fit_line_tls

DEFAULTS = dict(guided.DEFAULTS)
SCOPE = "guide_direction_local_unknown_axis_not_target_identity_or_general_topology"

__all__ = ["DEFAULTS", "SCOPE", "guide_direction", "readout"]


def readout(values, voxel_size, origin, support_fn, direction):
    values, origin = np.asarray(values, float), np.asarray(origin, float)
    if (values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all()
            or origin.shape != (3,) or not np.isfinite(origin).all()
            or isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Finite Nx3 geometry, fixed origin and positive voxel size required")
    direction = canonical(direction)
    result = dict(state="complete", resolution_state="unresolved", reason=None,
        segments=np.empty((0, 2, 3)), components=0, input_count=len(values), scope=SCOPE,
        defaults=dict(DEFAULTS), voxel_size=float(voxel_size), origin=origin.tolist(),
        guide_direction=direction, guide_location_used=False, output_axis_from="actual_3d_voxel_balanced_TLS",
        multiplicity_invariant=True, within_voxel_sampling_invariant=False,
        endpoint_policy="actual_radially_supported_projection_extrema_AND_original_RGB_without_gap_closing")
    if not len(values):
        return {**result, "reason": "empty_input", "unique_count": 0, "occupied_voxels": 0}
    unique = np.unique(values, axis=0)
    if len(unique) > DEFAULTS["maximum_unique_points"]:
        return {**result, "state": "unmeasurable", "reason": "unique_point_budget_exceeded"}
    try:
        unique, representatives = previous.voxel_representatives(unique, voxel_size, origin)
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    result.update(unique_count=len(unique), occupied_voxels=len(representatives))
    if len(representatives) > DEFAULTS["maximum_voxels"]:
        return {**result, "state": "unmeasurable", "reason": "occupied_voxel_budget_exceeded"}
    if len(representatives) < DEFAULTS["minimum_voxels"]:
        return {**result, "reason": "too_few_occupied_voxels"}
    evidence = transverse_evidence(unique, voxel_size, origin, direction)
    result["transverse_evidence"] = evidence
    local_unknown = evidence["reason"] == "nonpersistent_split_transverse_evidence"
    if evidence["state"] != "resolved" and not local_unknown:
        return {**result, "reason": evidence["reason"]}
    unknown_blocks = np.array(sorted({row["block"] for row in evidence["split_blocks"]}), np.int64)
    block_width = DEFAULTS["block_width_voxels"]*voxel_size
    point_blocks = np.floor(((unique-origin) @ direction)/block_width).astype(np.int64)
    known = ~np.isin(point_blocks, unknown_blocks)
    result.update(unknown_blocks=unknown_blocks, unknown_original_point_count=int((~known).sum()),
                  known_original_point_count=int(known.sum()), ambiguity_policy="local_unknown_retains_full_evaluation_domain")
    unique = unique[known]
    unique, representatives = previous.voxel_representatives(unique, voxel_size, origin)
    result["fitting_voxels"] = len(representatives)
    if len(representatives) < DEFAULTS["minimum_voxels"]:
        return {**result, "reason": "too_few_unambiguous_occupied_voxels"}
    try:
        model = fit_line_tls(representatives, min_points=DEFAULTS["minimum_voxels"])
    except LineFitDegenerate as exc:
        return {**result, "reason": str(exc)}
    result["model"] = model
    if 1-model["eigen_gap_ratio"] > DEFAULTS["maximum_second_eigen_ratio"]:
        return {**result, "reason": "not_a_unique_dominant_axis"}
    center, fitted = model["centroid"], model["direction"]
    delta = representatives-center
    distances = np.linalg.norm(delta-(delta @ fitted)[:, None]*fitted, axis=1)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances-median)))
    radius = max(voxel_size, median+DEFAULTS["radial_mad_multiplier"]*mad)
    offset = unique-center
    projections = offset @ fitted
    selected = np.linalg.norm(offset-projections[:, None]*fitted, axis=1) <= radius
    observed = projections[selected]
    result.update(radial_support_radius_m=radius, radial_median_m=median, radial_mad_m=mad,
                  supported_original_point_count=int(selected.sum()))
    if len(observed) < DEFAULTS["minimum_voxels"]:
        return {**result, "reason": "too_few_radially_supported_points"}
    begin, end = float(observed.min()), float(observed.max())
    if end-begin < DEFAULTS["minimum_length_voxels"]*voxel_size:
        return {**result, "reason": "below_minimum_axis_length"}
    count = int(np.ceil((end-begin)/(DEFAULTS["axis_sampling_step_voxels"]*voxel_size)))+1
    if count > DEFAULTS["maximum_axis_samples"]:
        return {**result, "state": "unmeasurable", "reason": "axis_sample_budget_exceeded"}
    coordinates = np.linspace(begin, end, count)
    samples = center+coordinates[:, None]*fitted
    geometry = previous.projection_support(coordinates, observed, voxel_size)
    rgb = np.asarray(support_fn(samples))
    if rgb.dtype != np.dtype(bool) or rgb.shape != (count,):
        raise ValueError("Normal support callback must return one boolean per axis sample")
    sample_blocks = np.floor(((samples-origin) @ direction)/block_width).astype(np.int64)
    known_samples = ~np.isin(sample_blocks, unknown_blocks)
    joint = geometry & rgb & known_samples
    segments, discarded = previous.supported_runs(samples, joint,
        DEFAULTS["minimum_length_voxels"]*voxel_size)
    result.update(segments=segments, components=len(segments), axis_sample_count=count,
        observed_projection_extent_m=end-begin, geometry_supported_samples=int(geometry.sum()),
        rgb_supported_samples=int(rgb.sum()), jointly_supported_samples=int(joint.sum()), unknown_axis_sample_count=int((~known_samples).sum()),
        discarded_short_runs=discarded,
        resolution_state=("partially_resolved" if local_unknown else "resolved") if len(segments) else "unresolved",
        reason=("local_ambiguity_preserved" if local_unknown else None) if len(segments) else "no_jointly_supported_finite_run")
    return result
