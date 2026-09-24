"""Development reader: do not turn an unsupported split into a claimed single axis.

Execution completion and geometric resolution are separate. Unresolved components
emit no curve, while independently supported components remain in the output.
"""
from __future__ import annotations

import numpy as np

from .common_readout import region_mask, sample_segments
from .common_readout_components import merged_segments
from .common_readout_sections import circle_section, coherent_side_branch, connected_indices
from .common_readout_split import axial_runs, principal_axes
from .common_readout_valley_closed import DEFAULTS, guarded_split_component


def readout(points, segments, config, region=None):
    p = {**DEFAULTS, **config}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown section-readout policy")
    integer_keys = ("minimum_component_voxels", "maximum_voxels", "maximum_curve_samples", "maximum_split_depth", "circle_trials", "split_valley_axial_slices", "split_valley_minimum_points_per_side")
    for key in integer_keys:
        if type(p[key]) is not int or not 1 <= p[key] <= (4096 if key == "circle_trials" else 2000000):
            raise ValueError("Bounded positive integer reader policy required")
    if not 2 <= p["split_valley_axial_slices"] <= 32 or not 0 < p["split_valley_minimum_fraction"] <= 1 or not 0 < p["split_valley_maximum_density_ratio"] < 1:
        raise ValueError("Invalid persistent-valley policy")
    for key, value in p.items():
        if key not in integer_keys and key != "origin" and (not np.isfinite(value) or value <= 0):
            raise ValueError("Positive finite reader policy required")
    if not 0 < p["maximum_second_eigen_ratio"] < 1 or p["neighbor_radius_voxels"] > 2 or p["maximum_split_depth"] > 8:
        raise ValueError("Invalid component geometry policy")
    if not 0 < p["split_minimum_fraction"] < .5 or not 0 < p["split_minimum_axial_overlap"] <= 1 or not 0 < p["minimum_circle_fraction"] <= 1:
        raise ValueError("Invalid support policy")
    if any(p[key] >= 90 for key in ("merge_angle_degrees", "split_maximum_axis_angle_degrees", "branch_minimum_angle_degrees")) or p["minimum_circle_coverage_degrees"] > 360:
        raise ValueError("Invalid angular policy")
    if np.asarray(p["origin"]).shape != (3,) or not np.isfinite(p["origin"]).all():
        raise ValueError("Fixed finite grid origin required")
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Finite Nx3 points and Mx2x3 segments required")
    result = dict(state="complete", config=p, segments=np.empty((0, 2, 3)), input_point_count=len(points), input_segment_count=len(segments),
        scope="abstaining_component_section_diagnostic_not_general_skeleton_or_graph_topology",
        resolution_state="not_evaluated", unresolved_component_count=0, unresolved_components=[])
    try:
        samples = sample_segments(segments, p["voxel_size"] / 2, p["maximum_curve_samples"])
    except OverflowError as error:
        return {**result, "state": "unmeasurable", "reason": str(error)}
    values = np.concatenate((points, samples))
    keep = region_mask(values, region)
    result.update(region_point_count=int(keep[:len(points)].sum()), region_curve_sample_count=int(keep[len(points):].sum()),
        region_policy="all_input" if region is None else "fixed_RGB_strip_union_distinct_view_vote")
    scaled = (values[keep] - p["origin"]) / p["voxel_size"]
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**52):
        raise ValueError("Coordinates exceed exact voxel-index precision")
    grid = np.unique(np.floor(scaled).astype(np.int64), axis=0)
    result["occupied_voxels"] = len(grid)
    if len(grid) > p["maximum_voxels"]:
        return {**result, "state": "unmeasurable", "reason": "occupied_voxel_budget_exceeded"}
    centers = (grid.astype(float) + .5) * p["voxel_size"] + p["origin"]
    pending = [(centers[ids], 0, f"component-{number:06d}")
               for number, ids in enumerate(connected_indices(grid, p["neighbor_radius_voxels"]))]
    diagnostics = dict(connected_components=len(pending), rejected_short_components=0, rejected_nonlinear_components=0,
        rejected_unsupported_sections=0, rejected_side_branches=0, accepted_splits=0, split_decisions={}, sections=[], split_valley_checks=[], unresolved_components=[])
    output = []
    while pending:
        cloud, depth, component_id = pending.pop()
        if len(cloud) < p["minimum_component_voxels"]:
            diagnostics["rejected_short_components"] += 1
            continue
        mean, eigen, vectors, axis = principal_axes(cloud)
        if eigen[-1] <= 1e-20 or eigen[-2] / eigen[-1] > p["maximum_second_eigen_ratio"]:
            diagnostics["rejected_nonlinear_components"] += 1
            continue
        # 到深度上限也得看有没有二分提议，不能借预算耗尽偷偷回退成单轴。
        children, reason, valley = guarded_split_component(cloud, eigen, vectors, axis, p)
        unresolved = valley is not None and (children is None or depth >= p["maximum_split_depth"])
        if unresolved and children is not None:
            reason = "supported_split_exceeds_depth_budget"
        if valley is not None:
            valley = {**valley, "component_id": component_id,
                      "decision": "abstain_unresolved" if unresolved else "accept_split"}
            diagnostics["split_valley_checks"].append(valley)
        decisions = diagnostics["split_decisions"]
        decisions[reason] = decisions.get(reason, 0) + 1
        if unresolved:
            # 看不清是几根，就老实说未决。少画一根不等于认出了背景。
            diagnostics["unresolved_components"].append(dict(
                component_id=component_id, state="unresolved", reason=reason, depth=depth,
                occupied_voxels=len(cloud), bounds_min=cloud.min(axis=0).tolist(),
                bounds_max=cloud.max(axis=0).tolist(), proposed_parts=2,
                emitted_segments=0, valley=valley,
                alternatives=["whole_component_not_fitted", "proposed_split_not_emitted"]))
            continue
        if children is not None:
            pending.extend((child, depth + 1, component_id + f".{number}")
                           for number, child in enumerate(children))
            diagnostics["accepted_splits"] += 1
            continue
        projection = (cloud - mean) @ axis
        transverse = (cloud - mean) @ vectors[:, :2]
        circle = circle_section(transverse, np.ptp(projection), p)
        if circle is not None:
            center, radius, inside, detail = circle
            # 半边圆的质心不是杆心。这里终于用拟合圆心，别再只拿圆拟合当否决票。
            anchor = mean + vectors[:, :2] @ center
            detail["model"] = "circle"
        else:
            radius = float(np.quantile(np.linalg.norm(transverse, axis=1), .95))
            if radius > p["maximum_line_radius_voxels"] * p["voxel_size"]:
                diagnostics["rejected_unsupported_sections"] += 1
                continue
            inside = np.linalg.norm(transverse, axis=1) <= p["maximum_line_radius_voxels"] * p["voxel_size"]
            anchor = mean
            detail = dict(model="unresolved_thin_section", radius=radius, inlier_fraction=float(inside.mean()))
        if coherent_side_branch(cloud, inside, axis, radius, p):
            diagnostics["rejected_side_branches"] += 1
            continue
        diagnostics["sections"].append(detail)
        # 只跟随真的占据支持；稳健拟合不授权跨过没有观测的长缺口。
        for run in axial_runs(projection[inside], p["voxel_size"], p["neighbor_radius_voxels"]):
            if run[-1] - run[0] >= p["minimum_run_length_voxels"] * p["voxel_size"]:
                output.append(np.array([anchor + run[0] * axis, anchor + run[-1] * axis]))
    result.update(segments=merged_segments(output, p), components_before_overlap_merge=len(output), **diagnostics)
    count = len(diagnostics["unresolved_components"])
    resolution = ("partial" if output else "unresolved") if count else ("resolved" if output else "no_supported_curve")
    reason = ("some_components_unresolved" if output else "ambiguous_components_abstained") if count else (None if output else "no_supported_curve")
    result.update(components=len(result["segments"]), unresolved_component_count=count,
                  resolution_state=resolution, reason=reason)
    return result
