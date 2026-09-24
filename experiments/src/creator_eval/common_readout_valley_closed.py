"""Experimental v5 boundary correction: require a persistent resolvable valley before transverse splitting.

Only the old split proposal receives a new gate. If it fails, the whole component
still goes through v4's circle/line tests. No semantic target labels are available.
"""
from __future__ import annotations

import numpy as np

from .common_readout import region_mask, sample_segments
from .common_readout_components import merged_segments
from .common_readout_sections import DEFAULTS as SECTION_DEFAULTS
from .common_readout_sections import circle_section, coherent_side_branch, connected_indices
from .common_readout_split import axial_runs, principal_axes
from .common_readout_split import split_component as legacy_split_component

DEFAULTS = {**SECTION_DEFAULTS, "split_valley_width_voxels": 1., "split_valley_maximum_density_ratio": .2,
            "split_valley_axial_slices": 6, "split_valley_minimum_fraction": 2 / 3,
            "split_valley_minimum_points_per_side": 3}


def guarded_split_component(cloud, eigen, vectors, axis, p):
    children, reason = legacy_split_component(cloud, eigen, vectors, axis, p)
    if children is None:
        return None, reason, None
    transverse_axis = vectors[:, 1]
    middle = (np.mean(children[0] @ transverse_axis) + np.mean(children[1] @ transverse_axis)) / 2
    major, axial = cloud @ transverse_axis, cloud @ axis
    width = p["split_valley_width_voxels"] * p["voxel_size"]
    limits = np.quantile(axial, [.05, .95])
    boundaries = np.linspace(*limits, p["split_valley_axial_slices"] + 1)
    slices = []
    band_epsilon = 64 * np.finfo(float).eps * max(width, abs(middle), float(np.max(np.abs(major))))
    for number, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        keep = (axial >= low) & ((axial <= high) if number == len(boundaries) - 2 else (axial < high))
        values = major[keep]
        left, right = values[values < middle], values[values >= middle]
        enough = min(len(left), len(right)) >= p["split_valley_minimum_points_per_side"]
        # 边界上的占据点不能凭浮点比较消失，否则均匀墙面会被算成空隔带。
        gap_count = int(np.count_nonzero(np.abs(values - middle) <= width / 2 + band_epsilon))
        ratio = None
        if enough:
            left_width = max(p["voxel_size"], float(np.diff(np.quantile(left, [.05, .95]))[0]))
            right_width = max(p["voxel_size"], float(np.diff(np.quantile(right, [.05, .95]))[0]))
            shoulder_density = min(len(left) / left_width, len(right) / right_width)
            ratio = (gap_count / width) / shoulder_density
        supported = enough and ratio <= p["split_valley_maximum_density_ratio"]
        slices.append(dict(left_points=len(left), right_points=len(right), gap_points=gap_count,
            density_ratio=ratio, supported=bool(supported)))
    count = sum(item["supported"] for item in slices)
    required = int(np.ceil(p["split_valley_minimum_fraction"] * len(slices)))
    detail = dict(band_width=width, axial_slices=slices, supported_slices=count, required_slices=required,
                  decision="accept_split" if count >= required else "keep_whole_component")
    # 二均值总能把墙切成两半。得先看到持续的稀疏隔带，才有资格叫“两根”。
    if count < required:
        return None, "transverse_valley_not_persistent", detail
    return children, "persistent_transverse_valley", detail


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
        scope="persistent_valley_closed_band_section_diagnostic_not_general_skeleton_or_graph_topology")
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
    pending = [(centers[ids], 0) for ids in connected_indices(grid, p["neighbor_radius_voxels"])]
    diagnostics = dict(connected_components=len(pending), rejected_short_components=0, rejected_nonlinear_components=0,
        rejected_unsupported_sections=0, rejected_side_branches=0, accepted_splits=0, split_decisions={}, sections=[], split_valley_checks=[])
    output = []
    while pending:
        cloud, depth = pending.pop()
        if len(cloud) < p["minimum_component_voxels"]:
            diagnostics["rejected_short_components"] += 1
            continue
        mean, eigen, vectors, axis = principal_axes(cloud)
        if eigen[-1] <= 1e-20 or eigen[-2] / eigen[-1] > p["maximum_second_eigen_ratio"]:
            diagnostics["rejected_nonlinear_components"] += 1
            continue
        children, reason, valley = guarded_split_component(cloud, eigen, vectors, axis, p) if depth < p["maximum_split_depth"] else (None, "depth_budget", None)
        if valley is not None:
            diagnostics["split_valley_checks"].append(valley)
        decisions = diagnostics["split_decisions"]
        decisions[reason] = decisions.get(reason, 0) + 1
        if children is not None:
            pending.extend((child, depth + 1) for child in children)
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
    result.update(components=len(result["segments"]), reason=None if output else "no_supported_curve")
    return result
