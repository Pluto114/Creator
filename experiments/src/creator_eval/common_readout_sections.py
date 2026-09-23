"""Experimental common reader with robust circular sections and explicit side-branch rejection.

All geometry first becomes the same occupied voxels. Circle fitting estimates a
surface axis; it is not a privileged shortcut for input curve primitives. This
reader still assumes straight members and must earn qualification independently.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common_readout import region_mask, sample_segments
from .common_readout_components import merged_segments
from .common_readout_split import DEFAULTS as SPLIT_DEFAULTS
from .common_readout_split import axial_runs, principal_axes, split_component

DEFAULTS = {k: v for k, v in SPLIT_DEFAULTS.items() if k != "maximum_transverse_voxels"}
DEFAULTS.update(circle_trials=128, circle_residual_voxels=.85, minimum_circle_radius_voxels=1.25,
                maximum_radius_span_fraction=.25, minimum_circle_fraction=.55,
                minimum_circle_coverage_degrees=120., maximum_line_radius_voxels=2.,
                branch_minimum_length_voxels=6., branch_minimum_angle_degrees=20.,
                branch_maximum_second_eigen_ratio=.12)


def connected_indices(grid, radius):
    lookup = {tuple(cell): i for i, cell in enumerate(grid)}
    extent = int(np.ceil(radius))
    offsets = [o for o in itertools.product(range(-extent, extent + 1), repeat=3)
               if 0 < sum(t*t for t in o) <= radius**2]
    seen, output = set(), []
    for start in range(len(grid)):
        if start in seen:
            continue
        pending, component = [start], []
        seen.add(start)
        while pending:
            i = pending.pop()
            component.append(i)
            for delta in offsets:
                j = lookup.get(tuple(grid[i] + delta))
                if j is not None and j not in seen:
                    seen.add(j)
                    pending.append(j)
        output.append(np.asarray(sorted(component)))
    return output


def algebraic_circle(values):
    design = np.c_[2 * values, np.ones(len(values))]
    solution, _, rank, _ = np.linalg.lstsq(design, np.sum(values**2, axis=1), rcond=None)
    radius2 = solution[2] + solution[:2] @ solution[:2]
    if rank < 3 or not np.isfinite(radius2) or radius2 <= 0:
        return None
    return solution[:2], float(np.sqrt(radius2))


def circle_section(values, span, p):
    """Fit a deterministic consensus circle; no GT radius or target axis is read."""
    voxel, tolerance = p["voxel_size"], p["circle_residual_voxels"] * p["voxel_size"]
    random = np.random.default_rng(0)
    proposals = [algebraic_circle(values)]
    proposals.extend(algebraic_circle(values[random.choice(len(values), 3, replace=False)]) for _ in range(p["circle_trials"]))
    best, best_key = None, None
    for proposal in proposals:
        if proposal is None:
            continue
        center, radius = proposal
        if not p["minimum_circle_radius_voxels"] * voxel <= radius <= p["maximum_radius_span_fraction"] * span:
            continue
        residual = np.abs(np.linalg.norm(values - center, axis=1) - radius)
        inside = residual <= tolerance
        if inside.mean() < p["minimum_circle_fraction"]:
            continue
        key = (int(inside.sum()), -float(np.median(residual[inside])))
        if best_key is None or key > best_key:
            best, best_key = proposal, key
    if best is None:
        return None
    center, radius = best
    for _ in range(4):
        residual = np.abs(np.linalg.norm(values - center, axis=1) - radius)
        inside = residual <= tolerance
        fitted = algebraic_circle(values[inside])
        if fitted is None:
            return None
        center, radius = fitted
    residual = np.abs(np.linalg.norm(values - center, axis=1) - radius)
    inside = residual <= tolerance
    if inside.sum() < 3 or inside.mean() < p["minimum_circle_fraction"]:
        return None
    angles = np.sort(np.arctan2(*(values[inside] - center).T[::-1]))
    coverage = 2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
    if inside.mean() < p["minimum_circle_fraction"] or coverage < np.deg2rad(p["minimum_circle_coverage_degrees"]):
        return None
    if not p["minimum_circle_radius_voxels"] * voxel <= radius <= p["maximum_radius_span_fraction"] * span:
        return None
    return center, radius, inside, dict(radius=radius, inlier_fraction=float(inside.mean()),
        angular_coverage_degrees=float(np.rad2deg(coverage)), median_residual=float(np.median(residual[inside])))


def coherent_side_branch(cloud, inside, axis, radius, p):
    """Ignore scattered contamination, but do not silently erase a long side arm."""
    outside = cloud[~inside]
    if len(outside) < p["minimum_component_voxels"]:
        return False
    grid = np.floor((outside - p["origin"]) / p["voxel_size"]).astype(np.int64)
    for indices in connected_indices(grid, p["neighbor_radius_voxels"]):
        part = outside[indices]
        if len(part) < p["minimum_component_voxels"]:
            continue
        _, eigen, _, direction = principal_axes(part)
        span = np.ptp(part @ direction)
        if (span >= max(p["branch_minimum_length_voxels"] * p["voxel_size"], 2 * radius)
                and eigen[-1] > 1e-20 and eigen[-2] / eigen[-1] <= p["branch_maximum_second_eigen_ratio"]
                and abs(direction @ axis) < np.cos(np.deg2rad(p["branch_minimum_angle_degrees"]))):
            return True
    return False


def readout(points, segments, config, region=None):
    p = {**DEFAULTS, **config}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown section-readout policy")
    integer_keys = ("minimum_component_voxels", "maximum_voxels", "maximum_curve_samples", "maximum_split_depth", "circle_trials")
    for key in integer_keys:
        if type(p[key]) is not int or not 1 <= p[key] <= (4096 if key == "circle_trials" else 2000000):
            raise ValueError("Bounded positive integer reader policy required")
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
        scope="robust_straight_surface_section_diagnostic_not_general_skeleton_or_graph_topology")
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
        rejected_unsupported_sections=0, rejected_side_branches=0, accepted_splits=0, split_decisions={}, sections=[])
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
        children, reason = split_component(cloud, eigen, vectors, axis, p) if depth < p["maximum_split_depth"] else (None, "depth_budget")
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
