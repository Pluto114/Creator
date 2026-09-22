"""Third pilot: stable axial gaps and guarded transverse splitting of straight rods.

All complete candidates use the same voxel occupancy. This remains a diagnostic
reader, not a claim that every connected cloud has a physical centerline.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common_readout import region_mask, sample_segments
from .common_readout_components import merged_segments

DEFAULTS = dict(voxel_size=.01, maximum_second_eigen_ratio=.2, maximum_transverse_voxels=4.,
                minimum_component_voxels=5, minimum_run_length_voxels=4., neighbor_radius_voxels=1.75,
                maximum_voxels=80000, maximum_curve_samples=2000000, merge_angle_degrees=3.,
                merge_distance_voxels=.75, origin=[0., 0., 0.], split_cross_eigen_ratio=3.,
                split_minimum_separation_voxels=3., split_minimum_fraction=.15,
                split_minimum_axial_overlap=.7, maximum_split_depth=3,
                split_maximum_axis_angle_degrees=12., maximum_circle_residual_voxels=.6)


def principal_axes(cloud):
    mean = cloud.mean(axis=0)
    eigen, vectors = np.linalg.eigh((cloud - mean).T @ (cloud - mean) / len(cloud))
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return mean, eigen, vectors, axis


def axial_runs(projection, voxel, neighbor_radius):
    """A real distance gap survives; a floor boundary cannot invent one."""
    ordered = np.sort(projection)
    if not len(ordered):
        return []
    epsilon = 64 * np.finfo(float).eps * max(voxel, float(np.max(np.abs(ordered))))
    return np.split(ordered, np.flatnonzero(np.diff(ordered) > neighbor_radius * voxel + epsilon) + 1)


def split_component(cloud, eigen, vectors, axis, p):
    """Try a deterministic two-mode transverse split; reject weak/short branches."""
    voxel = p["voxel_size"]
    transverse = (cloud - cloud.mean(axis=0)) @ vectors[:, :2]
    if eigen[1] < p["split_cross_eigen_ratio"] * max(eigen[0], .01 * voxel**2):
        return None, "cross_section_not_elongated"
    # 半个圆环也可能看起来又扁又长。若一整个圆已经解释得通，别硬拆成两根。
    design = np.c_[2 * transverse, np.ones(len(cloud))]
    circle, _, rank, _ = np.linalg.lstsq(design, np.sum(transverse**2, axis=1), rcond=None)
    if rank == 3:
        distances = np.linalg.norm(transverse - circle[:2], axis=1)
        residual = np.sqrt(np.mean((distances - distances.mean())**2))
        if residual <= p["maximum_circle_residual_voxels"] * voxel:
            return None, "single_circle_explains_cross_section"
    major = transverse[:, 1]
    means = np.array([np.quantile(major, .25), np.quantile(major, .75)])
    groups = np.zeros(len(cloud), dtype=bool)
    for _ in range(32):
        candidate = major > means.mean()
        if not candidate.any() or candidate.all():
            return None, "empty_split"
        if np.array_equal(candidate, groups):
            break
        groups = candidate
        means = np.array([major[~groups].mean(), major[groups].mean()])
    if np.diff(means)[0] < p["split_minimum_separation_voxels"] * voxel:
        return None, "transverse_modes_too_close"
    if min(groups.mean(), 1 - groups.mean()) < p["split_minimum_fraction"]:
        return None, "minority_too_small"
    children = [cloud[~groups], cloud[groups]]
    if min(map(len, children)) < p["minimum_component_voxels"]:
        return None, "child_too_small"
    intervals = [np.quantile(child @ axis, [.05, .95]) for child in children]
    shared = min(i[1] for i in intervals) - max(i[0] for i in intervals)
    span = max(i[1] for i in intervals) - min(i[0] for i in intervals)
    if span <= 0 or shared / span < p["split_minimum_axial_overlap"]:
        return None, "short_axial_overlap"
    directions = [principal_axes(child)[3] for child in children]
    if min(abs(a @ axis) for a in directions) < np.cos(np.deg2rad(p["split_maximum_axis_angle_degrees"])):
        return None, "inconsistent_child_directions"
    return children, "split_supported"


def readout(points, segments, config, region=None):
    p = {**DEFAULTS, **config}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown split-readout policy")
    integer_keys = ("minimum_component_voxels", "maximum_voxels", "maximum_curve_samples", "maximum_split_depth")
    for key in integer_keys:
        if type(p[key]) is not int or p[key] < (1 if key == "maximum_split_depth" else 2):
            raise ValueError("Positive integer readout budget required")
    for key, value in p.items():
        if key not in integer_keys and key != "origin" and (not np.isfinite(value) or value <= 0):
            raise ValueError("Positive finite readout policy required")
    if not 0 < p["maximum_second_eigen_ratio"] < 1 or p["neighbor_radius_voxels"] > 2 or p["merge_angle_degrees"] >= 90 or p["split_maximum_axis_angle_degrees"] >= 90:
        raise ValueError("Invalid geometric policy")
    if not 0 < p["split_minimum_fraction"] < .5 or not 0 < p["split_minimum_axial_overlap"] <= 1 or p["maximum_split_depth"] > 8:
        raise ValueError("Invalid splitting limits")
    if np.asarray(p["origin"]).shape != (3,) or not np.isfinite(p["origin"]).all():
        raise ValueError("Fixed finite grid origin required")
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Finite Nx3 points and Mx2x3 segments required")
    result = dict(state="complete", config=p, segments=np.empty((0, 2, 3)), input_point_count=len(points), input_segment_count=len(segments),
                  scope="straight_transverse_split_diagnostic_not_general_skeleton_or_graph_topology")
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
    lookup = {tuple(cell): i for i, cell in enumerate(grid)}
    extent = int(np.ceil(p["neighbor_radius_voxels"]))
    offsets = [o for o in itertools.product(range(-extent, extent + 1), repeat=3) if 0 < sum(t*t for t in o) <= p["neighbor_radius_voxels"]**2]
    seen, pending_components, output = set(), [], []
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
        pending_components.append((centers[sorted(component)], 0))
    diagnostics = dict(connected_components=len(pending_components), rejected_short_components=0,
                       rejected_nonlinear_components=0, rejected_wide_components=0, accepted_splits=0, split_decisions={})
    while pending_components:
        cloud, depth = pending_components.pop()
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
            pending_components.extend((child, depth + 1) for child in children)
            diagnostics["accepted_splits"] += 1
            continue
        projection = (cloud - mean) @ axis
        transverse = np.linalg.norm(cloud - mean - projection[:, None] * axis, axis=1)
        if transverse.max() > p["maximum_transverse_voxels"] * p["voxel_size"]:
            diagnostics["rejected_wide_components"] += 1
            continue
        for run in axial_runs(projection, p["voxel_size"], p["neighbor_radius_voxels"]):
            if run[-1] - run[0] >= p["minimum_run_length_voxels"] * p["voxel_size"]:
                output.append(np.array([mean + run[0] * axis, mean + run[-1] * axis]))
    result.update(segments=merged_segments(output, p), components_before_overlap_merge=len(output), **diagnostics)
    result.update(components=len(result["segments"]), reason=None if output else "no_supported_curve")
    return result
