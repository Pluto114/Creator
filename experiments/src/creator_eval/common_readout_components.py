"""Straight-component common readout, a second pilot after local-link fragmentation.

Voxel connectivity precedes one PCA per component. Only occupied axial bins are
reported. No target labels, GT axes or privileged treatment of added curves.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common_readout import region_mask, sample_segments

DEFAULTS = dict(voxel_size=.01, maximum_second_eigen_ratio=.2, maximum_transverse_voxels=4.,
                minimum_component_voxels=5, minimum_run_length_voxels=4.,
                neighbor_radius_voxels=1.75, maximum_voxels=80000, maximum_curve_samples=2000000,
                merge_angle_degrees=3., merge_distance_voxels=.75, origin=[0., 0., 0.])


def merged_segments(segments, p):
    """Deduplicate near-collinear overlap; never close a positive axial gap."""
    output = []
    ordered = sorted(segments, key=lambda s: (-np.linalg.norm(s[1] - s[0]), tuple(s.reshape(-1))))
    for segment in ordered:
        a, b = segment
        direction = (b - a) / np.linalg.norm(b - a)
        merged = False
        for i, current in enumerate(output):
            first, last = current
            axis = (last - first) / np.linalg.norm(last - first)
            if abs(direction @ axis) < np.cos(np.deg2rad(p["merge_angle_degrees"])):
                continue
            delta = segment - first
            coordinates = delta @ axis
            separation = np.linalg.norm(delta - coordinates[:, None] * axis, axis=1)
            length = np.linalg.norm(last - first)
            if separation.max() > p["merge_distance_voxels"] * p["voxel_size"] or min(coordinates) > length or max(coordinates) < 0:
                continue
            output[i] = np.array([first + min(0., min(coordinates)) * axis, first + max(length, max(coordinates)) * axis])
            merged = True
            break
        if not merged:
            output.append(segment)
    return np.asarray(output).reshape(-1, 2, 3)


def readout(points, segments, config, region=None):
    p = {**DEFAULTS, **config}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown straight-component readout parameter")
    for key in ("voxel_size", "maximum_transverse_voxels", "minimum_run_length_voxels", "neighbor_radius_voxels", "merge_angle_degrees", "merge_distance_voxels"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError("Positive finite component policy required")
    for key in ("minimum_component_voxels", "maximum_voxels", "maximum_curve_samples"):
        if type(p[key]) is not int or p[key] < 2:
            raise ValueError("Integer component budgets required")
    if not 0 < p["maximum_second_eigen_ratio"] < 1 or p["neighbor_radius_voxels"] > 2 or p["merge_angle_degrees"] >= 90:
        raise ValueError("Invalid component geometry policy")
    if np.asarray(p["origin"]).shape != (3,) or not np.isfinite(p["origin"]).all():
        raise ValueError("Fixed finite grid origin required")
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Finite point/segment arrays required")
    result = dict(state="complete", config=p, segments=np.empty((0, 2, 3)), input_point_count=len(points), input_segment_count=len(segments),
                  scope="straight_connected_component_readout_not_general_skeleton_or_graph_topology")
    try:
        samples = sample_segments(segments, p["voxel_size"] / 2, p["maximum_curve_samples"])
    except OverflowError as error:
        return {**result, "state": "unmeasurable", "reason": str(error)}
    values = np.concatenate((points, samples))
    keep = region_mask(values, region)
    result.update(region_point_count=int(keep[:len(points)].sum()), region_curve_sample_count=int(keep[len(points):].sum()),
                  region_policy="all_input" if region is None else "fixed_RGB_strip_union_distinct_view_vote")
    values = values[keep]
    scaled = (values - p["origin"]) / p["voxel_size"]
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
    seen, output = set(), []
    diagnostics = dict(connected_components=0, rejected_short_components=0, rejected_nonlinear_components=0, rejected_wide_components=0)
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
        diagnostics["connected_components"] += 1
        if len(component) < p["minimum_component_voxels"]:
            diagnostics["rejected_short_components"] += 1
            continue
        cloud = centers[sorted(component)]
        mean = cloud.mean(axis=0)
        eigen, vectors = np.linalg.eigh((cloud - mean).T @ (cloud - mean))
        if eigen[-1] <= 1e-20 or eigen[-2] / eigen[-1] > p["maximum_second_eigen_ratio"]:
            diagnostics["rejected_nonlinear_components"] += 1
            continue
        axis = vectors[:, -1]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        t = (cloud - mean) @ axis
        transverse = np.linalg.norm(cloud - mean - t[:, None] * axis, axis=1)
        if transverse.max() > p["maximum_transverse_voxels"] * p["voxel_size"]:
            diagnostics["rejected_wide_components"] += 1
            continue
        bins = np.floor(t / p["voxel_size"]).astype(np.int64)
        occupied = np.unique(bins)
        # 看不到的长区间不连。体素以下的小缝仍不可分辨，要把分辨率限制说清楚。
        for run in np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1):
            observed = t[np.isin(bins, run)]
            low, high = observed.min(), observed.max()
            if high - low >= p["minimum_run_length_voxels"] * p["voxel_size"]:
                output.append(np.array([mean + low * axis, mean + high * axis]))
    result.update(segments=merged_segments(output, p), components_before_overlap_merge=len(output), **diagnostics)
    result.update(components=len(result["segments"]), reason=None if output else "no_supported_curve")
    return result
