"""Deterministic common curve readout for complete point + segment candidates.

Fixed world grid, voxel occupancy, local PCA and reciprocal directional links.
No GT, method label, line-ID shortcut or candidate-dependent scale is accepted.
This is an evaluator adapter; it can fail and is not a reconstruction algorithm.
"""
from __future__ import annotations

import itertools

import numpy as np

DEFAULTS = dict(voxel_size=.01, neighborhood_radius_voxels=3.0, link_radius_voxels=2.1,
                maximum_transverse_voxels=.8, minimum_neighbors=5, maximum_second_eigen_ratio=.2,
                maximum_tangent_angle_deg=25., minimum_component_length_voxels=4., maximum_voxels=80000,
                maximum_curve_samples=2000000, origin=[0., 0., 0.])


def policy(config):
    p = {**DEFAULTS, **config}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown common readout parameter")
    for key in ("voxel_size", "neighborhood_radius_voxels", "link_radius_voxels", "maximum_transverse_voxels", "minimum_component_length_voxels"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError("Finite positive readout scales required")
    for key in ("minimum_neighbors", "maximum_voxels", "maximum_curve_samples"):
        if type(p[key]) is not int or p[key] < 2:
            raise ValueError("Integer readout budgets required")
    if not 0 < p["maximum_second_eigen_ratio"] < 1 or not 0 < p["maximum_tangent_angle_deg"] < 90:
        raise ValueError("Invalid linearity/angle policy")
    if np.asarray(p["origin"]).shape != (3,) or not np.isfinite(p["origin"]).all():
        raise ValueError("Fixed finite world origin required")
    if p["neighborhood_radius_voxels"] > 8 or p["link_radius_voxels"] > p["neighborhood_radius_voxels"]:
        raise ValueError("Neighborhood policy exceeds the local-grid budget")
    return p


def sample_segments(segments, spacing, maximum_samples):
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("Zero-length input segment")
    counts = np.maximum(1, np.ceil(lengths / spacing)).astype(np.int64)
    if int(np.sum(counts + 1)) > maximum_samples:
        raise OverflowError("curve_sample_budget_exceeded")
    # 曲线也先变成采样点，再走同一个格子。不能直接把新增杆原样当读取成功。
    samples = [a + np.linspace(0, 1, n + 1)[:, None] * (b - a) for (a, b), n in zip(segments, counts)]
    return np.concatenate(samples) if samples else np.empty((0, 3))


def region_mask(points, region):
    """One frozen union of RGB guide strips, shared by base and complete candidate."""
    if region is None:
        return np.ones(len(points), dtype=bool)
    if set(region) != {"minimum_views", "views"} or type(region["minimum_views"]) is not int or not 1 <= region["minimum_views"] <= len(region["views"]):
        raise ValueError("Invalid region voting policy")
    counts = np.zeros(len(points), dtype=np.int32)
    identifiers = []
    for view in region["views"]:
        identifiers.append(view["view_id"])
        k, e = np.asarray(view["K_index"], float), np.asarray(view["world_to_camera_cv"], float)
        if k.shape != (3, 3) or e.shape != (3, 4) or not np.isfinite(k).all() or not np.isfinite(e).all():
            raise ValueError("Region requires explicit finite cameras")
        xyz = points @ e[:, :3].T + e[:, 3]
        image = xyz @ k.T
        valid = image[:, 2] > 1e-10
        xy = np.zeros((len(points), 2))
        xy[valid] = image[valid, :2] / image[valid, 2, None]
        inside = np.zeros(len(points), dtype=bool)
        for guide in view["guides"]:
            ends, width = np.asarray(guide["xyxy"], float), guide["half_width_px"]
            if ends.shape != (2, 2) or not np.isfinite(ends).all() or not np.isfinite(width) or width <= 0 or ends[1, 1] <= ends[0, 1]:
                raise ValueError("Ordered vertical guide interval required")
            expected = ends[0, 0] + (xy[:, 1] - ends[0, 1]) * (ends[1, 0] - ends[0, 0]) / (ends[1, 1] - ends[0, 1])
            inside |= valid & (xy[:, 1] >= ends[0, 1]) & (xy[:, 1] <= ends[1, 1]) & (np.abs(xy[:, 0] - expected) <= width)
        counts += inside
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate region view IDs")
    return counts >= region["minimum_views"]


def readout(points, segments, config, region=None):
    p = policy(config)
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Finite Nx3 points and Mx2x3 segments required")
    result = dict(state="complete", config=p, segments=np.empty((0, 2, 3)),
                  input_point_count=len(points), input_segment_count=len(segments),
                  scope="common_point_and_curve_readout_not_graph_topology_or_surface_recovery")
    try:
        samples = sample_segments(segments, p["voxel_size"] / 2, p["maximum_curve_samples"])
    except OverflowError as error:
        return {**result, "state": "unmeasurable", "reason": str(error)}
    values = np.concatenate((points, samples))
    mask = region_mask(values, region)
    result.update(region_point_count=int(mask[:len(points)].sum()), region_curve_sample_count=int(mask[len(points):].sum()), region_policy="all_input" if region is None else "fixed_RGB_strip_union_distinct_view_vote")
    values = values[mask]
    if not len(values):
        return {**result, "reason": "empty_input", "occupied_voxels": 0, "components": 0}
    scaled = (values - p["origin"]) / p["voxel_size"]
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**52):
        raise ValueError("Coordinates exceed exact voxel-index precision")
    grid = np.unique(np.floor(scaled).astype(np.int64), axis=0)
    result["occupied_voxels"] = len(grid)
    if len(grid) > p["maximum_voxels"]:
        return {**result, "state": "unmeasurable", "reason": "occupied_voxel_budget_exceeded"}
    centers = (grid.astype(float) + .5) * p["voxel_size"] + p["origin"]
    lookup = {tuple(cell): i for i, cell in enumerate(grid)}
    radius = p["neighborhood_radius_voxels"]
    extent = int(np.ceil(radius))
    offsets = [o for o in itertools.product(range(-extent, extent + 1), repeat=3) if sum(t * t for t in o) <= radius**2 + 1e-10]
    neighbors, tangents = [], np.zeros_like(centers)
    nodes, eligible = centers.copy(), np.zeros(len(grid), dtype=bool)
    for i, cell in enumerate(grid):
        adjacent = np.array(sorted(lookup[k] for delta in offsets if (k := tuple(cell + delta)) in lookup), dtype=int)
        neighbors.append(adjacent)
        if len(adjacent) < p["minimum_neighbors"]:
            continue
        patch = centers[adjacent]
        mean = patch.mean(axis=0)
        values, vectors = np.linalg.eigh((patch - mean).T @ (patch - mean))
        if values[-1] <= 1e-20 or values[-2] / values[-1] > p["maximum_second_eigen_ratio"]:
            continue
        tangent = vectors[:, -1]
        if tangent[np.argmax(np.abs(tangent))] < 0:
            tangent = -tangent
        tangents[i] = tangent
        nodes[i] = mean + np.dot(centers[i] - mean, tangent) * tangent
        eligible[i] = True
    cos_limit = np.cos(np.deg2rad(p["maximum_tangent_angle_deg"]))
    choices = [set() for _ in grid]
    for i in np.flatnonzero(eligible):
        candidates = [j for j in neighbors[i] if j != i and eligible[j]]
        for sign in (-1, 1):
            supported = []
            for j in candidates:
                delta = nodes[j] - nodes[i]
                distance = float(np.linalg.norm(delta))
                along = float(delta @ tangents[i])
                across = np.linalg.norm(delta - along * tangents[i])
                if sign * along <= 1e-10 or distance > p["link_radius_voxels"] * p["voxel_size"] or across > p["maximum_transverse_voxels"] * p["voxel_size"]:
                    continue
                if abs(tangents[i] @ tangents[j]) < cos_limit or abs(delta @ tangents[j]) / max(distance, 1e-20) < cos_limit:
                    continue
                supported.append((distance, j))
            if supported:
                choices[i].add(min(supported)[1])
    edges = {(i, j) for i, selected in enumerate(choices) for j in selected if i < j and i in choices[j]}
    adjacency = {i: set() for edge in edges for i in edge}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    seen, segments_out = set(), []
    kept, discarded, cycles = 0, 0, 0
    for start in sorted(adjacency):
        if start in seen:
            continue
        pending, component = [start], set()
        while pending:
            i = pending.pop()
            if i in component:
                continue
            component.add(i)
            pending.extend(adjacency[i] - component)
        seen.update(component)
        endpoints = sorted(i for i in component if len(adjacency[i]) == 1)
        if len(endpoints) != 2 or any(len(adjacency[i]) > 2 for i in component):
            cycles += 1
            continue
        path, previous, current = [], None, endpoints[0]
        while current is not None:
            path.append(current)
            following = sorted(adjacency[current] - ({previous} if previous is not None else set()))
            previous, current = current, (following[0] if following else None)
        polyline = nodes[path]
        length = np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum()
        if length < p["minimum_component_length_voxels"] * p["voxel_size"]:
            discarded += 1
            continue
        kept += 1
        segments_out.extend(np.stack((polyline[:-1], polyline[1:]), axis=1))
    result.update(segments=np.asarray(segments_out).reshape(-1, 2, 3), components=kept,
                  discarded_short_components=discarded, rejected_nonchain_components=cycles,
                  eligible_voxels=int(eligible.sum()), reciprocal_edges=len(edges),
                  quantization_policy="occupied_voxel_centers_then_local_axis_projection; multiplicity_does_not_change_occupancy",
                  reason=None if kept else "no_supported_curve")
    return result
