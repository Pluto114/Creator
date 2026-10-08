"""Guide-oriented transverse evidence; output is fitted to actual 3D geometry.

The guide defines a partition direction, never a target location or output axis.
Finite-bin evidence is not a complete multimodality or semantic-identity test.
"""

from __future__ import annotations

import numpy as np

from . import single_axis_readout as previous
from .line_controls import LineFitDegenerate, fit_line_tls

DEFAULTS = {**previous.DEFAULTS, "maximum_unique_points": 2000000,
    "block_width_voxels": 4., "transverse_bin_width_voxels": .5,
    "minimum_local_axial_cells": 2, "minimum_global_axial_cells": 3,
    "transverse_gap_voxels": 1., "persistent_blocks": 3,
    "track_motion_voxels": 2., "radial_mad_multiplier": 3.,
    "minimum_guide_rank_ratio": .05, "maximum_guide_residual_ratio": .2}
SCOPE = "guide_direction_partitioned_local_axis_not_target_identity_or_general_topology"


def canonical(direction):
    direction = np.asarray(direction, float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError("Finite nonzero guide direction required")
    direction = direction/np.linalg.norm(direction)
    return direction if direction[np.argmax(np.abs(direction))] >= 0 else -direction


def guide_direction(views):
    """Intersect image-line planes using normal estimated cameras only."""
    if len(views) < 3 or len({view["view_id"] for view in views}) != len(views):
        raise ValueError("At least three distinct original guide views required")
    normals = []
    for view in views:
        endpoints = np.asarray(view["guide_xyxy"], float)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all() or not view["guide_source"]:
            raise ValueError("Original finite two-endpoint guide and provenance required")
        line = np.cross(np.r_[endpoints[0], 1.], np.r_[endpoints[1], 1.])
        if np.linalg.norm(line[:2]) < 1e-12:
            raise ValueError("Guide endpoints must differ")
        k, e = np.asarray(view["K_index"], float), np.asarray(view["world_to_camera_cv"], float)
        if k.shape != (3, 3) or e.shape != (3, 4) or not np.isfinite(k).all() or not np.isfinite(e).all():
            raise ValueError("Finite pinhole camera matrices required")
        normal = e[:, :3].T @ k.T @ line
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError("Degenerate guide camera plane")
        normals.append(normal/np.linalg.norm(normal))
    _, singular, vt = np.linalg.svd(np.asarray(normals), full_matrices=False)
    if (singular[1] < DEFAULTS["minimum_guide_rank_ratio"]*singular[0]
            or singular[2] > DEFAULTS["maximum_guide_residual_ratio"]*singular[1]):
        return dict(state="unresolved", reason="guide_direction_degenerate", singular_values=singular)
    return dict(state="resolved", reason=None, direction=canonical(vt[-1]), singular_values=singular,
                location_used=False, source="original_image_guides_and_estimated_cameras")


def transverse_frame(points, direction):
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    first = canonical(np.cross(direction, axis))
    second = np.cross(direction, first)
    frame = np.c_[first, second]
    coordinates = points @ frame
    centered = coordinates-coordinates.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    first = canonical(frame @ vectors[:, -1])
    return np.c_[first, np.cross(direction, first)]


def bands(coordinates, axial_cells, size, minimum_cells):
    """Only repeated longitudinal support can bridge transverse bins."""
    bins = np.floor(coordinates/(DEFAULTS["transverse_bin_width_voxels"]*size)).astype(np.int64)
    order = np.argsort(bins, kind="stable")
    occupied, starts = np.unique(bins[order], return_index=True)
    pairs = np.unique(np.c_[bins, axial_cells], axis=0)
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    lows = np.minimum.reduceat(coordinates[order], starts)
    highs = np.maximum.reduceat(coordinates[order], starts)
    reliable = []
    for index, value in enumerate(occupied):
        if counts[index] >= minimum_cells:
            reliable.append((float(lows[index]), float(highs[index]), int(value)))
    groups = []
    for left, right, key in reliable:
        if not groups or left-groups[-1][1] > DEFAULTS["transverse_gap_voxels"]*size:
            groups.append([left, right, [key]])
        else:
            groups[-1][1] = right
            groups[-1][2].append(key)
    return groups, bins


def transverse_evidence(points, size, origin, direction):
    offsets = points-origin
    axial = offsets @ direction
    axial_cells = np.floor(axial/size).astype(np.int64)
    blocks = np.floor(axial/(DEFAULTS["block_width_voxels"]*size)).astype(np.int64)
    frame = transverse_frame(offsets, direction)
    transverse = offsets @ frame
    block_order = np.argsort(blocks, kind="stable")
    block_ids, starts = np.unique(blocks[block_order], return_index=True)
    stops = np.r_[starts[1:], len(block_order)]
    global_counts, local_split_blocks = [], []
    for dimension in range(2):
        global_bands, _ = bands(transverse[:, dimension], axial_cells, size,
                                DEFAULTS["minimum_global_axial_cells"])
        global_counts.append(len(global_bands))
        previous_states, previous_block = [], None
        for block, start, stop in zip(block_ids, starts, stops):
            selected = block_order[start:stop]
            groups, bins = bands(transverse[selected, dimension], axial_cells[selected], size,
                                  DEFAULTS["minimum_local_axial_cells"])
            states = []
            if len(groups) >= 2:
                local_split_blocks.append(dict(dimension=dimension, block=int(block), bands=len(groups)))
                for left, right in zip(groups[:-1], groups[1:]):
                    a = np.median(transverse[selected][np.isin(bins, left[2])], axis=0)
                    b = np.median(transverse[selected][np.isin(bins, right[2])], axis=0)
                    run = 1
                    if previous_block is not None and block == previous_block+1:
                        for old_a, old_b, old_run in previous_states:
                            if max(np.linalg.norm(a-old_a), np.linalg.norm(b-old_b)) <= DEFAULTS["track_motion_voxels"]*size:
                                run = max(run, old_run+1)
                    states.append((a, b, run))
                    if run >= DEFAULTS["persistent_blocks"]:
                        return dict(state="unresolved", reason="persistent_multiple_transverse_tracks",
                            global_band_counts=global_counts, split_blocks=local_split_blocks,
                            persistent_run_blocks=run, frame=frame)
            previous_states, previous_block = states, block
    # Inadequate persistence is NOT evidence that short split structures are one rod.
    if any(count != 1 for count in global_counts):
        return dict(state="unresolved", reason="insufficient_single_transverse_band_evidence",
                    global_band_counts=global_counts, split_blocks=local_split_blocks, frame=frame)
    if local_split_blocks:
        return dict(state="unresolved", reason="nonpersistent_split_transverse_evidence",
                    global_band_counts=global_counts, split_blocks=local_split_blocks, frame=frame)
    return dict(state="resolved", reason=None, global_band_counts=global_counts,
                split_blocks=[], frame=frame, proof_scope="two_projected_histograms_not_general_unimodality")


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
    if evidence["state"] != "resolved":
        return {**result, "reason": evidence["reason"]}
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
    segments, discarded = previous.supported_runs(samples, geometry & rgb,
        DEFAULTS["minimum_length_voxels"]*voxel_size)
    result.update(segments=segments, components=len(segments), axis_sample_count=count,
        observed_projection_extent_m=end-begin, geometry_supported_samples=int(geometry.sum()),
        rgb_supported_samples=int(rgb.sum()), jointly_supported_samples=int((geometry & rgb).sum()),
        discarded_short_runs=discarded, resolution_state="resolved" if len(segments) else "unresolved",
        reason=None if len(segments) else "no_jointly_supported_finite_run")
    return result
