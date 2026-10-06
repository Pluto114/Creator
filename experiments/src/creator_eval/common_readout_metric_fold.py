"""Fixed-partition ridge extraction with unchanged upstream section ownership.

Point and sampled-segment geometry is unified before fitting. Input primitive
identities are never used to nominate or accept a line. This is a research
reader for a locally straight ROI, not a general skeleton or identity model.
"""

from __future__ import annotations

import numpy as np

from . import ridge_bundle_evidence as ridge_local_evidence
from . import ridge_sampling_evidence
from . import ridge_sampling_proposals as ridge_proposals
from . import section_metric_evidence as section_evidence
from .common_readout import region_mask, sample_segments
from .common_readout_components import merged_segments
from .common_readout_split import axial_runs, principal_axes

DEFAULTS = dict(voxel_size=.01, origin=[0., 0., 0.], maximum_voxels=80000,
    maximum_curve_samples=2000000, maximum_unique_points=200000, maximum_raw_points=2000000,
    maximum_ridge_runs=1024, maximum_output_segments=1024, trials=192,
    maximum_transverse_proposals=96, ridge_radius_voxels=.5, inner_radius_voxels=.15, expanded_radius_ratio=2.,
    minimum_run_length_voxels=12., minimum_axial_coverage=.8,
    minimum_validation_coverage=.3, peer_radius_voxels=3., peer_coverage_fraction=.6,
    neighbor_radius_voxels=1.75, maximum_second_eigen_ratio=.2,
    parallel_angle_degrees=12., parallel_overlap_fraction=.7,
    merge_angle_degrees=3., merge_distance_voxels=.5)
DEFAULTS.update(section_evidence.DEFAULTS)
DEFAULTS.update(ridge_proposals.DEFAULTS)
DEFAULTS.update(ridge_local_evidence.DEFAULTS)
DEFAULTS.update(ridge_sampling_evidence.DEFAULTS)
# Independent salted ridge folds; the unchanged section ownership is computed
# with the original partition, and cannot be expanded by a later ridge fold.
DEFAULTS["local_proposal_small_neighbors"] = 6
DEFAULTS["ridge_fold_index"] = 0


def policy(config):
    p = {**DEFAULTS, **config}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown ridge-reader parameter")
    integers = {"maximum_voxels", "maximum_curve_samples", "maximum_unique_points", "maximum_raw_points", "maximum_ridge_runs", "maximum_output_segments", "trials",
                "maximum_transverse_proposals"}
    if type(p["ridge_fold_index"]) is not int or not 0 <= p["ridge_fold_index"] <= 2:
        raise ValueError("Fixed fold index 0, 1 or 2 required")
    section_evidence.policy(p)
    ridge_proposals.policy(p)
    ridge_local_evidence.policy(p)
    ridge_sampling_evidence.policy(p)
    for key, value in p.items():
        if key in section_evidence.DEFAULTS or key in ridge_proposals.DEFAULTS or key in ridge_local_evidence.DEFAULTS or key in ridge_sampling_evidence.DEFAULTS:
            continue
        if key == "ridge_fold_index":
            continue
        if key == "origin":
            if np.asarray(value).shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Fixed finite world origin required")
        elif key in integers:
            if type(value) is not int or not 1 <= value <= 2000000:
                raise ValueError("Bounded positive integer budgets required")
        elif not np.isfinite(value) or value <= 0:
            raise ValueError("Positive finite geometry policy required")
    if any(p[key] > 1 for key in ("minimum_axial_coverage", "minimum_validation_coverage", "peer_coverage_fraction", "parallel_overlap_fraction")):
        raise ValueError("Fractions cannot exceed one")
    if (not 0 < p["maximum_second_eigen_ratio"] < 1 or p["expanded_radius_ratio"] <= 1
            or p["peer_radius_voxels"] <= p["ridge_radius_voxels"]*p["expanded_radius_ratio"]
            or p["trials"] > 4096
            or p["maximum_transverse_proposals"] > 4096 or p["merge_angle_degrees"] >= 90
            or p["inner_radius_voxels"] >= p["ridge_radius_voxels"]
            or p["parallel_angle_degrees"] >= 90
            or p["maximum_ridge_runs"] > 4096 or p["maximum_output_segments"] > 4096):
        raise ValueError("Invalid ridge/circle geometry policy")
    if not np.isfinite(p["voxel_size"]*p["inner_radius_voxels"]) or p["voxel_size"]*p["inner_radius_voxels"] <= 0:
        raise ValueError("Unrepresentable inner geometry scale")
    return p


def _distance(points, anchor, axis):
    delta = points-anchor
    projection = delta @ axis
    return projection, np.linalg.norm(delta-projection[:, None]*axis, axis=1)


def _resolution(segments, sections, ambiguous=False):
    if not len(segments):
        return "unresolved"
    unresolved = ambiguous or any(
        group.get("state") == "surface_unresolved"
        or any(not run["accepted"] for run in group.get("runs", []))
        for group in sections.get("groups", []))
    return "partially_resolved" if unresolved else "resolved"


def _training_mask(grid, fold=0):
    """Stable nonlinear voxel hash; linear parity aliases diagonal lattice lines."""
    cells = grid.astype(np.uint64)
    value = (cells[:, 0]*np.uint64(0x9E3779B185EBCA87)
             ^ cells[:, 1]*np.uint64(0xC2B2AE3D27D4EB4F)
             ^ cells[:, 2]*np.uint64(0x165667B19E3779F9))
    value ^= np.uint64((fold*0xD6E8FEB86659FD93) & ((1 << 64)-1))
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return (value & np.uint64(1)) == 0


def _fit_ridge(train, anchor, axis, radius, inner):
    for selected_radius in [radius]*4+[inner]*2:
        _, distance = _distance(train, anchor, axis)
        inside = train[distance <= selected_radius]
        if len(inside) < 3:
            return None
        anchor, eigen, _, axis = principal_axes(inside)
        if eigen[-1] <= 1e-20:
            return None
    return anchor, axis


def _ridge_runs(points, train_mask, anchor, axis, p):
    voxel, radius = p["voxel_size"], p["ridge_radius_voxels"]*p["voxel_size"]
    inner = p["inner_radius_voxels"]*voxel
    t, distance = _distance(points, anchor, axis)
    core = distance <= inner
    scale = ridge_sampling_evidence.train_scale(t[core & train_mask], p)
    audit = dict(state=scale["state"], sampling_scale=scale, gate_counts={},
                 observed_runs=0, measured_runs=[])
    if scale["state"] != "complete":
        audit["reason"] = scale["reason"]
        return [], [], audit
    width = scale["bin_width_m"]
    seed = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    transverse_all = (points-anchor) @ np.column_stack((u, np.cross(axis, u)))
    details, output = [], []
    # Only the axis sampling metric changes; radial tubes and actual endpoints
    # remain physical, and no point is synthesized across an observed break.
    for run in axial_runs(t[core], width, scale["gap_limit_m"]/width):
        audit["observed_runs"] += 1
        low, high = float(run[0]), float(run[-1])
        span = high-low
        if span < p["minimum_run_length_voxels"]*voxel:
            key = "below_minimum_physical_length"
            audit["gate_counts"][key] = audit["gate_counts"].get(key, 0)+1
            continue
        window = (t >= low) & (t <= high)
        support = core & window
        # Cadence cells are centered on observed sample stations, not their
        # boundaries: tiny axial jitter must not collapse adjacent stations.
        phase = .5 if scale["learning_state"] == "training_cadence" else 0.
        bins = np.floor((t-low)/width+phase).astype(np.int64)
        expected = max(1, int(np.floor(span/width+phase))+1)
        coverage = len(np.unique(bins[support]))/expected
        validation = len(np.unique(bins[support & ~train_mask]))/expected
        fitting = len(np.unique(bins[support & train_mask]))/expected
        expanded = window & (distance <= radius*p["expanded_radius_ratio"])
        occupancy = np.column_stack((bins, np.floor(transverse_all/inner).astype(np.int64)))
        fraction = len(np.unique(occupancy[support], axis=0))/max(1, len(np.unique(occupancy[expanded], axis=0)))
        reason = ("axial_coverage" if coverage < p["minimum_axial_coverage"]
                  else "independent_fold_coverage" if min(validation, fitting) < p["minimum_validation_coverage"]
                  else None)
        if reason is None:
            transverse = transverse_all[window]
            cells, inverse = np.unique(np.floor(transverse/voxel).astype(np.int64), axis=0, return_inverse=True)
            for i in range(len(cells)):
                members = transverse[inverse == i]
                mean = members.mean(axis=0)
                center = members[np.argmin(np.linalg.norm(members-mean, axis=1))]
                separation = np.linalg.norm(center)
                if not 2*inner <= separation <= p["peer_radius_voxels"]*voxel:
                    continue
                peer = np.linalg.norm(transverse-center, axis=1) <= inner
                peer_bins = len(np.unique(bins[window][peer]))
                if peer_bins >= p["peer_coverage_fraction"]*len(np.unique(bins[support])):
                    reason = "persistent_local_parallel_peer"
                    break
        row = dict(low=low, high=high, span=span, axial_coverage=coverage,
                   train_coverage=fitting, validation_coverage=validation,
                   core_fraction=fraction, core_fraction_is_diagnostic=True,
                   bin_width_m=width, gap_limit_m=scale["gap_limit_m"],
                   accepted=reason is None, reason=reason, parallel_peer=reason == "persistent_local_parallel_peer")
        key = reason or "accepted"
        audit["gate_counts"][key] = audit["gate_counts"].get(key, 0)+1
        audit["measured_runs"].append(row)
        if reason is None:
            output.append(np.array([anchor+low*axis, anchor+high*axis]))
            details.append(row)
    return output, details, audit


def _parallel_ambiguity(ridges, p):
    """Reject members of three-way separated, coextensive parallel peak groups.

    Three disconnected runs on one axis are not three surface mother-lines.
    Unrelated nonparallel lines neither cause nor cancel this local ambiguity.
    """
    adjacent = np.zeros((len(ridges), len(ridges)), dtype=bool)
    for i, first in enumerate(ridges):
        axis = first[1]-first[0]
        length = np.linalg.norm(axis)
        axis /= length
        for j in range(i+1, len(ridges)):
            second = ridges[j]
            other = second[1]-second[0]
            other /= np.linalg.norm(other)
            if abs(axis @ other) < np.cos(np.deg2rad(p["parallel_angle_degrees"])):
                continue
            t, distance = _distance(second, first[0], axis)
            overlap = max(0., min(length, t.max())-max(0., t.min()))
            if (distance.min() >= 2*p["inner_radius_voxels"]*p["voxel_size"]
                    and overlap >= p["parallel_overlap_fraction"]*min(length, np.ptp(t))):
                adjacent[i, j] = adjacent[j, i] = True
    rejected = np.zeros(len(ridges), dtype=bool)
    for i in range(len(ridges)):
        for j in np.flatnonzero(adjacent[i]):
            common = adjacent[i] & adjacent[j]
            if common.any():
                rejected[i] = rejected[j] = True
                rejected |= common
    return rejected


def readout(points, segments, config, region=None):
    p = policy(config)
    points, segments = np.asarray(points, float), np.asarray(segments, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Finite Nx3 points and Mx2x3 segments required")
    result = dict(state="complete", config=p, segments=np.empty((0, 2, 3)), input_point_count=len(points),
        input_segment_count=len(segments), resolution_state="no_supported_curve", components=0,
        scope="training_cadence_ridges_with_layer_local_sections_not_general_identity_or_topology")
    if len(points) > p["maximum_raw_points"]:
        return {**result, "state": "unmeasurable", "reason": "raw_point_budget_exceeded"}
    # Check endpoints and floating-point sample estimates before any int64 cast
    # or allocation in the shared sampler. Oversize inputs are not empty success.
    for geometry in (points, segments.reshape(-1, 3)):
        with np.errstate(over="ignore", invalid="ignore"):
            coordinates = 4*(geometry-p["origin"])/(p["voxel_size"]*min(1., p["inner_radius_voxels"]))
        if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) >= 2**52):
            raise ValueError("Coordinates exceed exact voxel-index precision")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sample_estimate = np.sum(np.ceil(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)/(p["voxel_size"]/2))+1)
    if not np.isfinite(sample_estimate) or sample_estimate > p["maximum_curve_samples"]:
        return {**result, "state": "unmeasurable", "reason": "curve_sample_budget_exceeded"}
    try:
        samples = sample_segments(segments, p["voxel_size"]/2, p["maximum_curve_samples"])
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    values = np.concatenate((points, samples))
    values = np.unique(values[region_mask(values, region)], axis=0)
    scaled = (values-p["origin"])/p["voxel_size"]
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**52):
        raise ValueError("Coordinates exceed exact voxel-index precision")
    grid = np.floor(scaled).astype(np.int64)
    occupied = len(np.unique(grid, axis=0))
    result.update(unique_points=len(values), occupied_voxels=occupied)
    if occupied > p["maximum_voxels"] or len(values) > p["maximum_unique_points"]:
        return {**result, "state": "unmeasurable", "reason": "geometry_budget_exceeded"}
    if len(values) < 6:
        return result
    # Every source point in the same cell shares a fold; duplication cannot leak it.
    train_mask = _training_mask(grid)
    if min(train_mask.sum(), (~train_mask).sum()) < 3:
        return {**result, "resolution_state": "unresolved", "reason": "insufficient_separate_validation_cells"}
    train = values[train_mask]
    anchor, eigen, vectors, axis = principal_axes(train)
    if eigen[-1] <= 1e-20 or eigen[-2]/eigen[-1] > p["maximum_second_eigen_ratio"]:
        return {**result, "resolution_state": "unresolved", "reason": "not_a_straight_ROI"}
    surface, consumed, section_detail = section_evidence.analyze(values, train_mask, p)
    if section_detail.get("state") == "unmeasurable":
        return {**result, "state": "unmeasurable", "reason": section_detail["reason"], "section_evidence": section_detail}
    if len(surface) > p["maximum_output_segments"]:
        return {**result, "state": "unmeasurable", "reason": "output_segment_budget_exceeded"}
    # Accepted surfaces own their shell/center samples. Locally ambiguous
    # surfaces also consume their support: failed sections must not reappear
    # as arbitrary short mother-lines through ridge fallback.
    ridge_values = values[~consumed]
    # Section ownership stays upstream and unchanged; each additional ridge
    # fold nominates and fits only from its own fixed hash partition.
    ridge_mask = _training_mask(grid[~consumed], p["ridge_fold_index"])
    train = ridge_values[ridge_mask]
    if min(ridge_mask.sum(), (~ridge_mask).sum()) < 3:
        chosen = np.asarray(surface).reshape(-1, 2, 3)
        return {**result, "segments": chosen, "components": len(chosen),
                "section_evidence": section_detail, "surface_consumed_points": int(consumed.sum()),
                "training_points": int(train_mask.sum()), "validation_points": int((~train_mask).sum()),
                "resolution_state": _resolution(chosen, section_detail)}
    if p["ridge_fold_index"]:
        anchor, _, vectors, axis = principal_axes(train)
    proposals = [(anchor, axis)]
    transverse = (train-anchor) @ vectors[:, :2]
    _, inverse, counts = np.unique(np.floor(transverse/p["voxel_size"]).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    for i in np.argsort(-counts, kind="stable")[:p["maximum_transverse_proposals"]]:
        proposals.append((train[inverse == i].mean(axis=0), axis))
    random = np.random.default_rng(0)
    for _ in range(p["trials"]):
        a, b = train[random.choice(len(train), 2, replace=False)]
        delta = b-a
        length = np.linalg.norm(delta)
        if length >= p["minimum_run_length_voxels"]*p["voxel_size"]:
            proposals.append((a, delta/length))
    local, local_detail = ridge_proposals.propose(train, p)
    if local_detail["state"] == "unmeasurable":
        return {**result, "state": "unmeasurable", "reason": local_detail["reason"],
                "local_proposal_evidence": local_detail, "section_evidence": section_detail}
    proposals.extend(local)
    radius = p["ridge_radius_voxels"]*p["voxel_size"]
    inner = p["inner_radius_voxels"]*p["voxel_size"]
    output, evidence, seen, fit_audits = [], [], set(), []
    for first, direction in proposals:
        fitted = _fit_ridge(train, first, direction, radius, inner)
        if fitted is None:
            continue
        first, direction = fitted
        # Repeat proposals of the same trained support need not be revalidated.
        _, distance = _distance(train, first, direction)
        key = np.packbits(distance <= inner).tobytes()
        if key in seen:
            continue
        seen.add(key)
        lines, detail, audit = _ridge_runs(ridge_values, ridge_mask, first, direction, p)
        fit_audits.append(audit)
        if audit["state"] != "complete":
            return {**result, "state": "unmeasurable", "reason": audit["reason"],
                    "ridge_fit_audits": fit_audits, "section_evidence": section_detail}
        output.extend(lines)
        evidence.extend(detail)
        if len(output) > p["maximum_ridge_runs"]:
            return {**result, "state": "unmeasurable", "reason": "ridge_run_budget_exceeded"}
    merged = merged_segments(output, p)
    supported, merged_evidence, merged_audits = [], [], []
    for segment in merged:
        direction = segment[1]-segment[0]
        length = np.linalg.norm(direction)
        direction /= length
        # The old helper can extend one axis with a nearby overlapping run.
        # Re-establish actual inner continuity after merging; never merge again.
        t = (ridge_values-segment[0]) @ direction
        window = (t >= -1e-8*p["voxel_size"]) & (t <= length+1e-8*p["voxel_size"])
        lines, detail, audit = _ridge_runs(ridge_values[window], ridge_mask[window], segment[0], direction, p)
        merged_audits.append(audit)
        if audit["state"] != "complete":
            return {**result, "state": "unmeasurable", "reason": audit["reason"],
                    "ridge_fit_audits": fit_audits, "post_merge_ridge_audits": merged_audits,
                    "section_evidence": section_detail}
        supported.extend(lines)
        merged_evidence.extend(detail)
        if len(supported) > p["maximum_ridge_runs"]:
            return {**result, "state": "unmeasurable", "reason": "ridge_run_budget_exceeded"}
    ridges = np.unique(np.asarray(supported).reshape(-1, 6), axis=0).reshape(-1, 2, 3)
    array_evidence = []
    for segment in ridges:
        detail = ridge_local_evidence.analyze(ridge_values, ridge_mask, segment, p)
        if detail["state"] == "unmeasurable":
            return {**result, "state": "unmeasurable", "reason": detail["reason"],
                    "ridge_array_evidence": array_evidence+[detail], "section_evidence": section_detail}
        array_evidence.append(detail)
    array_rejected = np.asarray([row["parallel_array"] for row in array_evidence], dtype=bool)
    # Three+ coextensive narrow peaks may be a sparsely sampled surface, not rods.
    parallel = _parallel_ambiguity(ridges, p)
    # Three non-collinear transverse row centers do not witness a sheet.
    # Keep the legacy triplet count as a diagnostic, not a blanket rod veto.
    ridges = ridges[~array_rejected]
    if len(surface)+len(ridges) > p["maximum_output_segments"]:
        return {**result, "state": "unmeasurable", "reason": "output_segment_budget_exceeded"}
    chosen = np.concatenate((np.asarray(surface).reshape(-1, 2, 3), ridges))
    result.update(segments=chosen, components=len(chosen), ridge_proposals=len(proposals), unique_fits=len(seen),
        accepted_ridge_runs=len(output), ridge_evidence=evidence, local_proposal_evidence=local_detail,
        sparse_parallel_ambiguity=bool(array_rejected.any()),
        ambiguous_parallel_runs=int(array_rejected.sum()),
        legacy_parallel_triplet_runs=int(parallel.sum()),
        ridge_fit_audits=fit_audits, post_merge_ridge_audits=merged_audits,
        ridge_array_evidence=array_evidence, rejected_parallel_arrays=int(array_rejected.sum()),
        post_merge_ridge_evidence=merged_evidence,
        section_evidence=section_detail, surface_consumed_points=int(consumed.sum()),
        training_points=int(train_mask.sum()), validation_points=int((~train_mask).sum()),
        resolution_state=_resolution(chosen, section_detail, bool(array_rejected.any())),
        reason=None if len(chosen) else "no_supported_straight_geometric_model")
    return result
