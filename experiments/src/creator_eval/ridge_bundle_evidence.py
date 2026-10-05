"""Training-local straight-row and circular-row ambiguity evidence.

Three coextensive, approximately equally spaced collinear row centers establish
an unresolved two-dimensional arrangement. Two lines alone never establish it.
This is geometric ambiguity evidence, not a semantic assertion that a sheet or
three independent physical rods is present. No input primitive IDs are used.
"""

from __future__ import annotations

import numpy as np

from . import ridge_plane_evidence, ridge_sampling_evidence

DEFAULTS = dict(ridge_array_context_span_fraction=.5, ridge_array_maximum_spacing_ratio=1.5,
    ridge_array_support_fraction=.6, ridge_array_minimum_axial_bins=3,
    ridge_array_maximum_points=200000, ridge_array_maximum_seeds=20000,
    ridge_array_maximum_rows=256, ridge_array_maximum_families=256)
DEFAULTS.update(ridge_sampling_evidence.DEFAULTS)
DEFAULTS.update(ridge_plane_evidence.DEFAULTS)
DEFAULTS.update(ridge_bundle_minimum_ring_rows=4, ridge_bundle_minimum_ring_degrees=150.,
    ridge_bundle_ring_tolerance_inner=2., ridge_bundle_maximum_circle_condition=100.,
    ridge_bundle_maximum_ring_models=256)


def policy(config):
    p = {**DEFAULTS, **{key: value for key, value in config.items() if key.startswith(("ridge_array_", "ridge_sampling_", "ridge_bundle_", "ridge_plane_"))}}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown local ridge array policy")
    for key in ("ridge_array_minimum_axial_bins", "ridge_array_maximum_points", "ridge_array_maximum_seeds",
                "ridge_array_maximum_rows", "ridge_array_maximum_families",
                "ridge_bundle_minimum_ring_rows", "ridge_bundle_maximum_ring_models"):
        if type(p[key]) is not int or not 1 <= p[key] <= 200000:
            raise ValueError("Bounded positive integer ridge array budgets required")
    if (p["ridge_array_minimum_axial_bins"] < 3 or p["ridge_array_maximum_rows"] > 4096
            or p["ridge_array_maximum_families"] > 4096):
        raise ValueError("Invalid ridge array support or combinatorial budget")
    for key in ("ridge_array_context_span_fraction", "ridge_array_maximum_spacing_ratio", "ridge_array_support_fraction"):
        if isinstance(p[key], (bool, np.bool_)) or not np.isfinite(p[key]):
            raise ValueError("Finite ridge array geometry policy required")
    if (not 0 < p["ridge_array_context_span_fraction"] <= 1
            or not 1 <= p["ridge_array_maximum_spacing_ratio"] <= 2
            or not 0 < p["ridge_array_support_fraction"] <= 1):
        raise ValueError("Invalid ridge array geometry policy")
    if not 4 <= p["ridge_bundle_minimum_ring_rows"] <= 32 or p["ridge_bundle_maximum_ring_models"] > 4096:
        raise ValueError("Invalid bundle integer budgets")
    for key in ("ridge_bundle_minimum_ring_degrees", "ridge_bundle_ring_tolerance_inner",
                "ridge_bundle_maximum_circle_condition"):
        value = p[key]
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
            raise ValueError("Finite bundle geometry scales required")
    if (not 90 <= p["ridge_bundle_minimum_ring_degrees"] <= 360
            or not 0 < p["ridge_bundle_ring_tolerance_inner"] <= 4
            or not 1 < p["ridge_bundle_maximum_circle_condition"] <= 10000):
        raise ValueError("Invalid bundle geometry policy")
    ridge_sampling_evidence.policy(config)
    ridge_plane_evidence.policy(config)
    return p


def _coordinates(points, origin, direction, basis):
    delta = points-origin
    return delta @ direction, delta @ basis


def _rows_analyze(points, train_mask, segment, config):
    """Return a veto only for held-out-confirmed training-proposed row arrays.

    Call on the original remaining ridge points and a fitted finite candidate,
    before emission (and after any merge). The candidate only selects a local
    finite scope; training core points set its measured neighborhood bounds,
    search radius, row centers, spacings, and complete family inventory. Held-out
    points cannot nominate rows, expand this neighborhood, or refit parameters.
    An ``unmeasurable`` state is a budget failure, never a negative array test.
    """
    from scipy.spatial import cKDTree

    p = policy(config)
    points, train_mask, segment = np.asarray(points, float), np.asarray(train_mask), np.asarray(segment, float)
    voxel, inner = config["voxel_size"], config.get("inner_radius_voxels", .15)*config["voxel_size"]
    fold_fraction = config.get("minimum_validation_coverage", .3)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or train_mask.shape != (len(points),) or train_mask.dtype != np.dtype(bool)
            or segment.shape != (2, 3) or not np.isfinite(segment).all()
            or not np.isfinite(voxel) or voxel <= 0 or not np.isfinite(inner) or inner <= 0
            or not np.isfinite(fold_fraction) or not 0 < fold_fraction <= 1):
        raise ValueError("Finite points, boolean training mask, finite segment and positive scales required")
    delta = segment[1]-segment[0]
    length = float(np.linalg.norm(delta))
    if length <= 0 or not np.isfinite(length):
        raise ValueError("A nondegenerate finite ridge candidate is required")
    direction = delta/length
    # The candidate orientation is unoriented; canonicalizing also makes the
    # complete diagnostic invariant when the two endpoints are reversed.
    if direction[np.argmax(np.abs(direction))] < 0:
        segment, direction = segment[::-1], -direction
    seed = np.eye(3)[np.argmin(np.abs(direction))]
    first = np.cross(direction, seed)
    first /= np.linalg.norm(first)
    basis = np.column_stack((first, np.cross(direction, first)))
    diagnostic = dict(state="complete", parallel_array=False, training_only_proposals=True,
        validation_refits_model=False, minimum_rows=3, training_model=None, heldout_support=[],
        scope="Unresolved local three-or-more parallel-row array, not a physical sheet or foreground identity claim")
    if len(points) > p["ridge_array_maximum_points"]:
        return {**diagnostic, "state": "unmeasurable", "reason": "ridge_array_point_budget_exceeded"}
    training, validation = np.unique(points[train_mask], axis=0), np.unique(points[~train_mask], axis=0)
    t, xy = _coordinates(training, segment[0], direction, basis)
    numeric = 256*np.finfo(float).eps*max(1., length, float(np.abs(training).max(initial=0.)))
    core = (t >= -numeric) & (t <= length+numeric) & (np.linalg.norm(xy, axis=1) <= inner)
    if core.sum() < p["ridge_array_minimum_axial_bins"]:
        return {**diagnostic, "reason": "insufficient_training_core"}
    low, high = float(t[core].min()), float(t[core].max())
    if high-low <= numeric:
        return {**diagnostic, "reason": "degenerate_training_core"}
    scale = ridge_sampling_evidence.train_scale(t[core], config)
    if scale["state"] != "complete":
        return {**diagnostic, "state": "unmeasurable", "reason": scale["reason"]}
    axial_step = scale["bin_width_m"]
    phase = .5 if scale["learning_state"] == "training_cadence" else 0.
    center = xy[core].mean(axis=0)
    radius = (high-low)*p["ridge_array_context_span_fraction"]
    window = (t >= low-numeric) & (t <= high+numeric) & (np.linalg.norm(xy-center, axis=1) <= radius)
    t, xy = t[window], xy[window]
    coordinates = (t-low)/axial_step
    transverse_grid = xy/inner
    if (not np.isfinite(coordinates).all() or not np.isfinite(transverse_grid).all()
            or np.any(np.abs(coordinates) >= 2**52) or np.any(np.abs(transverse_grid) >= 2**52)):
        raise ValueError("Local ridge array coordinates exceed exact grid precision")
    bins = np.floor(coordinates+phase+numeric/axial_step).astype(np.int64)
    core_members = np.linalg.norm(xy-center, axis=1) <= inner
    core_bins = len(np.unique(bins[core_members]))
    # Reuse the ridge reader's absolute fold coverage. Relative-to-core counts
    # would make a neighbor's geometry depend on random fold imbalance in the
    # candidate row. Joint persistence is checked separately after validation.
    expected = max(1, int(np.floor((high-low)/axial_step+phase))+1)
    required = max(p["ridge_array_minimum_axial_bins"], fold_fraction*expected)
    if core_bins < p["ridge_array_minimum_axial_bins"]:
        return {**diagnostic, "reason": "insufficient_training_core_bins"}
    _, seeds = np.unique(np.floor(transverse_grid).astype(np.int64), axis=0, return_index=True)
    model = dict(origin=segment[0].tolist(), direction=direction.tolist(), transverse_basis=basis.tolist(),
        axial_bounds=[low, high], context_radius=radius, tube_radius=inner, core_training_bins=core_bins,
        expected_axial_bins=expected, required_fold_axial_bins=required, sampling_scale=scale,
        seed_count=len(seeds), row_centers=[center.tolist()], training_bins=[core_bins], families=[])
    diagnostic["training_model"] = model
    if len(seeds) > p["ridge_array_maximum_seeds"]:
        return {**diagnostic, "state": "unmeasurable", "reason": "ridge_array_seed_budget_exceeded"}
    tree, seen, candidates = cKDTree(xy), set(), []
    for index in seeds:
        members = tuple(sorted(tree.query_ball_point(xy[index], inner)))
        if members in seen:
            continue
        seen.add(members)
        count = len(np.unique(bins[list(members)]))
        if count >= required:
            local = xy[list(members)].mean(axis=0)
            candidates.append((count, local))
    candidates.sort(key=lambda item: (-item[0], *item[1].tolist()))
    centers, counts = [center], [core_bins]
    for count, local in candidates:
        if min(np.linalg.norm(local-old) for old in centers) <= 2*inner:
            continue
        centers.append(local)
        counts.append(count)
        if len(centers) > p["ridge_array_maximum_rows"]:
            return {**diagnostic, "state": "unmeasurable", "reason": "ridge_array_row_budget_exceeded"}
    model.update(row_centers=np.asarray(centers).tolist(), training_bins=counts)
    if len(centers) < 3:
        return {**diagnostic, "reason": "fewer_than_three_persistent_training_rows"}
    relative = np.asarray(centers)-center
    families = set()
    # A training fold may omit one physical row. Enumerate the finite observed
    # three-row subsets, not just consecutive surviving peaks; e.g. rows 0/2/4
    # still witness an array without inventing or completing missing rows.
    for i in range(1, len(relative)):
        distance = np.linalg.norm(relative[i])
        axis = relative[i]/distance
        for j in range(i+1, len(relative)):
            projection = float(relative[j] @ axis)
            if np.linalg.norm(relative[j]-projection*axis) > 2*inner:
                continue
            spacing = np.diff(np.sort([0., distance, projection]))
            if (spacing.min() <= 2*inner
                    or spacing.max() > p["ridge_array_maximum_spacing_ratio"]*spacing.min()):
                continue
            families.add((0, i, j))
            if len(families) > p["ridge_array_maximum_families"]:
                return {**diagnostic, "state": "unmeasurable", "reason": "ridge_array_family_budget_exceeded"}
    model["families"] = [list(item) for item in sorted(families)]
    ring_models = {}
    if len(centers) >= p["ridge_bundle_minimum_ring_rows"]:
        tolerance = p["ridge_bundle_ring_tolerance_inner"]*inner
        # Each circle contains this candidate (row zero); three rows nominate,
        # a fourth observed row and held-out support are required to confirm.
        for i in range(1, len(relative)):
            for j in range(i+1, len(relative)):
                matrix = 2*np.array([relative[i], relative[j]])
                condition = np.linalg.cond(matrix)
                if not np.isfinite(condition) or condition > p["ridge_bundle_maximum_circle_condition"]:
                    continue
                circle_center = np.linalg.solve(matrix, np.sum(np.array([relative[i], relative[j]])**2, axis=1))
                circle_radius = float(np.linalg.norm(circle_center))
                if circle_radius <= 2*inner:
                    continue
                members = tuple(np.flatnonzero(np.abs(np.linalg.norm(relative-circle_center, axis=1)-circle_radius) <= tolerance).tolist())
                if 0 not in members or len(members) < p["ridge_bundle_minimum_ring_rows"]:
                    continue
                offsets = relative[list(members)]-circle_center
                angles = np.sort(np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2*np.pi))
                coverage = float(np.rad2deg(2*np.pi-np.diff(np.r_[angles, angles[0]+2*np.pi]).max()))
                if coverage < p["ridge_bundle_minimum_ring_degrees"] or members in ring_models:
                    continue
                ring_models[members] = dict(rows=list(members), center=(circle_center+center).tolist(),
                    radius=circle_radius, angle_coverage_degrees=coverage, condition=float(condition))
                if len(ring_models) > p["ridge_bundle_maximum_ring_models"]:
                    return {**diagnostic, "state": "unmeasurable", "reason": "ridge_bundle_ring_budget_exceeded"}
    model["ring_models"] = [ring_models[key] for key in sorted(ring_models)]
    if not families and not ring_models:
        return {**diagnostic, "reason": "no_local_collinear_repeated_spacing"}
    vt, vxy = _coordinates(validation, segment[0], direction, basis)
    in_window = (vt >= low-numeric) & (vt <= high+numeric)
    vt, vxy = vt[in_window], vxy[in_window]
    validation_bins = np.floor((vt-low)/axial_step+phase+numeric/axial_step).astype(np.int64)
    training_indices = [np.unique(bins[np.linalg.norm(xy-row, axis=1) <= inner]) for row in centers]
    validation_indices = [np.unique(validation_bins[np.linalg.norm(vxy-row, axis=1) <= inner]) for row in centers]
    validation_counts = [len(indices) for indices in validation_indices]
    union_counts = [len(np.union1d(a, b)) for a, b in zip(training_indices, validation_indices)]
    required_joint = p["ridge_array_support_fraction"]*expected
    for family, kind in ([(f, "collinear_rows") for f in model["families"]]
                         + [(m["rows"], "circular_rows") for m in model["ring_models"]]):
        supported = [validation_counts[index] for index in family]
        train_supported = [len(training_indices[index]) for index in family]
        joint = [union_counts[index] for index in family]
        accepted = min(supported) >= required and min(train_supported) >= required and min(joint) >= required_joint
        diagnostic["heldout_support"].append(dict(rows=family, kind=kind, axial_bins=supported,
            training_axial_bins=train_supported, joint_axial_bins=joint,
            required_axial_bins=required, required_joint_axial_bins=required_joint, accepted=bool(accepted)))
    diagnostic["parallel_array"] = any(item["accepted"] for item in diagnostic["heldout_support"])
    diagnostic["ambiguity_kinds"] = sorted({item["kind"] for item in diagnostic["heldout_support"] if item["accepted"]})
    diagnostic["reason"] = "validated_local_parallel_array" if diagnostic["parallel_array"] else "heldout_array_not_confirmed"
    return diagnostic


def analyze(points, train_mask, segment, config):
    result = _rows_analyze(points, train_mask, segment, config)
    if result["state"] != "complete" or result["parallel_array"]:
        return result
    plane = ridge_plane_evidence.analyze(points, train_mask, segment, config)
    if plane["state"] != "complete":
        return {**result, "state": plane["state"], "reason": plane["reason"], "plane_evidence": plane}
    return {**result, "parallel_array": plane["planar_sheet"], "plane_evidence": plane,
        "ambiguity_kinds": ["planar_sheet"] if plane["planar_sheet"] else [],
        "reason": "validated_local_plane" if plane["planar_sheet"] else result.get("reason")}
