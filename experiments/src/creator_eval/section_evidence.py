"""Train-only transverse grouping and conditional circular-section evidence.

This is geometric model evidence, not foreground identity. Validation may reject
a trained section but cannot move its center, radius, direction, or groups.
Sparse globally circular support that lacks local angular evidence is consumed
as unresolved surface support instead of becoming privileged ridge candidates.
"""

from __future__ import annotations

import itertools

import numpy as np

from .common_readout_sections import algebraic_circle
from .common_readout_split import axial_runs, principal_axes

DEFAULTS = dict(
    section_cell_voxels=.5, section_connection_voxels=1.5,
    section_maximum_cells=20000, section_maximum_groups=256, section_maximum_points=200000,
    section_slices=6, section_radius_relative_residual=.08, section_residual_voxels=.35,
    section_minimum_shell_fraction=.9, section_minimum_coverage_degrees=150.,
    section_angular_bin_degrees=10., section_minimum_angular_bins=9,
    section_maximum_internal_gap_degrees=30., section_circle_line_rmse_ratio=.25,
    section_maximum_radius_span_fraction=.25, section_center_drift_radius_fraction=.15,
    section_radius_drift_fraction=.15, section_center_support_radius_fraction=.15,
    section_minimum_run_length_voxels=12., section_neighbor_radius_voxels=1.75,
    section_axis_fit_evaluations=64, section_axis_proposal_degrees=15.,
)


def _policy(policy):
    p = {**DEFAULTS, **{key: value for key, value in policy.items() if key.startswith("section_")}}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown section-evidence policy")
    integers = {"section_maximum_cells", "section_maximum_groups", "section_maximum_points",
                "section_slices", "section_minimum_angular_bins", "section_axis_fit_evaluations"}
    for key, value in p.items():
        if key in integers:
            if type(value) is not int or not 1 <= value <= 2000000:
                raise ValueError("Bounded positive integer section policy required")
        elif not np.isfinite(value) or value <= 0:
            raise ValueError("Positive finite section policy required")
    if (not 2 <= p["section_slices"] <= 32
            or p["section_connection_voxels"]/p["section_cell_voxels"] > 6
            or p["section_connection_voxels"] < p["section_cell_voxels"]
            or not 0 < p["section_minimum_shell_fraction"] <= 1
            or p["section_circle_line_rmse_ratio"] >= 1
            or p["section_radius_relative_residual"] >= .5
            or p["section_minimum_coverage_degrees"] > 360
            or p["section_maximum_internal_gap_degrees"] >= 180
            or p["section_angular_bin_degrees"] > 45
            or p["section_minimum_angular_bins"] > 360/p["section_angular_bin_degrees"]):
        raise ValueError("Invalid geometric section-evidence policy")
    if p["section_axis_fit_evaluations"] > 256 or p["section_axis_proposal_degrees"] >= 45:
        raise ValueError("Bounded locally straight axis refinement required")
    voxel = policy.get("voxel_size", .01)
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("Positive finite voxel size required")
    p["voxel_size"] = voxel
    return p


def policy(config):
    """Validate this module's prefixed keys within a full reader policy."""
    return _policy(config)


def _basis(axis):
    first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(axis, first)))


def _groups(points, train_mask, anchor, axis, p):
    xy = (points-anchor) @ _basis(axis)
    cell_size = p["section_cell_voxels"]*p["voxel_size"]
    coordinates = xy/cell_size
    if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) >= 2**52):
        raise ValueError("Section coordinates exceed exact grid precision")
    cells, inverse = np.unique(np.floor(coordinates).astype(np.int64), axis=0, return_inverse=True)
    if len(cells) > p["section_maximum_cells"]:
        return None, dict(state="unmeasurable", reason="section_transverse_cell_budget_exceeded")
    train_cells = set(inverse[train_mask].tolist())
    lookup = {tuple(cells[i]): i for i in train_cells}
    radius = p["section_connection_voxels"]/p["section_cell_voxels"]
    extent = int(np.ceil(radius))
    offsets = [delta for delta in itertools.product(range(-extent, extent+1), repeat=2)
               if sum(x*x for x in delta) <= radius*radius+1e-12]
    group_of, count = {}, 0
    for start in sorted(train_cells):
        if start in group_of:
            continue
        if count >= p["section_maximum_groups"]:
            return None, dict(state="unmeasurable", reason="section_group_budget_exceeded")
        pending = [start]
        group_of[start] = count
        while pending:
            current = pending.pop()
            for delta in offsets:
                neighbor = lookup.get(tuple(cells[current]+delta))
                if neighbor is not None and neighbor not in group_of:
                    group_of[neighbor] = count
                    pending.append(neighbor)
        count += 1
    # Held-out cells cannot bridge training groups or nominate new components.
    assignment = np.full(len(cells), -1, dtype=np.int64)
    for i, cell in enumerate(cells):
        if i in group_of:
            assignment[i] = group_of[i]
            continue
        candidates = {group_of[j] for delta in offsets
                      if (j := lookup.get(tuple(cell+delta))) is not None}
        if len(candidates) == 1:
            assignment[i] = candidates.pop()
    labels = assignment[inverse]
    return [np.flatnonzero(labels == i) for i in range(count)], dict(
        transverse_cells=len(cells), training_transverse_cells=len(train_cells),
        training_group_count=count, unassigned_validation_points=int((labels < 0).sum()),
    )


def _circle(xy, span, p, numeric):
    if len(xy) < 9 or span <= numeric:
        return None
    fit = algebraic_circle(xy)
    if fit is None:
        return None
    for _ in range(4):
        center, radius = fit
        if not numeric < radius <= p["section_maximum_radius_span_fraction"]*span:
            return None
        tolerance = min(p["section_residual_voxels"]*p["voxel_size"], p["section_radius_relative_residual"]*radius)
        residual = np.abs(np.linalg.norm(xy-center, axis=1)-radius)
        inside = residual <= tolerance
        if inside.mean() < p["section_minimum_shell_fraction"] or inside.sum() < 9:
            return None
        fit = algebraic_circle(xy[inside])
        if fit is None:
            return None
    center, radius = fit
    if not numeric < radius <= p["section_maximum_radius_span_fraction"]*span:
        return None
    tolerance = min(p["section_residual_voxels"]*p["voxel_size"], p["section_radius_relative_residual"]*radius)
    residual = np.abs(np.linalg.norm(xy-center, axis=1)-radius)
    inside = residual <= tolerance
    if inside.mean() < p["section_minimum_shell_fraction"] or inside.sum() < 9:
        return None
    selected = xy[inside]
    eigen = np.linalg.eigvalsh(np.cov(selected.T, bias=True))
    line_rmse = float(np.sqrt(max(0., eigen[0])))
    circle_rmse = float(np.sqrt(np.mean(residual[inside]**2)))
    # A straight sheet or a numerically flat arc cannot win merely because a
    # very large, ill-conditioned circle interpolates three coordinates.
    if line_rmse <= 4*numeric or circle_rmse+numeric >= p["section_circle_line_rmse_ratio"]*line_rmse:
        return None
    return dict(center=center, radius=float(radius), tolerance=float(tolerance), inside=inside,
                circle_rmse=circle_rmse, line_rmse=line_rmse,
                shell_fraction=float(inside.mean()))


def _trained_model(points, train_mask, initial_axis, p, numeric):
    training = points[train_mask]
    anchor, axis = training.mean(axis=0), initial_axis.copy()
    # Angular position can correlate with z (a helix), biasing whole-cloud PCA.
    # A bounded train-only cylinder proposal can identify that global shell;
    # the independent local angular test must still refuse it as a full section.
    initial_xy = (training-anchor) @ _basis(axis)
    initial_span = float(np.ptp((training-anchor) @ axis))
    if _circle(initial_xy, initial_span, p, numeric) is None:
        refined = _cylinder_axis(training, anchor, axis, initial_span, p, numeric)
        if refined is not None:
            axis = refined
    # The small axis refinement uses training section centers only, never
    # validation coordinates or a supplied curve primitive direction.
    for iteration in range(3):
        basis = _basis(axis)
        axial = (training-anchor) @ axis
        xy = (training-anchor) @ basis
        circle = _circle(xy, np.ptp(axial), p, numeric)
        if iteration == 2:
            return (anchor, axis, basis, circle) if circle is not None else None
        centers = []
        boundaries = np.linspace(axial.min(), axial.max(), p["section_slices"]+1)
        for i, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            part = (axial >= low) & ((axial <= high) if i == len(boundaries)-2 else (axial < high))
            local = _circle(xy[part], np.ptp(axial), p, numeric)
            if local is not None:
                centers.append(anchor+float(axial[part].mean())*axis+basis @ local["center"])
        if len(centers) < 3:
            return (anchor, axis, basis, circle) if circle is not None else None
        _, eigen, _, direction = principal_axes(np.asarray(centers))
        if eigen[-1] <= numeric**2:
            return (anchor, axis, basis, circle) if circle is not None else None
        axis = direction
    raise AssertionError("Bounded section refinement did not terminate")


def _cylinder_axis(training, anchor, axis, span, p, numeric):
    from scipy.optimize import least_squares

    basis = _basis(axis)
    initial = algebraic_circle((training-anchor) @ basis)
    maximum_radius = p["section_maximum_radius_span_fraction"]*span
    if initial is None or not numeric < initial[1] < maximum_radius:
        return None
    center, radius = initial
    angle_bound = np.tan(np.deg2rad(p["section_axis_proposal_degrees"]))

    def residual(parameters):
        direction = axis+basis @ parameters[:2]
        direction /= np.linalg.norm(direction)
        delta = training-anchor-basis @ parameters[2:4]
        transverse = delta-(delta @ direction)[:, None]*direction
        return np.linalg.norm(transverse, axis=1)-parameters[4]

    fitted = least_squares(residual, [0., 0., *center, radius],
        bounds=([-angle_bound, -angle_bound, -np.inf, -np.inf, numeric],
                [angle_bound, angle_bound, np.inf, np.inf, maximum_radius]),
        loss="soft_l1", f_scale=max(numeric, p["section_radius_relative_residual"]*radius),
        max_nfev=p["section_axis_fit_evaluations"], x_scale="jac")
    if not fitted.success or not np.isfinite(fitted.x).all():
        return None
    direction = axis+basis @ fitted.x[:2]
    return direction/np.linalg.norm(direction)


def _angular_evidence(xy, center, p):
    angles = np.sort(np.mod(np.arctan2(*(xy-center).T[::-1]), 2*np.pi))
    if len(angles) < 2:
        return dict(angular_bins=0, coverage_degrees=0., maximum_internal_gap_degrees=360., accepted=False)
    gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
    missing = int(np.argmax(gaps))
    coverage = float(np.rad2deg(2*np.pi-gaps[missing]))
    internal = float(np.rad2deg(np.delete(gaps, missing).max(initial=0.)))
    bins = len(np.unique(np.floor(np.rad2deg(angles)/p["section_angular_bin_degrees"]).astype(np.int64)))
    return dict(angular_bins=bins, coverage_degrees=coverage, maximum_internal_gap_degrees=internal,
                accepted=(bins >= p["section_minimum_angular_bins"]
                          and coverage >= p["section_minimum_coverage_degrees"]
                          and internal <= p["section_maximum_internal_gap_degrees"]))


def analyze(points, train_mask, policy):
    """Return accepted finite axes, consumed support mask, and full diagnostics.

    ``consumed`` prevents ridge fallback only for globally supported circular
    shells and their observed near-center samples. Other members, including
    radial outliers/side branches in the same transverse group, are preserved.
    An unmeasurable return must not be relabelled successful empty geometry.
    """
    p = _policy(policy)
    points, train_mask = np.asarray(points, float), np.asarray(train_mask)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or train_mask.shape != (len(points),) or train_mask.dtype != np.dtype(bool)):
        raise ValueError("Finite Nx3 points and a matching boolean training mask required")
    empty, consumed = np.empty((0, 2, 3)), np.zeros(len(points), dtype=bool)
    diagnostic = dict(state="complete", groups=[], accepted_surface_groups=0, unresolved_surface_groups=0,
                      fit_uses_training_only=True, semantic_identity_claimed=False,
                      scope="Locally circular straight extrusions; angular evidence pooled within finite axial slices, not arbitrary surface/topology qualification")
    if len(points) > p["section_maximum_points"]:
        return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": "section_point_budget_exceeded"}
    if train_mask.sum() < 9:
        return empty, consumed, {**diagnostic, "reason": "insufficient_training_support"}
    numeric = 256*np.finfo(float).eps*max(1., float(np.abs(points[train_mask]).max()))
    anchor, eigen, _, axis = principal_axes(points[train_mask])
    if eigen[-1] <= numeric**2:
        return empty, consumed, {**diagnostic, "reason": "degenerate_training_extent"}
    groups, grouping = _groups(points, train_mask, anchor, axis, p)
    diagnostic.update(grouping)
    if groups is None:
        return empty, consumed, diagnostic
    output = []
    for number, indices in enumerate(groups):
        cloud, fold = points[indices], train_mask[indices]
        if fold.sum() < 9:
            diagnostic["groups"].append(dict(group=number, state="not_surface", reason="insufficient_training_group"))
            continue
        model = _trained_model(cloud, fold, axis, p, numeric)
        if model is None:
            diagnostic["groups"].append(dict(group=number, state="not_surface", reason="no_trained_curved_shell"))
            continue
        origin, direction, basis, circle = model
        axial, xy = (cloud-origin) @ direction, (cloud-origin) @ basis
        radial = np.linalg.norm(xy-circle["center"], axis=1)
        shell = np.abs(radial-circle["radius"]) <= circle["tolerance"]
        centers = radial <= p["section_center_support_radius_fraction"]*circle["radius"]
        consumed[indices[shell | centers]] = True
        center_world = origin+basis @ circle["center"]
        detail = dict(group=number, state="surface_unresolved", points=len(cloud), consumed_points=int((shell | centers).sum()),
            retained_radial_outliers=int((~(shell | centers)).sum()), training_shell_fraction=circle["shell_fraction"],
            validation_shell_fraction=float(shell[~fold].mean()) if (~fold).any() else None,
            trained_center=center_world.tolist(), trained_axis=direction.tolist(), trained_radius=circle["radius"],
            training_circle_rmse=circle["circle_rmse"], training_line_rmse=circle["line_rmse"], runs=[])
        accepted = 0
        runs = axial_runs(axial[shell], p["voxel_size"], p["section_neighbor_radius_voxels"])
        for run in runs:
            low, high = float(run[0]), float(run[-1])
            if high-low < p["section_minimum_run_length_voxels"]*p["voxel_size"]:
                continue
            checks, usable = [], True
            boundaries = np.linspace(low, high, p["section_slices"]+1)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                part = (axial >= start) & ((axial <= end) if i == len(boundaries)-2 else (axial < end))
                local = _circle(xy[part & fold], high-low, p, numeric)
                validation = part & ~fold
                angular = _angular_evidence(xy[validation & shell], circle["center"], p)
                fraction = float(shell[validation].mean()) if validation.any() else 0.
                drift = float(np.linalg.norm(local["center"]-circle["center"])) if local else None
                radius_drift = abs(local["radius"]-circle["radius"]) if local else None
                valid = (local is not None and angular["accepted"]
                         and fraction >= p["section_minimum_shell_fraction"]
                         and drift <= p["section_center_drift_radius_fraction"]*circle["radius"]
                         and radius_drift <= p["section_radius_drift_fraction"]*circle["radius"])
                checks.append({**angular, "validation_shell_fraction": fraction, "center_drift": drift,
                               "radius_drift": radius_drift, "accepted": bool(valid)})
                usable &= valid
            detail["runs"].append(dict(low=low, high=high, accepted=bool(usable), slices=checks))
            if usable:
                output.append(np.array([center_world+low*direction, center_world+high*direction]))
                accepted += 1
        if accepted:
            detail["state"] = "accepted_surface"
            detail["accepted_runs"] = accepted
            diagnostic["accepted_surface_groups"] += 1
        else:
            diagnostic["unresolved_surface_groups"] += 1
        diagnostic["groups"].append(detail)
    diagnostic["consumed_points"] = int(consumed.sum())
    return np.asarray(output).reshape(-1, 2, 3), consumed, diagnostic
