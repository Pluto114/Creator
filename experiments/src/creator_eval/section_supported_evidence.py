"""Separate section identifiability, held-out prediction, and actual support.

Training alone fixes transverse groups, circular models, sampling scales, and
axial windows. The held-out fold checks predictions, not whether that arbitrary
fold itself contains an uninterrupted circular arc. The unchanged angular-shape
gate applies to all actual shell samples within each already fixed window.
Finite measured endpoints use actual support inside a training-fixed envelope;
neither good nor missing validation can enlarge that envelope or its windows.
"""

from __future__ import annotations

import numpy as np

from . import section_evidence as frozen
from . import section_sampling_evidence as sampling
from .common_readout_split import principal_axes

SUPPORTED_DEFAULTS = dict(
    section_supported_maximum_training_condition=40.,
    section_supported_minimum_validation_bins=3,
)
DEFAULTS = {**sampling.DEFAULTS, **SUPPORTED_DEFAULTS}


def policy(config):
    supported = {**SUPPORTED_DEFAULTS, **{key: value for key, value in config.items()
                                        if key.startswith("section_supported_")}}
    if set(supported) != set(SUPPORTED_DEFAULTS):
        raise ValueError("Unknown supported-section policy")
    limit, bins = supported["section_supported_maximum_training_condition"], supported["section_supported_minimum_validation_bins"]
    if not np.isfinite(limit) or not 1 < limit <= 1000000:
        raise ValueError("Bounded normalized training-condition limit required")
    if type(bins) is not int or not 3 <= bins <= 36:
        raise ValueError("At least three bounded held-out angular bins required")
    base = sampling.policy({key: value for key, value in config.items() if not key.startswith("section_supported_")})
    return {**base, **supported}


def _training_identifiability(xy, circle, p):
    """Dimensionless, equal-angular-bin circle-design conditioning.

    Repeated axial samples on one angular ray cannot improve geometric rank or
    turn a nearly straight arc into well-determined center/radius parameters.
    This does not require an arbitrary training fold to reproduce full support.
    """
    selected = xy[circle["inside"]]
    normalized = (selected-circle["center"])/circle["radius"]
    angles = np.mod(np.rad2deg(np.arctan2(normalized[:, 1], normalized[:, 0])), 360.)
    bins, inverse = np.unique(np.floor(angles/p["section_angular_bin_degrees"]).astype(np.int64), return_inverse=True)
    representatives = np.asarray([normalized[inverse == i].mean(axis=0) for i in range(len(bins))])
    condition = None
    if len(representatives) >= 3:
        singular = np.linalg.svd(np.c_[2*representatives, np.ones(len(representatives))], compute_uv=False)
        if singular[-1] > 64*np.finfo(float).eps*singular[0]:
            condition = float(singular[0]/singular[-1])
    return dict(angular_bins=len(bins), normalized_design_condition=condition,
                maximum_condition=p["section_supported_maximum_training_condition"],
                accepted=condition is not None and condition <= p["section_supported_maximum_training_condition"],
                policy="Training-only equal-angular-bin normalized circle design, plus unchanged curvature and stability gates")


def _validation_prediction(xy, inliers, center, p):
    selected = xy[inliers]-center
    angles = np.mod(np.rad2deg(np.arctan2(selected[:, 1], selected[:, 0])), 360.)
    bins = len(np.unique(np.floor(angles/p["section_angular_bin_degrees"]).astype(np.int64)))
    fraction = float(inliers.mean()) if len(inliers) else 0.
    return dict(point_count=len(xy), angular_bins=bins, shell_fraction=fraction,
                accepted=(fraction >= p["section_minimum_shell_fraction"]
                          and bins >= p["section_supported_minimum_validation_bins"]),
                policy="Residual support of frozen training model; no held-out arc completion or refitting")


def _endpoint_allowances(low, high, sampling_detail):
    """Read the already frozen training gap field at either support boundary."""
    layers = np.asarray(sampling_detail["training_layer_positions_m"])
    limits = np.asarray(sampling_detail["maximum_allowed_gaps_m"])
    if not len(limits):
        return [sampling_detail["original_voxel_gap_m"]]*2
    mids = (layers[:-1]+layers[1:])/2
    return [float(limits[np.argmin(np.abs(mids-value))]) for value in (low, high)]


def analyze(points, train_mask, config):
    """Return finite axes, consumed support, and per-window evidence provenance."""
    p = policy(config)
    sampling_policy = {key: value for key, value in p.items() if not key.startswith("section_supported_")}
    points, train_mask = np.asarray(points, float), np.asarray(train_mask)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or train_mask.shape != (len(points),) or train_mask.dtype != np.dtype(bool)):
        raise ValueError("Finite Nx3 points and a matching boolean training mask required")
    empty, consumed = np.empty((0, 2, 3)), np.zeros(len(points), dtype=bool)
    diagnostic = dict(state="complete", groups=[], accepted_surface_groups=0, unresolved_surface_groups=0,
        fit_uses_training_only=True, sampling_uses_training_only=True, windows_use_training_only=True,
        semantic_identity_claimed=False,
        scope="Training-identifiable circular sections with held-out prediction and actual angular support; internal folds are not independent physical observations")
    if len(points) > p["section_maximum_points"]:
        return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": "section_point_budget_exceeded"}
    if train_mask.sum() < 9:
        return empty, consumed, {**diagnostic, "reason": "insufficient_training_support"}
    numeric = 256*np.finfo(float).eps*max(1., float(np.abs(points[train_mask]).max()))
    anchor, eigen, _, axis = principal_axes(points[train_mask])
    if eigen[-1] <= numeric**2:
        return empty, consumed, {**diagnostic, "reason": "degenerate_training_extent"}
    groups, grouping = frozen._groups(points, train_mask, anchor, axis, p)
    diagnostic.update(grouping)
    if groups is None:
        return empty, consumed, diagnostic
    output = []
    for number, indices in enumerate(groups):
        cloud, fold = points[indices], train_mask[indices]
        if fold.sum() < 9:
            diagnostic["groups"].append(dict(group=number, state="not_surface", reason="insufficient_training_group"))
            continue
        model = frozen._trained_model(cloud, fold, axis, p, numeric)
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
            training_circle_rmse=circle["circle_rmse"], training_line_rmse=circle["line_rmse"], runs=[], training_runs=[],
            window_policy="Fixed equal-width slices per training-only support run; no validation-guided expansion",
            extent_policy="Actual observed runs inside training-fixed endpoint envelopes, with unchanged angular gates on added boundary support")
        training_axial = axial[shell & fold]
        training_runs, training_sampling = sampling.sampling_runs(training_axial, training_axial, sampling_policy)
        observed_runs, sampling_detail = sampling.sampling_runs(axial[shell], training_axial, sampling_policy)
        detail["training_sampling_evidence"] = training_sampling
        detail["sampling_evidence"] = sampling_detail
        if sampling_detail["state"] == "unmeasurable":
            diagnostic["groups"].append(detail)
            return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": sampling_detail["reason"]}
        # Projected points of one physical ring need not have identical axial
        # coordinates. Reuse the frozen layer resolution, not a new tolerance.
        edge_tolerance = max(numeric, training_sampling["layer_merge_tolerance_m"])
        detail["endpoint_layer_tolerance_m"] = edge_tolerance
        inside_training_windows = np.zeros(len(cloud), dtype=bool)
        for run in training_runs:
            low, high = float(run[0]), float(run[-1])
            boundaries = np.linspace(low, high, p["section_slices"]+1)
            left, right = _endpoint_allowances(low, high, training_sampling)
            run_detail = dict(low=low, high=high, training_window_edges=boundaries.tolist(),
                             measurement_envelope=[low-left, high+right], accepted=False, slices=[], boundary_support=[])
            for start, end, side in ((low-left, low-edge_tolerance, "left"), (high+edge_tolerance, high+right, "right")):
                boundary = ((axial >= start) & (axial < end) if side == "left"
                            else (axial > start) & (axial <= end))
                angular = frozen._angular_evidence(xy[boundary & shell], circle["center"], p)
                fraction = float(shell[boundary].mean()) if boundary.any() else 0.
                run_detail["boundary_support"].append({**angular, "side": side, "window": [start, end],
                    "shell_fraction": fraction,
                    "accepted": bool(angular["accepted"] and fraction >= p["section_minimum_shell_fraction"])})
            if high-low < p["section_minimum_run_length_voxels"]*p["voxel_size"]:
                run_detail["reason"] = "below_minimum_run_length"
                detail["training_runs"].append(run_detail)
                continue
            usable = True
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                part = (axial >= start) & ((axial <= end) if i == len(boundaries)-2 else (axial < end))
                inside_training_windows |= part
                training_xy = xy[part & fold]
                local = frozen._circle(training_xy, high-low, p, numeric)
                identification = (_training_identifiability(training_xy, local, p) if local is not None
                                  else dict(accepted=False, reason="no_training_curved_section"))
                validation = part & ~fold
                prediction = _validation_prediction(xy[validation], shell[validation], circle["center"], p)
                # This is a property of actual geometry, not of one random fold.
                # It can reject a shape, but cannot alter trained parameters.
                angular = frozen._angular_evidence(xy[part & shell], circle["center"], p)
                drift = float(np.linalg.norm(local["center"]-circle["center"])) if local else None
                radius_drift = abs(local["radius"]-circle["radius"]) if local else None
                valid = (local is not None and identification["accepted"] and prediction["accepted"]
                         and angular["accepted"]
                         and drift <= p["section_center_drift_radius_fraction"]*circle["radius"]
                         and radius_drift <= p["section_radius_drift_fraction"]*circle["radius"])
                run_detail["slices"].append({**angular, "angular_evidence_scope": "all_actual_shell_support_in_fixed_window",
                    "training_identifiability": identification, "validation_prediction": prediction,
                    "validation_shell_fraction": prediction["shell_fraction"], "center_drift": drift,
                    "radius_drift": radius_drift, "accepted": bool(valid)})
                usable &= valid
            run_detail["accepted"] = bool(usable)
            detail["training_runs"].append(run_detail)
        accepted = 0
        train_lows = np.asarray([run["low"] for run in detail["training_runs"]])
        train_highs = np.asarray([run["high"] for run in detail["training_runs"]])
        for run in observed_runs:
            low, high = float(run[0]), float(run[-1])
            first = int(np.searchsorted(train_highs, low-numeric))
            last = int(np.searchsorted(train_lows, high+numeric, side="right"))
            support = detail["training_runs"][first:last]
            measured = dict(low=low, high=high, accepted=False, training_run_indices=list(range(first, last)),
                training_window_edges=sorted({edge for item in support for edge in item["training_window_edges"]}),
                slices=[section for item in support for section in item["slices"]])
            if high-low < p["section_minimum_run_length_voxels"]*p["voxel_size"]:
                measured["reason"] = "below_minimum_run_length"
            elif not support:
                measured["reason"] = "no_training_section_support"
            elif not all(item["accepted"] for item in support):
                measured["reason"] = "training_window_evidence_rejected"
            else:
                covered_to = low
                covered = True
                boundaries_valid = True
                for item in support:
                    start, end = item["measurement_envelope"]
                    covered &= start <= covered_to+numeric
                    covered_to = max(covered_to, end)
                    for side, needed in enumerate((low < item["low"]-edge_tolerance, high > item["high"]+edge_tolerance)):
                        if needed:
                            boundaries_valid &= item["boundary_support"][side]["accepted"]
                covered &= covered_to >= high-numeric
                measured["accepted"] = bool(covered and boundaries_valid)
                if not covered:
                    measured["reason"] = "outside_training_measurement_envelope"
                elif not boundaries_valid:
                    measured["reason"] = "added_boundary_support_rejected"
            detail["runs"].append(measured)
            if measured["accepted"]:
                # Coordinates are actual measured shell limits, never training
                # order statistics or extrapolated envelope boundaries.
                output.append(np.array([center_world+low*direction, center_world+high*direction]))
                accepted += 1
        detail["shell_points_outside_training_windows"] = int((shell & ~inside_training_windows).sum())
        if accepted:
            detail["state"] = "accepted_surface"
            detail["accepted_runs"] = accepted
            diagnostic["accepted_surface_groups"] += 1
        else:
            diagnostic["unresolved_surface_groups"] += 1
        diagnostic["groups"].append(detail)
    diagnostic["consumed_points"] = int(consumed.sum())
    return np.asarray(output).reshape(-1, 2, 3), consumed, diagnostic
