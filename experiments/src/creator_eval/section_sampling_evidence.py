"""Circular-section evidence with a train-only local axial sampling scale.

The frozen section fitter and local angular evidence are reused unchanged. Only
the observed-support run splitter changes: repeated training layer intervals
can explain sparse sampling, while an interval cannot inflate its own allowed
gap. No missing points are generated and no validation coordinate fits a scale.
This is conditional sampling evidence, not proof of continuity between samples.
"""

from __future__ import annotations

import numpy as np

from . import section_evidence as frozen
from .common_readout_split import principal_axes

SAMPLING_DEFAULTS = dict(
    section_sampling_layer_merge_voxels=.05,
    section_sampling_neighbor_intervals=8,
    section_sampling_minimum_neighbors=6,
    section_sampling_maximum_relative_mad=.5,
    section_sampling_gap_factor=2.5,
    section_sampling_maximum_layers=8192,
)
DEFAULTS = {**frozen.DEFAULTS, **SAMPLING_DEFAULTS}


def policy(config):
    """Accept the caller's full policy, validating our own prefixed keys."""
    sampling = {**SAMPLING_DEFAULTS, **{key: value for key, value in config.items()
                                      if key.startswith("section_sampling_")}}
    if set(sampling) != set(SAMPLING_DEFAULTS):
        raise ValueError("Unknown section-sampling policy")
    base = frozen.policy({key: value for key, value in config.items() if not key.startswith("section_sampling_")})
    integers = {"section_sampling_neighbor_intervals", "section_sampling_minimum_neighbors", "section_sampling_maximum_layers"}
    for key, value in sampling.items():
        if key in integers:
            if type(value) is not int or not 1 <= value <= 200000:
                raise ValueError("Bounded integer sampling policy required")
        elif not np.isfinite(value) or value <= 0:
            raise ValueError("Positive finite sampling policy required")
    if (not 3 <= sampling["section_sampling_minimum_neighbors"] <= sampling["section_sampling_neighbor_intervals"] <= 64
            or not 0 < sampling["section_sampling_layer_merge_voxels"] < .25
            or not 0 < sampling["section_sampling_maximum_relative_mad"] < 1
            or not 1 < sampling["section_sampling_gap_factor"] <= 4
            or sampling["section_sampling_maximum_layers"] < sampling["section_sampling_minimum_neighbors"]+2):
        raise ValueError("Invalid local sampling-evidence policy")
    return {**base, **sampling}


def _typical_interval(intervals, p):
    if len(intervals) < p["section_sampling_minimum_neighbors"]:
        return None
    median = float(np.median(intervals))
    mad = float(np.median(np.abs(intervals-median)))
    if median <= 0 or mad > p["section_sampling_maximum_relative_mad"]*median:
        return None
    return median


def sampling_runs(projection, training_projection, config):
    """Split actual support with a frozen, leave-current-interval-out scale field.

    A train layer is a connected cluster of very close projected coordinates.
    A continuous dense cloud may form one cluster; it then retains the original
    voxel rule rather than inventing a sampling scale. Local left/right medians
    preserve genuine changes in sampling density. Long anomalous intervals are
    excluded from their own estimates and remain breaks in observed support.
    """
    p = policy(config)
    observed = np.sort(np.asarray(projection, float))
    training = np.sort(np.asarray(training_projection, float))
    if observed.ndim != 1 or training.ndim != 1 or not np.isfinite(observed).all() or not np.isfinite(training).all():
        raise ValueError("Finite one-dimensional axial projections required")
    if max(len(observed), len(training)) > p["section_maximum_points"]:
        return [], dict(state="unmeasurable", reason="sampling_projection_budget_exceeded")
    voxel = p["voxel_size"]
    base_limit = p["section_neighbor_radius_voxels"]*voxel
    numerical = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(training), initial=0.)), voxel)
    merge = max(numerical, p["section_sampling_layer_merge_voxels"]*voxel)
    parts = np.split(training, np.flatnonzero(np.diff(training) > merge)+1) if len(training) else []
    if len(parts) > p["section_sampling_maximum_layers"]:
        return [], dict(state="unmeasurable", reason="sampling_layer_budget_exceeded",
                        training_layer_count=len(parts), maximum_layers=p["section_sampling_maximum_layers"])
    layers = np.asarray([np.median(part) for part in parts])
    intervals = np.diff(layers)
    mids = (layers[1:]+layers[:-1])/2
    estimates = np.full(len(intervals), np.nan)
    sources = []
    width = p["section_sampling_neighbor_intervals"]
    for i in range(len(intervals)):
        left, right = intervals[max(0, i-width):i], intervals[i+1:i+1+width]
        candidates = [_typical_interval(side, p) for side in (left, right)]
        trusted = [value for value in candidates if value is not None]
        if trusted:
            estimates[i] = max(trusted)
            sources.append("independent_neighbor_side_median")
        else:
            pooled = _typical_interval(np.r_[left, right], p)
            if pooled is not None:
                estimates[i] = pooled
                sources.append("pooled_neighbor_median")
            else:
                sources.append("insufficient_or_irregular_neighbors")
    limits = np.full(len(intervals), base_limit)
    learned = np.isfinite(estimates)
    limits[learned] = np.maximum(base_limit, p["section_sampling_gap_factor"]*estimates[learned])
    if len(observed) > 1 and len(mids):
        observed_mids = (observed[:-1]+observed[1:])/2
        right = np.clip(np.searchsorted(mids, observed_mids), 0, len(mids)-1)
        left = np.maximum(0, right-1)
        nearest = np.where(np.abs(observed_mids-mids[left]) <= np.abs(observed_mids-mids[right]), left, right)
        allowed = limits[nearest]
    else:
        allowed = np.full(max(0, len(observed)-1), base_limit)
    differences = np.diff(observed)
    epsilon = 64*np.finfo(float).eps*max(voxel, float(np.max(np.abs(observed), initial=0.)))
    split = differences > allowed+epsilon
    runs = np.split(observed, np.flatnonzero(split)+1) if len(observed) else []
    diagnostic = dict(
        state="complete", training_only=True, current_interval_excluded_from_its_scale=True,
        training_layer_count=len(layers), layer_merge_tolerance_m=merge,
        training_layer_positions_m=layers.tolist(), training_intervals_m=intervals.tolist(),
        typical_intervals_m=[float(value) if np.isfinite(value) else None for value in estimates],
        scale_sources=sources, maximum_allowed_gaps_m=limits.tolist(),
        learned_interval_count=int(learned.sum()), original_voxel_gap_m=base_limit,
        learning_state="local_sampling_scale" if learned.any() else "voxel_rule_insufficient_layer_evidence",
        observed_run_count=len(runs), observed_split_gaps_m=differences[split].tolist(),
        observed_split_limits_m=allowed[split].tolist(),
        continuity_claim="Conditional on repeated local sampling; physical gaps below the reported allowance remain unresolved",
    )
    return runs, diagnostic


def analyze(points, train_mask, config):
    """Frozen section models/local gates plus sampling-aware observed runs."""
    p = policy(config)
    points, train_mask = np.asarray(points, float), np.asarray(train_mask)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or train_mask.shape != (len(points),) or train_mask.dtype != np.dtype(bool)):
        raise ValueError("Finite Nx3 points and a matching boolean training mask required")
    empty, consumed = np.empty((0, 2, 3)), np.zeros(len(points), dtype=bool)
    diagnostic = dict(state="complete", groups=[], accepted_surface_groups=0, unresolved_surface_groups=0,
        fit_uses_training_only=True, sampling_uses_training_only=True, semantic_identity_claimed=False,
        scope="Locally circular straight extrusions with training-sampling-conditioned continuity; not arbitrary topology or physical continuity qualification")
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
            training_circle_rmse=circle["circle_rmse"], training_line_rmse=circle["line_rmse"], runs=[])
        runs, sampling = sampling_runs(axial[shell], axial[shell & fold], p)
        detail["sampling_evidence"] = sampling
        if sampling["state"] == "unmeasurable":
            diagnostic["groups"].append(detail)
            return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": sampling["reason"]}
        accepted = 0
        for run in runs:
            low, high = float(run[0]), float(run[-1])
            if high-low < p["section_minimum_run_length_voxels"]*p["voxel_size"]:
                detail["runs"].append(dict(low=low, high=high, accepted=False, reason="below_minimum_run_length", slices=[]))
                continue
            checks, usable = [], True
            boundaries = np.linspace(low, high, p["section_slices"]+1)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                part = (axial >= start) & ((axial <= end) if i == len(boundaries)-2 else (axial < end))
                local = frozen._circle(xy[part & fold], high-low, p, numeric)
                validation = part & ~fold
                angular = frozen._angular_evidence(xy[validation & shell], circle["center"], p)
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
