"""Metric-connected training groups with frozen shell/context acceptance.

Actual bounded-diameter slabs, finite endpoints, and observed run splitting are
unchanged from section_context_evidence. Only transverse group connectivity
adds observed training-point distances at the original connection radius.
"""

from __future__ import annotations

import numpy as np

from . import section_evidence as frozen
from . import section_sampling_evidence as sampling
from . import section_supported_evidence as supported
from .common_readout_split import principal_axes
from .section_metric_grouping import groups as metric_groups

LOCAL_DEFAULTS = dict(
    section_local_resolution_voxels=.25,
    section_local_context_radius_radii=4.,
    section_local_maximum_contexts=8192,
    section_local_maximum_slabs=8192,
)
CONTEXT_DEFAULTS = dict(section_context_maximum_expansions=3)
DEFAULTS = {**supported.DEFAULTS, **LOCAL_DEFAULTS, **CONTEXT_DEFAULTS}


def policy(config):
    context = {**CONTEXT_DEFAULTS, **{key: value for key, value in config.items() if key.startswith("section_context_")}}
    if set(context) != set(CONTEXT_DEFAULTS):
        raise ValueError("Unknown section-context policy")
    expansions = context["section_context_maximum_expansions"]
    if type(expansions) is not int or not 1 <= expansions <= 5:
        raise ValueError("Bounded integer training-context expansion count required")
    config = {key: value for key, value in config.items() if not key.startswith("section_context_")}
    local = {**LOCAL_DEFAULTS, **{key: value for key, value in config.items() if key.startswith("section_local_")}}
    if set(local) != set(LOCAL_DEFAULTS):
        raise ValueError("Unknown local-section policy")
    for key, value in local.items():
        if key.endswith(("contexts", "slabs")):
            if type(value) is not int or not 1 <= value <= 200000:
                raise ValueError("Bounded local-section budget required")
        elif not np.isfinite(value) or value <= 0:
            raise ValueError("Positive finite local-section geometry required")
    if not .05 <= local["section_local_resolution_voxels"] <= .5 or not 1 <= local["section_local_context_radius_radii"] <= 8:
        raise ValueError("Bounded local-section resolution and context required")
    base = supported.policy({key: value for key, value in config.items() if not key.startswith("section_local_")})
    return {**base, **local, **context}


def _nearest_indices(sorted_values, values):
    right = np.clip(np.searchsorted(sorted_values, values), 0, len(sorted_values)-1)
    left = np.maximum(0, right-1)
    return np.where(np.abs(values-sorted_values[left]) <= np.abs(values-sorted_values[right]), left, right)


def _sampling_field(sampling_detail):
    layers = np.asarray(sampling_detail["training_layer_positions_m"])
    limits = np.asarray(sampling_detail["maximum_allowed_gaps_m"])
    pitches = np.asarray([value if value is not None else np.nan for value in sampling_detail["typical_intervals_m"]])
    return (layers[:-1]+layers[1:])/2, limits, sampling_detail["original_voxel_gap_m"], pitches


def _allowed_distances(values, field):
    mids, limits, base, _ = field
    if not len(limits):
        return np.full(len(values), base)
    return limits[_nearest_indices(mids, values)]


def _contexts(axial, xy, fold, shell, circle, sampling_field, p, numeric):
    """Choose from bounded training candidates, then validate exactly once."""
    training = np.sort(axial[fold & shell])
    low, high = float(training[0]), float(training[-1])
    radius = max(p["voxel_size"], p["section_local_context_radius_radii"]*circle["radius"])
    count = int(np.ceil((high-low)/radius))+1
    if count > p["section_local_maximum_contexts"]:
        return None, dict(state="unmeasurable", reason="section_local_context_budget_exceeded", context_count=count)
    anchors = np.linspace(low, high, count)
    order = np.argsort(axial, kind="stable")
    sorted_axial = axial[order]
    central = np.linalg.norm(xy-circle["center"], axis=1) <= p["section_center_support_radius_fraction"]*circle["radius"]
    mids, _, _, pitches = sampling_field
    contexts = []
    for center in anchors:
        pitch = float(pitches[_nearest_indices(mids, np.array([center]))[0]]) if len(mids) else np.nan
        minimum_radius = max(radius, p["section_sampling_neighbor_intervals"]*pitch) if np.isfinite(pitch) else radius
        candidates = np.unique(np.minimum(high-low, minimum_radius*2.**np.arange(p["section_context_maximum_expansions"])))
        attempts = []
        for half_width in candidates:
            start, end = center-half_width, center+half_width
            part = order[np.searchsorted(sorted_axial, start):np.searchsorted(sorted_axial, end, side="right")]
            train_part = part[fold[part]]
            fit_part = train_part[~central[train_part]]
            training_xy = xy[fit_part]
            # Do not prefilter to shell inliers: ordinary radial outliers must
            # still count against the original circle fit/inlier fraction.
            local = frozen._circle(training_xy, high-low, p, numeric)
            identification = (supported._training_identifiability(training_xy, local, p) if local is not None
                              else dict(accepted=False, reason="no_training_curved_section"))
            drift = float(np.linalg.norm(local["center"]-circle["center"])) if local else None
            radius_drift = abs(local["radius"]-circle["radius"]) if local else None
            trained = (local is not None and identification["accepted"]
                       and drift <= p["section_center_drift_radius_fraction"]*circle["radius"]
                       and radius_drift <= p["section_radius_drift_fraction"]*circle["radius"])
            attempts.append(dict(half_width_m=float(half_width), training_points=len(training_xy),
                training_identifiability=identification, center_drift=drift, radius_drift=radius_drift, accepted=bool(trained)))
            if trained:
                break
        # No held-out coordinate, count, angular coverage or residual chooses a
        # candidate. A held-out failure cannot trigger another expansion.
        validation_raw = part[~fold[part]]
        validation = validation_raw[~central[validation_raw]]
        prediction = supported._validation_prediction(xy[validation], shell[validation], circle["center"], p)
        contexts.append(dict(center=float(center), window=[float(start), float(end)],
            training_points=len(training_xy), raw_training_points=len(train_part),
            training_center_points=int(central[train_part].sum()), training_shell_points=int(shell[train_part].sum()),
            training_other_points=int((~(central[train_part] | shell[train_part])).sum()),
            validation_center_points=int(central[validation_raw].sum()), raw_validation_points=len(validation_raw),
            validation_other_points=int((~(central[validation_raw] | shell[validation_raw])).sum()),
            training_identifiability=identification, validation_prediction=prediction,
            center_drift=drift, radius_drift=radius_drift, accepted=bool(trained and prediction["accepted"]),
            training_pitch_m=pitch if np.isfinite(pitch) else None, minimum_context_radius_m=float(minimum_radius),
            training_candidates=attempts, selected_candidate=len(attempts)-1, heldout_evaluations=1,
            center_support_policy="Global training model classifies near-axis compatibility; ordinary radial outliers remain in fit and validation"))
    return contexts, dict(state="complete", context_count=count, context_radius_m=radius,
        context_anchor_policy="Uniform training-extent anchors; training-pitch minimum radius and train-only finite candidate selection",
        expansion_multipliers=(2.**np.arange(p["section_context_maximum_expansions"])).tolist(),
        context_selection_uses_validation=False,
        measurement_resolution_m=max(numeric, p["section_local_resolution_voxels"]*p["voxel_size"]),
        training_endpoint_allowances_m=_allowed_distances(np.array([low, high]), sampling_field).tolist())


def analyze(points, train_mask, config):
    p = policy(config)
    sampling_policy = {key: value for key, value in p.items()
                       if not key.startswith(("section_local_", "section_supported_", "section_context_"))}
    points, train_mask = np.asarray(points, float), np.asarray(train_mask)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or train_mask.shape != (len(points),) or train_mask.dtype != np.dtype(bool)):
        raise ValueError("Finite Nx3 points and a matching boolean training mask required")
    empty, consumed = np.empty((0, 2, 3)), np.zeros(len(points), dtype=bool)
    diagnostic = dict(state="complete", groups=[], accepted_surface_groups=0, unresolved_surface_groups=0,
        fit_uses_training_only=True, sampling_uses_training_only=True, windows_use_training_only=True,
        semantic_identity_claimed=False,
        scope="Training-fixed circular models and bounded-diameter actual axial slabs; neither physical identity nor sub-resolution surface continuity is claimed")
    if len(points) > p["section_maximum_points"]:
        return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": "section_point_budget_exceeded"}
    if train_mask.sum() < 9:
        return empty, consumed, {**diagnostic, "reason": "insufficient_training_support"}
    numeric = 256*np.finfo(float).eps*max(1., float(np.abs(points[train_mask]).max()))
    anchor, eigen, _, axis = principal_axes(points[train_mask])
    if eigen[-1] <= numeric**2:
        return empty, consumed, {**diagnostic, "reason": "degenerate_training_extent"}
    groups, grouping = metric_groups(points, train_mask, anchor, axis, p)
    diagnostic.update(grouping)
    if groups is None:
        return empty, consumed, diagnostic
    output, slab_count = [], 0
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
        training = np.sort(axial[shell & fold])
        observed_runs, sampling_detail = sampling.sampling_runs(axial[shell], training, sampling_policy)
        detail = dict(group=number, state="surface_unresolved", points=len(cloud), consumed_points=int((shell | centers).sum()),
            retained_radial_outliers=int((~(shell | centers)).sum()), training_shell_fraction=circle["shell_fraction"],
            validation_shell_fraction=float(shell[~fold].mean()) if (~fold).any() else None,
            trained_center=center_world.tolist(), trained_axis=direction.tolist(), trained_radius=circle["radius"],
            training_circle_rmse=circle["circle_rmse"], training_line_rmse=circle["line_rmse"],
            sampling_evidence=sampling_detail, runs=[])
        diagnostic["groups"].append(detail)
        if sampling_detail["state"] == "unmeasurable":
            return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": sampling_detail["reason"]}
        sampling_field = _sampling_field(sampling_detail)
        contexts, context_detail = _contexts(axial, xy, fold, shell, circle, sampling_field, p, numeric)
        detail["local_evidence"] = context_detail
        if contexts is None:
            return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": context_detail["reason"]}
        detail["training_contexts"] = contexts
        context_centers = np.asarray([context["center"] for context in contexts])
        resolution = context_detail["measurement_resolution_m"]
        order = np.argsort(axial, kind="stable")
        sorted_axial = axial[order]
        accepted = 0
        for observed in observed_runs:
            low, high = float(observed[0]), float(observed[-1])
            run_detail = dict(low=low, high=high, accepted=False, slices=[], reported_segments=[])
            detail["runs"].append(run_detail)
            if high-low < p["section_minimum_run_length_voxels"]*p["voxel_size"]:
                run_detail["reason"] = "below_minimum_observed_run_length"
                continue
            members = order[np.searchsorted(sorted_axial, low):np.searchsorted(sorted_axial, high, side="right")]
            shell_members = members[shell[members]]
            shell_axial = axial[shell_members]
            position, blocks, block = 0, [], None
            while position < len(shell_members):
                slab_count += 1
                if slab_count > p["section_local_maximum_slabs"]:
                    return empty, consumed, {**diagnostic, "state": "unmeasurable", "reason": "section_local_slab_budget_exceeded"}
                slab_low = float(axial[shell_members[position]])
                stop = int(np.searchsorted(shell_axial, slab_low+resolution, side="right"))
                slab = shell_members[position:stop]
                slab_high = float(axial[slab[-1]])
                center = (slab_low+slab_high)/2
                context_index = int(_nearest_indices(context_centers, np.array([center]))[0])
                angular = frozen._angular_evidence(xy[slab], circle["center"], p)
                endpoints = np.array([slab_low, slab_high])
                distances = np.abs(endpoints-training[_nearest_indices(training, endpoints)])
                allowances = _allowed_distances(endpoints, sampling_field)
                in_envelope = bool(np.all(distances <= allowances+numeric))
                valid = bool(angular["accepted"] and contexts[context_index]["accepted"] and in_envelope)
                run_detail["slices"].append({**angular, "low": slab_low, "high": slab_high,
                    "measurement_points": len(slab), "training_context": context_index,
                    "training_distances_m": distances.tolist(), "training_allowances_m": allowances.tolist(),
                    "within_training_envelope": in_envelope, "accepted": valid,
                    "angular_evidence_scope": "one_actual_slab_with_train_fixed_maximum_diameter"})
                if valid:
                    block = [slab_low, slab_high] if block is None else [block[0], slab_high]
                elif block is not None:
                    blocks.append(block)
                    block = None
                position = stop
            if block is not None:
                blocks.append(block)
            for start, end in blocks:
                if end-start >= p["section_minimum_run_length_voxels"]*p["voxel_size"]:
                    output.append(np.array([center_world+start*direction, center_world+end*direction]))
                    run_detail["reported_segments"].append([start, end])
                    accepted += 1
            run_detail["accepted"] = bool(run_detail["reported_segments"] and all(item["accepted"] for item in run_detail["slices"]))
            if not run_detail["accepted"]:
                run_detail["reason"] = "local_support_incomplete" if run_detail["reported_segments"] else "no_long_locally_supported_run"
        if accepted:
            detail["state"] = "accepted_surface"
            detail["accepted_runs"] = accepted
            diagnostic["accepted_surface_groups"] += 1
        else:
            diagnostic["unresolved_surface_groups"] += 1
    diagnostic["consumed_points"] = int(consumed.sum())
    diagnostic["measured_slabs"] = slab_count
    return np.asarray(output).reshape(-1, 2, 3), consumed, diagnostic
