"""Train-fitted local sheet evidence, independent of a persistent row lattice."""

import numpy as np

from . import ridge_sampling_evidence

DEFAULTS = dict(ridge_plane_minimum_inlier_fraction=.9, ridge_plane_thickness_voxels=.25,
    ridge_plane_minimum_station_fraction=.6, ridge_plane_minimum_transverse_stations=3,
    ridge_plane_transverse_separation_voxels=1., ridge_plane_window_steps=2.,
    ridge_plane_maximum_points=200000, ridge_plane_maximum_stations=4096)


def policy(config):
    p = {**DEFAULTS, **{k: v for k, v in config.items() if k.startswith("ridge_plane_")}}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown ridge plane policy")
    for key, value in p.items():
        if key in ("ridge_plane_minimum_transverse_stations", "ridge_plane_maximum_points", "ridge_plane_maximum_stations"):
            if type(value) is not int or not 1 <= value <= 200000:
                raise ValueError("Bounded positive integer plane budgets required")
        elif isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Positive finite plane geometry required")
    if (p["ridge_plane_minimum_transverse_stations"] < 3
            or p["ridge_plane_maximum_stations"] > 20000
            or p["ridge_plane_minimum_station_fraction"] > 1
            or p["ridge_plane_minimum_inlier_fraction"] > 1):
        raise ValueError("Invalid plane support policy")
    return p


def _spread_count(values, separation):
    ordered = np.sort(values)
    count, last = 0, -np.inf
    for value in ordered:
        if value-last >= separation:
            count, last = count+1, value
    return count


def analyze(points, train_mask, segment, config):
    p = policy(config)
    points, mask, segment = np.asarray(points, float), np.asarray(train_mask), np.asarray(segment, float)
    voxel = config["voxel_size"]
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or mask.shape != (len(points),) or mask.dtype != np.dtype(bool)
            or segment.shape != (2, 3) or not np.isfinite(segment).all() or not np.isfinite(voxel) or voxel <= 0):
        raise ValueError("Finite points, boolean folds and segment required")
    diagnostic = dict(state="complete", planar_sheet=False, training_only_proposals=True,
        validation_refits_model=False, training_model=None, validation_station_fraction=None)
    if len(points) > p["ridge_plane_maximum_points"]:
        return {**diagnostic, "state": "unmeasurable", "reason": "ridge_plane_point_budget_exceeded"}
    axis = segment[1]-segment[0]
    length = np.linalg.norm(axis)
    if length <= 0 or not np.isfinite(length):
        raise ValueError("Nondegenerate finite segment required")
    axis /= length
    if axis[np.argmax(np.abs(axis))] < 0:
        segment, axis = segment[::-1], -axis
    training, validation = np.unique(points[mask], axis=0), np.unique(points[~mask], axis=0)
    t = (training-segment[0]) @ axis
    radial = np.linalg.norm(training-segment[0]-t[:, None]*axis, axis=1)
    numeric = 256*np.finfo(float).eps*max(1., length, float(np.abs(training).max(initial=0.)))
    core = (radial <= config.get("inner_radius_voxels", .15)*voxel) & (t >= -numeric) & (t <= length+numeric)
    if core.sum() < 6:
        return {**diagnostic, "reason": "insufficient_training_core"}
    low, high = float(t[core].min()), float(t[core].max())
    scale = ridge_sampling_evidence.train_scale(t[core], config)
    if scale["state"] != "complete":
        return {**diagnostic, "state": "unmeasurable", "reason": scale["reason"]}
    radius = (high-low)*config.get("ridge_array_context_span_fraction", .5)
    window = (t >= low-numeric) & (t <= high+numeric) & (radial <= radius)
    local = training[window]
    if len(local) < 9:
        return {**diagnostic, "reason": "insufficient_training_neighborhood"}
    center = local.mean(axis=0)
    _, vectors = np.linalg.eigh((local-center).T @ (local-center)/len(local))
    normal = vectors[:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    if abs(normal @ axis) > np.sin(np.deg2rad(3.)):
        return {**diagnostic, "reason": "plane_not_parallel_to_candidate"}
    thickness = p["ridge_plane_thickness_voxels"]*voxel
    inliers = np.abs((local-center) @ normal) <= thickness
    if float(inliers.mean()) < p["ridge_plane_minimum_inlier_fraction"]:
        return {**diagnostic, "reason": "training_not_a_thin_plane"}
    transverse = np.cross(normal, axis)
    transverse /= np.linalg.norm(transverse)
    step = scale["bin_width_m"]
    count = int(np.floor((high-low)/step))+1
    if count > p["ridge_plane_maximum_stations"]:
        return {**diagnostic, "state": "unmeasurable", "reason": "ridge_plane_station_budget_exceeded"}
    if count < 6:
        return {**diagnostic, "reason": "insufficient_training_stations"}
    stations = low+np.arange(count)*step
    halfwidth = p["ridge_plane_window_steps"]*step
    separation = p["ridge_plane_transverse_separation_voxels"]*voxel
    lt, lu = t[window][inliers], (local[inliers]-center) @ transverse
    train_counts = [_spread_count(lu[np.abs(lt-at) <= halfwidth], separation) for at in stations]
    train_supported = np.asarray(train_counts) >= p["ridge_plane_minimum_transverse_stations"]
    model = dict(origin=center.tolist(), normal=normal.tolist(), axis=axis.tolist(), transverse=transverse.tolist(),
        axial_bounds=[low, high], context_radius=radius, halfwidth=halfwidth, station_count=count,
        transverse_counts=train_counts, training_station_fraction=float(train_supported.mean()), sampling_scale=scale)
    diagnostic["training_model"] = model
    if train_supported.mean() < p["ridge_plane_minimum_station_fraction"]:
        return {**diagnostic, "reason": "no_persistent_training_sheet_width"}
    vt = (validation-segment[0]) @ axis
    vr = np.linalg.norm(validation-segment[0]-vt[:, None]*axis, axis=1)
    vwindow = (vt >= low-numeric) & (vt <= high+numeric) & (vr <= radius)
    heldout = validation[vwindow]
    if not len(heldout):
        return {**diagnostic, "reason": "no_validation_neighborhood"}
    vin = np.abs((heldout-center) @ normal) <= thickness
    if vin.mean() < p["ridge_plane_minimum_inlier_fraction"]:
        return {**diagnostic, "reason": "validation_plane_not_confirmed"}
    vt, vu = vt[vwindow][vin], (heldout[vin]-center) @ transverse
    validated = np.asarray([_spread_count(vu[np.abs(vt-at) <= halfwidth], separation)
        >= p["ridge_plane_minimum_transverse_stations"] for at in stations])
    fraction = float(np.mean(validated & train_supported))
    return {**diagnostic, "planar_sheet": fraction >= p["ridge_plane_minimum_station_fraction"],
        "validation_station_fraction": fraction, "reason": "independently_checked_local_plane"}
