"""Train-only repeated axial cadence for narrow-ridge support counting.

This is a conditional sampling model, not evidence that every sub-allowance
physical gap is absent. Geometry, radial tubes, and measured endpoints are not
expanded. A lattice must be supported by repeated training intervals; missing
cell-fold observations may account for integer multiples of the base cadence.
"""

from __future__ import annotations

import numpy as np

DEFAULTS = dict(ridge_sampling_layer_merge_voxels=.05, ridge_sampling_minimum_layers=6,
    ridge_sampling_lower_quantile=.25, ridge_sampling_cluster_relative_tolerance=.2,
    ridge_sampling_minimum_repeats=3, ridge_sampling_minimum_repeat_fraction=.2,
    ridge_sampling_minimum_lattice_fraction=.8, ridge_sampling_lattice_relative_tolerance=.15,
    ridge_sampling_maximum_pitch_voxels=8., ridge_sampling_gap_factor=1.75,
    ridge_sampling_maximum_points=200000, ridge_sampling_maximum_layers=20000)


def policy(config):
    p = {**DEFAULTS, **{k: v for k, v in config.items() if k.startswith("ridge_sampling_")}}
    if set(p) != set(DEFAULTS):
        raise ValueError("Unknown ridge-sampling policy")
    integers = {"ridge_sampling_minimum_layers", "ridge_sampling_minimum_repeats",
                "ridge_sampling_maximum_points", "ridge_sampling_maximum_layers"}
    for key, value in p.items():
        if key in integers:
            if type(value) is not int or not 1 <= value <= 200000:
                raise ValueError("Bounded positive integer ridge-sampling budgets required")
        elif isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Positive finite ridge-sampling scales required")
    if (not 3 <= p["ridge_sampling_minimum_layers"] <= p["ridge_sampling_maximum_layers"]
            or not 2 <= p["ridge_sampling_minimum_repeats"] < p["ridge_sampling_minimum_layers"]
            or not 0 < p["ridge_sampling_layer_merge_voxels"] < .25
            or not 0 < p["ridge_sampling_lower_quantile"] <= .5
            or not 0 < p["ridge_sampling_cluster_relative_tolerance"] < .5
            or not 0 < p["ridge_sampling_lattice_relative_tolerance"] < .5
            or not 0 < p["ridge_sampling_minimum_repeat_fraction"] <= 1
            or not 0 < p["ridge_sampling_minimum_lattice_fraction"] <= 1
            or not 1 <= p["ridge_sampling_maximum_pitch_voxels"] <= 16
            or not 1 < p["ridge_sampling_gap_factor"] <= 2):
        raise ValueError("Invalid training cadence policy")
    for key, default in (("voxel_size", .01), ("neighbor_radius_voxels", 1.75)):
        value = config.get(key, default)
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Positive finite geometry scales required")
        p[key] = value
    return p


def train_scale(training_projection, config):
    """Fit one bounded sampling model using training near-axis points only."""
    p = policy(config)
    training = np.asarray(training_projection, float)
    if training.ndim != 1 or not np.isfinite(training).all():
        raise ValueError("Finite one-dimensional training projection required")
    voxel = p["voxel_size"]
    result = dict(state="complete", training_only=True, training_layer_count=0,
        trained_pitch_m=None, bin_width_m=float(voxel),
        gap_limit_m=float(voxel*p["neighbor_radius_voxels"]),
        learning_state="voxel_fallback", reason=None, lattice_fraction=None,
        repeated_short_intervals=0,
        continuity_claim="Conditional on repeated training cadence; gaps below the reported allowance are not certified absent")
    if len(training) > p["ridge_sampling_maximum_points"]:
        return {**result, "state": "unmeasurable", "reason": "ridge_sampling_point_budget_exceeded"}
    training = np.unique(training)
    numeric = 64*np.finfo(float).eps*max(voxel, float(np.abs(training).max(initial=0.)))
    merge = max(numeric, p["ridge_sampling_layer_merge_voxels"]*voxel)
    parts = np.split(training, np.flatnonzero(np.diff(training) > merge)+1) if len(training) else []
    result["training_layer_count"] = len(parts)
    if len(parts) > p["ridge_sampling_maximum_layers"]:
        return {**result, "state": "unmeasurable", "reason": "ridge_sampling_layer_budget_exceeded"}
    if len(parts) < p["ridge_sampling_minimum_layers"]:
        return {**result, "reason": "insufficient_training_layers"}
    layers = np.asarray([np.median(part) for part in parts])
    intervals = np.diff(layers)
    lower = float(np.quantile(intervals, p["ridge_sampling_lower_quantile"]))
    cluster = intervals[np.abs(intervals-lower) <= p["ridge_sampling_cluster_relative_tolerance"]*lower+numeric]
    result["repeated_short_intervals"] = len(cluster)
    required = max(p["ridge_sampling_minimum_repeats"], int(np.ceil(p["ridge_sampling_minimum_repeat_fraction"]*len(intervals))))
    if len(cluster) < required:
        return {**result, "reason": "insufficient_repeated_training_intervals"}
    pitch = float(np.median(cluster))
    if pitch <= numeric or pitch > p["ridge_sampling_maximum_pitch_voxels"]*voxel+numeric:
        return {**result, "reason": "training_pitch_outside_bounded_model"}
    multiples = np.maximum(1., np.rint(intervals/pitch))
    residual = np.abs(intervals-multiples*pitch)
    fraction = float(np.mean(residual <= p["ridge_sampling_lattice_relative_tolerance"]*pitch+numeric))
    result["lattice_fraction"] = fraction
    if fraction < p["ridge_sampling_minimum_lattice_fraction"]:
        return {**result, "reason": "irregular_training_intervals"}
    result.update(trained_pitch_m=pitch, learning_state="voxel_dense_spacing")
    if pitch > voxel+numeric:
        result.update(bin_width_m=pitch,
            gap_limit_m=max(result["gap_limit_m"], p["ridge_sampling_gap_factor"]*pitch),
            learning_state="training_cadence")
    return result
