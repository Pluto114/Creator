"""Explicit fixed-lens prior and optional training-only outlier trimming."""
from __future__ import annotations

import numpy as np

from creator_eval.camera_bundle import (
    CameraGauge,
    centers,
    optimize_cameras,
    project,
    validate_cameras,
)


def shared_square_intrinsics(intrinsics):
    k = np.asarray(intrinsics, float).copy()
    focal = float(np.exp(np.mean(np.log(k[:, [0, 1], [0, 1]]))))
    k[:, 0, 0] = focal
    k[:, 1, 1] = focal
    # Principal points stay at their model values. Fixed lens / square pixels
    # is an explicit acquisition assumption, not a value copied from GT.
    return k


def trim_training(training, result, threshold, minimum_per_view):
    k, e = validate_cameras(result["intrinsics"], result["extrinsics"])
    points = {i: p for i, p in zip(result["kept_track_ids"], result["points"])}
    kept, removed, diagnostics = [], [], []
    for track in training:
        point = points.get(track["track_id"])
        errors, front = [], point is not None
        if point is not None:
            for observation in track["observations"]:
                view = observation["view"]
                uv, depth = project(np.asarray(point)[None], k[view], e[view])
                front &= bool(depth[0] > 0 and np.isfinite(uv).all())
                errors.append(float(np.linalg.norm(uv[0] - observation["xy"])))
        eligible = front and max(errors, default=np.inf) <= threshold
        (kept if eligible else removed).append(track)
        diagnostics.append(dict(track_id=track["track_id"], kept=bool(eligible), maximum_error_px=max(errors) if errors and np.isfinite(errors).all() else None))
    coverage = [sum(any(o["view"] == i for o in t["observations"]) for t in kept) for i in range(len(k))]
    return kept, dict(removed_track_ids=[t["track_id"] for t in removed], coverage=coverage,
                     enough_coverage=min(coverage) >= minimum_per_view, threshold_px=threshold, diagnostics=diagnostics)


def exceeds_initial_bounds(k, e, result, config):
    from scipy.spatial.transform import Rotation
    gauge = CameraGauge(k, e, True)
    final = np.asarray(result["extrinsics"])
    rotation = Rotation.from_matrix(final[:, :, :3] @ np.asarray(e)[:, :, :3].transpose(0, 2, 1)).as_rotvec()
    locations = (centers(final) - gauge.origin) / gauge.scale
    direction = locations[-1]
    denominator = float(direction @ gauge.centers[-1])
    if denominator <= 0:
        return True
    last_parameters = np.array([direction @ gauge.u, direction @ gauge.v]) / denominator
    displacement = locations[1:-1] - gauge.centers[1:-1]
    return bool(np.max(abs(rotation)) >= config["rotation_bound_rad"] - 1e-6
                or np.max(abs(displacement), initial=0) >= config["center_bound_baseline_fraction"] - 1e-6
                or np.max(abs(last_parameters)) >= config["center_bound_baseline_fraction"] - 1e-6
                or not config["focal_scale_bounds"][0] < result["focal_scale"] < config["focal_scale_bounds"][1])


def fit_shared(intrinsics, extrinsics, training, config, variant):
    k = shared_square_intrinsics(intrinsics) if variant["common_intrinsics"] else np.asarray(intrinsics).copy()
    if not variant["trim"]:
        return optimize_cameras(k, extrinsics, training, config, focal=variant["focal"], loss=variant["loss"])
    warm_config = {**config, "max_nfev": config["warm_max_nfev"]}
    warm = optimize_cameras(k, extrinsics, training, warm_config, focal=True, loss="soft_l1")
    if "extrinsics" not in warm:
        return warm
    kept, trim = trim_training(training, warm, config["trim_maximum_error_px"], config["minimum_trimmed_tracks_per_view"])
    if not trim["enough_coverage"]:
        return {**warm, "success": False, "state": "insufficient_support_after_training_trim", "trim": trim}
    next_config = {**config, "focal_scale_bounds": [v / warm["focal_scale"] for v in config["focal_scale_bounds"]]}
    result = optimize_cameras(np.asarray(warm["intrinsics"]), np.asarray(warm["extrinsics"]), kept, next_config, focal=True, loss="soft_l1")
    if "extrinsics" in result:
        result["discarded_track_ids"] = sorted(set(result["discarded_track_ids"]) | set(trim["removed_track_ids"]))
        result["focal_scale_second_stage"] = result["focal_scale"]
        result["focal_scale"] *= warm["focal_scale"]
        result["two_stage_exceeds_initial_bounds"] = exceeds_initial_bounds(k, extrinsics, result, config)
        result["at_parameter_bound"] |= result["two_stage_exceeds_initial_bounds"]
    result["trim"] = trim
    result["warm_start"] = {key: warm[key] for key in ("state", "nfev", "train_before", "train_after", "focal_scale")}
    return result
