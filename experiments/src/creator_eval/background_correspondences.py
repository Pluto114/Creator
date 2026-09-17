"""RGB correspondence controls, with camera/mesh truth confined to separate callers."""

import hashlib

import numpy as np

from creator_eval.camera_diagnostics import sampson_distances


def _points(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("Expected finite Nx2 image coordinates")
    return value


def spatial_split(first_xy, second_xy, config):
    """Split whole source-image cells; never split by fitted residual or match quality."""
    first, second = _points(first_xy), _points(second_xy)
    if first.shape != second.shape or config["cell_size_px"] <= 0:
        raise ValueError("Invalid paired coordinates or cell size")
    cells = np.floor(first / config["cell_size_px"]).astype(np.int64)
    second_cells = np.floor(second / config["cell_size_px"]).astype(np.int64)
    validation = np.array([
        int.from_bytes(hashlib.sha256(f"{config['seed']}:{x}:{y}".encode()).digest()[:8], "big")
        % config["validation_modulus"] == config["validation_residue"] for x, y in cells
    ], dtype=bool)
    train_cells = {tuple(v) for v in second_cells[~validation]}
    val_cells = {tuple(v) for v in second_cells[validation]}
    return validation, {
        "train_count": int((~validation).sum()), "validation_count": int(validation.sum()),
        "source_train_cells": len({tuple(v) for v in cells[~validation]}),
        "source_validation_cells": len({tuple(v) for v in cells[validation]}),
        "target_cells_shared_across_split": len(train_cells & val_cells),
        "policy": "sha256(seed:source_cell_x:source_cell_y); no residual-dependent rebalance",
    }


def residual_summary(values):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    usable = values[finite]
    return {
        "sample_count": len(values), "finite_count": len(usable),
        "invalid_count": int((~finite).sum()),
        "median_px": float(np.median(usable)) if len(usable) else None,
        "p95_px": float(np.quantile(usable, .95)) if len(usable) else None,
        # Invalid residuals stay in the denominator, rather than vanishing as easy wins.
        "fraction_within_2px": float(np.mean(finite & (values <= 2))) if len(values) else None,
        "fraction_within_5px": float(np.mean(finite & (values <= 5))) if len(values) else None,
    }


def homography_distances(first_xy, second_xy, matrix):
    """Symmetric forward/backward transfer RMS, in original image pixels."""
    first, second = _points(first_xy), _points(second_xy)
    matrix = np.asarray(matrix, float)
    if first.shape != second.shape or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid homography inputs")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.full(len(first), np.nan)
    a = np.c_[first, np.ones(len(first))] @ matrix.T
    b = np.c_[second, np.ones(len(second))] @ inverse.T
    valid = (abs(a[:, 2]) > 1e-12) & (abs(b[:, 2]) > 1e-12)
    errors = np.full(len(first), np.nan)
    errors[valid] = np.sqrt((np.sum((a[valid, :2] / a[valid, 2, None] - second[valid])**2, axis=1)
                             + np.sum((b[valid, :2] / b[valid, 2, None] - first[valid])**2, axis=1)) / 2)
    return errors


def exclusion_mask(size_wh, guides, guide_y, half_width):
    width, height = size_wh
    mask = np.full((height, width), 255, np.uint8)
    start, end = guide_y
    if end <= start or half_width < 0:
        raise ValueError("Invalid exclusion strip")
    for y in range(max(0, start), min(height, end + 1)):
        for first, last in guides.values():
            center = first + (y - start) / (end - start) * (last - first)
            lo, hi = max(0, int(np.floor(center - half_width))), min(width, int(np.ceil(center + half_width + 1)))
            mask[y, lo:hi] = 0
    return mask


def detect_features(rgb, mask, detector, settings):
    import cv2

    factory = {"ORB": cv2.ORB_create, "SIFT": cv2.SIFT_create}[detector]
    points, descriptors = factory(**settings).detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)
    xy = np.array([point.pt for point in points], float).reshape(-1, 2)
    return {"xy": xy, "descriptors": descriptors, "sizes": np.array([p.size for p in points]),
            "angles": np.array([p.angle for p in points]), "responses": np.array([p.response for p in points])}


def mutual_ratio_matches(first, second, detector, ratio):
    import cv2

    a, b = first["descriptors"], second["descriptors"]
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return np.empty((0, 2), np.int64), np.empty(0)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if detector == "ORB" else cv2.NORM_L2)

    def select(query, train):
        return {m.queryIdx: (m.trainIdx, float(m.distance)) for pair in matcher.knnMatch(query, train, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance}

    forward, reverse = select(a, b), select(b, a)
    matches = [(i, j, distance) for i, (j, distance) in sorted(forward.items())
               if j in reverse and reverse[j][0] == i]
    return np.array([[i, j] for i, j, _ in matches], dtype=np.int64).reshape(-1, 2), np.array([v[2] for v in matches])


def fit_rgb_geometry(first_xy, second_xy, validation, config):
    """Only training matches reach OpenCV's robust estimators; validation never refits."""
    import cv2

    first, second = _points(first_xy), _points(second_xy)
    validation = np.asarray(validation, bool)
    if first.shape != second.shape or validation.shape != (len(first),):
        raise ValueError("Invalid geometry split")
    counts = {"match_count": len(first), "train_count": int((~validation).sum()), "validation_count": int(validation.sum())}
    if any(counts[key] < config[limit] for key, limit in
           [("match_count", "minimum_matches"), ("train_count", "minimum_train"), ("validation_count", "minimum_validation")]):
        return {"state": "inconclusive", "reason": "insufficient_matches_or_fixed_split_support", **counts, "models": {}}
    models = {}
    for kind in ("F", "H"):
        cv2.setRNGSeed(config["seed"])
        a, b = first[~validation], second[~validation]
        if kind == "F":
            matrix, inliers = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, config["ransac_threshold_px"], config["confidence"], config["maximum_iterations"])
        else:
            matrix, inliers = cv2.findHomography(a, b, cv2.RANSAC, config["ransac_threshold_px"], maxIters=config["maximum_iterations"], confidence=config["confidence"])
        if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            models[kind] = {"state": "degenerate", "reason": "robust_estimator_returned_no_single_finite_model"}
            continue
        error = sampson_distances(first, second, matrix) if kind == "F" else homography_distances(first, second, matrix)
        validation_report = residual_summary(error[validation])
        models[kind] = {"state": "validation_consistent" if validation_report["fraction_within_2px"] >= config["validation_fraction_within_2px"] else "validation_inconsistent",
                        "matrix": matrix, "train_inlier_mask": inliers.reshape(-1).astype(bool),
                        "train": residual_summary(error[~validation]), "validation": validation_report,
                        "all_residuals_px": error,
                        "singular_values": np.linalg.svd(matrix, compute_uv=False),
                        "error_kind": "Sampson" if kind == "F" else "symmetric_transfer_RMS"}
    f = models.get("F", {}).get("validation", {}).get("fraction_within_2px")
    h = models.get("H", {}).get("validation", {}).get("fraction_within_2px")
    planar = h is not None and h >= config["validation_fraction_within_2px"] and (f is None or h >= f - config["planar_fraction_margin"])
    return {"state": "assessed", **counts, "models": models,
            "planar_or_repetitive_explanation_possible": planar,
            "warning": "F/H self-consistency is not physical correspondence or calibrated camera accuracy; neither model changes cameras"}


def world_from_depth(xy, depth_z, camera):
    xy = _points(xy)
    depth_z = np.asarray(depth_z, float)
    if depth_z.shape != (len(xy),):
        raise ValueError("Invalid depth shape")
    inverse = np.linalg.inv(np.asarray(camera["world_to_camera_cv"]))
    camera_xyz = (np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(camera["K_index"]).T) * depth_z[:, None]
    return camera_xyz @ inverse[:3, :3].T + inverse[:3, 3]


def mesh_direction(first_xy, second_xy, first_camera, second_camera, cast, tolerance_m):
    """Evaluation only: first-hit world point, true reprojection, and target visibility.

    The injected mesh caster takes edge-origin pixels. Keypoints here use index
    coordinates, so the +0.5 belongs exactly at that boundary, not in K itself.
    """
    first, second = _points(first_xy), _points(second_xy)
    if first.shape != second.shape:
        raise ValueError("Mesh diagnostic requires matching coordinate arrays")
    depth, surface, _ = cast(first_camera, first + .5)
    hit = np.isfinite(depth) & (surface != 0)
    world = world_from_depth(first, depth, first_camera)
    transform = np.asarray(second_camera["world_to_camera_cv"])
    xyz = world @ transform[:3, :3].T + transform[:3, 3]
    projected = xyz @ np.asarray(second_camera["K_index"]).T
    with np.errstate(invalid="ignore", divide="ignore"):
        xy = projected[:, :2] / projected[:, 2, None]
    width, height = second_camera["size_wh"]
    eligible = hit & np.isfinite(xy).all(axis=1) & (xyz[:, 2] > second_camera["clip_start"]) & (xyz[:, 2] < second_camera["clip_end"])
    eligible &= (xy[:, 0] >= -.5) & (xy[:, 0] < width - .5) & (xy[:, 1] >= -.5) & (xy[:, 1] < height - .5)
    target_depth = np.full(len(first), np.nan)
    if eligible.any():
        target_depth[eligible] = cast(second_camera, xy[eligible] + .5)[0]
    visible = eligible & np.isfinite(target_depth) & (abs(target_depth - xyz[:, 2]) <= tolerance_m)
    occluded = eligible & np.isfinite(target_depth) & (target_depth < xyz[:, 2] - tolerance_m)
    inconsistent = eligible & np.isfinite(target_depth) & (target_depth > xyz[:, 2] + tolerance_m)
    error = np.linalg.norm(second - xy, axis=1)
    return {"source_hit": hit, "source_surface_id": surface, "target_in_frame": eligible,
            "occluded": occluded, "depth_inconsistent": inconsistent,
            "target_no_hit": eligible & ~np.isfinite(target_depth),
            "invalid_projection": hit & ~np.isfinite(xy).all(axis=1),
            "source_point_visible_in_target": visible, "expected_target_xy": xy,
            "transfer_error_px": np.where(visible, error, np.nan), "source_world": world}
