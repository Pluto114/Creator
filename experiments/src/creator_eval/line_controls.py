"""Small straight-line controls and arc-length diagnostics; never a GT-guided fitter."""

import numpy as np


class LineFitDegenerate(ValueError):
    """The supplied observations do not determine one usable line."""


def fit_line_tls(points, *, min_points=3, min_extent=1e-12, min_eigen_gap_ratio=1e-8):
    """Equal-weight 3D TLS; endpoints are the full observed projection extent.

    Point selection belongs to the method and must use allowed RGB/predictions only.
    No clipping, GT matching, confidence weighting, or best-looking subset happens here.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point array")
    if min_points < 2 or min_extent <= 0 or not 0 <= min_eigen_gap_ratio < 1:
        raise ValueError("Invalid line-fit degeneracy policy")
    if len(points) < min_points:
        raise LineFitDegenerate(f"too_few_points: {len(points)} < {min_points}")
    if not np.isfinite(points).all():
        raise LineFitDegenerate("nonfinite_points")
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0.0)
    if values[-1] <= min_extent**2:
        raise LineFitDegenerate("coincident_or_tiny_extent")
    eigen_gap_ratio = float((values[-1] - values[-2]) / values[-1])
    if eigen_gap_ratio <= min_eigen_gap_ratio:
        raise LineFitDegenerate("ambiguous_principal_axis")
    direction = vectors[:, -1]
    # 特征向量正负都对；固定符号只是让重复运行的端点顺序别乱跳。
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    projection = centered @ direction
    extent = float(np.ptp(projection))
    if extent <= min_extent:
        raise LineFitDegenerate("coincident_or_tiny_extent")
    segment = centroid + np.array([projection.min(), projection.max()])[:, None] * direction
    residuals = centered - projection[:, None] * direction
    return {
        "segment": segment,
        "centroid": centroid,
        "direction": direction,
        "point_count": len(points),
        "extent": extent,
        "perpendicular_rmse": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
        "principal_variance_fraction": float(values[-1] / values.sum()),
        "eigen_gap_ratio": eigen_gap_ratio,
        "endpoint_policy": "observed_projection_min_max_no_trimming",
        "weighting": "equal_weight_per_input_point_no_deduplication",
    }


def _segments(value, name="segments"):
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    if value.shape == (2, 3):
        value = value[None]
    if value.ndim != 3 or value.shape[1:] != (2, 3) or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite Nx2x3 segments")
    lengths = np.linalg.norm(value[:, 1] - value[:, 0], axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError(f"{name}: zero_or_tiny_length_segment")
    return value


def _reject_overlapping_segments(segments):
    # 相接的两段正常；把同一条线复制十份可不该给它十倍分量。
    # 这里只拒绝正长度的共线重叠，不假装实现了通用曲线去重。
    for index, first in enumerate(segments):
        axis = first[1] - first[0]
        length = float(np.linalg.norm(axis))
        unit = axis / length
        for second in segments[index + 1:]:
            second_axis = second[1] - second[0]
            second_length = float(np.linalg.norm(second_axis))
            numeric_tolerance = 1e-10 * max(1.0, length, second_length)
            if np.linalg.norm(np.cross(unit, second_axis / second_length)) > 1e-10:
                continue
            if np.max(np.linalg.norm(np.cross(second - first[0], unit), axis=1)) > numeric_tolerance:
                continue
            projected = (second - first[0]) @ unit
            overlap = min(length, float(projected.max())) - max(0.0, float(projected.min()))
            if overlap > numeric_tolerance:
                raise ValueError("overlapping_segments: arc-length multiplicity is not supported")


def sample_segments_midpoint(segments, spacing):
    """Sample bounded cells at their midpoints, weighted by each cell's exact length.

    Spacing is a maximum cell length. Splitting a line can change quadrature sites,
    but total weight remains its arc length; no sample-count weighting is used.
    Positive-length collinear overlap within the collection is rejected.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    segments = _segments(segments)
    _reject_overlapping_segments(segments)
    points, weights = [], []
    for start, end in segments:
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / spacing)))
        fractions = (np.arange(count) + 0.5) / count
        points.append(start + fractions[:, None] * (end - start))
        weights.append(np.full(count, length / count))
    if not points:
        return np.empty((0, 3)), np.empty(0)
    return np.concatenate(points), np.concatenate(weights)


def point_to_segments_distance(points, segments):
    """Exact point-to-finite-segment distance, not distance to an infinite line."""
    points = np.asarray(points, dtype=np.float64)
    segments = _segments(segments)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 points")
    if not len(segments):
        return np.full(len(points), np.inf)
    axis = segments[:, 1] - segments[:, 0]
    square_length = np.sum(axis**2, axis=1)
    output = np.empty(len(points))
    for begin in range(0, len(points), 4096):
        chunk = points[begin:begin + 4096]
        offsets = chunk[:, None, :] - segments[None, :, 0, :]
        fraction = np.sum(offsets * axis[None], axis=2) / square_length
        closest = segments[None, :, 0, :] + np.clip(fraction, 0, 1)[..., None] * axis[None]
        output[begin:begin + len(chunk)] = np.linalg.norm(chunk[:, None] - closest, axis=2).min(axis=1)
    return output


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    return float(values[order[min(index, len(order) - 1)]])


def _direction_metrics(source, target, *, tolerance, spacing, empty_source, empty_target):
    points, weights = sample_segments_midpoint(source, spacing)
    total_length = float(weights.sum())
    result = {
        "source_length": total_length,
        "sample_count": len(points),
        "covered_length": 0.0,
        "covered_fraction": None,
        "distance_median": None,
        "distance_p95": None,
        "reason": None,
    }
    if not len(points):
        result["reason"] = empty_source
        return result
    if not len(target):
        result.update(covered_fraction=0.0, reason=empty_target)
        return result
    distance = point_to_segments_distance(points, target)
    covered_length = float(weights[distance <= tolerance].sum())
    result.update(
        covered_length=covered_length,
        covered_fraction=covered_length / total_length,
        distance_median=_weighted_quantile(distance, weights, 0.5),
        distance_p95=_weighted_quantile(distance, weights, 0.95),
    )
    return result


def curve_metrics(predicted_segments, truth_segments, *, tolerance, spacing):
    """Arc-weighted R/P and separate directed errors for already-aligned curves.

    All distances/lengths use the caller's coordinate units. This routine does not
    align, match branches, select points, or infer graph topology.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    predicted = _segments(predicted_segments, "prediction")
    truth = _segments(truth_segments, "truth")
    _reject_overlapping_segments(predicted)
    _reject_overlapping_segments(truth)
    gt_to_prediction = _direction_metrics(
        truth, predicted, tolerance=tolerance, spacing=spacing,
        empty_source="empty_truth", empty_target="empty_prediction",
    )
    prediction_to_gt = _direction_metrics(
        predicted, truth, tolerance=tolerance, spacing=spacing,
        empty_source="empty_prediction", empty_target="empty_truth",
    )
    return {
        "recovery_fraction": gt_to_prediction["covered_fraction"],
        "precision_fraction": prediction_to_gt["covered_fraction"],
        "false_predicted_length": (
            prediction_to_gt["source_length"] - prediction_to_gt["covered_length"]
        ),
        "truth_to_prediction": gt_to_prediction,
        "prediction_to_truth": prediction_to_gt,
        "tolerance": float(tolerance),
        "sampling_max_spacing": float(spacing),
        "sampling": "per_segment_equal_cells_midpoint_exact_arc_weights",
        "overlap_policy": "reject_positive_length_collinear_overlap_within_each_collection",
        "units": "input_coordinate_units",
        "scope": "native_curve_geometry_diagnostic_not_graph_topology_or_pointcloud_recovery",
    }


def gap_coverage(predicted_segments, gap_segment, *, tolerance, spacing, endpoint_guard=None):
    """Prediction proximity over a real empty gap, not a topological connection rate.

    Also score the gap interior, removing endpoint_guard length at each end. By
    default the guard equals tolerance so correct rod endpoints alone do not look
    like a false bridge. A too-short interior is explicitly unmeasurable.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    guard = float(tolerance if endpoint_guard is None else endpoint_guard)
    if not np.isfinite(guard) or guard < 0:
        raise ValueError("endpoint_guard must be finite and nonnegative")
    predicted = _segments(predicted_segments, "prediction")
    gap = _segments(gap_segment, "gap")
    if len(gap) != 1:
        raise ValueError("Expected exactly one gap segment")
    _reject_overlapping_segments(predicted)
    whole = _direction_metrics(
        gap, predicted, tolerance=tolerance, spacing=spacing,
        empty_source="empty_gap", empty_target="empty_prediction",
    )
    axis = gap[0, 1] - gap[0, 0]
    length = float(np.linalg.norm(axis))
    # 真实断口边上本来就该有杆。容差扫到端点不能直接说算法乱补了。
    if length <= 2 * guard + 1e-12:
        interior = {
            "source_length": 0.0, "sample_count": 0, "covered_length": 0.0,
            "covered_fraction": None, "distance_median": None, "distance_p95": None,
            "reason": "gap_not_longer_than_two_endpoint_guards",
        }
    else:
        direction = axis / length
        guarded = np.array([gap[0, 0] + guard * direction, gap[0, 1] - guard * direction])
        interior = _direction_metrics(
            guarded[None], predicted, tolerance=tolerance, spacing=spacing,
            empty_source="empty_gap_interior", empty_target="empty_prediction",
        )
    return {
        "whole_gap": whole,
        "guarded_interior": interior,
        "endpoint_guard": guard,
        "tolerance": float(tolerance),
        "sampling_max_spacing": float(spacing),
        "units": "input_coordinate_units",
        "scope": "geometric_gap_proximity_not_topological_wrong_connection_rate",
    }


def fit_multiview_line(
    image_lines, intrinsics, extrinsics, *,
    min_second_to_first_ratio=1e-5, min_direction_gap_ratio=1e-8,
):
    """Intersect estimated-camera backprojection planes without reading truth.

    image_lines are homogeneous (a,b,c), in each native camera's pixel grid.
    Extrinsics map world to camera. Endpoints/visibility must be supplied by a
    separate RGB-supported method; this function returns an infinite line only.
    """
    lines = np.asarray(image_lines, dtype=np.float64)
    cameras = np.asarray(extrinsics, dtype=np.float64)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    if lines.ndim != 2 or lines.shape[1] != 3:
        raise ValueError("Expected Nx3 homogeneous image lines")
    if len(lines) < 3:
        raise LineFitDegenerate("too_few_views: need at least three image lines")
    if cameras.shape == (len(lines), 4, 4):
        cameras = cameras[:, :3]
    if calibration.shape == (3, 3):
        calibration = np.broadcast_to(calibration, (len(lines), 3, 3))
    if cameras.shape != (len(lines), 3, 4) or calibration.shape != (len(lines), 3, 3):
        raise ValueError("Expected one 3x3 intrinsic and 3x4/4x4 extrinsic per line")
    if not (
        np.isfinite(lines).all() and np.isfinite(cameras).all()
        and np.isfinite(calibration).all()
    ):
        raise LineFitDegenerate("nonfinite_line_or_camera")
    if not 0 < min_second_to_first_ratio < 1 or not 0 <= min_direction_gap_ratio < 1:
        raise ValueError("Invalid multiview degeneracy policy")
    if np.any(np.linalg.norm(lines[:, :2], axis=1) <= 1e-12):
        raise LineFitDegenerate("invalid_image_line")
    projection = calibration @ cameras
    planes = np.einsum("nij,ni->nj", projection, lines)
    normal_lengths = np.linalg.norm(planes[:, :3], axis=1)
    if np.any(normal_lengths <= 1e-12):
        raise LineFitDegenerate("zero_backprojection_plane")
    planes = planes / normal_lengths[:, None]
    normals, offsets = planes[:, :3], planes[:, 3]
    _, singular, vt = np.linalg.svd(normals, full_matrices=False)
    if singular[0] <= 1e-12 or singular[1] / singular[0] < min_second_to_first_ratio:
        raise LineFitDegenerate("coincident_or_nearly_parallel_backprojection_planes")
    direction_gap_ratio = float((singular[1] - singular[2]) / singular[1])
    if direction_gap_ratio <= min_direction_gap_ratio:
        raise LineFitDegenerate("ambiguous_multiview_direction")
    direction = vt[-1]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    # 平面有噪声时不会真交成一条线。先固定方向，再只在其正交平面找锚点，
    # 不往原点约束里塞一个随便的“大权重”来假装约束成立。
    orthogonal_basis = vt[:2].T
    coordinates = np.linalg.lstsq(normals @ orthogonal_basis, -offsets, rcond=None)[0]
    anchor = orthogonal_basis @ coordinates
    residuals = normals @ anchor + offsets
    return {
        "anchor": anchor,
        "direction": direction,
        "view_count": len(lines),
        "singular_values": singular,
        "second_to_first_singular_ratio": float(singular[1] / singular[0]),
        "direction_gap_ratio": direction_gap_ratio,
        "plane_residuals": residuals,
        "plane_rmse": float(np.sqrt(np.mean(residuals**2))),
        "direction_plane_residuals": normals @ direction,
        "anchor_constraint": "dot_anchor_direction_zero",
        "scope": "infinite_line_no_endpoint_or_gap_inference",
    }


def closest_ray_line_parameters(
    anchor, direction, ray_origins, ray_directions, *, minimum_parallel_sine_squared=1e-8,
):
    """Closest points of a forward ray and an infinite line, with explicit validity.

    The parameterization stays anchor + t * direction and origin + s * ray_direction;
    directions are not normalized behind the caller's back. A negative ray parameter
    is invalid observation support. Parallel/invalid rays get NaN parameters.
    """
    anchor, direction = np.asarray(anchor, float), np.asarray(direction, float)
    origins, rays = np.asarray(ray_origins, float), np.asarray(ray_directions, float)
    if anchor.shape != (3,) or direction.shape != (3,):
        raise ValueError("Expected one 3D anchor and direction")
    if origins.ndim != 2 or origins.shape[1] != 3 or rays.shape != origins.shape:
        raise ValueError("Expected matching Nx3 ray origins and directions")
    if not np.isfinite(anchor).all() or not np.isfinite(direction).all():
        raise ValueError("Line anchor and direction must be finite")
    direction_square = float(direction @ direction)
    if direction_square <= 1e-24:
        raise LineFitDegenerate("zero_line_direction")
    if not 0 < minimum_parallel_sine_squared < 1:
        raise ValueError("Parallel-ray tolerance must be between zero and one")
    ray_square = np.sum(rays**2, axis=1)
    mixed = rays @ direction
    determinant = direction_square * ray_square - mixed**2
    geometric_valid = (
        np.isfinite(origins).all(axis=1) & np.isfinite(rays).all(axis=1)
        & (ray_square > 1e-24)
        & (determinant > minimum_parallel_sine_squared * direction_square * ray_square)
    )
    line_t, ray_s, separation = (np.full(len(rays), np.nan) for _ in range(3))
    offsets = anchor - origins[geometric_valid]
    offset_dot_line = offsets @ direction
    offset_dot_ray = np.sum(offsets * rays[geometric_valid], axis=1)
    denom = determinant[geometric_valid]
    line_t[geometric_valid] = (
        mixed[geometric_valid] * offset_dot_ray - ray_square[geometric_valid] * offset_dot_line
    ) / denom
    ray_s[geometric_valid] = (
        direction_square * offset_dot_ray - mixed[geometric_valid] * offset_dot_line
    ) / denom
    delta = (
        anchor + line_t[geometric_valid, None] * direction - origins[geometric_valid]
        - ray_s[geometric_valid, None] * rays[geometric_valid]
    )
    separation[geometric_valid] = np.linalg.norm(delta, axis=1)
    # 方程能解不等于像素真的看到它：落在相机背后的交会不能拿来投票。
    valid = geometric_valid & (ray_s >= 0)
    return {
        "line_t": line_t,
        "ray_s": ray_s,
        "valid": valid,
        "closest_separation": separation,
    }


def support_intervals(
    per_view_t, bin_width, minimum_views=3, dilation_bins=1, minimum_run_bins=3,
):
    """Fixed-grid t intervals supported by distinct views, never by point density.

    Grid origin is floor(min(t) / bin_width) * bin_width. Occupancy is dilated in
    each view before voting. Endpoints are cell boundaries and can extend beyond
    observations by dilation; that is part of the protocol, not an endpoint fit.
    A caller must supply one array per distinct view and remove invalid ray matches.
    """
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    for name, value, lower in (
        ("minimum_views", minimum_views, 1),
        ("dilation_bins", dilation_bins, 0),
        ("minimum_run_bins", minimum_run_bins, 1),
    ):
        if not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    views = [np.asarray(value, dtype=np.float64) for value in per_view_t]
    if any(value.ndim != 1 or not np.isfinite(value).all() for value in views):
        raise ValueError("Each view must contain finite 1D t values after validity filtering")
    nonempty = [value for value in views if len(value)]
    if len(nonempty) < minimum_views:
        return np.empty((0, 2))
    grid_origin = float(np.floor(min(value.min() for value in nonempty) / bin_width) * bin_width)
    occupied_by_view = []
    for values in nonempty:
        bins = np.floor((values - grid_origin) / bin_width).astype(np.int64)
        # 一个像素复制一万次，还是只代表这一个视图看见了；先去重再计票。
        bins = np.unique(bins)
        dilated = np.unique(np.concatenate([bins + shift for shift in range(-dilation_bins, dilation_bins + 1)]))
        occupied_by_view.append(dilated)
    indices, votes = np.unique(np.concatenate(occupied_by_view), return_counts=True)
    accepted = indices[votes >= minimum_views]
    if not len(accepted):
        return np.empty((0, 2))
    split_positions = np.flatnonzero(np.diff(accepted) > 1) + 1
    intervals = [
        [grid_origin + run[0] * bin_width, grid_origin + (run[-1] + 1) * bin_width]
        for run in np.split(accepted, split_positions)
        if len(run) >= minimum_run_bins
    ]
    return np.asarray(intervals, dtype=np.float64).reshape(-1, 2)


def deterministic_ransac_line(
    points, view_ids, distance_threshold, min_pair_extent,
    min_inliers=12, min_views=3, trials=256, seed=0,
):
    """Fixed-budget two-point RANSAC followed by TLS on the winning consensus.

    Parameters must be frozen in method-visible units. Views are explicit input
    identities, not inferred from point coordinates. The returned mask is the
    winning hypothesis's consensus before TLS; refinement never adds new points.
    """
    points = np.asarray(points, dtype=np.float64)
    view_ids = np.asarray(view_ids)
    if points.ndim != 2 or points.shape[1] != 3 or view_ids.shape != (len(points),):
        raise ValueError("Expected Nx3 points and one view identity per point")
    if not np.isfinite(points).all():
        raise LineFitDegenerate("nonfinite_points")
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be finite and positive")
    if not np.isfinite(min_pair_extent) or min_pair_extent <= 0:
        raise ValueError("min_pair_extent must be finite and positive")
    for name, value, lower in (("min_inliers", min_inliers, 3), ("min_views", min_views, 1), ("trials", trials, 1)):
        if not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if len(points) < min_inliers:
        raise LineFitDegenerate("too_few_points_for_ransac_consensus")
    unique_views = np.unique(view_ids)
    if len(unique_views) < min_views:
        raise LineFitDegenerate("too_few_distinct_views_for_ransac")
    rng = np.random.default_rng(seed)
    best_mask, best_score, best_trial = None, None, None
    tested_pairs, supported_hypotheses = 0, 0
    for trial in range(trials):
        a, b = rng.choice(len(points), size=2, replace=False)
        axis = points[b] - points[a]
        extent = float(np.linalg.norm(axis))
        if extent < min_pair_extent:
            continue
        tested_pairs += 1
        axis /= extent
        offsets = points - points[a]
        residual_square = np.sum((offsets - (offsets @ axis)[:, None] * axis)**2, axis=1)
        mask = residual_square <= distance_threshold**2
        count = int(mask.sum())
        # 一张图上的点再多也不能伪装成多视图证据。先过门槛，再参与排名。
        if count < min_inliers or len(np.unique(view_ids[mask])) < min_views:
            continue
        supported_hypotheses += 1
        score = (count, -float(np.mean(residual_square[mask])))
        if best_score is None or score > best_score:
            best_mask, best_score, best_trial = mask, score, trial
    if best_mask is None:
        raise LineFitDegenerate("no_ransac_consensus_with_required_view_support")
    fitted = fit_line_tls(points[best_mask])
    return {
        **fitted,
        "inlier_mask": best_mask,
        "input_point_count": len(points),
        "inlier_count": int(best_mask.sum()),
        "view_support_counts": {str(view): int(np.sum(best_mask & (view_ids == view))) for view in unique_views},
        "distance_threshold": float(distance_threshold),
        "min_pair_extent": float(min_pair_extent),
        "trials": trials,
        "tested_pairs": tested_pairs,
        "supported_hypotheses": supported_hypotheses,
        "winning_trial": best_trial,
        "seed": seed,
        "winning_hypothesis_inlier_rmse": float(np.sqrt(-best_score[1])),
        "inlier_policy": "winning_two_point_consensus_before_tls_no_post_refit_growth",
        "score_policy": "max_point_inliers_then_min_mean_squared_error_then_first_trial",
    }
