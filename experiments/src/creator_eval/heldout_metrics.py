"""Read-only, finite-curve projection diagnostics for evaluation-only views.

This module receives already aligned candidates and labels. It never fits, matches
rods, loads data, or selects parameters. Pixel coordinates use K_index throughout.
"""

import numpy as np


def _inside(points, size_wh):
    width, height = size_wh
    return (
        (points[:, 0] >= -0.5)
        & (points[:, 0] < width - 0.5)
        & (points[:, 1] >= -0.5)
        & (points[:, 1] < height - 0.5)
    )


def _clip_image(segment, size_wh):
    """Liang-Barsky clipping in the index-origin image footprint."""
    start, end = segment
    delta = end - start
    lower = np.array([-0.5, -0.5])
    upper = np.nextafter(np.asarray(size_wh, float) - 0.5, -np.inf)
    lo, hi = 0.0, 1.0
    for axis in range(2):
        if abs(delta[axis]) < 1e-14:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        first = (lower[axis] - start[axis]) / delta[axis]
        last = (upper[axis] - start[axis]) / delta[axis]
        lo, hi = max(lo, min(first, last)), min(hi, max(first, last))
        if hi < lo:
            return None
    return np.array([start + lo * delta, start + hi * delta])


def _project_candidates(segments_world, intrinsic, transform, size_wh, clip_start, clip_end):
    segments = np.asarray(segments_world, dtype=np.float64)
    if segments.size == 0:
        segments = np.empty((0, 2, 3), dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("segments_world must be finite Nx2x3 endpoints")
    report = {
        "input_segments": len(segments),
        "accepted_segments": 0,
        "rejected_at_or_crossing_near_plane": 0,
        "rejected_at_or_crossing_far_plane": 0,
        "rejected_outside_image": 0,
        "clipped_to_image": 0,
        "zero_projected_length_segments": 0,
    }
    output = []
    for segment in segments:
        camera = np.c_[segment, np.ones(2)] @ transform.T
        # 穿过相机/近裁面时投影可以炸到无穷远。整段拒计，并把拒计数留在报告里。
        if np.any(camera[:, 2] <= clip_start):
            report["rejected_at_or_crossing_near_plane"] += 1
            continue
        if np.any(camera[:, 2] >= clip_end):
            report["rejected_at_or_crossing_far_plane"] += 1
            continue
        image = camera[:, :3] @ intrinsic.T
        projected = image[:, :2] / image[:, 2, None]
        if not np.isfinite(projected).all():
            raise ValueError("Nonfinite candidate projection")
        clipped = _clip_image(projected, size_wh)
        if clipped is None:
            report["rejected_outside_image"] += 1
            continue
        if not np.allclose(clipped, projected, rtol=0, atol=1e-12):
            report["clipped_to_image"] += 1
        if np.linalg.norm(clipped[1] - clipped[0]) <= 1e-12:
            report["zero_projected_length_segments"] += 1
        output.append(clipped)
    report["accepted_segments"] = len(output)
    report["has_depth_rejections"] = bool(
        report["rejected_at_or_crossing_near_plane"] or report["rejected_at_or_crossing_far_plane"]
    )
    return np.asarray(output, dtype=float).reshape(-1, 2, 2), report


def _distance(points, segments):
    """Point distance to finite segments, including zero-length point support."""
    if not len(segments):
        return np.full(len(points), np.inf)
    start, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
    length_squared = np.einsum("ni,ni->n", delta, delta)
    denominator = np.where(length_squared > 0, length_squared, 1.0)
    result = np.empty(len(points))
    for first in range(0, len(points), 256):
        part = points[first : first + 256]
        offset = part[:, None] - start[None]
        fraction = np.clip(np.einsum("nsi,si->ns", offset, delta) / denominator, 0, 1)
        error = offset - fraction[..., None] * delta[None]
        result[first : first + len(part)] = np.sqrt(np.min(np.sum(error * error, axis=-1), axis=1))
    return result


def _distance_summary(distance, weights=None):
    if not len(distance) or not np.isfinite(distance).all():
        return {
            "distance_mean_px": None,
            "distance_median_px": None,
            "distance_p95_px": None,
            "distance_max_px": None,
        }
    if weights is None:
        median, p95 = np.quantile(distance, [0.5, 0.95])
        mean = float(distance.mean())
    else:
        order = np.argsort(distance)
        cumulative = np.cumsum(weights[order])
        quantiles = [
            distance[order[np.searchsorted(cumulative, q * cumulative[-1], side="left")]]
            for q in (0.5, 0.95)
        ]
        median, p95 = quantiles
        mean = float(np.average(distance, weights=weights))
    return {
        "distance_mean_px": mean,
        "distance_median_px": float(median),
        "distance_p95_px": float(p95),
        "distance_max_px": float(distance.max()),
    }


def _point_report(points, candidates, tolerance, fraction_key, empty_reason):
    distance = _distance(points, candidates)
    covered = int(np.count_nonzero(distance <= tolerance))
    reason = (
        empty_reason
        if not len(points)
        else ("no_accepted_projected_candidate" if not len(candidates) else None)
    )
    return {
        "sample_count": len(points),
        "covered_count": covered,
        fraction_key: covered / len(points) if len(points) else None,
        **_distance_summary(distance),
        "reason": reason,
    }


def _visibility(value, size_wh):
    points = np.asarray(value["uv_index"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("visibility uv_index must be finite Nx2 coordinates")
    count = len(points)
    result = {"uv_index": points}
    for key in (
        "segment_surface_id",
        "rod_id",
        "segment_fraction",
        "visible",
        "inside_image",
        "within_clip",
        "intentional_gap",
        "gap_clear",
    ):
        item = np.asarray(value[key])
        if item.shape != (count,) or not np.isfinite(item).all():
            raise ValueError("Invalid visibility field: " + key)
        result[key] = item
    fraction = result["segment_fraction"]
    if np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("segment_fraction must be in [0,1]")
    result["eligible"] = (
        result["inside_image"].astype(bool)
        & result["within_clip"].astype(bool)
        & _inside(points, size_wh)
    )
    return result


def _visible_support(labels, rod_id, selected):
    """Keep adjacency from the full annotation, not from the filtered point list."""
    segments = []
    used = np.zeros(len(selected), bool)
    original_rod = (labels["rod_id"] == rod_id) & ~labels["intentional_gap"].astype(bool)
    for sid in np.unique(labels["segment_surface_id"][original_rod]):
        indices = np.flatnonzero(original_rod & (labels["segment_surface_id"] == sid))
        if np.any(np.diff(labels["segment_fraction"][indices]) <= 0):
            raise ValueError("GT segment samples must preserve strictly increasing fractions")
        for first, last in zip(indices[:-1], indices[1:]):
            # 先删遮挡点再连线会偷偷把洞补回去。必须看未过滤序列里的相邻关系。
            if last == first + 1 and selected[first] and selected[last]:
                segments.append(labels["uv_index"][[first, last]])
                used[[first, last]] = True
    for index in np.flatnonzero(selected & ~used):
        segments.append(np.repeat(labels["uv_index"][index : index + 1], 2, axis=0))
    return np.asarray(segments, dtype=float).reshape(-1, 2, 2)


def _precision(candidates, support, tolerance, spacing):
    points, weights = [], []
    for segment in candidates:
        length = float(np.linalg.norm(segment[1] - segment[0]))
        if length <= 1e-12:
            continue
        count = max(1, int(np.ceil(length / spacing)))
        fractions = (np.arange(count) + 0.5) / count
        points.append(segment[0] + fractions[:, None] * (segment[1] - segment[0]))
        weights.append(np.full(count, length / count))
    samples = np.concatenate(points) if points else np.empty((0, 2))
    arc_weights = np.concatenate(weights) if weights else np.empty(0)
    distance = _distance(samples, support)
    total = float(arc_weights.sum())
    covered = float(arc_weights[distance <= tolerance].sum())
    return {
        "sample_count": len(samples),
        "projected_length_px": total,
        "supported_length_px": covered,
        "precision_fraction": covered / total if total else None,
        **_distance_summary(distance, arc_weights if total else None),
        "reason": "no_positive_length_in_frame_candidate"
        if not total
        else ("no_visible_gt_support" if not len(support) else None),
        "sampling": "projected_segment_equal_arc_cells_midpoints_with_exact_cell_length_weights",
        "sampling_max_spacing_px": float(spacing),
        "gt_support": "finite_adjacent_visible_samples_per_surface_id; isolated_samples_remain_points",
        "limitation": "sampled_projection_precision_approximation; visibility_between_adjacent_GT_samples_is_not_measured; overlapping_candidate_arcs_count_separately",
    }


def evaluate_projected_candidate(
    segments_world,
    K_index,
    world_to_camera_cv,
    size_wh,
    visibility_dict,
    rod_id,
    tolerance_px=2.0,
    *,
    clip_start=1e-6,
    clip_end=np.inf,
    sample_spacing_px=0.5,
):
    """Score an already-selected rod candidate without fitting or parameter choice.

    Candidates are finite world-space endpoint pairs (Nx2x3). Pass the real camera
    clip_start/clip_end when available; the default only guards the camera plane.
    A segment touching/crossing either depth cutoff is wholly rejected and counted.
    In-front segments are clipped to the image footprint before all statistics.
    GT recall/gap coverage are unweighted over the exported samples. Precision is
    an arc-weighted approximation over candidate midpoint samples, using only
    contiguous visible GT support. These are 2D diagnostics, not 3D correctness.
    """
    if not np.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("tolerance_px must be finite and nonnegative")
    if not np.isfinite(sample_spacing_px) or sample_spacing_px <= 0:
        raise ValueError("sample_spacing_px must be finite and positive")
    if not np.isfinite(clip_start) or clip_start < 0 or not clip_end > clip_start:
        raise ValueError("Invalid camera depth cutoffs")
    size = np.asarray(size_wh)
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or np.any(size != np.floor(size))
    ):
        raise ValueError("size_wh must be two positive integer dimensions")
    intrinsic = np.asarray(K_index, dtype=float)
    transform = np.asarray(world_to_camera_cv, dtype=float)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-12, rtol=0)
    ):
        raise ValueError("K_index must be a finite pinhole 3x3 intrinsic")
    if transform.shape not in ((3, 4), (4, 4)) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 3x4 or 4x4 world_to_camera_cv")
    if transform.shape == (4, 4) and not np.allclose(
        transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0
    ):
        raise ValueError("Invalid homogeneous world_to_camera_cv")
    labels = _visibility(visibility_dict, size)
    candidates, projection = _project_candidates(
        segments_world, intrinsic, transform, size, clip_start, clip_end
    )
    own_rod = labels["rod_id"] == rod_id
    selected = (
        own_rod
        & labels["eligible"]
        & labels["visible"].astype(bool)
        & ~labels["intentional_gap"].astype(bool)
    )
    support = _visible_support(labels, rod_id, selected)
    all_gap = own_rod & labels["intentional_gap"].astype(bool)
    interior = (labels["segment_fraction"] > 0) & (labels["segment_fraction"] < 1)
    gap = all_gap & labels["eligible"] & labels["gap_clear"].astype(bool) & interior
    endpoints = labels["uv_index"][all_gap & ~interior]
    guarded = gap.copy()
    if len(endpoints):
        endpoint_distance = np.min(
            np.linalg.norm(labels["uv_index"][:, None] - endpoints[None], axis=-1), axis=1
        )
        guarded &= endpoint_distance > tolerance_px
    elif gap.any():
        raise ValueError("Gap interior requires explicit endpoint samples for the tolerance guard")
    return {
        "scope": "same_asset_heldout_projection_diagnostic_not_3d_recovery_or_new_object_generalization",
        "rod_id": int(rod_id),
        "gt_sample_weights": "one_per_selected_exported_sample; not_image_arc_weighted",
        "tolerance_px": float(tolerance_px),
        "units": "native_image_pixels",
        "pixel_convention": "K_index_and_uv_index; image footprint [-0.5,width-0.5) x [-0.5,height-0.5)",
        "camera_depth_cutoffs": {
            "clip_start": float(clip_start),
            "clip_end": float(clip_end) if np.isfinite(clip_end) else None,
            "policy": "reject_entire_segment_touching_or_crossing_either_plane",
        },
        "candidate_projection": projection,
        "visible_gt_to_candidate": _point_report(
            labels["uv_index"][selected],
            candidates,
            tolerance_px,
            "recall_fraction",
            "no_visible_gt_samples",
        ),
        "gap_clear_interior_to_candidate": _point_report(
            labels["uv_index"][gap],
            candidates,
            tolerance_px,
            "false_coverage_fraction",
            "no_clear_gap_interior_samples",
        ),
        "gap_clear_guarded_interior_to_candidate": {
            **_point_report(
                labels["uv_index"][guarded],
                candidates,
                tolerance_px,
                "false_coverage_fraction",
                "no_clear_gap_samples_beyond_endpoint_guard",
            ),
            "endpoint_guard_px": float(tolerance_px),
            "guard_rule": "exclude_gap_samples_within_tolerance_of_either_projected_gap_endpoint",
        },
        "candidate_to_visible_gt": _precision(candidates, support, tolerance_px, sample_spacing_px),
    }
