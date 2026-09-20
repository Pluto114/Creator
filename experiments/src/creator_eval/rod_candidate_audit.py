"""Posthoc surface-ID probes for saved RGB candidates; never used by inference."""

from __future__ import annotations

import math

import numpy as np


def _index(value, length, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer index")
    if not 0 <= value < length:
        raise ValueError(f"{name} is outside its source collection")
    return int(value)


def _finite_coordinate(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite coordinate")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite coordinate") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite coordinate")
    return number


def audit_candidate_edges(observations, hypothesis, surface_ids, target_id=1, probe_offset_px=1.0):
    """Probe the raw edge pair selected by each saved ``row_matches`` triple.

    Coordinates use integer pixel centers, rounded with ``floor(x + 0.5)``.
    Every reported fraction has all matched rows as its denominator. Out-of-frame
    probes are counted explicitly and never produce a successful boundary test.
    Empty hypotheses have null fractions, not perfect or zero success rates.

    This is a diagnostic about sampled IDs, not a silhouette certification. A
    one-pixel probe can jump across a thin internal stripe or into another object;
    center-ray IDs do not describe RGB antialiasing or subpixel coverage.
    """
    raster = np.asarray(surface_ids)
    if raster.ndim != 2 or not all(raster.shape) or not np.issubdtype(raster.dtype, np.integer):
        raise ValueError("surface_ids must be a nonempty 2D integer ID raster")
    if np.any(raster < 0):
        raise ValueError("surface_ids must contain nonnegative IDs")
    if isinstance(target_id, (bool, np.bool_)) or not isinstance(target_id, (int, np.integer)) or target_id < 0:
        raise ValueError("target_id must be a nonnegative integer")
    target_id = int(target_id)
    offset = _finite_coordinate(probe_offset_px, "probe_offset_px")
    if offset <= 0:
        raise ValueError("probe_offset_px must be positive")
    rows = observations["rows"]
    matches = hypothesis["row_matches"]
    height, width = raster.shape
    histogram = {}
    center_valid = target_centers = center_outside = 0
    both_boundaries = outside_probes = outside_rows = 0
    edges = {
        name: {"valid_probe_rows": 0, "target_boundary_count": 0,
               "out_of_bounds_probe_count": 0}
        for name in ("left", "right")
    }
    seen_rows = set()

    def sample(x, y):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Probe coordinates must remain finite")
        column, row = math.floor(x + 0.5), math.floor(y + 0.5)
        if not (0 <= column < width and 0 <= row < height):
            return None
        return int(raster[row, column])

    for match in matches:
        if not isinstance(match, (list, tuple, np.ndarray)) or len(match) != 3:
            raise ValueError("Expected row_matches triples")
        row_index = _index(match[0], len(rows), "matched row")
        if row_index in seen_rows:
            raise ValueError("A hypothesis may match each raw row only once")
        seen_rows.add(row_index)
        row = rows[row_index]
        candidate_index = _index(match[1], len(row["candidates"]), "matched raw candidate")
        candidate = row["candidates"][candidate_index]
        y = _finite_coordinate(row["y"], "row y")
        center_x = _finite_coordinate(candidate["center_x"], "candidate center_x")
        center_id = sample(center_x, y)
        row_outside = int(center_id is None)
        if center_id is None:
            center_outside += 1
        else:
            center_valid += 1
            key = str(center_id)
            histogram[key] = histogram.get(key, 0) + 1
            target_centers += int(center_id == target_id)
        boundary_flags = []
        for name in ("left", "right"):
            x = _finite_coordinate(candidate[f"{name}_edge"]["x"], f"{name} edge x")
            first, second = sample(x - offset, y), sample(x + offset, y)
            outside = int(first is None) + int(second is None)
            row_outside += outside
            edges[name]["out_of_bounds_probe_count"] += outside
            valid = outside == 0
            edges[name]["valid_probe_rows"] += int(valid)
            # A missing probe is unknown, not the background side of a boundary.
            boundary = valid and ((first == target_id) != (second == target_id))
            edges[name]["target_boundary_count"] += int(boundary)
            boundary_flags.append(boundary)
        both_boundaries += int(all(boundary_flags))
        outside_probes += row_outside
        outside_rows += int(row_outside > 0)

    count = len(matches)
    for edge in edges.values():
        edge["target_boundary_fraction"] = edge["target_boundary_count"] / count if count else None
    return {
        "matched_rows": count,
        "target_id": target_id,
        "probe_offset_px": offset,
        "center_surface_id_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "valid_center_rows": center_valid,
        "target_center_count": target_centers,
        "target_center_fraction": target_centers / count if count else None,
        "center_out_of_bounds_count": center_outside,
        "edges": edges,
        "both_edges_target_boundary_count": both_boundaries,
        "both_edges_target_boundary_fraction": both_boundaries / count if count else None,
        "out_of_bounds_probe_count": outside_probes,
        "out_of_bounds_rows": outside_rows,
        "fraction_denominator": "all_matched_rows_including_out_of_bounds",
        "fraction_reason": None if count else "no_matched_rows",
        "sampling": "integer_pixel_centers_nearest_floor_coordinate_plus_half",
        "scope": "Posthoc sampled surface-ID diagnostic only. One-pixel probes can cross thin internal bands or adjacent objects; successful straddles do not prove true silhouettes, RGB antialias coverage, or physical-axis identity. Never an inference input.",
    }
