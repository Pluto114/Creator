"""RGB-only rod observations for the near-vertical pilot, with explicit uncertainty.

No truth, depth, camera, or model output is accepted here. Pixel coordinates refer
to original RGB integer pixel centers; horizontal differences live at half pixels.
"""

import numpy as np

DEFAULT_OBSERVATION_CONFIG = {
    "scan_half_width": 20,
    "row_stride": 4,
    "y_window_radius": 2,
    "edge_min_contrast": 8.0,
    "center_min_contrast": 8.0,
    "min_width": 2.0,
    "max_width": 24.0,
    "background_samples": 2,
    "edge_min_horizontal_fraction": 0.8,
    "edge_min_row_support": 0.6,
    "edge_match_tolerance": 1.5,
    "min_resolved_center_pixels": 2,
    "flat_max_std": 3.0,
    "flat_max_gradient": 6.0,
}

DEFAULT_LINE_CONFIG = {
    "minimum_rows": 12,
    "minimum_y_span": 50.0,
    "minimum_pair_y_span": 16.0,
    "inlier_distance_px": 1.5,
    "trials": 256,
    "seed": 0,
    "ambiguity_support_fraction": 0.9,
    "ambiguity_separation_px": 3.0,
}


def _configuration(defaults, supplied):
    supplied = {} if supplied is None else dict(supplied)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = {**defaults, **supplied}
    if any(not np.isfinite(value) or value < 0 for value in config.values()):
        raise ValueError("Configuration values must be finite and nonnegative")
    return config


def _gray_image(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or min(rgb.shape[:2]) < 3:
        raise ValueError("Expected HxWx3 RGB with at least three pixels on each axis")
    rgb = rgb[..., :3].astype(np.float64)
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError("Expected finite RGB intensities in 0..255 units")
    # 这里固定亮度定义；不能某次图暗了就临时换归一化，把噪声也拉成杆。
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def _gradient_peaks(profile, start, stop, minimum):
    peaks = []
    for index in range(start, stop):
        value = profile[index]
        if abs(value) < minimum:
            continue
        sign = np.sign(value)
        previous = sign * profile[index - 1] if index else 0.0
        following = sign * profile[index + 1] if index + 1 < len(profile) else 0.0
        if abs(value) >= previous and abs(value) >= following:
            if peaks and index == peaks[-1][-1] + 1 and value == profile[peaks[-1][0]]:
                peaks[-1].append(index)
            else:
                peaks.append([index])
    return [
        {"x": float(np.mean(indices) + 0.5), "gradient": float(profile[indices[0]])}
        for indices in peaks
    ]


def _edge_evidence(edge, y, sign, guide_slope, gx, gy_edge, config):
    matched, horizontal_fractions = [], []
    radius = int(config["y_window_radius"])
    tolerance = config["edge_match_tolerance"]
    for neighbor_y in range(y - radius, y + radius + 1):
        expected = edge + guide_slope * (neighbor_y - y)
        indices = np.arange(
            max(0, int(np.ceil(expected - tolerance - 0.5))),
            min(gx.shape[1], int(np.floor(expected + tolerance - 0.5)) + 1),
        )
        if not len(indices):
            horizontal_fractions.append(0.0)
            continue
        index = indices[np.argmax(sign * gx[neighbor_y, indices])]
        horizontal = float(gx[neighbor_y, index])
        vertical = float(gy_edge[neighbor_y, index])
        fraction = abs(horizontal) / max(float(np.hypot(horizontal, vertical)), 1e-12)
        horizontal_fractions.append(fraction)
        if (
            sign * horizontal >= config["edge_min_contrast"]
            and fraction >= config["edge_min_horizontal_fraction"]
        ):
            matched.append(neighbor_y)
    return {
        "support_fraction": len(matched) / (2 * radius + 1),
        "center_row_horizontal_fraction": horizontal_fractions[radius],
        "supported_rows": matched,
        "horizontal_gradient_fraction_median": float(np.median(horizontal_fractions)),
        "window_rows": [y - radius, y + radius],
    }


def extract_rod_observations(rgb, guide_xyxy, config=None):
    """Keep all resolved double-edge candidates in a coarse two-endpoint strip.

    guide_xyxy has shape (2,2): [[x0,y0],[x1,y1]], measured in original RGB.
    It describes where to search, not the true rod endpoints or its missing parts.
    `absent` means only a flat-background-like local observation; flat occlusion
    cannot be distinguished here. Failed pairing/low SNR/boundaries stay unknown.
    """
    config = _configuration(DEFAULT_OBSERVATION_CONFIG, config)
    for name in (
        "scan_half_width",
        "row_stride",
        "background_samples",
        "min_resolved_center_pixels",
    ):
        if config[name] < 1 or int(config[name]) != config[name]:
            raise ValueError(f"{name} must be a positive integer")
        config[name] = int(config[name])
    if int(config["y_window_radius"]) != config["y_window_radius"]:
        raise ValueError("y_window_radius must be an integer")
    if config["min_width"] <= 0 or config["max_width"] < config["min_width"]:
        raise ValueError("Invalid resolved width interval")
    for name in ("edge_min_horizontal_fraction", "edge_min_row_support"):
        if not 0 < config[name] <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    gray = _gray_image(rgb)
    guide = np.asarray(guide_xyxy, dtype=np.float64)
    if guide.shape != (2, 2) or not np.isfinite(guide).all() or guide[1, 1] <= guide[0, 1]:
        raise ValueError("Expected two finite guide points with increasing y")
    height, width = gray.shape
    if guide[0, 1] < 0 or guide[1, 1] > height - 1:
        raise ValueError("Guide y range is outside the RGB image")
    slope = float((guide[1, 0] - guide[0, 0]) / (guide[1, 1] - guide[0, 1]))
    gx = np.diff(gray, axis=1)
    gy_pixels = np.gradient(gray, axis=0)
    gy_edge = (gy_pixels[:, 1:] + gy_pixels[:, :-1]) / 2
    rows = []
    radius = int(config["y_window_radius"])
    for y in range(
        int(np.ceil(guide[0, 1])), int(np.floor(guide[1, 1])) + 1, int(config["row_stride"])
    ):
        expected = float(guide[0, 0] + slope * (y - guide[0, 1]))
        x0 = int(np.floor(expected - config["scan_half_width"]))
        x1 = int(np.ceil(expected + config["scan_half_width"])) + 1
        row = {
            "y": y,
            "guide_x": expected,
            "status": "unknown",
            "reason": None,
            "candidates": [],
            "rejected_pair_counts": {},
            "edge_count": 0,
            "absence_window_xyxy": [x0, y - radius, x1, y + radius + 1],
            "absence_evidence": None,
        }
        rows.append(row)
        if (
            x0 < config["background_samples"]
            or x1 + config["background_samples"] >= width
            or y - radius < 0
            or y + radius >= height
        ):
            row["reason"] = "incomplete_search_or_background_window"
            continue
        window = gray[y - radius : y + radius + 1, x0:x1]
        maximum_gradient = max(
            float(np.max(np.abs(gx[y - radius : y + radius + 1, x0 : x1 - 1]))),
            float(np.max(np.abs(gy_pixels[y - radius : y + radius + 1, x0:x1]))),
        )
        window_std = float(np.std(window))
        background_difference = float(
            abs(
                np.median(window[:, : config["background_samples"]])
                - np.median(window[:, -config["background_samples"] :])
            )
        )
        flat = (
            window_std <= config["flat_max_std"]
            and maximum_gradient <= config["flat_max_gradient"]
            and background_difference <= config["flat_max_gradient"]
        )
        row["absence_evidence"] = {
            "window_std": window_std,
            "maximum_gradient": maximum_gradient,
            "side_background_difference": background_difference,
            "flat_background_like": bool(flat),
            "limitation": "Flat foreground occlusion or an unresolved low-contrast rod cannot be ruled out.",
        }
        edges = _gradient_peaks(gx[y], x0, x1 - 1, config["edge_min_contrast"])
        row["edge_count"] = len(edges)
        for left_index, left in enumerate(edges):
            for right in edges[left_index + 1 :]:
                if left["gradient"] * right["gradient"] >= 0:
                    continue
                separation = right["x"] - left["x"]
                reason = None
                if not config["min_width"] <= separation <= config["max_width"]:
                    reason = "unresolved_or_out_of_width_range"
                interior_x = np.arange(int(np.floor(left["x"])) + 1, int(np.ceil(right["x"])))
                if len(interior_x) < config["min_resolved_center_pixels"]:
                    reason = "unresolved_center_support"
                center = (left["x"] + right["x"]) / 2
                sign = float(np.sign(left["gradient"]))
                outer_steps = np.arange(int(config["background_samples"])) + 0.5
                left_background = float(
                    np.median(np.interp(left["x"] - outer_steps, np.arange(width), gray[y]))
                )
                right_background = float(
                    np.median(np.interp(right["x"] + outer_steps, np.arange(width), gray[y]))
                )
                center_value = float(np.interp(center, np.arange(width), gray[y]))
                contrast = float(
                    min(
                        sign * (center_value - left_background),
                        sign * (center_value - right_background),
                    )
                )
                if contrast < config["center_min_contrast"]:
                    reason = reason or "center_not_distinct_from_both_background_sides"
                left_evidence = _edge_evidence(left["x"], y, sign, slope, gx, gy_edge, config)
                right_evidence = _edge_evidence(right["x"], y, -sign, slope, gx, gy_edge, config)
                continuity = min(
                    left_evidence["support_fraction"], right_evidence["support_fraction"]
                )
                if (
                    min(
                        left_evidence["center_row_horizontal_fraction"],
                        right_evidence["center_row_horizontal_fraction"],
                    )
                    < config["edge_min_horizontal_fraction"]
                ):
                    reason = reason or "nonvertical_center_row_edge"
                if continuity < config["edge_min_row_support"]:
                    reason = reason or "insufficient_persistent_vertical_edges"
                if reason:
                    row["rejected_pair_counts"][reason] = (
                        row["rejected_pair_counts"].get(reason, 0) + 1
                    )
                    continue
                row["candidates"].append(
                    {
                        "center_x": center,
                        "width": separation,
                        "polarity": "bright" if sign > 0 else "dark",
                        "contrast": contrast,
                        "center_value": center_value,
                        "left_background": left_background,
                        "right_background": right_background,
                        "left_edge": left,
                        "right_edge": right,
                        "orientation": {
                            "left": left_evidence["horizontal_gradient_fraction_median"],
                            "right": right_evidence["horizontal_gradient_fraction_median"],
                        },
                        "continuity": {
                            "minimum_support_fraction": continuity,
                            "left": left_evidence,
                            "right": right_evidence,
                        },
                        "quantization_half_width_px": 0.5,
                    }
                )
        if len(row["candidates"]) == 1:
            row.update(status="observed", reason="one_resolved_double_edge_candidate")
        elif len(row["candidates"]) > 1:
            row.update(status="ambiguous", reason="multiple_resolved_double_edge_candidates")
        elif flat and not edges:
            row.update(status="absent", reason="flat_background_like_not_occlusion_disambiguated")
        else:
            row["reason"] = "no_resolved_candidate_missing_edges_or_insufficient_evidence"
    return {
        "schema_version": "1.0.0",
        "config": config,
        "guide_xyxy": guide.tolist(),
        "rows": rows,
        "pixel_convention": "original_RGB_integer_pixel_centers_edges_at_half_pixels",
        "scope": "Near-vertical resolved double-edge observations; no gap filling or occlusion inference.",
    }


def _line_support(rows, slope, intercept, threshold):
    matches = []
    divisor = float(np.hypot(1, slope))
    for row_index, row in rows:
        centers = np.array([candidate["center_x"] for candidate in row["candidates"]])
        residuals = np.abs(centers - (slope * row["y"] + intercept)) / divisor
        candidate_index = int(np.argmin(residuals))
        if residuals[candidate_index] <= threshold:
            matches.append((row_index, candidate_index, float(residuals[candidate_index])))
    return matches


def fit_robust_image_line(observations, config=None):
    """Fit x=a*y+b with one vote per observed row and deterministic two-row RANSAC.

    All candidates are retained in the source observations. The fitted infinite
    line does not create evidence at absent/unknown rows or infer finite endpoints.
    An alternative supported line makes the result explicitly ambiguous.
    """
    config = _configuration(DEFAULT_LINE_CONFIG, config)
    if config["minimum_rows"] < 2 or int(config["minimum_rows"]) != config["minimum_rows"]:
        raise ValueError("minimum_rows must be an integer >=2")
    if config["trials"] < 1 or int(config["trials"]) != config["trials"]:
        raise ValueError("trials must be a positive integer")
    if (
        config["inlier_distance_px"] <= 0
        or config["minimum_pair_y_span"] <= 0
        or config["ambiguity_separation_px"] <= 0
    ):
        raise ValueError("Line distances and minimum pair span must be positive")
    if not 0 < config["ambiguity_support_fraction"] <= 1:
        raise ValueError("ambiguity_support_fraction must lie in (0,1]")
    original_rows = observations["rows"]
    original_y = np.array([row["y"] for row in original_rows], dtype=float)
    if not np.isfinite(original_y).all() or len(np.unique(original_y)) != len(original_y):
        raise ValueError("Each observation row must have a unique finite y coordinate")
    if any(
        not np.isfinite(candidate["center_x"])
        for row in original_rows
        for candidate in row["candidates"]
    ):
        raise ValueError("Candidate centers must be finite")
    rows = [(index, row) for index, row in enumerate(original_rows) if row["candidates"]]
    result = {
        "state": "insufficient_support",
        "reason": None,
        "usable": False,
        "slope": None,
        "intercept": None,
        "inlier_rows": [],
        "row_matches": [],
        "residual_median_px": None,
        "residual_p95_px": None,
        "config": config,
        "observed_row_count": len(rows),
        "alternative_models": [],
        "scope": "Infinite image line only; absent and unknown rows remain unfilled.",
    }
    if len(rows) < config["minimum_rows"]:
        result["reason"] = "too_few_rows_with_candidates"
        return result
    y_values = np.array([row["y"] for _, row in rows], dtype=float)
    if np.ptp(y_values) < config["minimum_y_span"]:
        result["reason"] = "insufficient_y_span"
        return result
    rng = np.random.default_rng(int(config["seed"]))
    hypotheses = []
    for _ in range(int(config["trials"])):
        first, second = rng.choice(len(rows), size=2, replace=False)
        row_a, row_b = rows[first][1], rows[second][1]
        if abs(row_b["y"] - row_a["y"]) < config["minimum_pair_y_span"]:
            continue
        x_a = row_a["candidates"][rng.integers(len(row_a["candidates"]))]["center_x"]
        x_b = row_b["candidates"][rng.integers(len(row_b["candidates"]))]["center_x"]
        slope = (x_b - x_a) / (row_b["y"] - row_a["y"])
        intercept = x_a - slope * row_a["y"]
        matches = _line_support(rows, slope, intercept, config["inlier_distance_px"])
        if len(matches) < config["minimum_rows"]:
            continue
        matched_y = np.array([original_rows[i]["y"] for i, _, _ in matches], dtype=float)
        if np.ptp(matched_y) < config["minimum_y_span"]:
            continue
        matched_x = np.array([original_rows[i]["candidates"][j]["center_x"] for i, j, _ in matches])
        slope, intercept = np.linalg.lstsq(
            np.c_[matched_y, np.ones(len(matches))], matched_x, rcond=None
        )[0]
        matches = _line_support(rows, float(slope), float(intercept), config["inlier_distance_px"])
        if len(matches) < config["minimum_rows"]:
            continue
        if np.ptp([original_rows[i]["y"] for i, _, _ in matches]) < config["minimum_y_span"]:
            continue
        score = (len(matches), -float(np.mean([residual**2 for _, _, residual in matches])))
        hypotheses.append((score, float(slope), float(intercept), matches))
    if not hypotheses:
        result["reason"] = "no_consistent_line_with_required_rows_and_span"
        return result
    hypotheses.sort(key=lambda value: value[0], reverse=True)
    score, slope, intercept, matches = hypotheses[0]
    alternatives = []
    for alternative_score, alternative_slope, alternative_intercept, _ in hypotheses[1:]:
        if alternative_score[0] < config["ambiguity_support_fraction"] * score[0]:
            continue
        separation = float(
            np.median(
                np.abs((alternative_slope - slope) * y_values + alternative_intercept - intercept)
            )
        )
        if separation >= config["ambiguity_separation_px"]:
            alternatives.append(
                {
                    "slope": alternative_slope,
                    "intercept": alternative_intercept,
                    "support_rows": alternative_score[0],
                    "median_separation_px": separation,
                }
            )
            break
    residuals = np.array([residual for _, _, residual in matches])
    result.update(
        state="ambiguous" if alternatives else "fitted",
        usable=not alternatives,
        reason="competing_supported_image_lines"
        if alternatives
        else "unique_supported_line_within_fixed_budget",
        slope=slope,
        intercept=intercept,
        inlier_rows=[index for index, _, _ in matches],
        row_matches=[
            {
                "row_index": index,
                "y": original_rows[index]["y"],
                "candidate_index": candidate,
                "perpendicular_residual_px": residual,
                "source_status": original_rows[index]["status"],
            }
            for index, candidate, residual in matches
        ],
        residual_median_px=float(np.median(residuals)),
        residual_p95_px=float(np.quantile(residuals, 0.95)),
        alternative_models=alternatives,
    )
    return result
