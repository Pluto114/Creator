"""Frozen conventional point-fit controls for complete fixture candidates.

Both controls receive the same original RGB-supported base points; neither
receives candidate curves, truth, score tolerances or readout scale. A single
observed projection span is intentional: real-gap bridging must remain visible
in evaluation, not be removed using the proposed method's finite endpoints.
"""

import hashlib
import time

import numpy as np

from .line_controls import LineFitDegenerate, deterministic_ransac_line, fit_line_tls

DEFAULTS = dict(distance_camera_span_fraction=.005, minimum_pair_extent_camera_span_fraction=.03,
                minimum_inliers=12, minimum_views=3, trials=256, seed=0)
METHODS = ("depth_tls_single_span", "depth_ransac_single_span")


def fit_controls(points, point_ids, camera_span, mask):
    """Return two explicit normal results; caller persists actual point patches."""
    points, point_ids, mask = np.asarray(points, float), np.asarray(point_ids), np.asarray(mask)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or point_ids.shape != points.shape or point_ids.dtype != np.dtype("uint32")
            or mask.shape != (len(points),) or mask.dtype != np.dtype(bool)):
        raise ValueError("Finite Nx3 points, uint32 source IDs and matching boolean mask required")
    if isinstance(camera_span, (bool, np.bool_)) or not np.isfinite(camera_span) or camera_span <= 0:
        raise ValueError("Positive finite camera span required")
    # Stable source identities, not arbitrary point-array order, fix the RANSAC
    # sampling order. Different source IDs at equal coordinates remain distinct.
    indices = np.flatnonzero(mask)
    ids = point_ids[indices]
    order = np.lexsort(ids.T[::-1])
    ids, selected = ids[order], points[indices[order]]
    if len(ids) > 1 and np.any(np.all(ids[1:] == ids[:-1], axis=1)):
        raise ValueError("Supported source point IDs must be unique")
    identity = hashlib.sha256(b"fixture-naive-supported-points-v1\0")
    identity.update(np.asarray(ids, dtype="<u4").tobytes())
    identity.update(np.asarray(selected, dtype="<f8").tobytes())
    rows = []
    for method in METHODS:
        started = time.perf_counter()
        row = dict(method=method, state="complete", reason=None, segments=np.empty((0, 2, 3)), model=None,
            outcome="no_supported_change", input_point_count=len(points), input_support_count=len(selected),
            input_support_sha256=identity.hexdigest(), config=dict(DEFAULTS), camera_span_m=float(camera_span),
            source_view_counts={str(int(view)): int(np.sum(ids[:, 0] == view)) for view in np.unique(ids[:, 0])},
            endpoint_policy="observed_projection_min_max_single_span_no_gap_inference",
            point_weighting="equal_per_source_point_no_geometric_deduplication",
            selection_policy="same_frozen_original_RGB_mask_for_both_controls",
            readout_scale_used=False, gt_read=False)
        try:
            if method == METHODS[0]:
                fitted = fit_line_tls(selected)
            else:
                fitted = deterministic_ransac_line(selected, ids[:, 0],
                    distance_threshold=camera_span*DEFAULTS["distance_camera_span_fraction"],
                    min_pair_extent=camera_span*DEFAULTS["minimum_pair_extent_camera_span_fraction"],
                    min_inliers=DEFAULTS["minimum_inliers"], min_views=DEFAULTS["minimum_views"],
                    trials=DEFAULTS["trials"], seed=DEFAULTS["seed"])
            row.update(segments=fitted["segment"][None].copy(), model=fitted, outcome="accepted_change")
        except LineFitDegenerate as exc:
            row["reason"] = str(exc)
        row["elapsed_seconds"] = time.perf_counter()-started
        rows.append(row)
    return rows
