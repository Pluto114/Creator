"""RGB-only known-fixture calibration, with marker identities split before capture.

The fixture supplies manufactured metric coordinates. It does not supply camera
parameters, target geometry, or projected image points.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

POLICY = dict(
    dictionary="DICT_5X5_100", marker_border_bits=1, corner_refinement="SUBPIX",
    refinement_window=3, minimum_training_markers=8, minimum_training_markers_per_plane=2,
    minimum_validation_markers=4, minimum_validation_markers_per_plane=1,
    minimum_rank_ratio=1e-3, maximum_training_p95_px=2., maximum_validation_p95_px=2.,
    initial_focal_image_max_fraction=1.2, shared_intrinsics=True, square_pixels=True,
    zero_skew=True, zero_distortion=True, principal_point_fitted=True,
    maximum_iterations=300, epsilon=1e-10,
)


def validate_cad(cad):
    if cad["units"] != "metres" or cad["dictionary"] != POLICY["dictionary"]:
        raise ValueError("Expected declared metric fixture and dictionary")
    markers = cad["markers"]
    ids = [m["marker_id"] for m in markers]
    if len(set(ids)) != len(ids) or any(not isinstance(i, int) or not 0 <= i < 100 for i in ids):
        raise ValueError("Marker IDs must be unique dictionary IDs")
    planes = {m["plane_id"] for m in markers}
    if len(planes) != 2:
        raise ValueError("Exactly two declared planes required")
    for marker in markers:
        points = np.asarray(marker["corners_world"], float)
        if points.shape != (4, 3) or not np.isfinite(points).all():
            raise ValueError("Four finite ordered CAD corners required")
        if marker["role"] not in ("training", "validation"):
            raise ValueError("Fixed marker role required")
    for role in ("training", "validation"):
        chosen = [m for m in markers if m["role"] == role]
        if {m["plane_id"] for m in chosen} != planes:
            raise ValueError("Both roles must cover both planes")
        if geometry_rank(np.array([m["corners_world"] for m in chosen]).reshape(-1, 3))["rank"] != 3:
            raise ValueError("Each role must contain noncoplanar declared points")
    return {m["marker_id"]: m for m in markers}


def marker_image(marker_id, side_pixels=224, margin_pixels=28):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels, borderBits=1)
    return np.pad(image, margin_pixels, constant_values=255)


def detect_markers(rgb, cad):
    declared = validate_cad(cad)
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("uint8 RGB image required")
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = POLICY["refinement_window"]
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = .001
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, rejected = detector.detectMarkers(cv2.cvtColor(array, cv2.COLOR_RGB2GRAY))
    decoded = [] if ids is None else [int(i) for i in ids.ravel()]
    counts = Counter(decoded)
    unique = {i: np.asarray(x).reshape(4, 2).tolist() for i, x in zip(decoded, corners) if counts[i] == 1}
    rows = []
    for marker_id, marker in sorted(declared.items()):
        state = "duplicate" if counts[marker_id] > 1 else "detected" if marker_id in unique else "missing"
        rows.append(dict(marker_id=marker_id, plane_id=marker["plane_id"], role=marker["role"],
            state=state, corners_xy=unique.get(marker_id) if state == "detected" else None))
    return dict(markers=rows, unknown_ids=sorted(set(decoded) - set(declared)),
        duplicate_ids=sorted(i for i, n in counts.items() if n > 1),
        rejected_quad_count=len(rejected), detected_count=sum(r["state"] == "detected" for r in rows),
        source_kind="decoded_rgb_marker", pixel_convention="integer-index centres; no half-pixel shift")


def geometry_rank(points):
    value = np.asarray(points, float).reshape(-1, 3)
    singular = np.linalg.svd(value - value.mean(axis=0), compute_uv=False) if len(value) else np.zeros(3)
    ratio = float(singular[-1] / singular[0]) if len(singular) == 3 and singular[0] > 0 else 0.
    return dict(point_count=len(value), singular_values=singular.tolist(),
        rank=int(np.sum(singular > (singular[0] * 1e-10 if len(singular) else 0))),
        smallest_to_largest_ratio=ratio)


def observations(detection, cad, role):
    declared = validate_cad(cad)
    records = detection["markers"]
    if len({r["marker_id"] for r in records}) != len(records) or set(r["marker_id"] for r in records) != set(declared):
        raise ValueError("Every declared marker must have exactly one detection record")
    selected = []
    for row in records:
        marker = declared[row["marker_id"]]
        if row["role"] != marker["role"] or row["plane_id"] != marker["plane_id"]:
            raise ValueError("Detection role or CAD identity changed")
        if row["state"] == "detected" and marker["role"] == role:
            xy = np.asarray(row["corners_xy"], float)
            if xy.shape != (4, 2) or not np.isfinite(xy).all():
                raise ValueError("Malformed RGB corners")
            selected.append((marker, xy))
    xyz = np.array([m["corners_world"] for m, _ in selected], float).reshape(-1, 3)
    xy = np.array([p for _, p in selected], float).reshape(-1, 2)
    ids = [m["marker_id"] for m, _ in selected]
    coverage = dict(Counter(m["plane_id"] for m, _ in selected))
    return dict(marker_ids=ids, plane_marker_counts=coverage, points_world=xyz.tolist(),
        corners_xy=xy.tolist(), geometry=geometry_rank(xyz))


def coverage_reasons(data, cad, role):
    minimum = POLICY["minimum_" + role + "_markers"]
    per_plane = POLICY["minimum_" + role + "_markers_per_plane"]
    reasons = []
    if len(data["marker_ids"]) < minimum:
        reasons.append(role + "_marker_count")
    if any(data["plane_marker_counts"].get(p, 0) < per_plane for p in {m["plane_id"] for m in cad["markers"]}):
        reasons.append(role + "_plane_coverage")
    if data["geometry"]["rank"] != 3 or data["geometry"]["smallest_to_largest_ratio"] <= POLICY["minimum_rank_ratio"]:
        reasons.append(role + "_rank")
    return reasons


def score_observations(data, k, e):
    points = np.asarray(data["points_world"], float).reshape(-1, 3)
    if not len(points):
        return dict(count=0, median_px=None, p95_px=None, maximum_px=None, positive_depth=False, errors_px=[])
    camera = points @ np.asarray(e)[:, :3].T + np.asarray(e)[:, 3]
    values = camera @ np.asarray(k).T
    pixels = values[:, :2] / values[:, 2, None]
    errors = np.linalg.norm(pixels - np.asarray(data["corners_xy"]), axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite projection")
    return dict(count=len(errors), median_px=float(np.median(errors)), p95_px=float(np.quantile(errors, .95)),
        maximum_px=float(errors.max()), positive_depth=bool(np.all(camera[:, 2] > 0)), errors_px=errors.tolist())


def calibrate_case(frames, cad):
    """Each five-view capture has its own K; validation never reaches the solver."""
    validate_cad(cad)
    if len(frames) != 5 or len({f["view_id"] for f in frames}) != 5:
        raise ValueError("Five uniquely identified views required")
    size = frames[0]["size_wh"]
    if any(f["size_wh"] != size for f in frames):
        raise ValueError("All images must share their declared dimensions")
    train = [observations(f["detection"], cad, "training") for f in frames]
    validation = [observations(f["detection"], cad, "validation") for f in frames]
    failures = [coverage_reasons(x, cad, "training") for x in train]
    w, h = size
    initial = np.array([[POLICY["initial_focal_image_max_fraction"] * max(size), 0., (w-1)/2],
                        [0., POLICY["initial_focal_image_max_fraction"] * max(size), (h-1)/2], [0., 0., 1.]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
        cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    cameras = []
    for frame, a, b, reasons in zip(frames, train, validation, failures):
        cameras.append(dict(view_id=frame["view_id"], state="unavailable", reasons=list(reasons),
            K_index=None, world_to_camera_cv=None, training=a, validation=b,
            training_score=None, validation_score=None))
    if any(failures):
        return dict(state="unavailable", cameras=cameras, flags=int(flags), initial_K=initial.tolist(),
            solver_rms_px=None, distortion=None, solver_error="Training coverage failed; validation not substituted")
    try:
        rms, k, distortion, rvecs, tvecs = cv2.calibrateCamera(
            [np.array(x["points_world"], np.float32) for x in train],
            [np.array(x["corners_xy"], np.float32) for x in train], tuple(size), initial.copy(),
            np.zeros(5), flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, POLICY["maximum_iterations"], POLICY["epsilon"]))
    except cv2.error as exc:
        return dict(state="unavailable", cameras=cameras, flags=int(flags), initial_K=initial.tolist(),
            solver_rms_px=None, distortion=None, solver_error=str(exc))
    if not np.isfinite(k).all() or not np.isfinite(rms) or k[0, 0] <= 0 or not np.allclose(distortion, 0, atol=0, rtol=0):
        raise ValueError("Invalid fixed-model calibration")
    for row, a, b, rv, tv in zip(cameras, train, validation, rvecs, tvecs):
        rotation, _ = cv2.Rodrigues(rv)
        e = np.c_[rotation, np.asarray(tv).reshape(3)]
        ts, vs = score_observations(a, k, e), score_observations(b, k, e)
        reasons = coverage_reasons(b, cad, "validation")
        for role, score in (("training", ts), ("validation", vs)):
            if not score["positive_depth"]:
                reasons.append(role + "_cheirality")
            if score["p95_px"] is None or score["p95_px"] > POLICY["maximum_" + role + "_p95_px"]:
                reasons.append(role + "_residual")
        row.update(K_index=k.tolist(), world_to_camera_cv=e.tolist(), training_score=ts, validation_score=vs,
                   state="validated" if not reasons else "withheld", reasons=reasons)
    return dict(state="validated" if all(r["state"] == "validated" for r in cameras) else "withheld",
        cameras=cameras, flags=int(flags), initial_K=initial.tolist(), solver_rms_px=float(rms),
        distortion=distortion.ravel().tolist(), solver_error=None,
        convergence_certificate=None, solver_note="OpenCV returns fit output, not a per-iteration convergence certificate",
        coordinate_system=cad["coordinate_system"], units=cad["units"])
