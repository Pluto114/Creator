"""Read-only pixel checks for supplied endpoint correspondences and target claims.

These checks do not establish that their observations are independent of fitting.
The caller must retain provenance and disclose reused clicks/pixels. A supported
result is pixel consistency at the inherited 2 px tolerance, never 3D accuracy.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .camera_bundle import project, triangulate

THRESHOLD_PX = 2.0
MINIMUM_ENDPOINT_VIEWS = 4
MINIMUM_TARGET_VIEWS = 2
SOURCE_KINDS = {"rgb", "synthetic_pixel_measurements"}


def _threshold(value):
    if isinstance(value, bool) or value != THRESHOLD_PX:
        raise ValueError("This diagnostic inherits the frozen finite-stage 2 px tolerance")
    return THRESHOLD_PX


def _array(value, shape):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.shape == shape and np.isfinite(array).all() else None


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _frames(frames):
    """Bad camera inputs remain unavailable; do not repair rotations or pixels."""
    indexed, reasons = {}, []
    ids = [f.get("view_id") for f in frames]
    if any(not isinstance(vid, str) or not vid for vid in ids):
        return {}, ["invalid_view_id"]
    duplicates = {vid for vid, count in Counter(ids).items() if count > 1}
    if duplicates:
        reasons.append("duplicate_view_id")
    for frame in frames:
        vid = frame["view_id"]
        errors = []
        if vid in duplicates:
            errors.append("duplicate_view_id")
        if not _sha(frame.get("source_sha256")):
            errors.append("invalid_source_sha256")
        if frame.get("source_kind") not in SOURCE_KINDS:
            errors.append("invalid_source_kind")
        size = _array(frame.get("size_wh"), (2,))
        if size is None or np.any(size <= 0) or np.any(size != np.floor(size)):
            errors.append("invalid_image_size")
        k = _array(frame.get("K_index"), (3, 3))
        e = _array(frame.get("world_to_camera_cv"), (3, 4))
        if e is None:
            homogeneous = _array(frame.get("world_to_camera_cv"), (4, 4))
            if homogeneous is not None and np.array_equal(homogeneous[3], [0, 0, 0, 1]):
                e = homogeneous[:3]
        if k is None or e is None:
            errors.append("missing_or_nonfinite_camera")
        else:
            r = e[:, :3]
            if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0)
                    or abs(np.linalg.det(k)) <= 1e-12):
                errors.append("invalid_pinhole_camera")
            if (not np.allclose(r @ r.T, np.eye(3), atol=1e-4, rtol=0)
                    or not np.isclose(np.linalg.det(r), 1., atol=1e-4, rtol=0)):
                errors.append("improper_camera_rotation")
        indexed[vid] = dict(frame=frame, k=k, e=e, size=size, reasons=errors,
                            center=-np.linalg.solve(e[:, :3], e[:, 3]) if not errors else None)
        if errors:
            reasons.append("invalid_frame:" + vid)
    if not frames:
        reasons.append("missing_frames")
    return indexed, list(dict.fromkeys(reasons))


def _observation(observation, indexed, *, anchor=False):
    vid = observation.get("view_id")
    info = indexed.get(vid) if isinstance(vid, str) else None
    errors = []
    if info is None:
        errors.append("missing_frame")
    elif info["reasons"]:
        errors.extend(info["reasons"])
    if not _sha(observation.get("source_sha256")):
        errors.append("invalid_observation_source_sha256")
    elif info is not None and observation["source_sha256"] != info["frame"].get("source_sha256"):
        errors.append("observation_source_mismatch")
    xy = _array(observation.get("xy"), (2,))
    uncertainty = _array(observation.get("uncertainty_xy_px"), (2,)) if anchor else np.zeros(2)
    if xy is None:
        errors.append("missing_or_nonfinite_observation")
    if uncertainty is None or np.any(uncertainty < 0):
        errors.append("invalid_anchor_uncertainty")
    if xy is not None and uncertainty is not None and info is not None and info["size"] is not None:
        if np.any(xy - uncertainty < -.5) or np.any(xy + uncertainty > info["size"] - .5):
            errors.append("observation_outside_image")
    return xy, uncertainty, list(dict.fromkeys(errors))


def _independent(view_ids, indexed):
    centers = np.asarray([indexed[vid]["center"] for vid in view_ids])
    if len(centers) < 2:
        return True
    differences = np.linalg.norm(centers[:, None] - centers, axis=2)
    scale = max(1., float(differences.max()))
    return bool(np.all(differences[np.triu_indices(len(centers), 1)] > 1e-9 * scale))


def _state(rows, reasons):
    # One real contradiction can falsify a claim. Missing data can never
    # turn a partial collection of supported observations into acceptance.
    if any(row["state"] == "contradicted" for row in rows):
        return "contradicted"
    if reasons or not rows or any(row["state"] != "supported" for row in rows):
        return "unresolved"
    return "supported"


def _identifier(value):
    return value if isinstance(value, str) and value else None


def _finite_max(values):
    usable = [value for value in values if value is not None]
    return max(usable) if usable else None


def check_endpoint_reprojection(frames, endpoint_tracks, *, threshold_px=THRESHOLD_PX):
    """LOO triangulate each supplied endpoint from >=3 other observed views.

    Frames: view_id, K_index, world_to_camera_cv, size_wh, source_sha256,
    source_kind. Tracks: endpoint_id, observations [{view_id, xy, source_sha256}].
    Endpoints are supplied physical correspondences, not arbitrary points on rods.
    """
    threshold = _threshold(threshold_px)
    indexed, reasons = _frames(frames)
    ids = [track.get("endpoint_id") for track in endpoint_tracks]
    valid_ids = [tid for tid in ids if isinstance(tid, str) and tid]
    repeated = {tid for tid, count in Counter(valid_ids).items() if count > 1}
    if len(valid_ids) != len(ids):
        reasons.append("invalid_endpoint_id")
    if repeated:
        reasons.append("duplicate_endpoint_id")
    if not endpoint_tracks:
        reasons.append("missing_endpoint_tracks")
    endpoints = []
    for track in endpoint_tracks:
        endpoint_id = track.get("endpoint_id")
        observations = track.get("observations") or []
        view_ids = [o.get("view_id") for o in observations]
        valid_views = [vid for vid in view_ids if isinstance(vid, str)]
        track_reasons = []
        if not isinstance(endpoint_id, str) or not endpoint_id:
            track_reasons.append("invalid_endpoint_id")
        elif endpoint_id in repeated:
            track_reasons.append("duplicate_endpoint_id")
        if len(observations) < MINIMUM_ENDPOINT_VIEWS:
            track_reasons.append("insufficient_endpoint_views")
        if len(set(valid_views)) != len(valid_views):
            track_reasons.append("duplicate_endpoint_view")
        checked = [_observation(o, indexed) for o in observations]
        if any(errors for _, _, errors in checked):
            track_reasons.append("invalid_endpoint_observation")
        elif not _independent(view_ids, indexed):
            track_reasons.append("repeated_camera_center")
        held_out = []
        for oi, observation in enumerate(observations):
            xy, _, errors = checked[oi]
            row = dict(view_id=_identifier(observation.get("view_id")),
                fit_view_ids=[_identifier(vid) for j, vid in enumerate(view_ids) if j != oi],
                observed_xy=xy.tolist() if xy is not None else None,
                predicted_xy=None, residual_px=None, triangulated_point=None,
                state="unresolved", reasons=list(dict.fromkeys(track_reasons + errors)))
            if not row["reasons"]:
                ordered = [indexed[vid] for vid in view_ids]
                k = np.asarray([item["k"] for item in ordered])
                e = np.asarray([item["e"] for item in ordered])
                fitting = [dict(view=j, xy=checked[j][0]) for j in range(len(observations)) if j != oi]
                try:
                    equations = []
                    for item in fitting:
                        j = item["view"]
                        ray = np.linalg.solve(k[j], np.r_[item["xy"], 1.])
                        equations.extend((ray[0] * e[j, 2] - e[j, 0], ray[1] * e[j, 2] - e[j, 1]))
                    # Distinct camera centers alone do not give depth: a point
                    # on their common optical ray can still be unconstrained.
                    if np.linalg.matrix_rank(equations) < 3:
                        row["reasons"].append("degenerate_triangulation")
                        held_out.append(row)
                        continue
                    point = triangulate(fitting, k, e)
                    if not np.isfinite(point).all():
                        row["reasons"].append("nonfinite_triangulation")
                    else:
                        projections = [project(point[None], ki, ei) for ki, ei in zip(k, e)]
                        if any(depth[0] <= 1e-10 for _, depth in projections):
                            row["reasons"].append("point_behind_camera")
                        elif any(not np.isfinite(uv).all() for uv, _ in projections):
                            row["reasons"].append("nonfinite_projection")
                        else:
                            uv = projections[oi][0][0]
                            residual = float(np.linalg.norm(uv - xy))
                            row.update(predicted_xy=uv.tolist(), residual_px=residual,
                                triangulated_point=point.tolist(),
                                state="supported" if residual <= threshold else "contradicted")
                            if residual > threshold:
                                row["reasons"].append("heldout_endpoint_residual_above_threshold")
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    row["reasons"].append("triangulation_failure")
            held_out.append(row)
        state = _state(held_out, track_reasons)
        endpoints.append(dict(endpoint_id=_identifier(endpoint_id), state=state, reasons=track_reasons,
            observation_count=len(observations), scored_count=sum(r["residual_px"] is not None for r in held_out),
            maximum_residual_px=_finite_max([r["residual_px"] for r in held_out]), held_out=held_out))
    state = _state(endpoints, reasons)
    return dict(state=state, reasons=reasons, threshold_px=threshold,
        minimum_endpoint_views=MINIMUM_ENDPOINT_VIEWS, endpoint_count=len(endpoints),
        maximum_residual_px=_finite_max([e["maximum_residual_px"] for e in endpoints]),
        per_endpoint=endpoints, rod_geometry_modified=False,
        scope="Supplied endpoint correspondences; correlated leave-one-view-out pixel checks; independence must be audited by caller; not 3D accuracy")


def _distance_to_segment(point, a, b):
    delta = b - a
    length_squared = float(delta @ delta)
    fraction = float(np.clip((point - a) @ delta / length_squared, 0., 1.)) if length_squared > 0 else 0.
    return float(np.linalg.norm(point - (a + fraction * delta)))


def check_target_anchors(frames, segments, anchors, *, threshold_px=THRESHOLD_PX):
    """Check extra membership claims against unchanged finite projected segments.

    Anchors: view_id, xy, uncertainty_xy_px, source_sha256. The uncertainty
    rectangle's circumscribed disk is conservative; gaps remain separate segments.
    """
    threshold = _threshold(threshold_px)
    indexed, reasons = _frames(frames)
    try:
        finite = np.asarray(segments, float)
    except (TypeError, ValueError):
        finite = np.empty(0)
    valid_shape = finite.ndim == 3 and finite.shape[1:] == (2, 3) and len(finite) > 0
    if not valid_shape or not np.isfinite(finite).all():
        reasons.append("missing_or_nonfinite_segments")
    elif np.any(np.linalg.norm(finite[:, 1] - finite[:, 0], axis=1) <= 0):
        reasons.append("nonpositive_segment_length")
    geometry_errors = [r for r in reasons if "segment" in r]
    ids = [a.get("view_id") for a in anchors]
    valid_ids = [vid for vid in ids if isinstance(vid, str)]
    duplicates = {vid for vid, count in Counter(valid_ids).items() if count > 1}
    if duplicates:
        reasons.append("duplicate_anchor_view")
    if len(set(valid_ids)) < MINIMUM_TARGET_VIEWS:
        reasons.append("insufficient_anchor_views")
    if not anchors:
        reasons.append("missing_anchors")
    available_ids = list(dict.fromkeys(vid for vid in valid_ids if vid in indexed and not indexed[vid]["reasons"]))
    if len(available_ids) >= 2 and not _independent(available_ids, indexed):
        reasons.append("repeated_camera_center")
    rows = []
    for anchor in anchors:
        xy, uncertainty, errors = _observation(anchor, indexed, anchor=True)
        errors = list(dict.fromkeys(errors + geometry_errors))
        if anchor.get("view_id") in duplicates:
            errors.append("duplicate_anchor_view")
        row = dict(view_id=_identifier(anchor.get("view_id")), observed_xy=xy.tolist() if xy is not None else None,
            uncertainty_radius_px=float(np.hypot(*uncertainty)) if uncertainty is not None and np.all(uncertainty >= 0) else None,
            distance_px=None, minimum_distance_bound_px=None, maximum_distance_bound_px=None,
            state="unresolved", reasons=errors)
        if not errors:
            info = indexed[anchor["view_id"]]
            uv, depth = project(finite.reshape(-1, 3), info["k"], info["e"])
            if np.any(depth <= 1e-10):
                row["reasons"].append("segment_behind_camera")
            elif not np.isfinite(uv).all():
                row["reasons"].append("nonfinite_projection")
            else:
                # 不把缺口偷偷拉直，也不拿点击重拟合：它只是来检查已有结果。
                distance = min(_distance_to_segment(xy, a, b) for a, b in uv.reshape(-1, 2, 2))
                radius = row["uncertainty_radius_px"]
                lower, upper = max(0., distance - radius), distance + radius
                state = "supported" if upper <= threshold else ("contradicted" if lower > threshold else "unresolved")
                row.update(distance_px=distance, minimum_distance_bound_px=lower,
                    maximum_distance_bound_px=upper, state=state)
                if state != "supported":
                    row["reasons"].append("anchor_distance_above_threshold" if state == "contradicted"
                                          else "anchor_uncertainty_crosses_threshold")
        rows.append(row)
    return dict(state=_state(rows, reasons), reasons=reasons, threshold_px=threshold,
        minimum_anchor_views=MINIMUM_TARGET_VIEWS, anchor_count=len(rows),
        scored_count=sum(r["distance_px"] is not None for r in rows), per_anchor=rows,
        maximum_distance_px=_finite_max([r["distance_px"] for r in rows]),
        geometry_modified=False, target_selected=False,
        scope="Finite-segment projection consistency only; supplied membership claims may reuse fitting/selection data; not independent identity or 3D proof")
