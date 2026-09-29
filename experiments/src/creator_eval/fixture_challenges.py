"""Declared calibration fault injections, independent of scene/camera truth."""

from __future__ import annotations

import copy

import cv2
import numpy as np

from .fixture_calibration import validate_cad


def perturb_cad(cad, scenario):
    """Change the supplied ruler, keeping the original scene physically fixed."""
    result = copy.deepcopy(cad)
    kind = scenario["kind"]
    scale = float(scenario["factor"]) if kind == "cad_scale" else 1.0
    offset = np.asarray(scenario.get("offset_m", [0., 0., 0.]), float)
    if not np.isfinite(scale) or scale <= 0 or offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Finite positive scale and finite 3D offset required")
    if kind == "cad_panel_shift" and scenario["plane_id"] not in {p["plane_id"] for p in cad["panels"]}:
        raise ValueError("Unknown panel")
    for collection, field in ((result["markers"], "corners_world"), (result["panels"], "center_world")):
        for row in collection:
            points = np.asarray(row[field], float) * scale
            if kind == "cad_translation" or (kind == "cad_panel_shift" and row["plane_id"] == scenario["plane_id"]):
                points += offset
            row[field] = points.tolist()
    result["marker_side_m"] *= scale
    validate_cad(result)
    return result


def perturb_detections(frames, scenario):
    """Fault injection after decoding; roles stay attached to their declared IDs."""
    result = copy.deepcopy(frames)
    kind = scenario["kind"]
    if kind == "missing_markers" and scenario["view_id"] not in {f["view_id"] for f in frames}:
        raise ValueError("Unknown missing-marker view")
    for frame in result:
        rows = frame["detection"]["markers"]
        if kind == "missing_markers" and frame["view_id"] == scenario["view_id"]:
            for row in rows:
                if row["role"] == scenario["role"] and scenario["plane_id"] in (None, row["plane_id"]):
                    row.update(state="missing", corners_xy=None)
        if kind == "swap_observations":
            by_id = {r["marker_id"]: r for r in rows}
            a, b = [by_id[i] for i in scenario["marker_ids"]]
            if a["marker_id"] == b["marker_id"] or a["role"] != "training" or b["role"] != "training":
                raise ValueError("Swap two distinct training IDs only")
            a["state"], b["state"] = b["state"], a["state"]
            a["corners_xy"], b["corners_xy"] = b["corners_xy"], a["corners_xy"]
        frame["detection"]["detected_count"] = sum(r["state"] == "detected" for r in rows)
    return result


def radial_forward(points_xy, size_wh, k):
    points = np.asarray(points_xy, float)
    size = np.asarray(size_wh, float)
    if points.shape[-1] != 2 or size.shape != (2,) or np.any(size <= 1):
        raise ValueError("2D points and image dimensions >1 required")
    if not np.isfinite(points).all() or not np.isfinite(size).all() or not np.isfinite(k) or abs(k) > .1:
        raise ValueError("Finite coordinates and |k| <= 0.1 required")
    center, radius = (size - 1) / 2, np.linalg.norm((size - 1) / 2)
    q = (points - center) / radius
    return center + radius * q * (1 + k * np.sum(q*q, axis=-1, keepdims=True))


def radial_source_map(size_wh, k):
    """Inverse map via radial Newton solve; monotone over the sampled domain."""
    w, h = size_wh
    if int(w) != w or int(h) != h or min(w, h) <= 1:
        raise ValueError("Integer image dimensions >1 required")
    radial_forward(np.zeros((1, 2)), size_wh, k)
    y, x = np.indices((h, w), dtype=float)
    destination = np.stack((x, y), axis=-1)
    center = (np.asarray(size_wh) - 1) / 2
    radius = np.linalg.norm(center)
    q = (destination - center) / radius
    rd = np.linalg.norm(q, axis=-1)
    ru = rd.copy()
    for _ in range(12):
        derivative = 1 + 3*k*ru*ru
        if np.any(derivative <= 0):
            raise ValueError("Noninvertible radial map")
        ru -= (ru + k*ru**3 - rd) / derivative
    factor = np.divide(ru, rd, out=np.ones_like(ru), where=rd > 0)
    source = center + radius*q*factor[..., None]
    if np.max(np.abs(radial_forward(source, size_wh, k) - destination)) > 1e-7:
        raise ValueError("Radial inverse did not converge")
    return source


def distort_rgb(rgb, k):
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("uint8 RGB required")
    source = radial_source_map(list(array.shape[1::-1]), k)
    return cv2.remap(array, source[..., 0].astype(np.float32), source[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
