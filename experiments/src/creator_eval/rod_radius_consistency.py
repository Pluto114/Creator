"""RGB edge-ray/axis distance diagnostics, conditional on silhouette geometry.

A ray tangent to a circular cylinder stays one radius from its infinite axis.
These distances test that assumption; they do not establish physical identity,
prove that an image edge is a silhouette, or recover a surface radius by fiat.
"""
from __future__ import annotations

import numpy as np

from .rod_multiview_candidates import _validated_views


def axis_ray_distances(anchor, direction, origin, rays):
    """Shortest infinite-line distances, excluding parallel or backwards ray hits."""
    anchor, direction, origin, rays = [np.asarray(x, float) for x in (anchor, direction, origin, rays)]
    if anchor.shape != (3,) or direction.shape != (3,) or origin.shape != (3,) or rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError("Expected 3D axis, origin, and Nx3 rays")
    if any(not np.isfinite(x).all() for x in (anchor, direction, origin, rays)):
        raise ValueError("Nonfinite axis/rays")
    length, ray_lengths = np.linalg.norm(direction), np.linalg.norm(rays, axis=1)
    if length <= 1e-12 or np.any(ray_lengths <= 1e-12):
        raise ValueError("Zero direction")
    axis, unit = direction / length, rays / ray_lengths[:, None]
    normal = np.cross(axis, unit)
    squared = np.sum(normal * normal, axis=1)
    valid = squared > 1e-12
    displacement = anchor - origin
    b = unit @ axis
    forward = np.full(len(rays), np.nan)
    forward[valid] = ((unit @ displacement)[valid] - b[valid] * np.dot(axis, displacement)) / squared[valid]
    valid &= forward > 0
    distances = np.full(len(rays), np.nan)
    distances[valid] = np.abs(normal[valid] @ displacement) / np.sqrt(squared[valid])
    return distances, valid


def summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "median": None, "p10": None, "p90": None, "relative_p90_p10_spread": None}
    median = float(np.median(values))
    low, high = [float(v) for v in np.quantile(values, [.1, .9])]
    return {"count": len(values), "median": median, "p10": low, "p90": high,
            "relative_p90_p10_spread": (high - low) / median if median > 1e-12 else None}


def diagnose_edge_radii(hypothesis, views, observations):
    _validated_views(views)
    if len(views) != len(observations) or len(hypothesis["matches"]) != len(views):
        raise ValueError("Views, matches and raw observations must align")
    all_values, rows = [], []
    for vi in hypothesis["supporting_views"]:
        view, raw = views[vi], observations[vi]
        candidate = view["candidates"][hypothesis["matches"][vi]["candidate_index"]]
        ys, left, right = [], [], []
        for ri, ci, _ in candidate["row_matches"]:
            row, pair = raw["rows"][ri], raw["rows"][ri]["candidates"][ci]
            ys.append(row["y"])
            left.append(pair["left_edge"]["x"])
            right.append(pair["right_edge"]["x"])
        camera = np.asarray(view["world_to_camera_cv"], float)[:3]
        rotation, translation = camera[:, :3], camera[:, 3]
        origin = -rotation.T @ translation
        side_values, side_stats = [], {}
        for name, xs in (("left", left), ("right", right)):
            pixels = np.c_[xs, ys, np.ones(len(xs))]
            # 像素是原图index坐标，K也来自同一约定；这里不能再偷偷加半像素。
            camera_rays = np.linalg.solve(np.asarray(view["K_index"]), pixels.T).T
            world_rays = camera_rays @ rotation
            distances, valid = axis_ray_distances(hypothesis["model"]["anchor"], hypothesis["model"]["direction"], origin, world_rays)
            side_values.append(distances[valid])
            side_stats[name] = summary(distances)
            side_stats[name]["invalid_ray_count"] = int(np.count_nonzero(~valid))
        values = np.concatenate(side_values)
        all_values.extend(values.tolist())
        a, b = side_stats["left"]["median"], side_stats["right"]["median"]
        balance = abs(a - b) / ((a + b) / 2) if a is not None and b is not None and a + b > 1e-12 else None
        rows.append({"view_id": view["view_id"], "candidate_index": hypothesis["matches"][vi]["candidate_index"],
                     "matched_rows": len(ys), **side_stats, "combined": summary(values), "relative_side_imbalance": balance})
    return {"all_supported_edge_rays": summary(all_values), "views": rows,
            "units": "input_camera_world_units", "gate_applied": False,
            "scope": "conditional_circular_cylinder_tangent_consistency_not_identity_or_silhouette_proof"}
