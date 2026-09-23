"""Diagnostic held-out plans that score every observed camera view.

The old farthest-pair validation remains the primary experiment readout. This
adds coverage, not a new success gate or independent samples from reused tracks.
"""
from __future__ import annotations

import copy
from itertools import combinations

import numpy as np

from creator_eval.background_correspondences import residual_summary
from creator_eval.camera_bundle import centers, project, triangulate, validate_cameras


def all_view_plan(validation, initial_extrinsics, *, training_track_ids):
    """Freeze each unused observation against a pair picked from initial cameras."""
    e = np.asarray(initial_extrinsics, float)
    if e.ndim != 3 or e.shape[1:] != (3, 4) or not np.isfinite(e).all():
        raise ValueError("Finite initial camera matrices required")
    camera_centers = centers(e)
    train_ids = set(training_track_ids)
    seen, rows = set(), []
    for track in validation:
        tid = track["track_id"]
        if tid in train_ids or tid in seen:
            raise ValueError("Validation tracks must be unique and disjoint from fitting")
        seen.add(tid)
        observations = sorted(copy.deepcopy(track["observations"]), key=lambda o: o["view"])
        view_ids = [o["view"] for o in observations]
        if len(observations) < 3 or len(set(view_ids)) != len(view_ids):
            raise ValueError("At least three distinct observed views required")
        for observation in observations:
            view = observation["view"]
            xy = np.asarray(observation["xy"], float)
            if isinstance(view, bool) or not isinstance(view, int) or not 0 <= view < len(e) or xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError("Finite image observation and valid view index required")
        for scoring in observations:
            others = [o for o in observations if o["view"] != scoring["view"]]
            pair = max(combinations(others, 2), key=lambda ab: np.linalg.norm(camera_centers[ab[0]["view"]] - camera_centers[ab[1]["view"]]))
            baseline = float(np.linalg.norm(camera_centers[pair[0]["view"]] - camera_centers[pair[1]["view"]]))
            if baseline <= 1e-10:
                raise ValueError("Held-out view has no nondegenerate triangulation pair")
            rows.append(dict(track_id=tid, triangulation=list(pair), scoring=scoring, initial_pair_baseline=baseline))
    return dict(rows=rows, validation_track_ids=sorted(seen), training_track_ids=sorted(train_ids),
                scope="Every validation observation scored once; pair selected only from initial camera centers; correlated rows, no BA fit")


def summarize_vectors(vectors):
    values = np.asarray(vectors, float).reshape(-1, 2)
    valid = np.isfinite(values).all(axis=1)
    norms = np.linalg.norm(values, axis=1)
    finite = values[valid]
    return {**residual_summary(norms),
            "signed_mean_uv_px": finite.mean(axis=0).tolist() if len(finite) else None,
            "signed_median_uv_px": np.median(finite, axis=0).tolist() if len(finite) else None}


def score_all_views(intrinsics, extrinsics, plan):
    k, e = validate_cameras(intrinsics, extrinsics)
    if set(plan["validation_track_ids"]) & set(plan["training_track_ids"]):
        raise ValueError("Held-out plan overlaps camera fitting")
    seen, rows = set(), []
    for entry in plan["rows"]:
        scoring = entry["scoring"]
        view = scoring["view"]
        key = (entry["track_id"], view)
        if key in seen or entry["track_id"] not in plan["validation_track_ids"]:
            raise ValueError("Repeated or undeclared scored observation")
        seen.add(key)
        if view in {o["view"] for o in entry["triangulation"]}:
            raise ValueError("Scored observation must not enter triangulation")
        point = triangulate(entry["triangulation"], k, e)
        forward = np.isfinite(point).all() and all(project(point[None], k[o["view"]], e[o["view"]])[1][0] > 0 for o in entry["triangulation"])
        uv, depth = project(point[None], k[view], e[view])
        valid = bool(forward and depth[0] > 0 and np.isfinite(uv).all())
        vector = uv[0] - scoring["xy"] if valid else np.full(2, np.nan)
        rows.append(dict(track_id=entry["track_id"], view=view, triangulation_views=[o["view"] for o in entry["triangulation"]],
                         observed_xy=scoring["xy"], signed_uv_px=vector.tolist() if valid else None,
                         error_px=float(np.linalg.norm(vector)) if valid else None))
    # 第一和最后一个相机以前只负责三角化，没有被打分。先把盲区补上；
    # 分数漂亮也不等于三维正确，更别拿这些相关像素当一堆独立样本。
    vectors = [r["signed_uv_px"] if r["signed_uv_px"] is not None else [np.nan, np.nan] for r in rows]
    per_view = []
    for view in range(len(k)):
        selected = [r for r in rows if r["view"] == view]
        xy = np.asarray([r["observed_xy"] for r in selected]).reshape(-1, 2)
        residuals = [r["signed_uv_px"] if r["signed_uv_px"] is not None else [np.nan, np.nan] for r in selected]
        per_view.append(dict(view=view, **summarize_vectors(residuals),
                             observation_bounds_xy=[xy.min(axis=0).tolist(), xy.max(axis=0).tolist()] if len(xy) else None))
    return dict(summary=summarize_vectors(vectors), per_view=per_view, rows=rows,
                scope="Additional RGB-only diagnostic; retains original scores and gate; rows within a track are dependent")
