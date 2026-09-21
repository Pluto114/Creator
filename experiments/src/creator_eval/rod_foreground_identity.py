"""Explicit foreground anchors, kept separate from coarse search guides.

An anchor is an object-membership claim in one original image, not a 3D point
correspondence. Its uncertainty rectangle must fit measured support and a finite
projected segment. Missing evidence remains unresolved; no snapping to a rod.
"""
from __future__ import annotations

import copy

import numpy as np

from .rod_candidate_extent import bound_selected_candidate
from .rod_cylinder_gate import recheck_finite_cylinder
from .rod_multiview_candidates import _validated_views

DEFAULT_POLICY = {"minimum_anchor_views": 2}


def validate_anchors(anchors, views, config=None):
    policy = {**DEFAULT_POLICY, **(config or {})}
    minimum = policy["minimum_anchor_views"]
    if set(policy) != set(DEFAULT_POLICY) or isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise ValueError("At least two independent annotated views required")
    _validated_views(views)
    by_id = {v["view_id"]: v for v in views}
    normalized, seen, centers = [], set(), []
    for anchor in anchors:
        if set(anchor) != {"view_id", "xy", "uncertainty_xy_px", "rgb_sha256"}:
            raise ValueError("Unexpected foreground-anchor fields")
        view_id = anchor["view_id"]
        if view_id not in by_id or view_id in seen:
            raise ValueError("Unknown or duplicate anchor view")
        seen.add(view_id)
        view = by_id[view_id]
        xy, uncertainty = np.asarray(anchor["xy"], float), np.asarray(anchor["uncertainty_xy_px"], float)
        if xy.shape != (2,) or uncertainty.shape != (2,) or not np.isfinite([xy, uncertainty]).all() or np.any(uncertainty < 0):
            raise ValueError("Finite pixel point and nonnegative uncertainty required")
        size = np.asarray(view["size_wh"], float)
        if np.any(xy - uncertainty < -.5) or np.any(xy + uncertainty > size - .5):
            raise ValueError("Anchor uncertainty leaves the original image")
        expected = view.get("rgb_sha256")
        if not isinstance(expected, str) or len(expected) != 64 or anchor["rgb_sha256"] != expected:
            raise ValueError("Anchor is detached from its exact RGB frame")
        camera = np.asarray(view["world_to_camera_cv"], float)[:3]
        centers.append(-camera[:, :3].T @ camera[:, 3])
        normalized.append({**anchor, "xy": xy, "uncertainty_xy_px": uncertainty})
    if len(centers) > 1:
        distances = np.linalg.norm(np.asarray(centers)[:, None] - centers, axis=2)
        span = max(1.0, float(distances.max()))
        if np.any(distances[np.triu_indices(len(centers), 1)] <= span * 1e-9):
            raise ValueError("Repeated camera center is not an independent identity view")
    return normalized, policy


def build_anchor_proposals(association, views, observations, extent_config, cylinder_config=None):
    """Bound all retained explanations once, before any annotation is consulted."""
    hypotheses = ([association["selected"]] if association.get("selected") is not None else []) + association.get("alternatives", [])
    proposals, seen = [], set()
    for ordinal, hypothesis in enumerate(hypotheses):
        signature = tuple((i, hypothesis["matches"][i]["candidate_index"]) for i in sorted(hypothesis["supporting_views"]))
        if signature in seen:
            continue
        seen.add(signature)
        conditional = {**association, "state": "accepted", "selected": hypothesis}
        finite = bound_selected_candidate(conditional, views, observations, extent_config)
        if cylinder_config is not None:
            finite = recheck_finite_cylinder(finite, hypothesis, views, observations, cylinder_config)
        proposals.append({"ordinal": ordinal, "hypothesis": copy.deepcopy(hypothesis), "finite": finite})
    return {"search_complete": bool(association["search_complete"]), "proposals": proposals,
            "scope": "stored_best_and_competing_assignments_only_not_all_image_hypotheses"}


def _anchor_support(anchor, proposal, views, observations):
    vi = next(i for i, v in enumerate(views) if v["view_id"] == anchor["view_id"])
    view, finite = views[vi], proposal["finite"]
    xy, uncertainty = anchor["xy"], anchor["uncertainty_xy_px"]
    low, high = xy - uncertainty, xy + uncertainty
    camera = np.asarray(view["world_to_camera_cv"], float)[:3]
    projection = np.asarray(view["K_index"]) @ camera
    segments = np.asarray(finite["segments"], float).reshape(-1, 2, 3)
    points = np.c_[segments.reshape(-1, 3), np.ones(2 * len(segments))] @ projection.T
    if not len(points) or np.any(points[:, 2] <= 1e-10):
        return {"view_id": anchor["view_id"], "state": "unresolved", "reason": "no_forward_finite_projection"}
    ys = (points[:, 1] / points[:, 2]).reshape(-1, 2)
    first, last = ys.min(axis=1), ys.max(axis=1)
    finite_full = bool(np.any((first <= low[1] + 1e-9) & (last >= high[1] - 1e-9)))
    finite_any = bool(np.any((first <= high[1] + 1e-9) & (last >= low[1] - 1e-9)))
    if not finite_any:
        return {"view_id": anchor["view_id"], "state": "contradicted", "reason": "anchor_outside_finite_segments"}
    selection = finite["candidate_selection"][vi]
    if selection["view_id"] != view["view_id"]:
        raise ValueError("Finite candidate selection changed view order")
    rows = observations[vi]["rows"]
    # 每个像素行代表一个高度为1的单元；边界只相碰不算多占一行。
    lower = int(np.ceil(low[1] - .5 + 1e-9))
    upper = int(np.floor(high[1] + .5 - 1e-9))
    if upper < lower:
        lower = upper = int(np.floor(xy[1] + .5))
    matched = {}
    for ri, ci in selection["row_matches"]:
        if not isinstance(ri, (int, np.integer)) or not isinstance(ci, (int, np.integer)) or not 0 <= ri < len(rows):
            raise ValueError("Invalid finite row selection")
        row = rows[ri]
        y = float(row["y"])
        if y != np.floor(y):
            raise ValueError("Foreground anchor pilot requires original integer pixel rows")
        if int(y) in matched or not 0 <= ci < len(row["candidates"]):
            raise ValueError("Duplicate row or invalid edge pair")
        pair = row["candidates"][ci]
        left, right = float(pair["left_edge"]["x"]), float(pair["right_edge"]["x"])
        if not np.isfinite([left, right]).all() or left >= right:
            raise ValueError("Invalid measured band")
        matched[int(y)] = (left, right)
    details = []
    for y in range(lower, upper + 1):
        band = matched.get(y)
        if band is None:
            details.append({"y": y, "state": "unresolved", "reason": "no_selected_measured_row"})
        elif high[0] < band[0] - 1e-9 or low[0] > band[1] + 1e-9:
            details.append({"y": y, "state": "contradicted", "measured_band": band})
        elif low[0] >= band[0] - 1e-9 and high[0] <= band[1] + 1e-9:
            details.append({"y": y, "state": "supported", "measured_band": band})
        else:
            details.append({"y": y, "state": "unresolved", "reason": "uncertainty_crosses_band_boundary", "measured_band": band})
    states = [row["state"] for row in details]
    state = "contradicted" if "contradicted" in states else ("supported" if finite_full and all(s == "supported" for s in states) else "unresolved")
    return {"view_id": anchor["view_id"], "state": state, "finite_interval_fully_supported": finite_full, "rows": details}


def select_foreground_identity(bundle, views, observations, anchors, config=None):
    """Select only a unique fully supported candidate, including uncertain rivals."""
    anchors, policy = validate_anchors(anchors, views, config)
    if len(views) != len(observations):
        raise ValueError("Raw observations and views must align")
    output = {"state": "unresolved", "reason": "insufficient_foreground_anchor_views", "selected_ordinal": None,
              "segments": np.empty((0, 2, 3)), "proposal_audit": [], "anchor_view_count": len(anchors), "policy": policy,
              "scope": "conditional_on_explicit_foreground_claims_not_automatic_target_recognition",
              "geometry_changed": False, "anchors_used_as_3d_correspondences": False}
    if len(anchors) < policy["minimum_anchor_views"]:
        return output
    supported, possible = [], []
    for proposal in bundle["proposals"]:
        ordinal = proposal["ordinal"]
        if proposal["finite"]["state"] != "accepted":
            output["proposal_audit"].append({"ordinal": ordinal, "state": "geometric_rejection"})
            continue
        checks = [_anchor_support(a, proposal, views, observations) for a in anchors]
        states = [row["state"] for row in checks]
        state = "contradicted" if "contradicted" in states else ("supported" if all(s == "supported" for s in states) else "unresolved")
        output["proposal_audit"].append({"ordinal": ordinal, "state": state, "anchors": checks})
        if state != "contradicted":
            possible.append(proposal)
        if state == "supported":
            supported.append(proposal)
    if not bundle["search_complete"]:
        output.update(state="unresolved", reason="incomplete_retained_candidate_search")
    elif len(supported) == 1 and len(possible) == 1:
        # 不把点吸到最近的杆上，也不拿点击重拟合轴。它只确认已有几何对应谁。
        output.update(state="accepted", reason="unique_candidate_supported_by_explicit_foreground_anchors",
                      selected_ordinal=supported[0]["ordinal"], segments=copy.deepcopy(supported[0]["finite"]["segments"]))
    elif possible:
        output.update(state="unresolved", reason="competing_or_partially_supported_identity")
    else:
        output.update(state="rejected", reason="no_finite_candidate_supports_foreground_claims")
    return output
