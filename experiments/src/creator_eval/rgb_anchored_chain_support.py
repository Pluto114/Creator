"""Explicit image-membership claims condition the fixed pair-chain support.

Anchors are neither a 3D correspondence nor a line to which points are snapped.
Only fully supported chains contribute positive support. Unknown alternatives
remain in the audit, never become proof of absence or unique target identity.
"""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from .rgb_chain_support import build_context as build_all_chain_context
from .rod_fixture_finite import canonical_hash, json_ready
from .rod_foreground_identity import validate_anchors

SCOPE = ("Positive support conditional on explicit RGB foreground claims and fixed pair proposals; "
         "unknown chains excluded from positive support, not evidence of absence or unique identity")


def classify_anchor_rows(rows, left, right):
    """Tri-state box membership; one real interval must contain the whole box."""
    if not np.isfinite([left, right]).all() or left > right:
        raise ValueError("Ordered finite anchor x interval required")
    if not rows:
        return "unresolved"
    all_supported = True
    all_disjoint = True
    for intervals in rows:
        bands = np.asarray(intervals, float).reshape(-1, 2)
        if not np.isfinite(bands).all() or np.any(bands[:, 0] >= bands[:, 1]):
            raise ValueError("Ordered finite measured intervals required")
        if not len(bands):
            all_supported = False
            all_disjoint = False
            continue
        overlaps = (bands[:, 0] <= right) & (bands[:, 1] >= left)
        contains = (bands[:, 0] <= left) & (bands[:, 1] >= right)
        all_disjoint &= not overlaps.any()
        all_supported &= bool(contains.any())
    return "contradicted" if all_disjoint else ("supported" if all_supported else "unresolved")


def _covered_rows(anchor):
    xy, uncertainty = np.asarray(anchor["xy"]), np.asarray(anchor["uncertainty_xy_px"])
    low, high = xy-uncertainty, xy+uncertainty
    first = int(np.ceil(low[1]-.5+1e-9))
    last = int(np.floor(high[1]+.5-1e-9))
    if last < first:
        first = last = int(np.floor(xy[1]+.5))
    return range(first, last+1), float(low[0]), float(high[0])


def _anchor_states(anchor, view, members):
    rows, low, high = _covered_rows(anchor)
    raw, selected = [], [[] for _ in range(members.shape[2])]
    for y in rows:
        # Validation allows the outer half-pixel border; touching it cannot
        # wrap NumPy indexing onto a row at the other end of the image.
        if not 0 <= y < len(view["left"]):
            raw.append([])
            for entries in selected:
                entries.append([])
            continue
        slots = np.flatnonzero(np.isfinite(view["left"][y]))
        bands = [(float(view["left"][y, i]), float(view["right"][y, i])) for i in slots]
        raw.append(bands)
        for candidate, entries in enumerate(selected):
            entries.append([band for slot, band in zip(slots, bands) if members[y, slot, candidate]])
    states = [classify_anchor_rows(entries, low, high) for entries in selected]
    return dict(view_id=anchor["view_id"], raw_measurement_state=classify_anchor_rows(raw, low, high),
        covered_rows=list(rows), candidate_states=states, candidate_state_counts=dict(Counter(states)))


def condition_context(context, evidence, cameras, anchors):
    """No arm geometry is accepted here; reuse one context for every arm."""
    identity_views = [dict(view_id=f["view_id"], rgb_sha256=f["rgb_sha256"], size_wh=f["size_wh"],
        K_index=c["K_index"], world_to_camera_cv=c["world_to_camera_cv"],
        y_range=[f["guide_xyxy"][0][1], f["guide_xyxy"][1][1]], candidates=f["pool"]["candidates"])
        for f, c in zip(evidence, cameras)]
    normalized, identity_policy = validate_anchors(anchors, identity_views)
    original = context["association"]
    if (original["evidence_sha256"] != canonical_hash(evidence)
            or original["cameras_sha256"] != canonical_hash(cameras)
            or context["sha256"] != canonical_hash(original)):
        raise ValueError("Anchors must condition the exact raw evidence/camera context")
    association = copy.deepcopy(original)
    association.update(scope=SCOPE, unconditioned_context_sha256=context["sha256"],
        original_chain_count=original["chain_count"], anchor_policy=identity_policy,
        anchors=json_ready(normalized), anchors_sha256=canonical_hash(normalized),
        anchor_view_count=len(normalized), anchors_used_as_3d_correspondences=False,
        geometry_changed=False, target_identity_confirmed=False,
        positive_chain_selection_complete=False, anchor_audit=[], chain_state_counts={},
        identity_state="unresolved", positive_support_state="unresolved")
    if original["fallback_to_raw_union"]:
        association["reason"] = "incomplete_pair_search_raw_fallback_not_identity_conditioned"
    elif len(normalized) < identity_policy["minimum_anchor_views"]:
        association.update(assignments=[], chain_count=0, fallback_to_raw_union=False,
            reason="insufficient_explicit_anchor_views_no_positive_support")
    else:
        ids = {v["view_id"]: i for i, v in enumerate(context["views"])}
        audits = [_anchor_states(a, context["views"][ids[a["view_id"]]], context["row_members"][ids[a["view_id"]]])
                  for a in normalized]
        association["anchor_audit"] = audits
        positive, counts = set(), Counter()
        for assignment in original["assignments"]:
            reduced, states = list(assignment), []
            for audit in audits:
                vi = ids[audit["view_id"]]
                members = assignment[vi]
                supported = tuple(i for i in members if audit["candidate_states"][i] == "supported")
                if supported:
                    states.append("supported")
                elif members and all(audit["candidate_states"][i] == "contradicted" for i in members):
                    states.append("contradicted")
                else:
                    states.append("unresolved")
                reduced[vi] = supported
            state = "contradicted" if "contradicted" in states else (
                "supported" if all(s == "supported" for s in states) else "unresolved")
            counts[state] += 1
            if state == "supported":
                positive.add(tuple(tuple(row) for row in reduced))
        association.update(assignments=sorted(positive), chain_count=len(positive),
            chain_state_counts=dict(counts), positive_chain_selection_complete=True,
            positive_support_state="supported" if positive else "unresolved",
            reason="fully_anchor_supported_fixed_chains" if positive else "no_fully_anchor_supported_chain")
    association["assignment_sha256"] = canonical_hash(association["assignments"])
    # Shared read-only projection/membership arrays, new independent metadata.
    return dict(views=context["views"], row_members=context["row_members"], association=association,
        sha256=canonical_hash(association))


def build_context(evidence, cameras, anchors, policy=None):
    return condition_context(build_all_chain_context(evidence, cameras, policy), evidence, cameras, anchors)
