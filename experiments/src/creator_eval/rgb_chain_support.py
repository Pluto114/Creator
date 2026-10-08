"""All fixed pair proposals, with same-chain (not mixed-chain) RGB votes.

Completeness is over this fixed image pool and nondegenerate two-plane seeds,
not all noisy multi-view least-squares solutions or all RGB explanations.
No branch ranking, target selection, guide-distance gate, or native geometry.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .rgb_candidate_readout import POLICY, support_views, support_votes
from .rod_fixture_finite import canonical_hash

DEFAULTS = dict(reprojection_threshold_px=1.5, row_inlier_distance_px=1.2,
    minimum_chain_views=4, minimum_plane_sine=.05, maximum_pairs=500000,
    maximum_chains=100000, batch_size=512)
SCOPE = ("All nondegenerate pair-plane seeds of the fixed sampled image pool; "
         "not all feasible noisy assignments or target identity. Unknown is not absence.")
POINT_BATCH_SIZE = 4096


def consistent_chain_mask(memberships, assignments, minimum_views=3):
    """OR chains only after counting distinct supporting views within each chain."""
    arrays = [np.asarray(value) for value in memberships]
    if (not arrays or type(minimum_views) is not int or not 1 <= minimum_views <= len(arrays)
            or any(a.dtype != np.bool_ or a.ndim != 2 for a in arrays)
            or len({len(a) for a in arrays}) != 1):
        raise ValueError("Matching per-view NxM boolean memberships required")
    count = len(arrays[0])
    # Packed integer sets keep work proportional to machine words, not Nxchains.
    # Each cache entry is one view's own candidate union; views are only counted
    # inside a single complete assignment below, never mixed across assignments.
    columns, caches = [], []
    for array in arrays:
        packed = np.packbits(array, axis=0, bitorder="little")
        columns.append([int.from_bytes(packed[:, i].tobytes(), "little") for i in range(array.shape[1])])
        caches.append({(): 0})
    result, universe = 0, (1 << count)-1
    for assignment in assignments:
        if len(assignment) != len(arrays):
            raise ValueError("One candidate-index set per view required")
        at_least = [universe]+[0]*minimum_views
        for array, indices, bits, cache in zip(arrays, assignment, columns, caches):
            ids = np.asarray(indices)
            if not len(ids):
                continue
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= array.shape[1]):
                raise ValueError("In-range integer candidate indices required")
            key = tuple(int(i) for i in ids)
            if key not in cache:
                union = 0
                for index in key:
                    union |= bits[index]
                cache[key] = union
            supported = cache[key]
            for level in range(minimum_views, 0, -1):
                at_least[level] |= at_least[level-1] & supported
        result |= at_least[minimum_views]
    packed_result = np.frombuffer(result.to_bytes((count+7)//8, "little"), np.uint8)
    return np.unpackbits(packed_result, bitorder="little", count=count).astype(bool)


def _policy(policy):
    p = dict(DEFAULTS if policy is None else policy)
    if set(p) != set(DEFAULTS):
        raise ValueError("Complete explicit chain search policy required")
    for name in ("maximum_pairs", "maximum_chains", "batch_size", "minimum_chain_views"):
        if type(p[name]) is not int or p[name] < 1:
            raise ValueError("Positive integer search budgets and view count required")
    for name in ("reprojection_threshold_px", "row_inlier_distance_px", "minimum_plane_sine"):
        if isinstance(p[name], bool) or not np.isfinite(p[name]) or p[name] <= 0:
            raise ValueError("Positive finite distance/conditioning thresholds required")
    if p["minimum_plane_sine"] >= 1:
        raise ValueError("Plane sine must be less than one")
    return p


def _matching_candidates(projected, lines, ys, threshold):
    """Ragged exact median-nine matching; midpoint is a necessary prefilter."""
    count = len(projected)
    result = [[] for _ in range(count)]
    if not len(lines):
        return result
    valid = np.linalg.norm(projected[:, :2], axis=1) > 1e-12
    valid &= np.abs(projected[:, 0]) > 1e-10*np.linalg.norm(projected[:, :2], axis=1)
    rows = np.flatnonzero(valid)
    if not len(rows):
        return result
    slopes = -projected[rows, 1]/projected[rows, 0]
    intercepts = -projected[rows, 2]/projected[rows, 0]
    candidate_slope, candidate_intercept = -lines[:, 1]/lines[:, 0], -lines[:, 2]/lines[:, 0]
    midpoint = float(ys[4])
    mid_x = candidate_slope*midpoint+candidate_intercept
    order = np.argsort(mid_x, kind="stable")
    seed_mid = slopes*midpoint+intercepts
    # Only the prefilter gets a numerical margin; final criterion is unchanged.
    lower = np.searchsorted(mid_x[order], seed_mid-threshold-1e-10, side="left")
    upper = np.searchsorted(mid_x[order], seed_mid+threshold+1e-10, side="right")
    seed = np.repeat(np.arange(len(rows)), upper-lower)
    if not len(seed):
        return result
    slots = np.concatenate([order[a:b] for a, b in zip(lower, upper)])
    differences = ((slopes[seed]-candidate_slope[slots])[:, None]*ys
                   +(intercepts[seed]-candidate_intercept[slots])[:, None])
    accepted = np.median(np.abs(differences), axis=1) <= threshold
    for row, slot in zip(rows[seed[accepted]], slots[accepted]):
        result[int(row)].append(int(slot))
    return [tuple(sorted(indices)) for indices in result]


def _associate(lines, projections, ys, policy):
    counts = [len(value) for value in lines]
    pair_count = sum(counts[a]*counts[b] for a, b in combinations(range(len(lines)), 2))
    stats = dict(policy=dict(policy), scope=SCOPE, pool_counts=counts, pair_count=pair_count,
        attempted_pairs=0, nondegenerate_pairs=0, feasible_pairs=0, chain_count=0,
        search_complete=False, fallback_to_raw_union=True, target_identity_confirmed=False)
    if pair_count > policy["maximum_pairs"]:
        return dict(**stats, reason="pair_budget_exceeded", assignments=[])
    planes = []
    for image_lines, projection in zip(lines, projections):
        values = image_lines @ projection
        normals = np.linalg.norm(values[:, :3], axis=1)
        planes.append(values/normals[:, None])
    signatures = set()
    for a, b in combinations(range(len(lines)), 2):
        total = counts[a]*counts[b]
        for offset in range(0, total, policy["batch_size"]):
            flat = np.arange(offset, min(total, offset+policy["batch_size"]))
            first, second = planes[a][flat//counts[b]], planes[b][flat % counts[b]]
            stats["attempted_pairs"] += len(flat)
            cross = np.cross(first[:, :3], second[:, :3])
            sine = np.linalg.norm(cross, axis=1)
            keep = sine >= policy["minimum_plane_sine"]
            first, second, cross, sine = first[keep], second[keep], cross[keep], sine[keep]
            stats["nondegenerate_pairs"] += len(first)
            if not len(first):
                continue
            cosine = np.sum(first[:, :3]*second[:, :3], axis=1)
            denominator = sine*sine
            alpha = (-first[:, 3]+cosine*second[:, 3])/denominator
            beta = (-second[:, 3]+cosine*first[:, 3])/denominator
            anchors = alpha[:, None]*first[:, :3]+beta[:, None]*second[:, :3]
            directions = cross/sine[:, None]
            matches = []
            for image_lines, projection, sample_y in zip(lines, projections, ys):
                # Infinite-line projection; arbitrary anchor depth is irrelevant.
                projected = np.cross(anchors @ projection[:, :3].T+projection[:, 3],
                                     directions @ projection[:, :3].T)
                matches.append(_matching_candidates(projected, image_lines, sample_y,
                    policy["reprojection_threshold_px"]))
            for index in range(len(first)):
                signature = tuple(tuple(view[index]) for view in matches)
                if sum(bool(ids) for ids in signature) >= policy["minimum_chain_views"]:
                    stats["feasible_pairs"] += 1
                    signatures.add(signature)
                    if len(signatures) > policy["maximum_chains"]:
                        stats["chain_count"] = len(signatures)
                        return dict(**stats, reason="chain_budget_exceeded", assignments=[])
    assignments = sorted(signatures)
    stats.update(chain_count=len(assignments), search_complete=True, fallback_to_raw_union=False)
    return dict(**stats, reason="fixed_pair_proposals_exhausted", assignments=assignments)


def _row_members(frame, lines, view, threshold):
    """Keep every measured band compatible with a line, including nested pairs."""
    height, capacity = view["left"].shape
    members = np.zeros((height, capacity, len(lines)), bool)
    for row in frame["observations"]["rows"]:
        if row["status"] not in {"observed", "ambiguous"}:
            continue
        for slot, pair in enumerate(row["candidates"]):
            residual = np.abs(lines[:, 0]*pair["center_x"]+lines[:, 1]*row["y"]+lines[:, 2])
            members[row["y"], slot] = residual <= threshold
    return members


def build_context(evidence, cameras, policy=None):
    """Freeze once from raw RGB plus cameras, before inspecting any method arm."""
    p = _policy(policy)
    views = support_views(evidence, cameras)
    if p["minimum_chain_views"] > len(views):
        raise ValueError("Enough distinct cameras for chain support required")
    lines, projections, ys = [], [], []
    for frame, view in zip(evidence, views):
        if (canonical_hash(frame["observations"]) != frame["observation_sha256"]
                or canonical_hash(frame["pool"]) != frame["pool_sha256"]):
            raise ValueError("Exact raw observations and fixed image pool required")
        values = np.asarray([item["line"] for item in frame["pool"]["candidates"]], float).reshape(-1, 3)
        norms = np.linalg.norm(values[:, :2], axis=1)
        if not np.isfinite(values).all() or np.any(norms <= 1e-12) or np.any(np.abs(values[:, 0]) <= 1e-10*norms):
            raise ValueError("Finite nonhorizontal image lines required")
        lines.append(values/norms[:, None])
        projections.append(view["K_index"] @ view["world_to_camera_cv"])
        guide = np.asarray(frame["guide_xyxy"], float)
        if guide.shape != (2, 2) or not np.isfinite(guide).all() or guide[1, 1] <= guide[0, 1]:
            raise ValueError("Finite increasing-y search interval required")
        ys.append(np.linspace(guide[0, 1], guide[1, 1], 9))
    association = _associate(lines, projections, ys, p)
    association["assignment_sha256"] = canonical_hash(association["assignments"])
    association["evidence_sha256"] = canonical_hash(evidence)
    association["cameras_sha256"] = canonical_hash(cameras)
    row_members = [] if association["fallback_to_raw_union"] else [
        _row_members(frame, image_lines, view, p["row_inlier_distance_px"])
        for frame, image_lines, view in zip(evidence, lines, views)]
    return dict(views=views, association=association, row_members=row_members,
        sha256=canonical_hash(association))


def point_memberships(points, context):
    """Measured raw intervals only; no filled envelope, depth or guide tube."""
    result = []
    points = np.asarray(points, float)
    for view, rows in zip(context["views"], context["row_members"]):
        e, k = view["world_to_camera_cv"], view["K_index"]
        projected = (points @ e[:, :3].T+e[:, 3]) @ k.T
        valid = projected[:, 2] > 1e-10
        xy = np.zeros((len(points), 2))
        xy[valid] = projected[valid, :2]/projected[valid, 2, None]
        valid &= (xy[:, 1] >= -.5) & (xy[:, 1] < len(rows)-.5)
        ids = np.flatnonzero(valid)
        y = np.floor(xy[ids, 1]+.5).astype(int)
        members = np.zeros((len(points), rows.shape[2]), bool)
        pad = POLICY["horizontal_padding_px"]
        for slot in range(rows.shape[1]):
            left, right = view["left"][y, slot], view["right"][y, slot]
            inside = np.isfinite(left) & (xy[ids, 0] >= left-pad) & (xy[ids, 0] <= right+pad)
            members[ids[inside]] |= rows[y[inside], slot]
        result.append(members)
    return result


def chain_support_mask(points, context, raw_votes=None):
    votes = support_votes(points, context["views"], POLICY) if raw_votes is None else np.asarray(raw_votes)
    raw = votes >= POLICY["minimum_views"]
    if context["association"]["fallback_to_raw_union"]:
        return raw.copy()
    mask = np.zeros(len(points), bool)
    if not context["association"]["assignments"]:
        return mask
    indices = np.flatnonzero(raw)
    values = np.asarray(points, float)
    for start in range(0, len(indices), POINT_BATCH_SIZE):
        batch = indices[start:start+POINT_BATCH_SIZE]
        members = point_memberships(values[batch], context)
        mask[batch] = consistent_chain_mask(members, context["association"]["assignments"], POLICY["minimum_views"])
    return mask
