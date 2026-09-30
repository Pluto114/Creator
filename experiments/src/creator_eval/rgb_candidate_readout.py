"""Candidate-union photometric measurement volume; not target identity.

An ambiguous row retains each already accepted edge-pair interval separately.
It never contributes more than one vote per view and never fills their envelope.
"""

from __future__ import annotations

import numpy as np

from .common_readout import sample_segments
from .common_readout_abstention import readout

POLICY = dict(minimum_views=3, horizontal_padding_px=1.0)


def support_views(evidence, cameras):
    if not evidence or len(evidence) != len(cameras) or [f["view_id"] for f in evidence] != [c["view_id"] for c in cameras]:
        raise ValueError("Nonempty matching evidence/camera order required")
    if len({f["view_id"] for f in evidence}) != len(evidence):
        raise ValueError("Duplicate views cannot create votes")
    views = []
    for frame, camera in zip(evidence, cameras):
        size = frame["size_wh"]
        if len(size) != 2 or any(type(x) is not int or x < 1 for x in size):
            raise ValueError("Positive integer image dimensions required")
        width, height = size
        rows, seen = frame["observations"]["rows"], set()
        capacity = max([1] + [len(r["candidates"]) for r in rows if r["status"] in {"observed", "ambiguous"}])
        left, right = np.full((height, capacity), np.nan), np.full((height, capacity), np.nan)
        counts = dict(observed=0, ambiguous=0, unknown=0, absent=0)
        for row in rows:
            y, state = row["y"], row["status"]
            if type(y) is not int or not 0 <= y < height or y in seen or state not in counts:
                raise ValueError("Unique integer rows with explicit observation states required")
            seen.add(y)
            counts[state] += 1
            if state in {"unknown", "absent"}:
                continue
            candidates = row["candidates"]
            if (state == "observed" and len(candidates) != 1) or (state == "ambiguous" and len(candidates) < 2):
                raise ValueError("Observation state and candidate cardinality differ")
            for i, candidate in enumerate(candidates):
                a, b = candidate["left_edge"]["x"], candidate["right_edge"]["x"]
                if not np.isfinite([a, b]).all() or not 0 <= a < b <= width-1:
                    raise ValueError("Ordered finite in-image edge pair required")
                left[y, i], right[y, i] = a, b
        k, e = np.asarray(camera["K_index"], float), np.asarray(camera["world_to_camera_cv"], float)
        if camera["state"] != "validated" or k.shape != (3, 3) or e.shape != (3, 4) or not np.isfinite(k).all() or not np.isfinite(e).all():
            raise ValueError("Explicit finite validated pinhole cameras required")
        rotation = e[:, :3]
        if (not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0) or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
            raise ValueError("Positive focal lengths and proper rigid cameras required")
        views.append(dict(view_id=frame["view_id"], K_index=k.copy(), world_to_camera_cv=e.copy(),
                          left=left, right=right, size_wh=list(size), row_states=counts))
    return views


def support_votes(points, views, policy=None):
    p = POLICY if policy is None else policy
    if set(p) != set(POLICY) or type(p["minimum_views"]) is not int or not 1 <= p["minimum_views"] <= len(views):
        raise ValueError("Valid distinct-view support count required")
    pad = p["horizontal_padding_px"]
    if not np.isfinite(pad) or pad < 0 or len({v["view_id"] for v in views}) != len(views):
        raise ValueError("Finite nonnegative padding and unique views required")
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Finite Nx3 points required")
    votes = np.zeros(len(points), np.int32)
    for view in views:
        e, k = view["world_to_camera_cv"], view["K_index"]
        xyz = points @ e[:, :3].T + e[:, 3]
        uvh = xyz @ k.T
        valid = uvh[:, 2] > 1e-10
        xy = np.zeros((len(points), 2))
        xy[valid] = uvh[valid, :2]/uvh[valid, 2, None]
        valid &= (xy[:, 1] >= -.5) & (xy[:, 1] < len(view["left"])-.5)
        indices = np.flatnonzero(valid)
        y = np.floor(xy[indices, 1]+.5).astype(int)
        inside = np.zeros(len(indices), bool)
        # One interval at a time avoids an Nx(max candidates) memory multiplier.
        for i in range(view["left"].shape[1]):
            left, right = view["left"][y, i], view["right"][y, i]
            inside |= np.isfinite(left) & (xy[indices, 0] >= left-pad) & (xy[indices, 0] <= right+pad)
        votes[indices[inside]] += 1
    return votes


def support_mask(points, views, policy=None):
    p = POLICY if policy is None else policy
    return support_votes(points, views, p) >= p["minimum_views"]


class CandidateSupportedReadout:
    """Cache one possible-support base mask; all variants use the same geometry."""

    def __init__(self, points, views, policy=None):
        self.policy = dict(POLICY if policy is None else policy)
        self.views = views
        self.votes = support_votes(points, views, self.policy)
        self.mask = self.votes >= self.policy["minimum_views"]
        self.points = np.asarray(points, float)[self.mask].copy()
        self.input_point_count = len(points)

    def __call__(self, segments, reader_config):
        segments = np.asarray(segments, float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
            raise ValueError("Finite Mx2x3 segments required")
        samples = sample_segments(segments, reader_config["voxel_size"]/2, 2000000)
        votes = support_votes(samples, self.views, self.policy)
        selected = votes >= self.policy["minimum_views"]
        values = np.concatenate((self.points, samples[selected]))
        result = readout(values, np.empty((0, 2, 3)), reader_config)
        result.update(full_input_point_count=self.input_point_count, supported_base_point_count=len(self.points),
            full_input_segment_count=len(segments), full_curve_sample_count=len(samples),
            supported_curve_sample_count=int(selected.sum()), rgb_support_policy=self.policy,
            base_vote_histogram=np.bincount(self.votes, minlength=len(self.views)+1).tolist(),
            curve_vote_histogram=np.bincount(votes, minlength=len(self.views)+1).tolist(),
            scope="Same raw RGB candidate-union possible-support volume for full base and candidates; no semantic identity or whole-scene quality certification")
        return result
