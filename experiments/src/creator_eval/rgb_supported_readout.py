"""Shared readout restricted by raw, unambiguous RGB row support in multiple views.

The region uses no fitted axis, target truth, candidate labels or patch identity.
It is a conditional photometric volume, not foreground-object certification.
"""

from __future__ import annotations

import numpy as np

from .common_readout import sample_segments
from .common_readout_abstention import readout

POLICY = dict(minimum_views=3, horizontal_padding_px=1.0)


def support_views(evidence, cameras):
    if len(evidence) != len(cameras) or [f["view_id"] for f in evidence] != [c["view_id"] for c in cameras]:
        raise ValueError("Evidence/camera order differs")
    if len({f["view_id"] for f in evidence}) != len(evidence):
        raise ValueError("Duplicate views cannot create votes")
    views = []
    for frame, camera in zip(evidence, cameras):
        width, height = frame["size_wh"]
        left, right = np.full(height, np.nan), np.full(height, np.nan)
        seen = set()
        for row in frame["observations"]["rows"]:
            y = row["y"]
            if int(y) != y or not 0 <= y < height or y in seen:
                raise ValueError("Unique integer original-image rows required")
            seen.add(y)
            if row["status"] != "observed":
                continue
            if len(row["candidates"]) != 1:
                raise ValueError("Observed support must be unambiguous")
            candidate = row["candidates"][0]
            a, b = candidate["left_edge"]["x"], candidate["right_edge"]["x"]
            if not np.isfinite([a, b]).all() or not 0 <= a < b <= width-1:
                raise ValueError("Ordered finite RGB edge pair required")
            left[int(y)], right[int(y)] = a, b
        k, e = np.asarray(camera["K_index"], float), np.asarray(camera["world_to_camera_cv"], float)
        if camera["state"] != "validated" or k.shape != (3, 3) or e.shape != (3, 4) or not np.isfinite(k).all() or not np.isfinite(e).all():
            raise ValueError("Explicit validated cameras required")
        views.append(dict(view_id=frame["view_id"], K_index=k, world_to_camera_cv=e,
                          left=left, right=right, size_wh=[width, height]))
    return views


def support_mask(points, views, policy=None):
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
        # A row represents its own half-open pixel cell; no vertical gap dilation.
        h = len(view["left"])
        valid &= (xy[:, 1] >= -.5) & (xy[:, 1] < h-.5)
        indices = np.flatnonzero(valid)
        y = np.floor(xy[indices, 1]+.5).astype(int)
        left, right = view["left"][y], view["right"][y]
        good = np.isfinite(left) & (xy[indices, 0] >= left-pad) & (xy[indices, 0] <= right+pad)
        votes[indices[good]] += 1
    return votes >= p["minimum_views"]


class SupportedReadout:
    """Cache one identical base-point mask for all paired candidate variants."""

    def __init__(self, points, views, policy=None):
        self.policy = dict(POLICY if policy is None else policy)
        self.views = views
        self.mask = support_mask(points, views, self.policy)
        self.points = np.asarray(points, float)[self.mask].copy()
        self.input_point_count = len(points)

    def __call__(self, segments, reader_config):
        segments = np.asarray(segments, float).reshape(-1, 2, 3)
        if not np.isfinite(segments).all():
            raise ValueError("Finite segments required")
        samples = sample_segments(segments, reader_config["voxel_size"]/2, 2000000)
        selected = support_mask(samples, self.views, self.policy)
        values = np.concatenate((self.points, samples[selected]))
        result = readout(values, np.empty((0, 2, 3)), reader_config)
        result.update(full_input_point_count=self.input_point_count, supported_base_point_count=len(self.points),
            full_input_segment_count=len(segments), supported_curve_sample_count=int(selected.sum()),
            rgb_support_policy=self.policy, scope="Same RGB-supported photometric volume for full base and candidates; not semantic identity or whole-scene quality")
        return result
