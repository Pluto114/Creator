"""Point-local unambiguous RGB support, not whole-row uniqueness or identity.

Distinct candidate intervals can coexist on a row. A projected point gets one
view vote only when exactly one interval contains it after the fixed padding.
"""

from __future__ import annotations

import numpy as np

from .common_readout import sample_segments
from .common_readout_abstention import readout
from .rgb_candidate_readout import POLICY
from .rgb_candidate_readout import support_views as union_views


def support_views(evidence, cameras):
    views = union_views(evidence, cameras)
    for view in views:
        # Exact duplicate representations do not introduce a new hypothesis.
        for y in range(len(view["left"])):
            valid = np.isfinite(view["left"][y])
            pairs = sorted(set(zip(view["left"][y, valid], view["right"][y, valid])))
            view["left"][y] = np.nan
            view["right"][y] = np.nan
            for i, (a, b) in enumerate(pairs):
                view["left"][y, i], view["right"][y, i] = a, b
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
        hits = np.zeros(len(indices), np.int32)
        for i in range(view["left"].shape[1]):
            left, right = view["left"][y, i], view["right"][y, i]
            hits += np.isfinite(left) & (xy[indices, 0] >= left-pad) & (xy[indices, 0] <= right+pad)
        votes[indices[hits == 1]] += 1
    return votes


def support_mask(points, views, policy=None):
    p = POLICY if policy is None else policy
    return support_votes(points, views, p) >= p["minimum_views"]


class LocalUniqueReadout:
    """Same locally unique photometric volume for every complete candidate."""

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
        result = readout(np.concatenate((self.points, samples[selected])), np.empty((0, 2, 3)), reader_config)
        result.update(full_input_point_count=self.input_point_count, supported_base_point_count=len(self.points),
            full_input_segment_count=len(segments), full_curve_sample_count=len(samples),
            supported_curve_sample_count=int(selected.sum()), rgb_support_policy=self.policy,
            base_vote_histogram=np.bincount(self.votes, minlength=len(self.views)+1).tolist(),
            curve_vote_histogram=np.bincount(votes, minlength=len(self.views)+1).tolist(),
            scope="Same point-local unique raw RGB interval volume for all full inputs; not foreground identity or reader qualification")
        return result
