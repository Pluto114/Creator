"""Unchanged RGB union; original guides supply only a shared block direction."""

import numpy as np

from . import context_block_axis_readout as reader
from .common_readout import sample_segments
from .rgb_candidate_readout import POLICY, CandidateSupportedReadout, support_views, support_votes

SCOPE = reader.SCOPE


def guided_support_views(evidence, cameras):
    views = support_views(evidence, cameras)
    for view, frame in zip(views, evidence):
        view["guide_xyxy"] = np.asarray(frame["guide_xyxy"], float).copy()
        view["guide_source"] = frame["guide_source"]
    return views


class ContextBlockAxisReadout(CandidateSupportedReadout):
    def __init__(self, points, views, policy=None):
        if policy is not None and policy != POLICY:
            raise ValueError("Keep the original RGB possible-support volume unchanged")
        super().__init__(points, views, POLICY)
        self.guide = reader.guide_direction(views)

    def __call__(self, segments, call_policy):
        if set(call_policy) != {"voxel_size", "origin"}:
            raise ValueError("Only frozen paired voxel size and origin may enter readout")
        size = call_policy["voxel_size"]
        origin = np.asarray(call_policy["origin"], float)
        if (isinstance(size, (bool, np.bool_)) or not np.isfinite(size) or size <= 0
                or origin.shape != (3,) or not np.isfinite(origin).all()):
            raise ValueError("Positive finite scale and finite fixed origin required")
        segments = np.asarray(segments, float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
            raise ValueError("Finite Mx2x3 segments required")
        samples = sample_segments(segments, size/2, 2000000)
        votes = support_votes(samples, self.views, self.policy)
        selected = votes >= self.policy["minimum_views"]
        values = np.concatenate((self.points, samples[selected]))
        try:
            if self.guide["state"] != "resolved":
                result = dict(state="complete", resolution_state="unresolved", reason=self.guide["reason"],
                              segments=np.empty((0, 2, 3)), components=0)
            else:
                result = reader.readout(values, size, origin,
                    lambda points: support_votes(points, self.views, self.policy) >= self.policy["minimum_views"],
                    self.guide["direction"])
        except Exception as exc:
            result = dict(state="error", resolution_state="error", reason=type(exc).__name__+": "+str(exc),
                          segments=np.empty((0, 2, 3)), components=0)
        result.update(full_input_point_count=self.input_point_count,
            supported_base_point_count=len(self.points), full_input_segment_count=len(segments),
            full_curve_sample_count=len(samples), supported_curve_sample_count=int(selected.sum()),
            rgb_support_policy=dict(self.policy), guide_context=self.guide,
            base_vote_histogram=np.bincount(self.votes, minlength=len(self.views)+1).tolist(),
            curve_vote_histogram=np.bincount(votes, minlength=len(self.views)+1).tolist(),
            call_policy=call_policy, effective_policy={**reader.DEFAULTS, **call_policy}, measurement_scope=SCOPE)
        return result
