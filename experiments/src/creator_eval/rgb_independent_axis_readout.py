"""Same RGB volume and sample counts, with raw-point support preservation."""

import numpy as np

from . import independent_axis_readout as reader
from .common_readout import sample_segments
from .rgb_candidate_readout import POLICY, CandidateSupportedReadout, support_votes

SCOPE = reader.SCOPE


class IndependentAxisReadout(CandidateSupportedReadout):
    def __init__(self, points, views, policy=None):
        if policy is not None and policy != POLICY:
            raise ValueError("Keep the original RGB possible-support volume unchanged")
        super().__init__(points, views, POLICY)

    def __call__(self, segments, call_policy):
        if set(call_policy) != {"voxel_size", "origin"}:
            raise ValueError("Only frozen paired voxel size and origin may enter readout")
        size, origin = call_policy["voxel_size"], call_policy["origin"]
        if isinstance(size, (bool, np.bool_)) or not np.isfinite(size) or size <= 0:
            raise ValueError("Positive finite fixed voxel size required")
        segments = np.asarray(segments, float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
            raise ValueError("Finite Mx2x3 segments required")
        samples = sample_segments(segments, size/2, 2000000)
        votes = support_votes(samples, self.views, self.policy)
        selected = votes >= self.policy["minimum_views"]
        values = np.concatenate((self.points, samples[selected]))
        try:
            result = reader.readout(values, size, origin,
                lambda points: support_votes(points, self.views, self.policy) >= self.policy["minimum_views"])
        except Exception as exc:
            result = dict(state="error", resolution_state="error", reason=type(exc).__name__+": "+str(exc),
                segments=np.empty((0, 2, 3)), components=0)
        result.update(full_input_point_count=self.input_point_count,
            supported_base_point_count=len(self.points), full_input_segment_count=len(segments),
            full_curve_sample_count=len(samples), supported_curve_sample_count=int(selected.sum()),
            rgb_support_policy=dict(self.policy),
            base_vote_histogram=np.bincount(self.votes, minlength=len(self.views)+1).tolist(),
            curve_vote_histogram=np.bincount(votes, minlength=len(self.views)+1).tolist(),
            call_policy=call_policy, effective_policy={**reader.DEFAULTS, **call_policy}, measurement_scope=SCOPE)
        return result
