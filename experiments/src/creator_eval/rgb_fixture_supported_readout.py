"""New shared reader inside the unchanged frozen RGB possible-support volume."""

from __future__ import annotations

import numpy as np

from . import common_readout_supported as reader
from .common_readout import sample_segments
from .rgb_candidate_readout import POLICY, CandidateSupportedReadout, support_votes

SCOPE = ("Complete immutable base/candidate geometry is loaded, then the same frozen raw RGB "
         "candidate-union measurement mask is applied. Local finite-axis readout only; "
         "not target identity, whole-scene quality improvement, independent qualification or G1.")


class FixtureSupportedReadout(CandidateSupportedReadout):
    """Reuse exactly the old cached base mask; replace only the geometry reader."""

    def __init__(self, points, views, policy=None):
        if policy is not None and policy != POLICY:
            raise ValueError("The original measurement-volume policy is immutable")
        super().__init__(points, views, POLICY)

    def __call__(self, segments, call_policy):
        if set(call_policy) != {"voxel_size", "origin"}:
            raise ValueError("Only the frozen voxel size and origin may reach the new reader")
        effective = reader.policy(call_policy)
        segments = np.asarray(segments, float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
            raise ValueError("Finite Mx2x3 segments required")
        samples = sample_segments(segments, effective["voxel_size"]/2, effective["maximum_curve_samples"])
        votes = support_votes(samples, self.views, self.policy)
        selected = votes >= self.policy["minimum_views"]
        values = np.concatenate((self.points, samples[selected]))
        try:
            result = reader.readout(values, np.empty((0, 2, 3)), call_policy)
        except Exception as exc:
            # A failed reader is retained as a row, never relabelled a rejection.
            result = dict(state="error", resolution_state="error", segments=[], components=0,
                          reason=type(exc).__name__ + ": " + str(exc))
        result.update(full_input_point_count=self.input_point_count,
            supported_base_point_count=len(self.points), full_input_segment_count=len(segments),
            full_curve_sample_count=len(samples), supported_curve_sample_count=int(selected.sum()),
            rgb_support_policy=dict(self.policy),
            base_vote_histogram=np.bincount(self.votes, minlength=len(self.views)+1).tolist(),
            curve_vote_histogram=np.bincount(votes, minlength=len(self.views)+1).tolist(),
            call_policy=call_policy, effective_policy=effective, measurement_scope=SCOPE)
        return result
