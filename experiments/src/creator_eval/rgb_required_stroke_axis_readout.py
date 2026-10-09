"""Same TLS with mandatory foreground-view observations inside each chain."""

import numpy as np

from . import single_axis_readout as reader
from .common_readout import sample_segments
from .rgb_candidate_readout import POLICY, support_votes
from .rgb_required_view_support import required_stroke_support_mask as chain_support_mask
from .rgb_single_axis_readout import MAXIMUM_CURVE_SAMPLES

SCOPE = "Sparse-stroke positive support AND every annotated view in the same chain; actual 3D TLS, not identity or full-scene quality"


class RequiredStrokeAxisReadout:
    def __init__(self, points, context, policy=None):
        if policy is not None and policy != POLICY:
            raise ValueError("Original raw RGB support policy required")
        self.policy, self.context, self.views = dict(POLICY), context, context["views"]
        self.raw_votes = support_votes(points, self.views, POLICY)
        self.votes = self.raw_votes.copy()
        self.mask = chain_support_mask(points, context, self.raw_votes)
        self.points = np.asarray(points, float)[self.mask].copy()
        self.input_point_count = len(points)

    def __call__(self, segments, call_policy):
        if set(call_policy) != {"voxel_size", "origin"}:
            raise ValueError("Only frozen paired voxel size and origin may enter readout")
        size, origin = call_policy["voxel_size"], call_policy["origin"]
        if isinstance(size, (bool, np.bool_)) or not np.isfinite(size) or size <= 0:
            raise ValueError("Positive finite fixed voxel size required")
        segments = np.asarray(segments, float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
            raise ValueError("Finite Mx2x3 segments required")
        samples = sample_segments(segments, size/2, MAXIMUM_CURVE_SAMPLES)
        votes = support_votes(samples, self.views, POLICY)
        selected = chain_support_mask(samples, self.context, votes)
        values = np.concatenate((self.points, samples[selected]))
        result = reader.readout(values, size, origin, lambda p: chain_support_mask(p, self.context))
        histogram = np.bincount(self.raw_votes, minlength=len(self.views)+1).tolist()
        curve_histogram = np.bincount(votes, minlength=len(self.views)+1).tolist()
        result.update(full_input_point_count=self.input_point_count,
            supported_base_point_count=len(self.points), full_input_segment_count=len(segments),
            full_curve_sample_count=len(samples), supported_curve_sample_count=int(selected.sum()),
            rgb_support_policy=dict(POLICY), base_vote_histogram=histogram, curve_vote_histogram=curve_histogram,
            raw_base_vote_histogram=histogram, raw_curve_vote_histogram=curve_histogram,
            base_chain_mask_subset_raw=bool(np.all(~self.mask | (self.raw_votes >= POLICY["minimum_views"]))),
            curve_chain_mask_subset_raw=bool(np.all(~selected | (votes >= POLICY["minimum_views"]))),
            chain_context_sha256=self.context["sha256"],
            chain_association={k: v for k, v in self.context["association"].items() if k != "assignments"},
            target_identity_confirmed=False, input_support_intervention=True, reader_algorithm_unchanged=True,
            call_policy=call_policy, effective_policy={**reader.DEFAULTS, **call_policy}, measurement_scope=SCOPE)
        result.update(support_domain_kind="stroke_required_views_positive_only_not_all_possible",
            anchor_claims_sha256=self.context["association"]["anchors_sha256"],
            anchor_view_count=self.context["association"]["anchor_view_count"],
            anchor_point_count=self.context["association"]["anchor_point_count"],
            stroke_interpolation_performed=False,
            required_support_view_ids=sorted({a["view_id"] for a in self.context["association"]["anchors"]}),
            required_support_views_applied=not self.context["association"]["fallback_to_raw_union"],
            required_support_rule="same_assignment_all_annotated_views_and_minimum_three")
        return result
