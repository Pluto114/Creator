"""Separate axis eligibility from local width-qualified finite support.

Keep the stable four-view raw boundary. Inside it, use the original finite
method's three positive views after the unchanged apparent-width filter.
An accepted native association still needs at least four original fit views.
"""

import numpy as np

from . import rod_boundary_patch as previous
from . import rod_width_boundary_patch as width
from .anchored_boundary_readout import trim_outer_envelope
from .rgb_chain_support import consistent_chain_mask, point_memberships
from .rod_evidence import DEFAULT_CONFIG
from .rod_fixture_finite import canonical_hash

POLICY = dict(width.POLICY, method="baseline_width_finite", version="width-finite-patch-1",
    minimum_native_fit_views=DEFAULT_CONFIG["minimum_fit_views"],
    minimum_raw_boundary_views=previous.POLICY["minimum_boundary_views"],
    minimum_boundary_views=DEFAULT_CONFIG["minimum_positive_views"],
    quorum_source="unchanged_rod_evidence_finite_minimum_positive_views",
    stage_policy="original_eligible_native_axis_then_raw_four_view_cap_then_width_finite_support")


def build_width_context(frames, cameras, baseline):
    context = width.build_width_context(frames, cameras, baseline)
    assignment = context["association"]["assignments"][0]
    count = sum(bool(indices) for indices in assignment)
    if count < POLICY["minimum_native_fit_views"]:
        raise ValueError("At least four original native fit views required before finite support")
    association = dict(context["association"], width_four_view_context_sha256=context["sha256"],
        finite_stage_policy_sha256=canonical_hash(POLICY), original_native_fit_view_count=count)
    return dict(context, association=association, sha256=canonical_hash(association))


def finite_support_mask(points, context):
    view_ids = [view["view_id"] for view in context["views"]]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("Distinct cameras required for finite width support")
    return consistent_chain_mask(point_memberships(points, context),
        context["association"]["assignments"], POLICY["minimum_boundary_views"])


def refine_boundary(segments, shared_context, width_context):
    if width_context["association"]["fallback_to_raw_union"]:
        raise ValueError("Exact original selected width context required")
    parent = previous.refine_boundary(segments, shared_context)
    result = trim_outer_envelope(parent["segments"], width_context, support_mask=finite_support_mask)
    finite_removed = result["boundary_removed_length_m"]
    result.update(method=POLICY["method"], policy=dict(POLICY), method_sha256=canonical_hash(POLICY),
        state="accepted" if len(result["segments"]) else "rejected",
        rejection_reasons=[] if len(result["segments"]) else ["no_three_view_width_qualified_finite_support"],
        boundary_removed_length_m=parent["boundary_removed_length_m"]+finite_removed,
        width_boundary_removed_length_m=finite_removed,
        previous_boundary_removed_length_m=parent["boundary_removed_length_m"],
        previous_boundary_segments_sha256=canonical_hash(parent["segments"]),
        parent_native_sha256=parent["parent_native_sha256"], context_sha256=shared_context["sha256"],
        width_context_sha256=width_context["sha256"],
        width_diagnostics=width_context["association"]["width_diagnostics"],
        original_native_fit_view_count=width_context["association"]["original_native_fit_view_count"],
        gt_read=False, extra_annotations=0, native_axis_refitted=False, physical_radius_estimated=False,
        original_baseline_preserved_as_control=True, common_reader_changed=False)
    # This is an intersection, not a fallback that can resurrect the parent's
    # removed raw tail or fill a component gap when fewer views qualify.
    result["raw_parent_length_m"] = float(np.linalg.norm(
        parent["segments"][:, 1]-parent["segments"][:, 0], axis=1).sum())
    return result
