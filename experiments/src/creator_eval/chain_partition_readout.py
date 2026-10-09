"""Fit each exact same-chain support set separately, retaining alternatives.

This deliberately exposes competing geometries rather than averaging them or
choosing a native/GT-favoured winner. Resolution aliases retain their complete
audit graphs; every non-aliased output remains in the common score.
"""

from hashlib import sha256

import numpy as np

from . import single_axis_readout as axis
from .chain_segment_union import merge_collinear_segments
from .common_readout import sample_segments
from .line_controls import point_to_segments_distance
from .rgb_candidate_readout import POLICY
from .rgb_chain_support import point_memberships
from .rgb_required_view_support import consistent_required_view_mask, required_stroke_support_mask
from .rod_fixture_finite import canonical_hash

DEFAULTS = dict(maximum_assignments=4096, maximum_fit_groups=256,
    maximum_membership_checks=200000000, maximum_output_segments=5000,
    minimum_shared_support_fraction=.5, alias_distance_voxels=1.,
    alias_sampling_step_voxels=.5, maximum_alias_distance_checks=200000000)
SCOPE = "Distinct same-chain actual-geometry TLS hypotheses, not unique foreground identity or general topology"


def support_groups(values, context):
    """Exact point-set deduplication; no geometric merging or arm labels."""
    assignments = context["association"]["assignments"]
    if len(assignments) > DEFAULTS["maximum_assignments"]:
        raise OverflowError("partition_assignment_budget_exceeded")
    if len(values)*len(assignments) > DEFAULTS["maximum_membership_checks"]:
        raise OverflowError("partition_membership_budget_exceeded")
    memberships = point_memberships(values, context)
    names = {a["view_id"] for a in context["association"]["anchors"]}
    required = [i for i, view in enumerate(context["views"]) if view["view_id"] in names]
    if len(required) < 2 or len(required) != len(names):
        raise ValueError("Validated independent foreground views required")
    groups = {}
    for assignment in assignments:
        mask = consistent_required_view_mask(memberships, [assignment], required, POLICY["minimum_views"])
        signature = np.packbits(mask, bitorder="little").tobytes()
        if signature not in groups:
            if len(groups) >= DEFAULTS["maximum_fit_groups"]:
                raise OverflowError("partition_fit_group_budget_exceeded")
            groups[signature] = dict(indices=np.flatnonzero(mask), assignments=set(),
                support_set_sha256=sha256(signature).hexdigest())
        groups[signature]["assignments"].add(tuple(tuple(row) for row in assignment))
    result = []
    for signature in sorted(groups):
        group = groups[signature]
        group["assignments"] = sorted(group["assignments"])
        result.append(group)
    return result


def select_readouts(readouts, groups, voxel_size):
    """Resolution-scale alias suppression, not identity or exact equivalence.

Only a retained hypothesis may suppress another. Distinct or extending finite
geometry stays visible. No transitive clustering, averaging, or segment fill.
"""
    def ranking(index):
        segments = np.asarray(readouts[index]["graph"]["segments"], float)
        length = float(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1).sum())
        return (-length, tuple(segments.ravel()), readouts[index]["support_set_sha256"])

    eligible = [i for i, row in enumerate(readouts) if len(row["graph"]["segments"])]
    kept, aliases, checked = [], [], 0
    for index in sorted(eligible, key=ranking):
        points = sample_segments(np.asarray(readouts[index]["graph"]["segments"], float),
            DEFAULTS["alias_sampling_step_voxels"]*voxel_size, axis.DEFAULTS["maximum_axis_samples"])
        own = groups[index]["indices"]
        for other in kept:
            peer = groups[other]["indices"]
            shared = len(np.intersect1d(own, peer, assume_unique=True))/min(len(own), len(peer))
            if shared < DEFAULTS["minimum_shared_support_fraction"]:
                continue
            segments = np.asarray(readouts[other]["graph"]["segments"], float)
            checked += len(points)*len(segments)
            if checked > DEFAULTS["maximum_alias_distance_checks"]:
                raise OverflowError("partition_alias_work_budget_exceeded")
            distance = point_to_segments_distance(points, segments)
            maximum = float(distance.max())
            # Distance to a finite-segment union is 1-Lipschitz. Every point on
            # the candidate is within h/2 of an endpoint-inclusive sample, so
            # this margin certifies continuous coverage, including old gaps.
            bound = maximum + DEFAULTS["alias_sampling_step_voxels"]*voxel_size/2
            if bound <= DEFAULTS["alias_distance_voxels"]*voxel_size:
                aliases.append(dict(group=index, retained_group=other,
                    shared_fraction_of_smaller_set=shared, maximum_sample_distance_m=maximum,
                    continuous_distance_upper_bound_m=bound))
                break
        else:
            kept.append(index)
    return kept, aliases, checked


def readout(values, voxel_size, origin, context):
    values, origin = np.asarray(values, float), np.asarray(origin, float)
    if (values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all()
            or origin.shape != (3,) or not np.isfinite(origin).all()
            or isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Finite actual Nx3 coordinates and frozen voxel policy required")
    unique = np.unique(values, axis=0)
    result = dict(state="complete", resolution_state="unresolved", reason=None,
        segments=np.empty((0, 2, 3)), components=0, input_count=len(values), unique_count=len(unique),
        scope=SCOPE, defaults=dict(DEFAULTS), axis_defaults=dict(axis.DEFAULTS),
        voxel_size=float(voxel_size), origin=origin.tolist(), readouts=[],
        global_union_fit_performed=False, target_identity_confirmed=False,
        partition_algorithm="same_chain_required_views_actual_geometry_tls",
        geometry_merge_kind="machine_collinear_interval_union_no_averaging",
        resolution_distinct_geometries_retained=True, axis_search_exhaustive=False,
        alias_policy="shared_actual_points_and_lipschitz_certified_continuous_finite_geometry_containment_not_identity_or_exact_equivalence")
    if context["association"]["fallback_to_raw_union"]:
        return {**result, "state": "unmeasurable", "reason": "incomplete_chain_search_no_partition_claim"}
    if len(unique) > axis.DEFAULTS["maximum_axis_samples"]:
        return {**result, "state": "unmeasurable", "reason": "partition_point_budget_exceeded"}
    try:
        groups = support_groups(unique, context)
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    segments = []
    for group in groups:
        association = dict(context["association"], assignments=group["assignments"], chain_count=len(group["assignments"]))
        local = {**context, "association": association}
        graph = axis.readout(unique[group["indices"]], voxel_size, origin,
            lambda points: required_stroke_support_mask(points, local))
        result["readouts"].append(dict(support_set_sha256=group["support_set_sha256"],
            support_point_count=len(group["indices"]), assignment_count=len(group["assignments"]),
            assignments_sha256=canonical_hash(group["assignments"]), graph=graph))
        segments.extend(graph["segments"])
        if len(segments) > DEFAULTS["maximum_output_segments"]:
            return {**result, "state": "unmeasurable", "reason": "partition_output_budget_exceeded"}
    # A budget-limited group is not evidence that a competitor does not exist.
    if any(row["graph"]["state"] != "complete" for row in result["readouts"]):
        return {**result, "state": "unmeasurable", "reason": "incomplete_partition_geometry"}
    try:
        retained, aliases, checked = select_readouts(result["readouts"], groups, voxel_size)
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    retained_segments = [segment for index in retained for segment in result["readouts"][index]["graph"]["segments"]]
    merged = merge_collinear_segments(retained_segments)
    result.update(segments=merged, components=len(merged), fitted_support_groups=len(groups),
        positive_groups=sum(len(row["graph"]["segments"]) > 0 for row in result["readouts"]),
        raw_group_segment_count=len(segments), retained_groups=retained, suppressed_aliases=aliases,
        alias_point_segment_checks=checked,
        resolution_state="hypotheses_retained" if len(merged) else "unresolved",
        reason=None if len(merged) else "no_partition_supported_finite_run")
    return result
