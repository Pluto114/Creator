"""Fixed actual seeds, same-chain expansion, then finite-support competition.

This is a resolution-dependent greedy local readout, not foreground identity,
an exhaustive axis search, or a proof that unresolved close rods do not exist.
All candidate and residual fits remain in the audit, including discarded ones.
"""

import numpy as np

from . import foreground_support as support
from . import rgb_segment_cells as cells
from . import single_axis_readout as axis
from .chain_partition_readout import DEFAULTS as PARTITION_LIMITS
from .chain_partition_readout import support_groups
from .chain_segment_union import merge_collinear_segments
from .rgb_chain_support import chain_support_mask
from .rod_fixture_finite import canonical_hash

DEFAULTS = dict(maximum_points=2000000, maximum_output_segments=5000,
    maximum_distance_checks=200000000)
SEED_GROUP_LIMITS = {k: PARTITION_LIMITS[k] for k in
    ("maximum_assignments", "maximum_fit_groups", "maximum_membership_checks")}
SCOPE = "Greedy actual-support consensus conditional on fixed sparse strokes; not unique identity or general topology"


def readout(values, voxel_size, origin, context):
    values, origin = np.asarray(values, float), np.asarray(origin, float)
    if (values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all()
            or origin.shape != (3,) or not np.isfinite(origin).all()
            or isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Finite actual Nx3 coordinates and frozen voxel policy required")
    unique = np.unique(values, axis=0)
    result = dict(state="complete", resolution_state="unresolved", reason=None,
        segments=np.empty((0, 2, 3)), components=0, input_count=len(values), unique_count=len(unique),
        scope=SCOPE, defaults=dict(DEFAULTS), support_defaults=dict(support.DEFAULTS),
        seed_group_limits=dict(SEED_GROUP_LIMITS),
        axis_defaults=dict(axis.DEFAULTS), voxel_size=float(voxel_size), origin=origin.tolist(),
        readouts=[], retained_groups=[], target_identity_confirmed=False, axis_search_exhaustive=False,
        global_union_fit_performed=False, iterative_seed_expansion=False,
        candidate_ranking="occupied_actual_support_voxels_then_actual_fit_rmse_then_geometry_then_support_hash",
        competition_policy="discard_points_explained_by_retained_finite_geometry_then_refit_remaining_actual_support",
        unresolved_competition_radius_m=support.DEFAULTS["explained_radius_voxels"]*voxel_size,
        close_or_crossing_structure_may_be_unresolved=True)
    if context["association"]["fallback_to_raw_union"]:
        return {**result, "state": "unmeasurable", "reason": "incomplete_chain_search_no_consensus_claim"}
    if len(unique) > DEFAULTS["maximum_points"]:
        return {**result, "state": "unmeasurable", "reason": "consensus_point_budget_exceeded"}
    work = 0

    def novel(points, segments, radius):
        nonlocal work
        work += len(points)*len(segments)
        if work > DEFAULTS["maximum_distance_checks"]:
            raise OverflowError("consensus_distance_budget_exceeded")
        return support.novel_support_mask(points, segments, radius)

    radius = support.DEFAULTS["explained_radius_voxels"]*voxel_size
    # Each emitted interval is sampled at <= h. A distance margin h/2 makes
    # every point of a later segment farther than radius from earlier ones.
    separation = radius+axis.DEFAULTS["axis_sampling_step_voxels"]*voxel_size/2
    candidates = []
    retained_segments = np.empty((0, 2, 3))
    try:
        groups = support_groups(unique, context)
        for group in groups:
            association = dict(context["association"], assignments=group["assignments"],
                chain_count=len(group["assignments"]))
            local = {**context, "association": association}
            local_indices = np.flatnonzero(chain_support_mask(unique, local))
            seed_indices = np.searchsorted(local_indices, group["indices"])
            if not np.array_equal(local_indices[seed_indices], group["indices"]):
                raise ValueError("Required seed must belong to its same-chain stroke pool")
            expansion = support.fixed_seed_expansion(unique[local_indices], seed_indices, voxel_size, origin)
            expanded = local_indices[expansion["indices"]]
            row = dict(support_set_sha256=group["support_set_sha256"],
                seed_point_count=len(seed_indices), assignment_count=len(group["assignments"]),
                assignments_sha256=canonical_hash(group["assignments"]),
                same_chain_point_count=len(local_indices), expanded_point_count=len(expanded),
                expanded_indices_sha256=canonical_hash(expanded),
                expansion={k: v for k, v in expansion.items() if k != "indices"},
                candidate_graph=None, final_graph=None, ranking_supported_voxels=0,
                novel_point_count=0, novel_indices_sha256=None, retained=False)
            result["readouts"].append(row)
            if expansion["state"] != "complete":
                raise OverflowError("incomplete_consensus_seed")
            if expansion["reason"] is not None:
                candidates.append((expanded, local))
                continue
            graph = axis.readout(unique[expanded], voxel_size, origin,
                lambda q, current=local: chain_support_mask(q, current))
            row["candidate_graph"] = graph
            if graph["state"] != "complete":
                raise OverflowError("incomplete_consensus_candidate")
            if len(graph["segments"]):
                explained = ~novel(unique[expanded], graph["segments"], radius)
                _, representatives = axis.voxel_representatives(unique[expanded[explained]], voxel_size, origin)
                row["ranking_supported_voxels"] = len(representatives)
            candidates.append((expanded, local))

        def ranking(index):
            row = result["readouts"][index]
            graph = row["candidate_graph"]
            return (-row["ranking_supported_voxels"], graph["model"]["perpendicular_rmse"],
                tuple(np.asarray(graph["segments"]).ravel()), row["support_set_sha256"])

        eligible = [i for i, row in enumerate(result["readouts"])
            if row["candidate_graph"] is not None and len(row["candidate_graph"]["segments"])]
        order = sorted(eligible, key=ranking)
        result["competition_order"] = order
        for index in order:
            row = result["readouts"][index]
            expanded, local = candidates[index]
            remaining = expanded[novel(unique[expanded], retained_segments, radius)]
            row.update(novel_point_count=len(remaining), novel_indices_sha256=canonical_hash(remaining))
            def positive(q, current=local):
                return chain_support_mask(q, current) & novel(q, retained_segments, separation)
            graph = axis.readout(unique[remaining], voxel_size, origin, positive)
            row["final_graph"] = graph
            if graph["state"] != "complete":
                raise OverflowError("incomplete_consensus_residual")
            row["sampled_graph"] = graph
            clipped = cells.clip_segments(graph["segments"], local,
                axis.DEFAULTS["minimum_length_voxels"]*voxel_size)
            row["rgb_cell_clip"] = clipped
            graph = {**graph, "segments": clipped["segments"], "components": len(clipped["segments"]),
                "continuous_rgb_clipping_applied": True,
                "resolution_state": "resolved" if len(clipped["segments"]) else "unresolved",
                "reason": None if len(clipped["segments"]) else "no_continuously_rgb_supported_finite_run"}
            row["final_graph"] = graph
            if len(graph["segments"]):
                row["retained"] = True
                result["retained_groups"].append(index)
                retained_segments = np.concatenate((retained_segments, graph["segments"]))
                if len(retained_segments) > DEFAULTS["maximum_output_segments"]:
                    raise OverflowError("consensus_output_budget_exceeded")
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc), "distance_checks": work}
    merged = merge_collinear_segments(retained_segments)
    result.update(segments=merged, components=len(merged), fitted_support_groups=len(groups),
        distance_checks=work, continuous_cross_group_output_separation_m=radius,
        separation_scope="between_distinct_retained_groups_not_within_one_group",
        resolution_state="greedy_resolution_consensus" if len(merged) else "unresolved",
        reason=None if len(merged) else "no_independently_supported_finite_run")
    return result
