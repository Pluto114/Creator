"""Native outer limits from the baseline's exact selected RGB row assignments.

This additional patch arm intersects the previous foreground boundary with one
already selected native association. Unselected edge pairs never lend votes.
The baseline axis, existing interior components and common reader stay fixed.
"""

import numpy as np

from . import rod_boundary_patch as previous
from .consensus_boundary_readout import trim_consensus_envelope
from .rgb_candidate_readout import support_views
from .rod_candidate_extent import _index
from .rod_fixture_finite import canonical_hash

POLICY = dict(previous.POLICY, method="baseline_selected_boundary",
    version="selected-boundary-patch-1",
    input_context="previous_foreground_boundary_intersect_exact_native_selected_row_matches",
    row_policy="nearest_measured_integer_row_half_pixel_cell_no_interpolation",
    candidate_policy="one_original_native_association_no_alternative_edge_pair_votes")


def build_selected_context(frames, cameras, baseline):
    """Bind finite candidate_selection to the original selected pool row_matches."""
    views = support_views(frames, cameras)
    proposal, finite = baseline["proposal"], baseline["finite"]
    if (baseline["method"] != "baseline" or baseline["state"] != "accepted"
            or proposal["state"] != "accepted" or finite["state"] != "accepted"):
        raise ValueError("Original accepted native baseline association required")
    selected, selections = proposal["selected"], finite["candidate_selection"]
    if (len(selected["matches"]) != len(views) or len(selections) != len(views)
            or [s["view_id"] for s in selections] != [v["view_id"] for v in views]):
        raise ValueError("Native selection must follow the original camera order")
    supporting = [_index(i, len(views), "supporting view") for i in selected["supporting_views"]]
    if len(set(supporting)) != len(supporting):
        raise ValueError("Unique native supporting views required")
    members, assignment, counts = [], [], []
    for i, (frame, view, selection) in enumerate(zip(frames, views, selections)):
        if (canonical_hash(frame["observations"]) != frame["observation_sha256"]
                or canonical_hash(frame["pool"]) != frame["pool_sha256"]):
            raise ValueError("Unchanged original RGB observations and pool required")
        chosen = None
        expected = []
        if i in supporting:
            chosen = _index(selected["matches"][i]["candidate_index"],
                len(frame["pool"]["candidates"]), "native selected candidate")
            matches = frame["pool"]["candidates"][chosen]["row_matches"]
            expected = sorted([[m[0], m[1]] for m in matches])
        if selection["candidate_index"] != chosen or selection["row_matches"] != expected:
            raise ValueError("Finite row assignments differ from the native selected hypothesis")
        mask = np.zeros((*view["left"].shape, 1), bool)
        seen = set()
        rows = frame["observations"]["rows"]
        for row_id, candidate_id in expected:
            row_id = _index(row_id, len(rows), "selected raw row")
            row = rows[row_id]
            candidate_id = _index(candidate_id, len(row["candidates"]), "selected raw edge pair")
            if row_id in seen or row["status"] not in ("observed", "ambiguous"):
                raise ValueError("Unique actually measured selected rows required")
            seen.add(row_id)
            mask[row["y"], candidate_id, 0] = True
        # Remove irrelevant event boundaries as well as their votes; the raw
        # frame and caller's shared foreground context remain untouched.
        view["left"][~mask[..., 0]] = np.nan
        view["right"][~mask[..., 0]] = np.nan
        members.append(mask)
        assignment.append((0,) if chosen is not None else ())
        counts.append(dict(view_id=view["view_id"], candidate_index=chosen,
                           selected_row_count=len(seen)))
    association = dict(assignments=[assignment], chain_count=1,
        fallback_to_raw_union=False, search_complete=True,
        scope="single_original_native_selected_association_not_new_search",
        native_selection_sha256=canonical_hash(selections),
        evidence_sha256=canonical_hash(frames), cameras_sha256=canonical_hash(cameras),
        baseline_sha256=canonical_hash(baseline), selected_rows=counts)
    return dict(views=views, row_members=members, association=association,
                sha256=canonical_hash(association))


def refine_boundary(segments, shared_context, selected_context):
    """Intersect two measured envelopes; never change the native axis or gaps."""
    parent = previous.refine_boundary(segments, shared_context)
    if selected_context["association"]["fallback_to_raw_union"]:
        raise ValueError("Exact native selected context required")
    result = trim_consensus_envelope(parent["segments"], selected_context)
    removed = parent["boundary_removed_length_m"] + result["boundary_removed_length_m"]
    return dict(result, state="accepted" if len(result["segments"]) else "rejected",
        rejection_reasons=[] if len(result["segments"]) else ["no_four_view_selected_native_boundary_support"],
        method=POLICY["method"], policy=dict(POLICY), method_sha256=canonical_hash(POLICY),
        boundary_removed_length_m=removed,
        previous_boundary_removed_length_m=parent["boundary_removed_length_m"],
        selected_boundary_removed_length_m=result["boundary_removed_length_m"],
        parent_native_sha256=parent["parent_native_sha256"],
        previous_boundary_segments_sha256=canonical_hash(parent["segments"]),
        context_sha256=shared_context["sha256"], selected_context_sha256=selected_context["sha256"],
        native_selected_rows=selected_context["association"]["selected_rows"],
        gt_read=False, extra_annotations=0, native_axis_refitted=False,
        original_baseline_preserved_as_control=True, common_reader_changed=False)
