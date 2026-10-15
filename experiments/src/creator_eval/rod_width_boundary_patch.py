"""Reject over-wide native RGB pairs as terminal support, without moving axes.

The reference is each view's own measured central-half median apparent width.
A one-pixel allowance for each of the two width measurements leaves two pixels
of total slack. This is an image support heuristic, not a resolved radius,
visibility proof, or a guarantee that every width change is a junction.
"""

import numpy as np

from . import rod_selected_boundary_patch as selected
from .rod_fixture_finite import canonical_hash

POLICY = dict(selected.POLICY, method="baseline_width_boundary", version="width-boundary-patch-1",
    input_context="native_selected_pairs_with_per_view_body_width_continuity",
    width_reference="median_selected_pair_width_in_central_half_of_original_native_image_y_span",
    central_fraction=[.25, .75], minimum_reference_rows=3,
    observed_width_allowance_px=1., reference_width_allowance_px=1.,
    width_gate="one_sided_apparent_width_excess_only_no_physical_radius_claim",
    unavailable_reference="retain_original_selected_votes_no_added_width_filter",
    rejected_width_semantics="unknown_not_negative_no_interior_split")


def build_width_context(frames, cameras, baseline):
    context = selected.build_selected_context(frames, cameras, baseline)
    parent_sha = context["sha256"]
    segments = np.asarray(baseline["segments"], float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("Original finite native geometry required for width reference")
    diagnostics = []
    points = segments.reshape(-1, 3)
    slack = POLICY["observed_width_allowance_px"]+POLICY["reference_width_allowance_px"]
    for view, members in zip(context["views"], context["row_members"]):
        e, k = view["world_to_camera_cv"], view["K_index"]
        camera = points @ e[:, :3].T+e[:, 3]
        pixels = camera @ k.T
        valid = np.isfinite(pixels).all(axis=1) & (pixels[:, 2] > 1e-10)
        diagnostic = dict(view_id=view["view_id"], reference_state="unavailable",
            reference_row_count=0, reference_width_px=None, maximum_width_px=None,
            selected_row_count=int(members.sum()), retained_row_count=int(members.sum()),
            removed_rows=[])
        if not len(points) or not valid.all():
            diagnostics.append(dict(diagnostic, reason="native_extent_not_projectable"))
            continue
        y = pixels[:, 1]/pixels[:, 2]
        low, high = float(y.min()), float(y.max())
        limits = low+np.array(POLICY["central_fraction"])*(high-low)
        rows, slots = np.nonzero(members[..., 0])
        widths = view["right"][rows, slots]-view["left"][rows, slots]
        reference = (rows >= limits[0]) & (rows <= limits[1])
        diagnostic.update(projected_y_span=[low, high], reference_y_span=limits,
            reference_row_count=int(np.count_nonzero(reference)))
        if np.count_nonzero(reference) < POLICY["minimum_reference_rows"]:
            diagnostics.append(dict(diagnostic, reason="too_few_selected_body_rows"))
            continue
        median = float(np.median(widths[reference]))
        maximum = median+slack
        reject = widths > maximum
        removed = [dict(row=int(row), raw_candidate_index=int(slot), width_px=float(width))
                   for row, slot, width in zip(rows[reject], slots[reject], widths[reject])]
        members[rows[reject], slots[reject], 0] = False
        view["left"][rows[reject], slots[reject]] = np.nan
        view["right"][rows[reject], slots[reject]] = np.nan
        diagnostics.append(dict(diagnostic, reference_state="measured", reason=None,
            reference_width_px=median, maximum_width_px=maximum,
            retained_row_count=int(members.sum()), removed_rows=removed))
    association = dict(context["association"], width_parent_context_sha256=parent_sha,
        width_policy_sha256=canonical_hash(POLICY), width_diagnostics=diagnostics)
    return dict(context, association=association, sha256=canonical_hash(association))


def refine_boundary(segments, shared_context, width_context):
    result = selected.refine_boundary(segments, shared_context, width_context)
    result.update(method=POLICY["method"], policy=dict(POLICY), method_sha256=canonical_hash(POLICY),
        width_context_sha256=width_context["sha256"],
        width_diagnostics=width_context["association"]["width_diagnostics"],
        width_boundary_removed_length_m=result["selected_boundary_removed_length_m"],
        physical_radius_estimated=False)
    if result["state"] == "rejected":
        result["rejection_reasons"] = ["no_four_view_width_consistent_native_boundary_support"]
    return result
