"""Require chain-level distinct-view consensus only at the outer envelope.

An individual marked view's unknown tail is not a hard boundary. Reuse the
existing four-view chain eligibility count and measured, unpadded bands.
Interior geometry, actual axis, ranking and three-view input admission stay
unchanged. This positive support bound is not a physical endpoint detector.
"""

from . import bracketed_occupancy_readout as previous
from .anchored_boundary_readout import selected_context, trim_outer_envelope
from .rgb_chain_support import DEFAULTS, consistent_chain_mask, point_memberships


def boundary_support_mask(points, context):
    names = [view["view_id"] for view in context["views"]]
    if len(names) != len(set(names)):
        raise ValueError("Distinct view IDs required for boundary consensus")
    return consistent_chain_mask(point_memberships(points, context),
        context["association"]["assignments"], DEFAULTS["minimum_chain_views"])


def trim_consensus_envelope(segments, context):
    return trim_outer_envelope(segments, context, support_mask=boundary_support_mask)


def readout(values, voxel_size, origin, context):
    result = previous.readout(values, voxel_size, origin, context)
    result.update(output_support_policy="bracketed_interior_with_unpadded_chain_consensus_outer_limits",
        rgb_outer_endpoints_preserved=False, boundary_requires_all_marked_views=False,
        boundary_minimum_views=DEFAULTS["minimum_chain_views"],
        boundary_horizontal_padding_px=0., boundary_only_intervention=True,
        boundary_removed_length_m=0., boundary_positive_cells=0, boundary_cell_count=0)
    if result["state"] != "complete" or result["selected_candidate"] is None or not len(result["segments"]):
        return result
    result.update(trim_consensus_envelope(result["segments"], selected_context(values, result, context)))
    result["components"] = len(result["segments"])
    if not result["components"]:
        result["reason"] = "no_unpadded_chain_consensus_boundary_support"
    return result
