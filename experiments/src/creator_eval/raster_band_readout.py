"""Rank actual support within a fixed one-pixel measured-band tolerance.

Integer RGB edges do not justify lexicographic subpixel preference for a short
axis over a longer measured one. This adds no image-fitted geometry: it selects
among the original actual-point TLS hypotheses, using their own exact RGB
clipping and unchanged bracketed-hole rule. No admissible fit means fallback.
"""

import numpy as np

from . import bracketed_occupancy_readout as previous
from . import rgb_segment_cells as cells
from .chain_partition_readout import support_groups
from .occupied_axis_readout import NUMERICAL_LENGTH_FLOOR_M
from .rod_fixture_finite import canonical_hash

POLICY = dict(version="raster-band-readout-1", maximum_band_excess_px=1.,
    tolerance_basis="one_original_image_pixel_fixed_before_development_evaluation",
    ranking="admissible_final_actual_support_length_then_actual_rmse_then_image_residual",
    no_admissible_candidate="unchanged_bracketed_winner",
    axis_policy="original_actual_point_TLS_hypotheses_no_refit",
    support_policy="original_same_chain_RGB_clip_and_bracketed_occupancy",
    base_and_curve_identical=True, extra_annotations=0, gt_read=False)


def length(segments):
    segments = np.asarray(segments, float).reshape(-1, 2, 3)
    return float(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1).sum())


def select_existing(result, values, voxel_size, context):
    """Resolve fixed-band equivalence without changing any candidate's model."""
    old_index = result["selected_candidate"]
    result.update(raster_band_policy=dict(POLICY), raster_band_policy_sha256=canonical_hash(POLICY),
        previous_selected_candidate=old_index, selection_changed=False,
        ranking=POLICY["ranking"], raster_admissible_count=0,
        raster_evaluated_count=0, raster_selection_state="fallback_unchanged")
    if result["state"] != "complete":
        return result
    rows = result["readouts"]
    eligible = [i for i, row in enumerate(rows) if row["band_score"] is not None
        and row["band_score"]["maximum_band_excess_px"] <= POLICY["maximum_band_excess_px"]
        and len(row["graph"]["segments"])]
    result["raster_admissible_count"] = len(eligible)
    if not eligible:
        return result
    contexts = {None: context}
    # Recover assignment provenance only when a previously un-clipped group is
    # actually needed. Fitting, sample admission and band measurements are old.
    def local_context(signature):
        if signature not in contexts:
            for group in support_groups(np.unique(np.asarray(values, float), axis=0), context):
                contexts[group["support_set_sha256"]] = dict(context, association=dict(
                    context["association"], assignments=group["assignments"],
                    chain_count=len(group["assignments"])))
        if signature not in contexts:
            raise ValueError("Original candidate assignment provenance missing")
        return contexts[signature]

    winner = None
    # Unclipped occupied length is a safe upper bound, including any subsequent
    # short-hole bridging. Never rank by raw point count or source arm identity.
    for index in sorted(eligible, key=lambda i: (-length(rows[i]["graph"]["segments"]), i)):
        row = rows[index]
        occupied = row["graph"]["segments"]
        if winner is not None and length(occupied) < -winner[0][0]:
            continue
        clipped = row.get("rgb_cell_clip")
        if clipped is None:
            clipped = cells.clip_segments(occupied, local_context(row["support_set_sha256"]),
                                          NUMERICAL_LENGTH_FLOOR_M)
            row["rgb_cell_clip"] = clipped
        bridged = previous.bridge_internal_holes(clipped["segments"], occupied, voxel_size)
        result["raster_evaluated_count"] += 1
        row["raster_final_length_m"] = length(bridged["segments"])
        if not len(bridged["segments"]):
            continue
        score = row["band_score"]
        key = (-row["raster_final_length_m"], row["graph"]["model"]["perpendicular_rmse"],
               score["maximum_band_excess_px"], score["band_excess_rms_px"],
               tuple(np.asarray(bridged["segments"]).ravel()))
        if winner is None or key < winner[0]:
            winner = (key, index, bridged, len(clipped["segments"]))
    if winner is None:
        return result
    _, index, bridged, clipped_count = winner
    result.update(bridged, selected_candidate=index, selected_source=rows[index]["source"],
        selected_band_score=rows[index]["band_score"], components=len(bridged["segments"]),
        previous_rgb_clipped_components=clipped_count, reason=None,
        selection_changed=index != old_index, raster_selection_state="selected_admissible_actual_support")
    return result


def readout(values, voxel_size, origin, context):
    result = previous.readout(values, voxel_size, origin, context)
    try:
        return select_existing(result, values, voxel_size, context)
    except OverflowError as exc:
        # Partial candidate work must not masquerade as a complete comparison.
        return dict(result, state="unmeasurable", reason=str(exc),
                    segments=np.empty((0, 2, 3)), components=0)
