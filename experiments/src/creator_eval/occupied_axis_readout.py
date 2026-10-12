"""Exact actual-point occupancy on foreground-selected, unaltered TLS axes.

Voxel means balance the fit, not the support extent. Occupancy is the exact
union of the unchanged 1v neighbourhood of each unique actual projection,
bounded by observed endpoints. The 4v length gate qualifies the entire axis;
actual fragments on that axis are retained, not discarded for being sparse.
No sampling phase or gap interpolation.
"""

import numpy as np

from . import foreground_support as support
from . import rgb_segment_cells as cells
from . import single_axis_readout as axis
from .chain_partition_readout import support_groups
from .line_controls import LineFitDegenerate, fit_line_tls
from .rgb_chain_support import chain_support_mask
from .rod_fixture_finite import canonical_hash
from .stroke_selected_axis_readout import foreground_bands, projected_band_score

NUMERICAL_LENGTH_FLOOR_M = 1e-10


def occupied_intervals(observed, radius):
    """Closed union, clipped to actual projection min/max, without gap closing."""
    observed = np.asarray(observed, float)
    if (observed.ndim != 1 or not np.isfinite(observed).all()
            or isinstance(radius, (bool, np.bool_)) or not np.isfinite(radius) or radius <= 0):
        raise ValueError("Finite scalar projections and positive support radius required")
    observed = np.unique(observed)
    if len(observed) < 2:
        return np.empty((0, 2))
    starts = np.maximum(observed-radius, observed[0])
    ends = np.minimum(observed+radius, observed[-1])
    cut = np.flatnonzero(starts[1:] > ends[:-1])+1
    first, last = np.r_[0, cut], np.r_[cut-1, len(observed)-1]
    intervals = np.c_[starts[first], ends[last]]
    return intervals[intervals[:, 1] > intervals[:, 0]]


def actual_axis(values, voxel_size, origin):
    """Old voxel-balanced TLS fit, exact support from unchanged unique points."""
    unique, representatives = axis.voxel_representatives(values, voxel_size, origin)
    graph = dict(state="complete", reason=None, segments=np.empty((0, 2, 3)),
        occupied_voxels=len(representatives), unique_count=len(unique), components=0,
        support_policy="exact_unique_actual_projection_union_clipped_to_observed_extent",
        projection_support_radius_voxels=axis.DEFAULTS["projection_support_radius_voxels"],
        minimum_axis_extent_voxels=axis.DEFAULTS["minimum_length_voxels"],
        fragment_length_policy="retain_positive_actual_support_on_qualified_axis",
        model=None, discarded_short_runs=0)
    try:
        model = fit_line_tls(representatives, min_points=axis.DEFAULTS["minimum_voxels"])
    except LineFitDegenerate as exc:
        return {**graph, "reason": str(exc)}
    graph["model"] = model
    if 1-model["eigen_gap_ratio"] > axis.DEFAULTS["maximum_second_eigen_ratio"]:
        return {**graph, "reason": "not_a_unique_dominant_axis"}
    center, direction = model["centroid"], model["direction"]
    observed = (unique-center) @ direction
    extent = float(np.ptp(observed))
    graph["observed_projection_extent_m"] = extent
    if extent < axis.DEFAULTS["minimum_length_voxels"]*voxel_size:
        return {**graph, "reason": "below_minimum_axis_length"}
    intervals = occupied_intervals(observed, axis.DEFAULTS["projection_support_radius_voxels"]*voxel_size)
    lengths = intervals[:, 1]-intervals[:, 0]
    keep = lengths > NUMERICAL_LENGTH_FLOOR_M
    short = keep & (lengths < axis.DEFAULTS["minimum_length_voxels"]*voxel_size)
    segments = center+intervals[keep, :, None]*direction
    graph.update(segments=segments, components=len(segments),
        geometry_interval_count=len(intervals), geometry_support_length_m=float(lengths.sum()),
        retained_short_fragments=int(short.sum()), retained_short_length_m=float(lengths[short].sum()),
        discarded_short_runs=int((~keep).sum()), discarded_short_length_m=float(lengths[~keep].sum()),
        reason=None if len(segments) else "no_finite_actual_support_run")
    return graph


def readout(values, voxel_size, origin, context):
    values, origin = np.asarray(values, float), np.asarray(origin, float)
    if (values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all()
            or origin.shape != (3,) or not np.isfinite(origin).all()
            or isinstance(voxel_size, (bool, np.bool_)) or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Finite actual Nx3 coordinates and positive voxel size required")
    unique = np.unique(values, axis=0)
    result = dict(state="complete", segments=np.empty((0, 2, 3)), components=0,
        input_count=len(values), unique_count=len(unique), voxel_size=float(voxel_size), origin=origin.tolist(),
        reason=None, readouts=[], selected_candidate=None,
        scope="one_explicitly_marked_local_straight_member_with_observed_gaps",
        target_identity_confirmed=False, image_constructed_geometry=False,
        preserves_arbitrary_multiple_axes=False,
        ranking="unpad_measured_band_excess_then_excess_rms_then_exact_actual_length_then_actual_rmse",
        support_policy="exact_unique_actual_projection_union_clipped_to_observed_extent",
        fragment_length_policy="4v_axis_eligibility_not_individual_supported_fragment_pruning",
        numerical_length_floor_m=NUMERICAL_LENGTH_FLOOR_M,
        axis_defaults=dict(axis.DEFAULTS), expansion_defaults=dict(support.DEFAULTS))
    if context["association"]["fallback_to_raw_union"]:
        return {**result, "state": "unmeasurable", "reason": "incomplete_chain_search"}
    if len(unique) > axis.DEFAULTS["maximum_axis_samples"]:
        return {**result, "state": "unmeasurable", "reason": "point_budget_exceeded"}
    try:
        bands = foreground_bands(context)
        result["foreground_bands"] = [{k: v for k, v in b.items() if k != "projection"} for b in bands]
        candidates = [(np.arange(len(unique)), context, "full_stroke", None)]
        for group in support_groups(unique, context):
            local = {**context, "association": dict(context["association"],
                assignments=group["assignments"], chain_count=len(group["assignments"]))}
            pool = np.flatnonzero(chain_support_mask(unique, local))
            seeds = np.searchsorted(pool, group["indices"])
            if not np.array_equal(pool[seeds], group["indices"]):
                raise ValueError("Required seeds must belong to their same-chain pool")
            expansion = support.fixed_seed_expansion(unique[pool], seeds, voxel_size, origin)
            if expansion["state"] != "complete":
                raise OverflowError(expansion["reason"])
            if expansion["reason"] is None:
                candidates.append((pool[expansion["indices"]], local, "fixed_seed_expansion",
                                   group["support_set_sha256"]))
        ranked = []
        for indices, local, source, signature in candidates:
            graph = actual_axis(unique[indices], voxel_size, origin)
            row = dict(source=source, support_set_sha256=signature,
                actual_point_count=len(indices), actual_points_sha256=canonical_hash(unique[indices]),
                graph=graph, band_score=None)
            result["readouts"].append(row)
            if not len(graph["segments"]):
                continue
            score = projected_band_score(graph["model"], bands)
            row["band_score"] = score
            if score is None:
                continue
            length = float(np.linalg.norm(graph["segments"][:, 1]-graph["segments"][:, 0], axis=1).sum())
            ranking = (score["maximum_band_excess_px"], score["band_excess_rms_px"], -length,
                       graph["model"]["perpendicular_rmse"], tuple(np.asarray(graph["segments"]).ravel()))
            ranked.append((ranking, len(result["readouts"])-1, local))
        winner = None
        for ranking, index, local in sorted(ranked, key=lambda item: item[0]):
            if winner is not None:
                if ranking[:2] > winner[0][:2]:
                    break
                # Unclipped geometric length is an upper bound; do not spend
                # projective-cell work on candidates that cannot beat winner.
                if ranking[:2] == winner[0][:2] and ranking[2] > winner[0][2]:
                    continue
            row = result["readouts"][index]
            clipped = cells.clip_segments(row["graph"]["segments"], local,
                                          NUMERICAL_LENGTH_FLOOR_M)
            row["rgb_cell_clip"] = clipped
            if len(clipped["segments"]):
                length = float(np.linalg.norm(clipped["segments"][:, 1]-clipped["segments"][:, 0], axis=1).sum())
                key = (*ranking[:2], -length, ranking[3], tuple(clipped["segments"].ravel()))
                if winner is None or key < winner[0]:
                    winner = (key, index)
        if winner is not None:
            index = winner[1]
            row = result["readouts"][index]
            segments = row["rgb_cell_clip"]["segments"]
            result.update(segments=segments, components=len(segments), selected_candidate=index,
                selected_source=row["source"], selected_band_score=row["band_score"])
            return result
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    result["reason"] = "no_actual_axis_with_continuous_foreground_support"
    return result
