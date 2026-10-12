"""One explicitly marked local axis, selected among fits of actual geometry.

Image bands rank existing TLS hypotheses; they never construct, triangulate,
snap, or extend output geometry. This reader is for one marked straight member
(possibly with gaps), not arbitrary multi-axis scenes or a general skeletonizer.
"""

import numpy as np

from . import foreground_support as support
from . import rgb_segment_cells as cells
from . import single_axis_readout as axis
from .chain_partition_readout import support_groups
from .rgb_anchored_chain_support import _covered_rows
from .rgb_chain_support import chain_support_mask
from .rod_fixture_finite import canonical_hash


def foreground_bands(context):
    """Fixed, arm-independent narrowest measured band enclosing each claim."""
    views = {view["view_id"]: view for view in context["views"]}
    bands = []
    for anchor in context["association"]["anchors"]:
        view = views[anchor["view_id"]]
        rows, low, high = _covered_rows(anchor)
        for row in rows:
            if not 0 <= row < len(view["left"]):
                raise ValueError("Foreground claim outside measured image")
            left, right = view["left"][row], view["right"][row]
            valid = np.flatnonzero(np.isfinite(left) & (left <= low) & (right >= high))
            if not len(valid):
                raise ValueError("Foreground claim has no enclosing measured band")
            index = min(valid, key=lambda i: (right[i]-left[i], left[i], right[i]))
            bands.append(dict(view_id=view["view_id"], row=int(row),
                left=float(left[index]), right=float(right[index]),
                projection=view["K_index"] @ view["world_to_camera_cv"]))
    if len({band["view_id"] for band in bands}) < 2:
        raise ValueError("At least two foreground-annotated views required")
    return bands


def projected_band_score(model, bands):
    """Worst-view RMS relative to measured half-width; no geometry alteration."""
    center, direction = np.asarray(model["centroid"]), np.asarray(model["direction"])
    by_view, residuals, excesses = {}, [], []
    for band in bands:
        projection = band["projection"]
        line = np.cross(projection[:, :3] @ center+projection[:, 3],
                        projection[:, :3] @ direction)
        if abs(line[0]) <= 1e-10*np.linalg.norm(line[:2]):
            return None
        x = -(line[1]*band["row"]+line[2])/line[0]
        half_width = (band["right"]-band["left"])/2
        residual = abs(x-(band["left"]+band["right"])/2)/half_width
        if not np.isfinite(residual):
            return None
        by_view.setdefault(band["view_id"], []).append(float(residual))
        residuals.append(float(residual))
        excesses.append(float(max(band["left"]-x, x-band["right"], 0.)))
    view_rms = {name: float(np.sqrt(np.mean(np.square(values)))) for name, values in by_view.items()}
    return dict(maximum_band_excess_px=max(excesses), worst_view_rms=max(view_rms.values()),
        band_excess_rms_px=float(np.sqrt(np.mean(np.square(excesses)))),
        all_marks_rms=float(np.sqrt(np.mean(np.square(residuals)))),
        maximum_normalized_residual=max(residuals), view_rms=view_rms)


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
        ranking="unpad_measured_band_excess_then_excess_rms_then_finite_length_then_actual_rmse",
        band_policy="narrowest_actual_enclosing_band_at_fixed_foreground_claim_rows",
        axis_defaults=dict(axis.DEFAULTS), expansion_defaults=dict(support.DEFAULTS))
    if context["association"]["fallback_to_raw_union"]:
        return {**result, "state": "unmeasurable", "reason": "incomplete_chain_search"}
    if len(unique) > axis.DEFAULTS["maximum_axis_samples"]:
        return {**result, "state": "unmeasurable", "reason": "point_budget_exceeded"}
    try:
        bands = foreground_bands(context)
        result["foreground_bands"] = [{k: v for k, v in b.items() if k != "projection"} for b in bands]
        groups = support_groups(unique, context)
        # Preserve the full stroke-domain hypothesis as well as required-seed
        # hypotheses. No arm identity or native primitive label enters this list.
        candidates = [(np.arange(len(unique)), context, "full_stroke", None)]
        for group in groups:
            local = {**context, "association": dict(context["association"],
                assignments=group["assignments"], chain_count=len(group["assignments"]))}
            pool = np.flatnonzero(chain_support_mask(unique, local))
            seeds = np.searchsorted(pool, group["indices"])
            if not np.array_equal(pool[seeds], group["indices"]):
                raise ValueError("Required seeds must belong to their same-chain pool")
            expansion = support.fixed_seed_expansion(unique[pool], seeds, voxel_size, origin)
            if expansion["reason"] is None:
                candidates.append((pool[expansion["indices"]], local, "fixed_seed_expansion",
                                   group["support_set_sha256"]))
        ranked = []
        for indices, local, source, signature in candidates:
            graph = axis.readout(unique[indices], voxel_size, origin,
                lambda points, current=local: chain_support_mask(points, current))
            if graph["state"] != "complete":
                raise OverflowError("incomplete_actual_axis_candidate")
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
            # A claim identifies a measured interval, not its exact center.
            # Equally admissible axes compete by finite actual support length;
            # subpixel center residual must not favour a short fragment.
            ranking = (score["maximum_band_excess_px"], score["band_excess_rms_px"], -length,
                       graph["model"]["perpendicular_rmse"], tuple(np.asarray(graph["segments"]).ravel()))
            ranked.append((ranking, len(result["readouts"])-1, local))
        for _, index, local in sorted(ranked, key=lambda item: item[0]):
            row = result["readouts"][index]
            clipped = cells.clip_segments(row["graph"]["segments"], local,
                                          axis.DEFAULTS["minimum_length_voxels"]*voxel_size)
            row["rgb_cell_clip"] = clipped
            if len(clipped["segments"]):
                result.update(segments=clipped["segments"], components=len(clipped["segments"]),
                    selected_candidate=index, selected_source=row["source"], selected_band_score=row["band_score"])
                return result
    except OverflowError as exc:
        return {**result, "state": "unmeasurable", "reason": str(exc)}
    result["reason"] = "no_actual_axis_with_continuous_foreground_support"
    return result
