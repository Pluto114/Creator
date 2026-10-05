"""Preserve the guarded baseline; supplement only two-fold agreed finite ridges.

The additional two fixed voxel partitions each nominate, fit and validate their
own ridge axes. They are overlapping cross-fits, not independent experiments or
a statistical confidence multiplier. Upstream section ownership is unchanged.
"""

import numpy as np

from . import common_readout_bundle_small as baseline
from . import common_readout_fold as fold_reader
from .ridge_fold_consensus import agreed_segments, overlaps_existing

DEFAULTS = dict(baseline.DEFAULTS)


def policy(config):
    return baseline.policy(config)


def readout(points, segments, config, region=None):
    p = policy(config)
    result = baseline.readout(points, segments, p, region)
    result["fold_consensus"] = dict(fixed_additional_partitions=[1, 2], required_agreement=2,
        existing_geometry_preserved=True, added_segments=0, fold_results=[],
        folds_are_independent_experiments=False,
        scope="Conditional ridge supplement under unchanged upstream section ownership; not a confidence or identity claim")
    if result["state"] != "complete" or "ridge_fit_audits" not in result:
        return result
    rows = []
    for index in (1, 2):
        value = fold_reader.readout(points, segments, {**p, "ridge_fold_index": index}, region)
        rows.append(value)
        result["fold_consensus"]["fold_results"].append({key: val for key, val in value.items()
            if key not in {"config", "section_evidence", "scope"}})
        if value["state"] != "complete":
            return {**result, "state": "unmeasurable", "resolution_state": "unresolved",
                    "reason": "additional_fold_unmeasurable:"+str(value.get("reason"))}
    added = agreed_segments(rows[0]["segments"], rows[1]["segments"], result["segments"], p)
    nonredundant = []
    for segment in added:
        if not overlaps_existing(segment, [*result["segments"], *nonredundant], p):
            nonredundant.append(segment)
    result["fold_consensus"]["overlapping_alternatives_not_added"] = len(added)-len(nonredundant)
    added = np.asarray(nonredundant, float).reshape(-1, 2, 3)
    if len(result["segments"])+len(added) > p["maximum_output_segments"]:
        return {**result, "state": "unmeasurable", "reason": "output_segment_budget_exceeded"}
    result["fold_consensus"]["added_segments"] = len(added)
    if len(added):
        chosen = np.concatenate((result["segments"], added))
        result.update(segments=chosen, components=len(chosen), reason=None,
            resolution_state=fold_reader._resolution(chosen, result["section_evidence"],
                                                   result.get("sparse_parallel_ambiguity", False)))
    return result
