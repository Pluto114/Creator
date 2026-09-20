"""One-pass removal of unsupported views before cylinder-consistent refitting.

A failed side is unknown, never negative evidence. At least four initially
consistent views must remain; do not search subsets until something looks good.
"""
from __future__ import annotations

import copy

from .line_controls import LineFitDegenerate
from .rod_cylinder_gate import (
    policy_values,
    refit_hypothesis,
    screen_cylinder,
    select_cylinder_hypothesis,
)


def select_supported_cylinders(association, views, observations, extent_config, config=None):
    policy = policy_values(config)
    hypotheses = ([association["selected"]] if association.get("selected") is not None else []) + association["alternatives"]
    eligible, audits, ordinals = [], [], []
    for ordinal, hypothesis in enumerate(hypotheses):
        audit = {"ordinal": ordinal, "original_supporting_views": hypothesis["supporting_views"], "eligible": False}
        try:
            fitted = refit_hypothesis(hypothesis, views, extent_config)
            initial = screen_cylinder(fitted, views, observations, policy)
        except LineFitDegenerate as error:
            audits.append({**audit, "reason": "initial_fit_degenerate:" + str(error)})
            continue
        passed_ids = {v["view_id"] for v in initial["views"] if v["passed"]}
        kept = [i for i in hypothesis["supporting_views"] if views[i]["view_id"] in passed_ids]
        dropped = [i for i in hypothesis["supporting_views"] if i not in kept]
        audit.update(initial_screen=initial, kept_views=kept, dropped_views=dropped)
        # 退出投票也要退出拟合。只在报告里写“忽略坏视角”，轴却继续吃它的数据，是糊弄。
        if len(kept) >= policy["minimum_views"]:
            trimmed = copy.deepcopy(hypothesis)
            trimmed.update(supporting_views=kept, support_view_count=len(kept))
            eligible.append(trimmed)
            ordinals.append(ordinal)
            audit.update(eligible=True, reason="enough_initially_consistent_views")
        else:
            audit["reason"] = "fewer_than_four_initially_consistent_views"
        audits.append(audit)
    proposal = {**association, "selected": eligible[0] if eligible else None, "alternatives": eligible[1:]}
    # The strict gate now refits and tests every retained view again. No recursive dropping.
    output = select_cylinder_hypothesis(proposal, views, observations, extent_config, policy)
    output["cylinder_support"] = {
        "hypotheses": audits, "eligible_original_ordinals": ordinals,
        "surviving_original_ordinals": [ordinals[i] for i in output["cylinder_screen"]["surviving_ordinals"]],
        "view_policy": "one_initial_screen_then_drop_failed_views_refit_once_and_recheck_all_retained_views",
        "failed_view_semantics": "unknown_excluded_from_axis_fit_and_finite_support_never_negative",
        "subset_search_performed": False,
    }
    return output
