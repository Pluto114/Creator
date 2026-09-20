"""Cache fixed image candidates while preserving the legacy association result.

The reference remains in rod_multiview_candidates. This module changes only how
view matches are calculated; fitting, search order, budgets, ranking, and output
semantics intentionally stay the same. No truth or image files are read here.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .line_controls import LineFitDegenerate, fit_multiview_line
from .rod_multiview_candidates import _line_x, _project_world_line, _validated_views


class _CachedViewMatcher:
    """Keep each candidate's nine sampled x coordinates for the whole search."""

    def __init__(self, view):
        self.view = view
        self.ys = np.linspace(view["y_range"][0], view["y_range"][1], 9)
        self.candidate_x = np.array([
            _line_x(candidate["line"], self.ys) for candidate in view["candidates"]
        ]).reshape(-1, 9)

    def match(self, model):
        # 三维投影先照旧算；这里只缓存不变的图像候选，别顺手改了几何算术。
        projected = _project_world_line(model, self.view)
        if not len(self.candidate_x):
            return {"candidate_index": None, "residual_px": None, "projected_line": projected}
        projected_x = _line_x(projected, self.ys)
        distances = np.abs(projected_x[None, :] - self.candidate_x)
        residuals = np.median(distances, axis=1)
        # np.argmin and min((residual, index)) both keep the first candidate on a
        # tie. Keep the output scalar conversion and projected line unchanged.
        index = int(np.argmin(residuals))
        return {"candidate_index": index, "residual_px": float(residuals[index]),
                "projected_line": projected}


def associate_multiview_lines_cached(views, config):
    """Associate lines with explicit search completion; truncation cannot prove uniqueness."""
    _validated_views(views)
    fit_count = int(config["fit_view_count"])
    maximum = int(config["maximum_hypotheses"])
    max_attempts = int(config.get("maximum_fit_attempts", 100000))
    if fit_count < 3 or maximum < 1 or max_attempts < 1:
        raise ValueError("Invalid candidate search budget")
    matchers = [_CachedViewMatcher(view) for view in views]
    combinations = list(itertools.combinations(range(len(views)), fit_count))
    total = sum(math.prod(len(views[i]["candidates"]) for i in indices) for indices in combinations)
    generated = []
    attempted = 0
    stop = False
    for selected_views in combinations:
        candidate_ranges = [range(len(views[index]["candidates"])) for index in selected_views]
        for selected_candidates in itertools.product(*candidate_ranges):
            if len(generated) >= maximum or attempted >= max_attempts:
                stop = True
                break
            attempted += 1
            try:
                model = fit_multiview_line(
                    [views[v]["candidates"][c]["line"] for v, c in zip(selected_views, selected_candidates)],
                    [views[v]["K_index"] for v in selected_views],
                    [views[v]["world_to_camera_cv"] for v in selected_views],
                )
                matches = [matcher.match(model) for matcher in matchers]
            except (LineFitDegenerate, np.linalg.LinAlgError):
                continue
            supporting = [
                index for index, match in enumerate(matches)
                if match["residual_px"] is not None
                and match["residual_px"] <= config["reprojection_threshold_px"]
            ]
            if len(supporting) < int(config["minimum_support_views"]):
                continue
            residuals = [matches[index]["residual_px"] for index in supporting]
            # These projections were just calculated. Re-projecting every pair
            # during deduplication cost minutes without adding any information.
            signature = np.concatenate([
                _line_x(match["projected_line"], np.linspace(*view["y_range"], 9))
                for match, view in zip(matches, views)
            ])
            generated.append({
                "model": model, "matches": matches, "supporting_views": supporting,
                "support_view_count": len(supporting),
                "residual_median_px": float(np.median(residuals)),
                "source_fit_views": list(selected_views),
                "source_candidate_indices": list(selected_candidates),
                "_signature": signature,
            })
        if stop:
            break
    generated.sort(key=lambda row: (-row["support_view_count"], row["residual_median_px"]))
    unique, signatures = [], []
    for hypothesis in generated:
        signature = hypothesis.pop("_signature")
        if signatures and np.any(np.median(np.abs(np.asarray(signatures) - signature), axis=1) < config["deduplicate_separation_px"]):
            continue
        unique.append(hypothesis)
        signatures.append(signature)
    complete = attempted == total
    result = {
        "state": "rejected" if complete else "ambiguous",
        "reason": "no_3d_line_with_required_view_support" if complete else "search_budget_exhausted",
        "selected": None, "alternatives": [],
        "unique_hypothesis_count": len(unique),
        "generated_hypothesis_count": len(generated),
        "search_complete": complete,
        "attempted_combination_count": attempted,
        "total_combination_count": total,
        "scope": "infinite_3d_line_identity_only_no_extent_gap_or_physical_existence_proof",
    }
    if not unique:
        return result
    best = unique[0]
    alternatives = []
    for index, candidate in enumerate(unique[1:], 1):
        if candidate["support_view_count"] < config["ambiguity_support_fraction"] * best["support_view_count"]:
            continue
        separation = float(np.median(np.abs(signatures[0] - signatures[index])))
        if separation >= config["ambiguity_separation_px"]:
            alternatives.append({**candidate, "separation_from_best_px": separation})
    result.update(
        state="ambiguous" if alternatives or not complete else "accepted",
        reason=("search_budget_exhausted" if not complete else
                "competing_multiview_lines" if alternatives else "unique_multiview_line"),
        selected=best, alternatives=alternatives,
    )
    return result
