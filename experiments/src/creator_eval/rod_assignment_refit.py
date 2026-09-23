"""Refit retained image assignments before ranking; keep the old search untouched.

This is a development comparison, not a physical-axis certificate. Coverage is
measured RGB-row support, never truth overlap. The exhaustive claim is limited
to the same nearest-match seed search and capped image pools as the old method.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np

from .line_controls import LineFitDegenerate, fit_multiview_line
from .rod_candidate_association import _CachedViewMatcher
from .rod_multiview_candidates import _line_x, _project_world_line, _validated_views


def _refit_assignment(seed, views, config):
    indices = seed["supporting_views"]
    model = fit_multiview_line(
        [views[i]["candidates"][seed["matches"][i]["candidate_index"]]["line"] for i in indices],
        [views[i]["K_index"] for i in indices],
        [views[i]["world_to_camera_cv"] for i in indices],
    )
    matches, signatures, coverage = [], [], []
    for i, view in enumerate(views):
        line = _project_world_line(model, view)
        ys = np.linspace(*view["y_range"], 9)
        projected = _line_x(line, ys)
        ci = seed["matches"][i]["candidate_index"]
        residual = None if ci is None else float(np.median(np.abs(
            projected - _line_x(view["candidates"][ci]["line"], ys))))
        matches.append(dict(candidate_index=ci, residual_px=residual, projected_line=line))
        signatures.extend(projected)
        if i in indices:
            # 不用重拟合后的轴偷偷换一组像素，也不靠删视图把失败挪走。
            if residual > config["reprojection_threshold_px"]:
                return None
            counts = [len(c["row_matches"]) for c in view["candidates"]]
            coverage.append(counts[ci] / max(1, max(counts)))
    return {**copy.deepcopy(seed), "model": model, "matches": matches,
            "residual_median_px": float(np.median([matches[i]["residual_px"] for i in indices])),
            "row_support_fraction": float(np.median(coverage)), "_signature": np.asarray(signatures)}


def _select(rows, common, config, *, coverage=False, retain_assignments=False):
    ranked = sorted(rows, key=lambda r: (-r["support_view_count"],
        -r["row_support_fraction"] if coverage else 0, r["residual_median_px"]))
    unique, signatures = [], []
    for row in ranked:
        signature = row["_signature"]
        if not retain_assignments and signatures and np.any(
                np.median(np.abs(np.asarray(signatures) - signature), axis=1) < config["deduplicate_separation_px"]):
            continue
        unique.append({k: copy.deepcopy(v) for k, v in row.items() if k != "_signature"})
        signatures.append(signature)
    output = {**common, "selected": None, "alternatives": [],
              "unique_hypothesis_count": len(unique), "state": "rejected", "reason": "no_finally_supported_assignment",
              "ranking": "view_count_row_coverage_residual" if coverage else "view_count_final_residual",
              "retention": "distinct_assignments" if retain_assignments else "legacy_pixel_separation"}
    if unique:
        best = unique[0]
        alternatives = []
        for i, row in enumerate(unique[1:], 1):
            if row["support_view_count"] < config["ambiguity_support_fraction"] * best["support_view_count"]:
                continue
            separation = float(np.median(np.abs(signatures[0] - signatures[i])))
            if retain_assignments or separation >= config["ambiguity_separation_px"]:
                alternatives.append({**row, "separation_from_best_px": separation})
        output.update(selected=best, alternatives=alternatives,
            state="ambiguous" if alternatives else "accepted",
            reason="competing_refitted_assignments" if alternatives else "one_retained_refitted_assignment")
    if not common["search_complete"]:
        output.update(state="ambiguous", reason="search_budget_exhausted")
    return output


def associate_refitted_variants(views, config):
    """Generate once; compare final-residual ranking, row coverage and retention."""
    _validated_views(views)
    for view in views:
        for candidate in view["candidates"]:
            rows = candidate["row_matches"]
            if not rows or len({int(r[0]) for r in rows}) != len(rows):
                raise ValueError("Nonempty unique measured rows required for coverage")
    count = int(config["fit_view_count"])
    maximum, attempts = int(config["maximum_hypotheses"]), int(config["maximum_fit_attempts"])
    if count < 3 or min(maximum, attempts) < 1:
        raise ValueError("Invalid candidate search budget")
    combinations = list(itertools.combinations(range(len(views)), count))
    total = sum(math.prod(len(views[i]["candidates"]) for i in indices) for indices in combinations)
    matchers = [_CachedViewMatcher(v) for v in views]
    seeds, attempted, generated = {}, 0, 0
    for indices in combinations:
        for choices in itertools.product(*(range(len(views[i]["candidates"])) for i in indices)):
            if attempted >= attempts or generated >= maximum:
                break
            attempted += 1
            try:
                model = fit_multiview_line([views[i]["candidates"][j]["line"] for i, j in zip(indices, choices)],
                    [views[i]["K_index"] for i in indices], [views[i]["world_to_camera_cv"] for i in indices])
                matches = [m.match(model) for m in matchers]
            except (LineFitDegenerate, np.linalg.LinAlgError):
                continue
            support = [i for i, m in enumerate(matches) if m["residual_px"] is not None
                       and m["residual_px"] <= config["reprojection_threshold_px"]]
            if len(support) < config["minimum_support_views"]:
                continue
            generated += 1
            signature = tuple((i, matches[i]["candidate_index"]) for i in support)
            # 一个像素赋值只重拟合一次。不同三视图种子不该假装成多个独立证据。
            seeds.setdefault(signature, dict(model=model, matches=matches, supporting_views=support,
                support_view_count=len(support), source_fit_views=list(indices), source_candidate_indices=list(choices)))
        if attempted >= attempts or generated >= maximum:
            break
    rows, failures = [], []
    for signature, seed in seeds.items():
        try:
            fitted = _refit_assignment(seed, views, config)
            if fitted is None:
                failures.append(dict(assignment=signature, reason="final_support_residual_exceeded"))
            else:
                rows.append(fitted)
        except (LineFitDegenerate, np.linalg.LinAlgError) as error:
            failures.append(dict(assignment=signature, reason="refit_degenerate:" + str(error)))
    common = dict(search_complete=attempted == total, attempted_combination_count=attempted,
        total_combination_count=total, generated_hypothesis_count=generated, distinct_assignment_count=len(seeds),
        finally_supported_assignment_count=len(rows), refit_rejections=failures,
        scope="same_capped_nearest_match_seed_search_not_all_possible_pixel_assignments_or_physical_identity",
        refit_policy="all_fixed_supporting_views_then_validate_same_assignments_before_ranking")
    return {
        "refit_residual": _select(rows, common, config),
        "refit_coverage": _select(rows, common, config, coverage=True),
        "retained_assignments": _select(rows, common, config, retain_assignments=True),
    }
