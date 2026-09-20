"""A complete pool from the existing finite RANSAC sample, before the model cap.

The default preserves the legacy computation control; corrected experiments
opt into final support validation. Candidate prefixes share one raw observation
package, one random draw sequence, and one stable ranking.
The pool still cannot claim to contain every possible image explanation.
"""

from __future__ import annotations

import numpy as np

from .rod_multiview_candidates import _as_line, _line_x


class _PackedRows:
    """Pack ragged raw candidates once; keep their original row/column identities."""

    def __init__(self, rows):
        self.raw_indices = np.array([index for index, _ in rows], dtype=np.int64)
        self.y = np.array([row["y"] for _, row in rows], dtype=float)
        width = max(len(row["candidates"]) for _, row in rows)
        self.centers = np.full((len(rows), width), np.inf)
        for index, (_, row) in enumerate(rows):
            self.centers[index, :len(row["candidates"])] = [
                item["center_x"] for item in row["candidates"]
            ]
        self.packed_indices = np.arange(len(rows))

    def support(self, slope, intercept, threshold):
        divisor = float(np.hypot(1.0, slope))
        predicted = slope * self.y + intercept
        residuals = np.abs(self.centers - predicted[:, None]) / divisor
        # argmin still takes the first raw candidate on a tie. Padding is +inf,
        # so it cannot beat a finite observation or renumber a real candidate.
        chosen = np.argmin(residuals, axis=1)
        nearest = residuals[self.packed_indices, chosen]
        accepted = np.flatnonzero(nearest <= threshold)
        return [
            (int(self.raw_indices[index]), int(chosen[index]), float(nearest[index]))
            for index in accepted
        ]


def enumerate_image_line_pool(observations, config, *, require_refit_support=False):
    """Return the legacy sorted/deduplicated hypotheses without maximum_models.

    For ordinary finite numeric rows from extract_rod_observations/JSON, the
    candidate list matches enumerate_image_lines with maximum_models >= trials.
    Sampling, least-squares fitting, stable sorting, strict deduplication, and
    row_matches retain the old semantics. ``candidates[:8]`` and ``[:16]`` thus
    share a pool; raising the cap does not trigger new extraction or random draws.

    require_refit_support=True also checks the final matched rows and y span
    before ranking/deduplication. False intentionally preserves the frozen
    legacy control; corrected experiments must explicitly request True.

    ``sampling`` counts attempts and pre-dedup outcomes; ``deduplication`` counts
    discarded duplicates and retained candidates. Neither is a completeness
    claim about image content. Missing rows are not invented or interpolated.
    """
    rows = [(index, row) for index, row in enumerate(observations["rows"]) if row["candidates"]]
    sampling = {
        "seed": int(config["seed"]),
        "requested_trials": int(config["trials"]),
        "attempted_trials": 0,
        "raw_row_count": len(observations["rows"]),
        "rows_with_candidates": len(rows),
        "row_pair_too_close": 0,
        "sampled_candidate_pairs": 0,
        "insufficient_initial_support": 0,
        "insufficient_initial_y_span": 0,
        "hypotheses_before_deduplication": 0,
    }
    result = {
        "candidates": [],
        "sampling": sampling,
        "deduplication": {"input_count": 0, "duplicate_count": 0, "retained_count": 0},
        "scope": "complete_ranked_pool_from_fixed_legacy_ransac_sample_not_all_image_explanations",
        "maximum_models_applied": False,
    }
    if require_refit_support:
        sampling.update(insufficient_refit_support=0, insufficient_refit_y_span=0)
        result["scope"] = "complete_fixed_sample_pool_with_final_support_validation"
        result["refit_support_validated"] = True
    if len(rows) < config["minimum_rows"]:
        result["reason"] = "too_few_rows_with_candidates"
        return result
    packed = _PackedRows(rows)
    rng = np.random.default_rng(int(config["seed"]))
    hypotheses = []
    for _ in range(int(config["trials"])):
        sampling["attempted_trials"] += 1
        first, second = rng.choice(len(rows), 2, replace=False)
        a, b = rows[first][1], rows[second][1]
        if abs(b["y"] - a["y"]) < 20:
            sampling["row_pair_too_close"] += 1
            continue
        # RNG调用顺序也得留下来。先抽候选再筛行距，会悄悄换掉整池证据。
        xa = a["candidates"][rng.integers(len(a["candidates"]))]["center_x"]
        xb = b["candidates"][rng.integers(len(b["candidates"]))]["center_x"]
        sampling["sampled_candidate_pairs"] += 1
        slope = float((xb - xa) / (b["y"] - a["y"]))
        intercept = float(xa - slope * a["y"])
        matches = packed.support(slope, intercept, config["inlier_distance_px"])
        if len(matches) < config["minimum_rows"]:
            sampling["insufficient_initial_support"] += 1
            continue
        ys = np.array([observations["rows"][i]["y"] for i, _, _ in matches], float)
        if np.ptp(ys) < config["minimum_y_span"]:
            sampling["insufficient_initial_y_span"] += 1
            continue
        xs = np.array([
            observations["rows"][i]["candidates"][j]["center_x"] for i, j, _ in matches
        ])
        slope, intercept = np.linalg.lstsq(
            np.c_[ys, np.ones(len(ys))], xs, rcond=None
        )[0]
        matches = packed.support(float(slope), float(intercept), config["inlier_distance_px"])
        if require_refit_support:
            # 初筛过了不等于重新拟合后还合格。先挡住缩水的支持集，再排序去重；
            # 在最终池里删会太晚，不合格项可能已经挤掉旁边的有效解释。
            if len(matches) < config["minimum_rows"]:
                sampling["insufficient_refit_support"] += 1
                continue
            final_ys = np.array([observations["rows"][i]["y"] for i, _, _ in matches], float)
            if np.ptp(final_ys) < config["minimum_y_span"]:
                sampling["insufficient_refit_y_span"] += 1
                continue
        residuals = np.array([value for _, _, value in matches])
        widths, enclosed = [], []
        for row_index, candidate_index, _ in matches:
            candidates = observations["rows"][row_index]["candidates"]
            chosen = candidates[candidate_index]
            widths.append(chosen["width"])
            enclosed.append(any(
                other["width"] > chosen["width"] + 1
                and other["left_edge"]["x"] <= chosen["left_edge"]["x"]
                and other["right_edge"]["x"] >= chosen["right_edge"]["x"]
                for other_index, other in enumerate(candidates)
                if other_index != candidate_index
            ))
        # Keep these summaries and their order identical to the reference. This
        # control must not make a hypothesis look better while making it faster.
        hypotheses.append({
            "line": _as_line(slope, intercept),
            "slope": float(slope),
            "intercept": float(intercept),
            "support_rows": len(matches),
            "residual_median_px": float(np.median(residuals)),
            "width_median_px": float(np.median(widths)),
            "width_p10_px": float(np.quantile(widths, 0.1)),
            "width_p90_px": float(np.quantile(widths, 0.9)),
            "enclosing_wider_row_fraction": float(np.mean(enclosed)),
            "row_matches": matches,
        })
    sampling["hypotheses_before_deduplication"] = len(hypotheses)
    hypotheses.sort(key=lambda row: (-row["support_rows"], row["residual_median_px"]))
    kept = []
    ys = np.array([row["y"] for _, row in rows], float)
    for item in hypotheses:
        if any(
            np.median(np.abs(_line_x(item["line"], ys) - _line_x(old["line"], ys)))
            < config["deduplicate_separation_px"]
            for old in kept
        ):
            continue
        kept.append(item)
    result.update(
        candidates=kept,
        reason="fixed_sample_exhausted",
        deduplication={
            "input_count": len(hypotheses),
            "duplicate_count": len(hypotheses) - len(kept),
            "retained_count": len(kept),
        },
    )
    return result
