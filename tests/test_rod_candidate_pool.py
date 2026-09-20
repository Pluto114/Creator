"""Check the pool control against legacy sampling, rank, and exact row identity."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_candidate_pool import _PackedRows, enumerate_image_line_pool
from creator_eval.rod_multiview_candidates import _support, enumerate_image_lines

# Ranking and raw row/candidate identities must be identical, not merely close.
# Floating fields get a stated absolute tolerance for cross-platform LAPACK;
# both paths still call the same least-squares implementation in each test run.
FLOAT_ATOL = 1e-12


def candidate(center, width=4.0):
    return {
        "center_x": float(center), "width": float(width),
        "left_edge": {"x": float(center - width / 2)},
        "right_edge": {"x": float(center + width / 2)},
    }


def config(seed=9, **changes):
    return {
        "minimum_rows": 12, "minimum_y_span": 80.0,
        "inlier_distance_px": .6, "trials": 1024, "seed": seed,
        "deduplicate_separation_px": .75, "maximum_models": 8,
        **changes,
    }


def ragged_fixture():
    rng = np.random.default_rng(52)
    rows = []
    for index, y in enumerate(range(20, 301, 5)):
        values = []
        if index % 11:
            center = 35 + .07 * y + rng.normal(0, .09)
            values = [candidate(center, 2), candidate(center, 10)]
            if index % 4:
                values.append(candidate(78 - .03 * y + rng.normal(0, .11), 5))
            if index % 3:
                values.append(candidate(102 + .12 * y + rng.normal(0, .1), 3))
            values.append(candidate(rng.uniform(20, 180), rng.uniform(2, 8)))
        rows.append({"y": y, "guide_x": 60.0, "candidates": values})
    return {"rows": rows}


class RodCandidatePoolTests(unittest.TestCase):
    def assert_candidates_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            with self.subTest(candidate=index):
                self.assertEqual(set(got), set(wanted))
                self.assertEqual(got["support_rows"], wanted["support_rows"])
                got_matches, wanted_matches = got["row_matches"], wanted["row_matches"]
                self.assertEqual(
                    [(i, j) for i, j, _ in got_matches],
                    [(i, j) for i, j, _ in wanted_matches],
                )
                np.testing.assert_allclose(
                    [r for _, _, r in got_matches], [r for _, _, r in wanted_matches],
                    rtol=0, atol=FLOAT_ATOL,
                )
                for key in set(got) - {"row_matches", "support_rows"}:
                    np.testing.assert_allclose(got[key], wanted[key], rtol=0, atol=FLOAT_ATOL)

    def assert_legacy_pool(self, observations, settings):
        result = enumerate_image_line_pool(observations, settings)
        legacy = enumerate_image_lines(
            observations, {**settings, "maximum_models": settings["trials"]}
        )
        self.assert_candidates_equal(result["candidates"], legacy)
        return result

    def test_ragged_empty_rows_noise_and_random_seeds_match_every_legacy_field(self):
        observations = ragged_fixture()
        before = copy.deepcopy(observations)
        for seed in (0, 9, 19092027):
            with self.subTest(seed=seed):
                result = self.assert_legacy_pool(observations, config(seed))
                self.assertGreater(len(result["candidates"]), 1)
                for model in result["candidates"]:
                    for raw_row, raw_candidate, _ in model["row_matches"]:
                        self.assertTrue(observations["rows"][raw_row]["candidates"])
                        self.assertNotEqual(raw_candidate, 1)  # exact duplicate center loses the tie
        self.assertEqual(observations, before)

    def test_vectorized_support_preserves_first_tie_and_threshold_boundary(self):
        observations = {"rows": [
            {"y": 0, "candidates": []},
            {"y": 20, "candidates": [candidate(-1), candidate(1)]},
            {"y": 30, "candidates": [candidate(0), candidate(0), candidate(4)]},
            {"y": 40, "candidates": []},
            {"y": 50, "candidates": [candidate(1 + np.spacing(1.0))]},
        ]}
        rows = [(i, row) for i, row in enumerate(observations["rows"]) if row["candidates"]]
        packed = _PackedRows(rows)
        self.assertEqual(packed.support(0.0, 0.0, 1.0), [(1, 0, 1.0), (2, 0, 0.0)])
        for slope, intercept, threshold in ((0.0, 0.0, 1.0), (.125, -3.0, 2.0), (-.4, 7.0, .5)):
            self.assertEqual(packed.support(slope, intercept, threshold), _support(rows, slope, intercept, threshold))

    def test_exact_and_near_dedup_boundary_preserves_order(self):
        observations = {"rows": [
            {"y": y, "candidates": [candidate(60), candidate(63), candidate(72)]}
            for y in range(20, 261, 4)
        ]}
        for separation in (3.0, np.nextafter(3.0, 0), np.nextafter(3.0, np.inf)):
            with self.subTest(separation=separation):
                self.assert_legacy_pool(observations, config(5, deduplicate_separation_px=separation))

    def test_full_pool_and_cap_prefixes_share_exact_rank_without_new_random_draws(self):
        observations = {"rows": [
            {"y": y, "candidates": [candidate(20 + 9 * index) for index in range(20)]}
            for y in range(20, 301, 7)
        ]}
        settings = config(19092027, minimum_rows=10, minimum_y_span=120,
                          inlier_distance_px=.3, deduplicate_separation_px=1.0)
        result = self.assert_legacy_pool(observations, settings)
        self.assertGreaterEqual(len(result["candidates"]), 16)
        for cap in (8, 16):
            self.assert_candidates_equal(
                result["candidates"][:cap],
                enumerate_image_lines(observations, {**settings, "maximum_models": cap}),
            )
        self.assert_candidates_equal(result["candidates"][:8], result["candidates"][:16][:8])
        self.assert_candidates_equal(
            result["candidates"],
            enumerate_image_line_pool(observations, {**settings, "maximum_models": 1})["candidates"],
        )
        self.assertEqual(result["sampling"]["attempted_trials"], 1024)
        self.assertFalse(result["maximum_models_applied"])

    def test_sampling_and_dedup_counts_reconcile(self):
        result = enumerate_image_line_pool(ragged_fixture(), config())
        sampling, dedup = result["sampling"], result["deduplication"]
        self.assertEqual(sampling["attempted_trials"], sampling["row_pair_too_close"] + sampling["sampled_candidate_pairs"])
        self.assertEqual(
            sampling["sampled_candidate_pairs"],
            sampling["insufficient_initial_support"] + sampling["insufficient_initial_y_span"]
            + sampling["hypotheses_before_deduplication"],
        )
        self.assertEqual(dedup["input_count"], sampling["hypotheses_before_deduplication"])
        self.assertEqual(dedup["input_count"], dedup["duplicate_count"] + dedup["retained_count"])
        self.assertEqual(dedup["retained_count"], len(result["candidates"]))
        self.assertGreater(sampling["row_pair_too_close"], 0)
        self.assertGreater(dedup["duplicate_count"], 0)

    def test_insufficient_rows_or_span_preserves_legacy_empty_result(self):
        for observations in (
            {"rows": []},
            {"rows": [{"y": 20, "candidates": [candidate(60)]}]},
            {"rows": [{"y": y, "candidates": [candidate(60)]} for y in range(20, 55)]},
        ):
            result = self.assert_legacy_pool(observations, config())
            self.assertEqual(result["candidates"], [])
        empty = enumerate_image_line_pool({"rows": []}, config())
        self.assertEqual(empty["sampling"]["attempted_trials"], 0)
        self.assertEqual(empty["reason"], "too_few_rows_with_candidates")


if __name__ == "__main__":
    unittest.main()
