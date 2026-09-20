"""Regression fixtures for support lost during least-squares refitting."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_candidate_pool import enumerate_image_line_pool


def observation(rows):
    return {"rows": [{"y": y, "candidates": [
        {"center_x": x, "width": 4.0, "left_edge": {"x": x - 2}, "right_edge": {"x": x + 2}}
        for x in centers]} for y, centers in rows]}


def violations(pool, raw, config):
    counts = {"rows": 0, "span": 0}
    for h in pool["candidates"]:
        ys = [raw["rows"][i]["y"] for i, _, _ in h["row_matches"]]
        counts["rows"] += len(ys) < config["minimum_rows"]
        counts["span"] += not ys or np.ptp(ys) < config["minimum_y_span"]
    return counts


class RefitSupportTests(unittest.TestCase):
    def test_real_rgb_regressions_remove_short_final_support_without_emptying_pool(self):
        path = Path(__file__).parent / "fixtures/rod_refit_support_centers.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))
        for frame, failure in zip(fixture["frames"], ("span", "rows")):
            with self.subTest(view=frame["view_id"]):
                raw, config = observation(frame["rows"]), fixture["config"]
                old = enumerate_image_line_pool(raw, config)
                fixed = enumerate_image_line_pool(raw, config, require_refit_support=True)
                self.assertGreater(violations(old, raw, config)[failure], 0)
                self.assertEqual(violations(fixed, raw, config), {"rows": 0, "span": 0})
                self.assertGreater(len(fixed["candidates"]), 0)
                # The experiment changes validation, not the seed or sampled pairs.
                for key in ("seed", "requested_trials", "attempted_trials", "sampled_candidate_pairs",
                            "row_pair_too_close", "insufficient_initial_support", "insufficient_initial_y_span"):
                    self.assertEqual(old["sampling"][key], fixed["sampling"][key])
                self.assertGreater(fixed["sampling"]["insufficient_refit_support"] +
                                   fixed["sampling"]["insufficient_refit_y_span"], 0)

    def test_equality_at_both_thresholds_is_valid_and_missing_rows_are_not_filled(self):
        raw = observation([[y, [100.0]] for y in (0, 20, 40, 60)])
        config = {"seed": 3, "trials": 30, "minimum_rows": 4, "minimum_y_span": 60,
                  "inlier_distance_px": 1.0, "deduplicate_separation_px": .75}
        pool = enumerate_image_line_pool(raw, config, require_refit_support=True)
        self.assertEqual(len(pool["candidates"]), 1)
        self.assertEqual([i for i, _, _ in pool["candidates"][0]["row_matches"]], [0, 1, 2, 3])
        self.assertEqual(pool["sampling"]["insufficient_refit_support"], 0)
        self.assertEqual(pool["sampling"]["insufficient_refit_y_span"], 0)

    def test_empty_observations_remain_an_explicit_empty_pool(self):
        config = {"seed": 3, "trials": 30, "minimum_rows": 4, "minimum_y_span": 60}
        pool = enumerate_image_line_pool({"rows": []}, config, require_refit_support=True)
        self.assertEqual(pool["candidates"], [])
        self.assertTrue(pool["refit_support_validated"])
        self.assertEqual(pool["sampling"]["attempted_trials"], 0)


if __name__ == "__main__":
    unittest.main()
