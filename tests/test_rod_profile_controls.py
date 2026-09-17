"""Known analytic answers, independent of the project's images and real truth."""

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.rod_profile_controls import (  # noqa: E402 - repository-local source root
    evaluate_profile_method,
    reject_disagreeing_pair_centers,
    render_profile_fixture,
    run_profile_methods,
)


def protocol():
    return json.loads((ROOT / "configs/rod_profile_controls_v1.json").read_text(encoding="utf-8"))


class ProfileControlTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol()

    def render(self, case):
        return render_profile_fixture(case, self.config["generator"], seed=123)

    def test_physical_support_is_independent_of_illumination(self):
        case = self.config["cases"][0]
        image, truth = self.render(case)
        self.assertTrue(np.all(image[80, 59:71] == 220))
        self.assertTrue(np.all(image[80, 58] == 32))
        self.assertTrue(np.all(image[80, 71] == 32))
        changed = copy.deepcopy(case)
        changed["rods"][0]["profile"] = {
            "kind": "linear_knots",
            "positions": [0, 1],
            "values": [40, 220],
        }
        illuminated, other_truth = self.render(changed)
        self.assertFalse(np.array_equal(image, illuminated))
        self.assertEqual(truth["rows"], other_truth["rows"])
        self.assertEqual(truth["rows"][80]["objects"][0]["center_x"], 64.5)

    def test_subpixel_area_integration_does_not_enlarge_physical_width(self):
        image, truth = self.render(self.config["cases"][14])
        # 0.75px物体只贡献它覆盖的像素面积；亮度扩散不是几何宽度扩张。
        self.assertAlmostEqual(truth["rows"][80]["objects"][0]["width"], 0.75)
        self.assertAlmostEqual(
            float(np.sum(image[80, :, 0].astype(float) - 32)), 0.75 * (220 - 32), delta=1
        )

    def test_texture_inside_true_gap_does_not_create_truth_rod(self):
        image, truth = self.render(self.config["cases"][18])
        self.assertEqual(truth["rows"][80]["objects"], [])
        self.assertTrue(np.any(image[80, :, 0] > 32))
        self.assertEqual(len(truth["rows"][50]["objects"]), 1)

    def test_noise_and_rgb_channels_are_deterministic(self):
        case = self.config["cases"][21]
        first, _ = self.render(case)
        second, _ = self.render(case)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[..., 0], first[..., 2])
        different, _ = render_profile_fixture(case, self.config["generator"], seed=124)
        self.assertFalse(np.array_equal(first, different))

    def test_disagreement_rejects_whole_row_without_changing_original(self):
        raw = {
            "rows": [
                {
                    "y": 2,
                    "status": "ambiguous",
                    "reason": "multiple",
                    "candidates": [{"center_x": 3}, {"center_x": 7}],
                }
            ]
        }
        controlled = reject_disagreeing_pair_centers(raw, 1)
        self.assertEqual(len(raw["rows"][0]["candidates"]), 2)
        self.assertEqual(controlled["rows"][0]["status"], "unknown")
        self.assertEqual(controlled["rows"][0]["candidates"], [])
        self.assertEqual(len(controlled["rows"][0]["candidates_before_control"]), 2)

    def test_unique_internal_highlight_survives_both_methods_and_is_wrong(self):
        image, truth = self.render(self.config["cases"][8])
        raw, methods = run_profile_methods(image, self.config["guide_xyxy"], self.config["method"])
        self.assertTrue(all(len(row["candidates"]) == 1 for row in raw["rows"]))
        for prediction in methods.values():
            score, _ = evaluate_profile_method(prediction, truth, self.config["evaluation"])
            self.assertTrue(score["usable"])
            self.assertEqual(score["target_row_coverage"], 1)
            self.assertAlmostEqual(score["selected_center_error_px"]["median"], 4)
            self.assertEqual(score["accurate_target_row_coverage"]["1"], 0)
            self.assertEqual(score["center_disagreement_rejected_rows"], 0)

    def test_resolved_flat_bright_and_dark_rods_have_correct_centers(self):
        for case in self.config["cases"][:2]:
            image, truth = self.render(case)
            _, methods = run_profile_methods(
                image, self.config["guide_xyxy"], self.config["method"]
            )
            for prediction in methods.values():
                score, _ = evaluate_profile_method(prediction, truth, self.config["evaluation"])
                self.assertEqual(score["target_row_coverage"], 1)
                self.assertEqual(score["selected_center_error_px"]["max"], 0)
                self.assertEqual(score["selected_maximum_boundary_error_px"]["max"], 0)

    def test_true_flat_gap_is_not_filled_by_fitted_infinite_line(self):
        image, truth = self.render(self.config["cases"][17])
        _, methods = run_profile_methods(image, self.config["guide_xyxy"], self.config["method"])
        for prediction in methods.values():
            score, details = evaluate_profile_method(prediction, truth, self.config["evaluation"])
            self.assertTrue(score["usable"])
            self.assertEqual(score["gap_false_acceptance_fraction"], 0)
            self.assertTrue(all(row["selected"] is None for row in details if row["in_gap"]))

    def test_low_contrast_rod_can_produce_false_flat_absence(self):
        image, truth = self.render(self.config["cases"][13])
        _, methods = run_profile_methods(image, self.config["guide_xyxy"], self.config["method"])
        for prediction in methods.values():
            score, _ = evaluate_profile_method(prediction, truth, self.config["evaluation"])
            self.assertEqual(score["target_row_coverage"], 0)
            self.assertEqual(score["false_absence_fraction"], 1)

    def test_evaluator_scores_selected_pair_not_nearest_truth_pair(self):
        _, truth = self.render(self.config["cases"][0])
        candidates = [
            {"center_x": 60.5, "width": 4, "left_edge_x": 58.5, "right_edge_x": 62.5},
            {"center_x": 64.5, "width": 12, "left_edge_x": 58.5, "right_edge_x": 70.5},
        ]
        prediction = {
            "rows": [
                {
                    "y": 80,
                    "status": "ambiguous",
                    "candidates": candidates,
                    "pair_center_control": None,
                }
            ],
            "fitted": {
                "state": "fitted",
                "reason": "test",
                "usable": True,
                "row_matches": [{"row_index": 0, "candidate_index": 0}],
            },
        }
        score, _ = evaluate_profile_method(prediction, truth, self.config["evaluation"])
        self.assertEqual(score["selected_center_error_px"]["median"], 4)
        self.assertEqual(score["selected_maximum_boundary_error_px"]["median"], 8)
        prediction["fitted"]["usable"] = False
        prediction["fitted"]["state"] = "ambiguous"
        rejected, _ = evaluate_profile_method(prediction, truth, self.config["evaluation"])
        self.assertEqual(rejected["selected_rows"], 0)
        self.assertEqual(rejected["target_row_coverage"], 0)
        self.assertIsNone(rejected["selected_center_error_px"]["median"])

    def test_ambiguous_identity_never_assigned_to_closest_object(self):
        _, truth = self.render(self.config["cases"][19])
        prediction = {
            "rows": [
                {
                    "y": 80,
                    "status": "observed",
                    "pair_center_control": None,
                    "candidates": [
                        {"center_x": 58.5, "width": 5, "left_edge_x": 56, "right_edge_x": 61}
                    ],
                }
            ],
            "fitted": {
                "state": "fitted",
                "reason": "test",
                "usable": True,
                "row_matches": [{"row_index": 0, "candidate_index": 0}],
            },
        }
        score, _ = evaluate_profile_method(prediction, truth, self.config["evaluation"])
        self.assertIsNone(score["selected_center_error_px"]["median"])
        self.assertIsNone(score["target_row_coverage"])
        self.assertEqual(score["unsupported_identity_choice_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
