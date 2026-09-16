"""Analytic image checks for observations; no project images or truth are read."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_observations import (
    extract_rod_observations,
    fit_robust_image_line,
)


def rgb(gray):
    return np.repeat(np.asarray(gray, dtype=float)[..., None], 3, axis=2)


def scene(background=30.0):
    return np.full((144, 100), background)


GUIDE = [[49.5, 12], [49.5, 132]]


class RodObservationTests(unittest.TestCase):
    def test_bright_and_dark_plateau_centers_do_not_follow_first_bright_pixel(self):
        for background, foreground, polarity in ((30, 220, "bright"), (220, 30, "dark")):
            image = scene(background)
            image[:, 47:53] = foreground
            output = extract_rod_observations(rgb(image), GUIDE)
            for row in output["rows"]:
                self.assertEqual(row["status"], "observed")
                candidate = row["candidates"][0]
                self.assertAlmostEqual(candidate["center_x"], 49.5)
                self.assertAlmostEqual(candidate["width"], 6)
                self.assertEqual(candidate["polarity"], polarity)
                self.assertAlmostEqual(candidate["continuity"]["minimum_support_fraction"], 1)
            fitted = fit_robust_image_line(output)
            self.assertEqual(fitted["state"], "fitted")
            self.assertAlmostEqual(fitted["intercept"], 49.5)
            self.assertAlmostEqual(fitted["slope"], 0)
        # 平顶亮杆的argmax会选左边第一个亮像素；双边缘中点没有这个2.5px偏差。
        first_peak = int(
            np.argmax(
                (scene() + (np.arange(100)[None] >= 47) * (np.arange(100)[None] < 53) * 190)[60]
            )
        )
        self.assertEqual(first_peak, 47)
        self.assertGreater(abs(first_peak - 49.5), 2)

    def test_sloped_resolved_rod_follows_both_edges(self):
        image = scene()
        for y in range(len(image)):
            left = 38 + int(np.floor(y / 8))
            image[y, left : left + 6] = 220
        guide = [[38 + 12 / 8 + 2, 12], [38 + 132 / 8 + 2, 132]]
        output = extract_rod_observations(rgb(image), guide)
        observed = [row for row in output["rows"] if row["candidates"]]
        self.assertEqual(len(observed), len(output["rows"]))
        for row in observed:
            expected = 38 + int(np.floor(row["y"] / 8)) + 2.5
            self.assertAlmostEqual(row["candidates"][0]["center_x"], expected)
        fitted = fit_robust_image_line(output)
        self.assertEqual(fitted["state"], "fitted")
        self.assertAlmostEqual(fitted["slope"], 0.125, delta=0.002)
        self.assertLess(fitted["residual_p95_px"], 0.3)

    def test_true_gap_stays_absent_after_fitting_a_line(self):
        image = scene()
        image[:, 47:53] = 220
        image[56:88] = 30
        observations = extract_rod_observations(rgb(image), GUIDE)
        original = copy.deepcopy(observations)
        gap_rows = [index for index, row in enumerate(observations["rows"]) if 60 <= row["y"] <= 84]
        self.assertTrue(all(observations["rows"][i]["status"] == "absent" for i in gap_rows))
        fitted = fit_robust_image_line(observations)
        self.assertEqual(fitted["state"], "fitted")
        self.assertFalse(set(gap_rows) & set(fitted["inlier_rows"]))
        self.assertEqual(observations, original)
        evidence = observations["rows"][gap_rows[0]]["absence_evidence"]
        self.assertTrue(evidence["flat_background_like"])
        self.assertIn("occlusion", evidence["limitation"])

    def test_horizontal_clutter_does_not_make_persistent_vertical_edges(self):
        image = scene()
        image[64, 43:57] = 220
        image[104] = 220
        observations = extract_rod_observations(rgb(image), GUIDE)
        self.assertTrue(all(not row["candidates"] for row in observations["rows"]))
        self.assertEqual(fit_robust_image_line(observations)["state"], "insufficient_support")
        clutter = next(row for row in observations["rows"] if row["y"] == 64)
        self.assertEqual(clutter["status"], "unknown")
        self.assertGreater(
            clutter["rejected_pair_counts"].get("insufficient_persistent_vertical_edges", 0), 0
        )

    def test_one_pixel_rod_and_single_edge_are_unknown(self):
        narrow = scene()
        narrow[:, 50] = 220
        observations = extract_rod_observations(rgb(narrow), GUIDE)
        self.assertTrue(all(row["status"] == "unknown" for row in observations["rows"]))
        self.assertTrue(all(not row["candidates"] for row in observations["rows"]))
        edge = scene()
        edge[:, 50:] = 220
        observations = extract_rod_observations(rgb(edge), GUIDE)
        self.assertTrue(all(row["status"] == "unknown" for row in observations["rows"]))

    def test_competing_rods_are_preserved_and_fit_is_ambiguous(self):
        image = scene()
        image[:, 40:46] = 220
        image[:, 55:61] = 220
        observations = extract_rod_observations(rgb(image), GUIDE)
        self.assertTrue(all(row["status"] == "ambiguous" for row in observations["rows"]))
        for row in observations["rows"]:
            centers = [value["center_x"] for value in row["candidates"]]
            self.assertIn(42.5, centers)
            self.assertIn(57.5, centers)
        result = fit_robust_image_line(observations)
        self.assertEqual(result["state"], "ambiguous")
        self.assertFalse(result["usable"])
        self.assertTrue(result["alternative_models"])

    def test_low_noise_flat_background_and_low_snr_are_not_rod_evidence(self):
        rng = np.random.default_rng(100)
        smooth = scene() + rng.normal(0, 0.5, (144, 100))
        output = extract_rod_observations(rgb(smooth), GUIDE)
        self.assertTrue(all(row["status"] == "absent" for row in output["rows"]))
        noisy = scene() + rng.normal(0, 4, (144, 100))
        output = extract_rod_observations(rgb(noisy), GUIDE)
        # 随机噪声偶尔会凑出一行双边缘；不能把它提升为一根杆，也不能当平坦反证。
        self.assertTrue(all(row["status"] != "absent" for row in output["rows"]))
        self.assertEqual(fit_robust_image_line(output)["state"], "insufficient_support")

    def test_image_boundary_is_unknown_not_absence(self):
        output = extract_rod_observations(rgb(scene()), [[5, 0], [5, 120]])
        self.assertTrue(all(row["status"] == "unknown" for row in output["rows"]))
        self.assertTrue(all(row["absence_evidence"] is None for row in output["rows"]))

    def test_line_fit_is_robust_deterministic_and_counts_each_row_once(self):
        rows = []
        for index, y in enumerate(range(12, 133, 4)):
            center = 0.05 * y + 45
            candidates = (
                [{"center_x": center + 15}] * 50 if index % 5 == 0 else [{"center_x": center}]
            )
            rows.append(
                {
                    "y": y,
                    "status": "ambiguous" if index % 5 == 0 else "observed",
                    "candidates": candidates,
                }
            )
        observations = {"rows": rows}
        first = fit_robust_image_line(observations)
        self.assertEqual(first["state"], "fitted")
        self.assertAlmostEqual(first["slope"], 0.05)
        self.assertAlmostEqual(first["intercept"], 45)
        self.assertEqual(first, fit_robust_image_line(observations))
        self.assertEqual(len(first["inlier_rows"]), len(set(first["inlier_rows"])))
        self.assertTrue(all(i % 5 != 0 for i in first["inlier_rows"]))
        singleton = fit_robust_image_line(
            {"rows": [{"y": 60, "status": "ambiguous", "candidates": [{"center_x": 50}] * 1000}]}
        )
        self.assertEqual(singleton["state"], "insufficient_support")

    def test_line_span_requirement_is_not_replaced_by_sample_density(self):
        rows = [{"y": y, "status": "observed", "candidates": [{"center_x": 50}]} for y in range(20)]
        result = fit_robust_image_line({"rows": rows})
        self.assertEqual(result["reason"], "insufficient_y_span")
        with self.assertRaisesRegex(ValueError, "unique finite y"):
            fit_robust_image_line({"rows": rows + rows})


if __name__ == "__main__":
    unittest.main()
