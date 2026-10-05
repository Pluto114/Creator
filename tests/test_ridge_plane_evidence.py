"""Local plane model from training only; no scored inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import ridge_plane_evidence as reader  # noqa: E402

CONFIG = dict(voxel_size=.006)
LINE = np.array([[0., 0., .017], [0., 0., 1.217]])


def rows(offsets):
    t = .017+np.arange(97)*.0125
    points = np.concatenate([np.c_[np.full(len(t), x), np.full(len(t), y), t] for x, y in offsets])
    return points, np.tile(np.arange(97) % 2 == 0, len(offsets))


class PlaneEvidenceTests(unittest.TestCase):
    def test_seven_sparse_rows_confirm_a_plane(self):
        points, mask = rows([(x*.019, 0.) for x in range(-3, 4)])
        result = reader.analyze(points, mask, LINE, CONFIG)
        self.assertEqual(result["state"], "complete")
        self.assertTrue(result["planar_sheet"], result)

    def test_single_twin_and_noncoplanar_three_rods_do_not_form_sheet(self):
        for offsets in ([(0., 0.)], [(0., 0.), (.08, 0.)], [(0., 0.), (.08, 0.), (.039, .072)]):
            with self.subTest(offsets=offsets):
                points, mask = rows(offsets)
                self.assertFalse(reader.analyze(points, mask, LINE, CONFIG)["planar_sheet"])

    def test_short_side_branch_cannot_supply_persistent_sheet_width(self):
        points, mask = rows([(0., 0.)])
        branch = np.c_[np.linspace(0., .19, 31), np.zeros(31), np.full(31, .64)]
        result = reader.analyze(np.r_[points, branch], np.r_[mask, np.arange(31) % 2 == 0], LINE, CONFIG)
        self.assertFalse(result["planar_sheet"], result)

    def test_scatter_does_not_establish_a_plane(self):
        points, mask = rows([(0., 0.)])
        scatter = np.random.default_rng(618997).uniform([-.06, -.06, .017], [.06, .06, 1.217], (1600, 3))
        result = reader.analyze(np.r_[points, scatter], np.r_[mask, np.arange(1600) % 2 == 0], LINE, CONFIG)
        self.assertFalse(result["planar_sheet"], result)

    def test_validation_cannot_refit_training_plane(self):
        points, mask = rows([(x*.019, 0.) for x in range(-3, 4)])
        first = reader.analyze(points, mask, LINE, CONFIG)
        changed = points.copy()
        changed[~mask, 1] += .02
        second = reader.analyze(changed, mask, LINE, CONFIG)
        self.assertEqual(first["training_model"], second["training_model"])
        self.assertFalse(second["planar_sheet"])

    def test_order_duplicates_and_endpoint_reversal_preserve_model(self):
        points, mask = rows([(x*.019, 0.) for x in range(-3, 4)])
        first = reader.analyze(points, mask, LINE, CONFIG)
        second = reader.analyze(np.repeat(points[::-1], 2, axis=0), np.repeat(mask[::-1], 2), LINE[::-1], CONFIG)
        self.assertEqual(first, second)

    def test_budgets_and_invalid_inputs_are_not_rejections(self):
        points, mask = rows([(0., 0.)])
        for config, reason in ((dict(ridge_plane_maximum_points=3), "ridge_plane_point_budget_exceeded"),
                               (dict(ridge_plane_maximum_stations=3), "ridge_plane_station_budget_exceeded")):
            result = reader.analyze(points, mask, LINE, {**CONFIG, **config})
            self.assertEqual(result["state"], "unmeasurable")
            self.assertEqual(result["reason"], reason)
        with self.assertRaises(ValueError):
            reader.analyze(points, mask.astype(int), LINE, CONFIG)
        for config in ({"ridge_plane_unknown": 1}, {"ridge_plane_maximum_points": True},
                       {"ridge_plane_minimum_inlier_fraction": 2.}, {"ridge_plane_thickness_voxels": np.nan}):
            with self.assertRaises(ValueError):
                reader.policy(config)


if __name__ == "__main__":
    unittest.main()
