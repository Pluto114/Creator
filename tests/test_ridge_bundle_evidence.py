"""Local ridge-bundle model from training only; no scored inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import ridge_bundle_evidence as reader  # noqa: E402

CONFIG = dict(voxel_size=.006)
Z0, Z1 = .017, 1.217
RADIUS = .04

FULL = [(RADIUS * np.cos(a), RADIUS * np.sin(a))
        for a in np.linspace(0., 2. * np.pi, 4, endpoint=False)]
HALF = [(RADIUS * np.cos(a), RADIUS * np.sin(a))
        for a in np.linspace(0., np.pi, 4)]
TRIANGLE = [(0., 0.), (.08, 0.), (.039, .072)]


def rows(offsets):
    t = Z0 + np.arange(97) * .0125
    points = np.concatenate([np.c_[np.full(len(t), x), np.full(len(t), y), t] for x, y in offsets])
    return points, np.tile(np.arange(97) % 2 == 0, len(offsets))


def segment_for(offsets):
    x, y = offsets[0]
    return np.array([[x, y, Z0], [x, y, Z1]])


def train_only_rows(train_offsets, val_offsets):
    t = Z0 + np.arange(97) * .0125

    def col(x, y):
        return np.c_[np.full(len(t), x), np.full(len(t), y), t]

    points = np.concatenate([col(x, y) for x, y in train_offsets + val_offsets])
    mask = np.concatenate([np.ones(len(t), bool) for _ in train_offsets]
                          + [np.zeros(len(t), bool) for _ in val_offsets])
    return points, mask


def ring(z=(Z0 + Z1) / 2., n=48):
    a = np.linspace(0., 2. * np.pi, n, endpoint=False)
    return np.c_[RADIUS * np.cos(a), RADIUS * np.sin(a), np.full(n, z)]


class RidgeBundleEvidenceTests(unittest.TestCase):
    def test_four_full_circle_rows_are_ambiguous(self):
        points, mask = rows(FULL)
        result = reader.analyze(points, mask, segment_for(FULL), CONFIG)
        self.assertTrue(result["parallel_array"], result)
        self.assertTrue(result["training_only_proposals"], result)
        self.assertFalse(result["validation_refits_model"], result)

    def test_four_half_arc_rows_are_ambiguous(self):
        points, mask = rows(HALF)
        result = reader.analyze(points, mask, segment_for(HALF), CONFIG)
        self.assertTrue(result["parallel_array"], result)

    def test_few_noncollinear_rows_are_not_rejected_by_circular_evidence(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                offsets = TRIANGLE[:count]
                points, mask = rows(offsets)
                circular = ring()
                result = reader.analyze(np.r_[points, circular],
                                        np.r_[mask, np.arange(len(circular)) % 2 == 0],
                                        segment_for(offsets), CONFIG)
                self.assertFalse(result["parallel_array"], result)

    def test_validation_rows_cannot_supply_the_missing_fourth_train_row(self):
        points_3, mask_3 = train_only_rows(FULL[:3], [])
        points_4, mask_4 = train_only_rows(FULL[:3], FULL[3:])
        segment = segment_for(FULL)
        first = reader.analyze(points_3, mask_3, segment, CONFIG)
        second = reader.analyze(points_4, mask_4, segment, CONFIG)
        self.assertEqual(first["training_model"], second["training_model"])
        self.assertFalse(second["validation_refits_model"], second)
        self.assertFalse(first["parallel_array"], first)
        self.assertFalse(second["parallel_array"], second)

    def test_order_duplicates_and_endpoint_reversal_preserve_model_and_conclusion(self):
        points, mask = rows(FULL)
        segment = segment_for(FULL)
        first = reader.analyze(points, mask, segment, CONFIG)
        second = reader.analyze(np.repeat(points[::-1], 2, axis=0),
                                np.repeat(mask[::-1], 2),
                                segment[::-1], CONFIG)
        self.assertEqual(first["state"], second["state"])
        self.assertEqual(first["parallel_array"], second["parallel_array"])
        self.assertEqual(first["training_model"], second["training_model"])

    def test_point_budget_returns_unmeasurable_not_rejection(self):
        points, mask = rows(FULL)
        config = {**CONFIG, "ridge_array_maximum_points": 3}
        reader.policy(config)  # budget is an accepted parameter
        result = reader.analyze(points, mask, segment_for(FULL), config)
        self.assertEqual(result["state"], "unmeasurable")


if __name__ == "__main__":
    unittest.main()
