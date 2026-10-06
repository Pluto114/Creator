"""Protocol boundaries for conventional point-fit controls, without pack IO."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments/src"))
from creator_eval import fixture_naive_controls as control  # noqa: E402


def data():
    x = np.linspace(0., 1., 30)
    return np.c_[x, np.zeros((30, 2))], np.c_[np.arange(30)%3, np.arange(30), np.zeros(30)].astype(np.uint32)


class NaiveFitContracts(unittest.TestCase):
    def test_frozen_conventional_parameters_have_no_readout_scale(self):
        self.assertEqual(control.DEFAULTS, dict(distance_camera_span_fraction=.005,
            minimum_pair_extent_camera_span_fraction=.03, minimum_inliers=12, minimum_views=3, trials=256, seed=0))
        points, ids = data()
        for result in control.fit_controls(points, ids, 1., np.ones(30, bool)):
            self.assertFalse(result["readout_scale_used"])
            self.assertFalse(result["gt_read"])

    def test_nonselected_outlier_never_enters_either_fit(self):
        points, ids = data()
        points = np.r_[points, [[100., 100., 100.]]]
        ids = np.r_[ids, np.array([[4, 31, 0]], np.uint32)]
        actual = control.fit_controls(points, ids, 1., np.r_[np.ones(30, bool), False])
        for result in actual:
            self.assertEqual(result["input_support_count"], 30)
            self.assertLessEqual(float(np.abs(result["segments"]).max()), 1.+1e-12)

    def test_duplicate_source_ids_are_not_extra_evidence(self):
        points, ids = data()
        ids[1] = ids[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            control.fit_controls(points, ids, 1., np.ones(30, bool))

    def test_ransac_requires_three_source_views_not_three_copies(self):
        points, ids = data()
        ids[:, 0] = 0
        tls, ransac = control.fit_controls(points, ids, 1., np.ones(30, bool))
        self.assertEqual(len(tls["segments"]), 1)
        self.assertEqual(len(ransac["segments"]), 0)
        self.assertIn("distinct_views", ransac["reason"])

    def test_invalid_input_is_not_relabelled_empty_success(self):
        points, ids = data()
        for span in (True, 0., -1., np.nan, np.inf):
            with self.subTest(span=span), self.assertRaises(ValueError):
                control.fit_controls(points, ids, span, np.ones(30, bool))
        for other, mask in ((ids.astype(float), np.ones(30, bool)), (ids, np.ones(30, int)),
                            (ids, np.ones(29, bool))):
            with self.assertRaises(ValueError):
                control.fit_controls(points, other, 1., mask)

    def test_unexpected_fit_error_propagates(self):
        points, ids = data()
        with patch.object(control, "fit_line_tls", side_effect=RuntimeError("unexpected")):
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                control.fit_controls(points, ids, 1., np.ones(30, bool))


if __name__ == "__main__":
    unittest.main()
