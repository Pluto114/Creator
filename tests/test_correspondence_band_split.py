"""A positive dense image split and boundary exclusions independent of fitting."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.correspondence_band_split import band_split


class BandSplitTests(unittest.TestCase):
    def test_dense_tracks_keep_train_validation_and_guard_independent(self):
        features = [dict(xy=np.array([[20, 70], [30, 220], [40, 150], [50, 90]], float)) for _ in range(3)]
        features[2]["xy"][3, 1] = 210
        tracks = [[(v, i) for v in range(3)] for i in range(4)]
        labels, report = band_split(tracks, features, dict(cell_size_px=64, validation_max_cell_y=1, train_min_cell_y=3))
        np.testing.assert_array_equal(labels, ["validation", "train", "excluded", "excluded"])
        self.assertEqual(report["shared_cells"], 0)

    def test_must_leave_a_guard_row(self):
        with self.assertRaises(ValueError):
            band_split([], [], dict(cell_size_px=64, validation_max_cell_y=1, train_min_cell_y=2))


if __name__ == "__main__":
    unittest.main()
