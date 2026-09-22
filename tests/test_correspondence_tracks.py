"""Check conflict, leakage and a deliberately wrong but cycle-consistent match."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.correspondence_tracks import build_tracks, pair_track_labels, split_tracks


def pair(a, b, ids):
    return dict(first_view=a, second_view=b, keypoint_ids=ids)


class TrackTests(unittest.TestCase):
    def test_complete_triangle_and_incomplete_chain(self):
        pairs = [pair(0, 1, [[1, 2], [8, 8]]), pair(1, 2, [[2, 3], [8, 8]]), pair(0, 2, [[1, 3]])]
        graph = build_tracks(pairs)
        self.assertEqual(graph["tracks"], [[(0, 1), (1, 2), (2, 3)]])
        self.assertEqual(graph["triangle_edge_count"], 3)
        self.assertEqual(graph["rejected_components"][0]["reason"], "missing_direct_cross_view_match")
        labels = pair_track_labels(pairs, graph, [True])
        np.testing.assert_array_equal(labels[0]["track_ids"], [0, -1])
        np.testing.assert_array_equal(labels[0]["validation"], [True, False])

    def test_view_conflict_rejects_entire_component(self):
        graph = build_tracks([pair(0, 1, [[0, 0]]), pair(1, 2, [[0, 0]]), pair(0, 2, [[1, 0]])])
        self.assertFalse(graph["tracks"])
        self.assertEqual(graph["rejected_components"][0]["reason"], "multiple_features_in_one_view")

    def test_many_to_one_is_invalid(self):
        with self.assertRaises(ValueError):
            build_tracks([pair(0, 1, [[0, 0], [1, 0]])])

    def test_repeated_texture_wrong_permutation_also_closes(self):
        # The IDs below represent different physical bricks. Cycles alone cannot
        # discover that: this passing graph is explicitly NOT a correctness test.
        graph = build_tracks([pair(0, 1, [[0, 1]]), pair(1, 2, [[1, 2]]), pair(0, 2, [[0, 2]])])
        self.assertEqual(len(graph["tracks"]), 1)

    def test_any_view_spatial_collision_keeps_tracks_in_one_split(self):
        tracks = [[(0, i), (1, i), (2, i)] for i in range(3)]
        features = [dict(xy=np.array([[1, 1], [80, 1], [180, 1]], float)),
                    dict(xy=np.array([[1, 1], [2, 2], [180, 1]], float)),
                    dict(xy=np.array([[1, 1], [80, 1], [81, 2]], float))]
        config = dict(cell_size_px=64, seed=0, validation_modulus=3, validation_residue=0)
        validation, report = split_tracks(tracks, features, config)
        self.assertEqual(len(set(validation)), 1)
        self.assertEqual(report["shared_cells"], 0)
        self.assertEqual(len(report["groups"]), 1)

    def test_empty_graph_and_split(self):
        graph = build_tracks([])
        self.assertEqual(graph["tracks"], [])
        validation, report = split_tracks([], [], dict(cell_size_px=64, seed=0, validation_modulus=3, validation_residue=0))
        self.assertEqual(len(validation), 0)
        self.assertEqual(report["shared_cells"], 0)


if __name__ == "__main__":
    unittest.main()
