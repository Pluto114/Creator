"""Six focused checks for unpadded, same-chain distinct-view outer limits."""

import copy
import pickle
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import consensus_boundary_readout as reader  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_bracketed_occupancy_readout import BracketedOccupancyReadout  # noqa: E402
from creator_eval.rgb_chain_support import chain_support_mask  # noqa: E402
from creator_eval.rgb_consensus_boundary_readout import ConsensusBoundaryReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_anchored_boundary_readout import changed_context, line  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_stroke_selected_axis_readout import measured_context  # noqa: E402


def five_view_context_missing_marked_tail():
    frames, cameras = fixture()
    fifth, camera = copy.deepcopy(frames[1]), copy.deepcopy(cameras[1])
    fifth.update(view_id="4", rgb_sha256="b"*64)
    camera.update(view_id="4", world_to_camera_cv=np.c_[np.eye(3), [-.6, 0., 0.]].tolist())
    for row in fifth["observations"]["rows"]:
        for candidate in row["candidates"]:
            candidate["center_x"] -= 30.
            candidate["left_edge"]["x"] -= 30.
            candidate["right_edge"]["x"] -= 30.
    fifth["pool"]["candidates"][0]["line"] = [1., 0., -80.]
    fifth["pool_sha256"] = canonical_hash(fifth["pool"])
    frames.append(fifth)
    cameras.append(camera)
    for frame in frames:
        if frame["view_id"] == "0":
            for row in frame["observations"]["rows"]:
                if row["y"] >= 122:
                    row.update(status="unknown", candidates=[])
        frame["observation_sha256"] = canonical_hash(frame["observations"])
    return stroke.build_context(frames, cameras, measured_context()["association"]["anchors"])


class ConsensusBoundaryMechanisms(unittest.TestCase):
    def test_one_marked_view_missing_but_four_distinct_views_preserve_tail(self):
        context, source = five_view_context_missing_marked_tail(), line()
        before, original = pickle.dumps(context), source.copy()
        self.assertEqual(len(context["views"]), 5)
        result = reader.trim_consensus_envelope(source, context)
        np.testing.assert_array_equal(result["segments"], source)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(pickle.dumps(context), before)
        self.assertEqual(result["boundary_removed_length_m"], 0.)

    def test_only_three_view_tail_is_trimmed_not_extended(self):
        def change(frame, row):
            if frame["view_id"] == "0" and row["y"] >= 122:
                row.update(status="unknown", candidates=[])
        context, source = changed_context(change), line()
        self.assertTrue(chain_support_mask([[0., .9, 3.]], context)[0])
        result = reader.trim_consensus_envelope(source, context)
        self.assertEqual(len(result["segments"]), 1)
        np.testing.assert_array_equal(result["segments"][0, 0], source[0, 0])
        self.assertAlmostEqual(result["segments"][0, 1, 1], .645, places=8)
        self.assertGreater(result["boundary_removed_length_m"], .35)

    def test_admission_padding_is_not_measured_boundary_support(self):
        context = measured_context()
        source = np.array([[[0., -1., 3.], [.045, 1., 3.]]])
        self.assertTrue(chain_support_mask(source[0], context).all())
        result = reader.trim_consensus_envelope(source, context)
        np.testing.assert_array_equal(result["segments"][0, 0], source[0, 0])
        self.assertAlmostEqual(result["segments"][0, 1, 0], .03, places=10)
        self.assertAlmostEqual(result["segments"][0, 1, 1], 1./3, places=9)
        empty = reader.trim_consensus_envelope(line(x=.045), context)
        self.assertEqual(empty["segments"].shape, (0, 2, 3))

    def test_internal_unknown_and_existing_geometric_gap_are_not_changed(self):
        def change(frame, row):
            if row["y"] == 110:
                row.update(status="unknown", candidates=[])
        context = changed_context(change)
        source = np.concatenate((line(-1., .1), line(.2, 1.)))
        self.assertFalse(chain_support_mask([[0., .3, 3.]], context)[0])
        result = reader.trim_consensus_envelope(source, context)
        np.testing.assert_array_equal(result["segments"], source)
        self.assertEqual(result["boundary_removed_length_m"], 0.)

    def test_chains_and_candidate_hits_cannot_manufacture_distinct_view_votes(self):
        memberships = [np.array([[i < 2, i >= 2]], dtype=bool) for i in range(5)]
        assignments = [[(0,)]*5, [(1,)]*5]
        context = dict(association=dict(assignments=assignments),
                       views=[dict(view_id=str(i)) for i in range(5)])
        with patch.object(reader, "point_memberships", return_value=memberships):
            self.assertFalse(reader.boundary_support_mask([[0., 0., 3.]], context)[0])
        repeated_hits = [np.array([[i < 3, i < 3]], dtype=bool) for i in range(4)]
        context = dict(association=dict(assignments=[[(0, 1)]*4]),
                       views=[dict(view_id=str(i)) for i in range(4)])
        with patch.object(reader, "point_memberships", return_value=repeated_hits):
            self.assertFalse(reader.boundary_support_mask([[0., 0., 3.]], context)[0])
        context["views"][-1]["view_id"] = "0"
        with self.assertRaises(ValueError):
            reader.boundary_support_mask([[0., 0., 3.]], context)

    def test_fixture_base_curve_same_coordinates_and_selected_model_unchanged(self):
        context, source = measured_context(), line(x=.015)
        points = sample_segments(source, .01, 1000)
        policy, empty = dict(voxel_size=.02, origin=[0., 0., 0.]), np.empty((0, 2, 3))
        old = BracketedOccupancyReadout(points, context)(empty, policy)
        base = ConsensusBoundaryReadout(points, context)(empty, policy)
        curve = ConsensusBoundaryReadout(np.empty((0, 3)), context)(source, policy)
        self.assertGreater(base["components"], 0)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["selected_candidate"], old["selected_candidate"])
        self.assertEqual(base["selected_candidate"], curve["selected_candidate"])
        chosen = base["selected_candidate"]
        self.assertEqual(canonical_hash(base["readouts"][chosen]["graph"]["model"]),
                         canonical_hash(old["readouts"][chosen]["graph"]["model"]))
        self.assertEqual(canonical_hash(base["readouts"]), canonical_hash(curve["readouts"]))
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])
        self.assertFalse(base["boundary_requires_all_marked_views"])
        self.assertEqual(base["boundary_minimum_views"], 4)
        self.assertEqual(base["boundary_horizontal_padding_px"], 0.)


if __name__ == "__main__":
    unittest.main()
