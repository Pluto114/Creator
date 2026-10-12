"""Normal-only contracts for the common actual-support competition adapter."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments/src"))
from creator_eval import foreground_consensus_readout as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.line_controls import point_to_segments_distance  # noqa: E402
from creator_eval.rgb_foreground_consensus_axis_readout import (
    ForegroundConsensusAxisReadout,  # noqa: E402
)
from test_chain_partition_readout import line_points, mechanism_context  # noqa: E402


class ForegroundConsensusContracts(unittest.TestCase):
    def test_identical_coordinates_are_fair_between_base_and_curve(self):
        segments = np.array([[[-.3, -1., 3.], [-.3, 1., 3.]], [[.3, -1., 3.], [.3, 1., 3.]]])
        points = sample_segments(segments, .01, 2000000)
        context = mechanism_context()
        policy = dict(voxel_size=.02, origin=[0., 0., 0.])
        base = ForegroundConsensusAxisReadout(points, context)(np.empty((0, 2, 3)), policy)
        curve = ForegroundConsensusAxisReadout(np.empty((0, 3)), context)(segments, policy)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["unique_count"], curve["unique_count"])
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])
        self.assertEqual(base["competition_order"], curve["competition_order"])
        self.assertFalse(base["required_support_views_applied"])
        self.assertTrue(base["seed_requires_all_annotated_views"])
        self.assertFalse(base["target_identity_confirmed"])

    def test_joint_translation_of_points_camera_and_voxel_origin(self):
        points = np.r_[line_points(-.25), line_points(.25)]
        shift = np.array([8., -4., 16.])
        before = reader.readout(points, .03125, [0., 0., 0.], mechanism_context((-.25, .25)))
        after = reader.readout(points+shift, .03125, shift, mechanism_context((-.25, .25), shift))
        np.testing.assert_allclose(after["segments"]-shift, before["segments"], atol=1e-11)
        self.assertEqual(before["components"], after["components"])

    def test_actual_axis_not_snapped_to_image_proposal(self):
        values = line_points(.006)
        graph = reader.readout(values, .02, [0., 0., 0.], mechanism_context((0.,)))
        self.assertEqual(graph["components"], 1)
        np.testing.assert_allclose(graph["segments"][..., 0], .006, atol=1e-12)
        np.testing.assert_array_equal(values, line_points(.006))

    def test_degenerate_local_seed_does_not_veto_valid_chain(self):
        values = np.r_[line_points(-.3), line_points(.3, [-.001, .001])]
        graph = reader.readout(values, .02, [0., 0., 0.], mechanism_context())
        self.assertEqual(graph["state"], "complete")
        self.assertEqual(graph["components"], 1)
        np.testing.assert_allclose(graph["segments"][..., 0], -.3, atol=1e-12)
        self.assertEqual(sum(r["expansion"]["reason"] is not None for r in graph["readouts"]), 1)

    def test_budget_failures_do_not_publish_partial_output(self):
        values = np.r_[line_points(-.3), line_points(.3)]
        for field, limit in (("maximum_points", 1), ("maximum_distance_checks", 1),
                             ("maximum_output_segments", 1)):
            with self.subTest(field=field), patch.dict(reader.DEFAULTS, {field: limit}):
                graph = reader.readout(values, .02, [0., 0., 0.], mechanism_context())
                self.assertEqual(graph["state"], "unmeasurable")
                self.assertEqual(len(graph["segments"]), 0)

    def test_incomplete_chain_search_cannot_claim_consensus(self):
        context = mechanism_context()
        context["association"]["fallback_to_raw_union"] = True
        graph = reader.readout(line_points(-.3), .02, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "unmeasurable")
        self.assertEqual(len(graph["segments"]), 0)

    def test_assignment_multiplicity_does_not_change_output(self):
        context = mechanism_context()
        repeated = copy.deepcopy(context)
        rows = repeated["association"]["assignments"]
        repeated["association"]["assignments"] = list(reversed(rows+rows))
        values = np.r_[line_points(-.3), line_points(.3)]
        first = reader.readout(values, .02, [0., 0., 0.], context)
        second = reader.readout(values, .02, [0., 0., 0.], repeated)
        np.testing.assert_array_equal(first["segments"], second["segments"])
        self.assertEqual(first["competition_order"], second["competition_order"])
        self.assertEqual([r["assignment_count"] for r in first["readouts"]],
                         [r["assignment_count"] for r in second["readouts"]])

    def test_retained_group_geometry_has_continuous_resolution_separation(self):
        # The two independently sampled curves are >4v apart. Dense external
        # samples check the stated certificate, not only fitter grid vertices.
        graph = reader.readout(np.r_[line_points(-.3), line_points(.3)],
            .02, [0., 0., 0.], mechanism_context())
        old = np.empty((0, 2, 3))
        for index in graph["retained_groups"]:
            segments = graph["readouts"][index]["final_graph"]["segments"]
            if len(old):
                distance = point_to_segments_distance(sample_segments(segments, .0001, 2000000), old)
                self.assertGreater(float(distance.min()), graph["continuous_cross_group_output_separation_m"])
            old = np.concatenate((old, segments))

    def test_adapter_rejects_policy_or_label_selection_knobs(self):
        context = mechanism_context()
        with self.assertRaises(ValueError):
            ForegroundConsensusAxisReadout(line_points(-.3), context, {"minimum_views": 2})
        adapter = ForegroundConsensusAxisReadout(line_points(-.3), context)
        with self.assertRaises(ValueError):
            adapter(np.empty((0, 2, 3)), dict(voxel_size=.02, origin=[0., 0., 0.], method="best"))


if __name__ == "__main__":
    unittest.main()
