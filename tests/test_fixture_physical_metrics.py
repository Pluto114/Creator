"""Known-answer physical checks; all fixtures are in-memory CAD coordinates."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/src"))
from creator_eval.fixture_physical_metrics import score_finite_structure  # noqa: E402


def segment(a,b,y=0.):
    return [[a,y,0.],[b,y,0.]]


class FixturePhysicalMetricsTests(unittest.TestCase):
    def test_exact_complete_line_has_full_length_and_endpoints(self):
        result=score_finite_structure([segment(0,2)],[segment(0,2)])
        self.assertEqual(result["curves"]["recovery_fraction"],1.)
        self.assertEqual(result["length"]["absolute_error_m"],0.)
        self.assertEqual(result["boundary_bijection"]["maximum_error_m"],0.)
        self.assertFalse(result["alignment_performed"])

    def test_contiguous_subdivision_has_only_two_physical_boundaries(self):
        result=score_finite_structure([segment(0,.7),segment(.7,2)],[segment(0,2)])
        self.assertEqual(result["predicted_endpoints"]["native_endpoint_count"],4)
        self.assertEqual(result["predicted_endpoints"]["boundary_endpoint_count"],2)
        self.assertEqual(len(result["predicted_endpoints"]["shared_degree_two_points"]),1)
        self.assertEqual(result["boundary_bijection"]["maximum_error_m"],0.)
        self.assertEqual(result["length"]["prediction_union_m"],2.)

    def test_short_prediction_pays_missing_length_and_endpoint_error(self):
        result=score_finite_structure([segment(.2,1.8)],[segment(0,2)])
        self.assertAlmostEqual(result["length"]["signed_error_m"],-.4)
        self.assertAlmostEqual(result["boundary_bijection"]["maximum_error_m"],.2)
        self.assertLess(result["curves"]["recovery_fraction"],1.)

    def test_offset_is_not_removed_by_registration(self):
        result=score_finite_structure([segment(0,2,.1)],[segment(0,2)])
        self.assertEqual(result["curves"]["recovery_fraction"],0.)
        self.assertAlmostEqual(result["curves"]["prediction_to_truth"]["distance_p95"],.1)
        self.assertAlmostEqual(result["boundary_bijection"]["maximum_error_m"],.1)

    def test_real_gap_remains_clear_with_guarded_300mm_interior(self):
        pieces=[segment(0,.8),segment(1.15,2.)]
        result=score_finite_structure(pieces,pieces,[segment(.8,1.15)])
        self.assertEqual(result["guarded_gap"]["false_proximity_fraction"],0.)
        self.assertAlmostEqual(result["guarded_gap"]["interior_length_m"],.3)
        self.assertEqual(result["predicted_endpoints"]["boundary_endpoint_count"],4)

    def test_false_bridge_keeps_missing_topology_and_false_length_cost(self):
        result=score_finite_structure([segment(0,2)],[segment(0,.8),segment(1.15,2)],[segment(.8,1.15)])
        self.assertAlmostEqual(result["guarded_gap"]["false_proximity_length_m"],.3)
        self.assertEqual(result["guarded_gap"]["false_proximity_fraction"],1.)
        self.assertEqual(result["boundary_bijection"]["state"],"boundary_count_mismatch")
        self.assertLess(result["curves"]["precision_fraction"],1.)

    def test_empty_is_zero_recovery_not_perfect_precision_or_endpoints(self):
        result=score_finite_structure([],[segment(0,1)],[segment(1,1.35)])
        self.assertEqual(result["curves"]["recovery_fraction"],0.)
        self.assertIsNone(result["curves"]["precision_fraction"])
        self.assertIsNone(result["boundary_bijection"]["maximum_error_m"])
        self.assertEqual(result["truth_boundary_to_prediction_boundary"]["distances_m"],[None,None])
        self.assertTrue(result["empty_prediction"])
        json.dumps(result,allow_nan=False)

    def test_short_gap_is_unmeasurable_not_zero_false_fill(self):
        result=score_finite_structure([segment(0,1)],[segment(0,1)],[segment(.4,.44)])
        self.assertIsNone(result["guarded_gap"]["false_proximity_fraction"])

    def test_reverse_orientation_does_not_change_boundary_errors(self):
        result=score_finite_structure([segment(2,0)],[segment(0,2)])
        self.assertEqual(result["boundary_bijection"]["maximum_error_m"],0.)

    def test_overlap_and_invalid_geometry_are_not_silently_scored(self):
        for prediction in ([segment(0,2),segment(0,2)],np.full((1,2,3),np.nan),[segment(1,1)]):
            with self.assertRaises(ValueError):
                score_finite_structure(prediction,[segment(0,2)])


if __name__=="__main__":
    unittest.main()
