"""Surface-ID probe semantics on small masks, without loading any dataset."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_candidate_audit import audit_candidate_edges  # noqa: E402


def fixture(left=2.5, right=8.5, center=5.5, y=2.0):
    rows = [{"y": y, "candidates": [
        {"center_x": center, "left_edge": {"x": left}, "right_edge": {"x": right}}
    ]}]
    raster = np.zeros((5, 12), dtype=np.uint32)
    raster[:, 3:9] = 1
    return {"rows": rows}, {"row_matches": [[0, 0, 0.0]]}, raster


class RodCandidateAuditTests(unittest.TestCase):
    def test_true_pair_straddles_both_target_boundaries_without_mutation(self):
        observations, hypothesis, raster = fixture()
        before = copy.deepcopy((observations, hypothesis))
        mask_before = raster.copy()
        result = audit_candidate_edges(observations, hypothesis, raster)
        self.assertEqual((observations, hypothesis), before)
        np.testing.assert_array_equal(raster, mask_before)
        self.assertEqual(result["matched_rows"], 1)
        self.assertEqual(result["center_surface_id_histogram"], {"1": 1})
        self.assertEqual(result["target_center_fraction"], 1.0)
        self.assertEqual(result["both_edges_target_boundary_fraction"], 1.0)
        self.assertEqual(result["edges"]["left"]["target_boundary_count"], 1)
        self.assertEqual(result["edges"]["right"]["target_boundary_count"], 1)
        self.assertEqual(result["out_of_bounds_probe_count"], 0)

    def test_internal_bright_band_center_is_target_but_edges_are_not_boundaries(self):
        result = audit_candidate_edges(*fixture(left=4.5, right=6.5))
        self.assertEqual(result["target_center_fraction"], 1.0)
        self.assertEqual(result["both_edges_target_boundary_fraction"], 0.0)
        for edge in result["edges"].values():
            self.assertEqual(edge["valid_probe_rows"], 1)
            self.assertEqual(edge["target_boundary_count"], 0)

    def test_other_rod_in_empty_target_does_not_count_as_target(self):
        observations, hypothesis, raster = fixture()
        raster[raster == 1] = 2
        result = audit_candidate_edges(observations, hypothesis, raster)
        self.assertEqual(result["center_surface_id_histogram"], {"2": 1})
        self.assertEqual(result["target_center_fraction"], 0.0)
        self.assertEqual(result["both_edges_target_boundary_fraction"], 0.0)
        other_target = audit_candidate_edges(observations, hypothesis, raster, target_id=2)
        self.assertEqual(other_target["both_edges_target_boundary_fraction"], 1.0)

    def test_out_of_bounds_is_unknown_and_remains_in_fraction_denominator(self):
        observations, hypothesis, raster = fixture()
        observations["rows"].append(copy.deepcopy(observations["rows"][0]))
        observations["rows"][1]["y"] = -0.6
        hypothesis["row_matches"].append([1, 0, 0.0])
        result = audit_candidate_edges(observations, hypothesis, raster)
        self.assertEqual(result["matched_rows"], 2)
        self.assertEqual(result["valid_center_rows"], 1)
        self.assertEqual(result["out_of_bounds_probe_count"], 5)
        self.assertEqual(result["out_of_bounds_rows"], 1)
        self.assertEqual(result["target_center_fraction"], 0.5)
        self.assertEqual(result["both_edges_target_boundary_fraction"], 0.5)
        for edge in result["edges"].values():
            self.assertEqual(edge["out_of_bounds_probe_count"], 2)
            self.assertEqual(edge["target_boundary_fraction"], 0.5)

    def test_one_invalid_edge_probe_cannot_be_the_background_side(self):
        observations, hypothesis, raster = fixture(left=-0.5, right=8.5, center=3)
        raster[:, :9] = 1
        result = audit_candidate_edges(observations, hypothesis, raster)
        self.assertEqual(result["edges"]["left"]["out_of_bounds_probe_count"], 1)
        self.assertEqual(result["edges"]["left"]["target_boundary_count"], 0)
        self.assertEqual(result["edges"]["right"]["target_boundary_count"], 1)
        self.assertEqual(result["both_edges_target_boundary_fraction"], 0.0)

    def test_nearest_center_uses_floor_half_up_not_bankers_rounding(self):
        observations, hypothesis, raster = fixture(left=2.5, right=8.5, center=2.5, y=1.5)
        raster[2, 3] = 7
        result = audit_candidate_edges(observations, hypothesis, raster)
        self.assertEqual(result["center_surface_id_histogram"], {"7": 1})

    def test_empty_hypothesis_has_no_fake_success_fraction(self):
        observations, hypothesis, raster = fixture()
        hypothesis["row_matches"] = []
        result = audit_candidate_edges(observations, hypothesis, raster)
        self.assertEqual(result["matched_rows"], 0)
        self.assertIsNone(result["target_center_fraction"])
        self.assertIsNone(result["both_edges_target_boundary_fraction"])
        self.assertEqual(result["fraction_reason"], "no_matched_rows")

    def test_invalid_indices_and_duplicate_rows_fail(self):
        for match in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                      [0.0, 0, 0], [0, True, 0], [0, 0]):
            observations, hypothesis, raster = fixture()
            hypothesis["row_matches"] = [match]
            with self.subTest(match=match), self.assertRaises(ValueError):
                audit_candidate_edges(observations, hypothesis, raster)
        observations, hypothesis, raster = fixture()
        hypothesis["row_matches"].append([0, 0, 0])
        with self.assertRaisesRegex(ValueError, "only once"):
            audit_candidate_edges(observations, hypothesis, raster)

    def test_invalid_coordinates_raster_and_probe_offset_fail(self):
        for kwargs in ({"center": np.nan}, {"left": np.inf}, {"right": -np.inf}, {"y": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "finite"):
                audit_candidate_edges(*fixture(**kwargs))
        observations, hypothesis, raster = fixture()
        for invalid in (raster.astype(float), np.zeros((0, 2), int), np.zeros((2, 2, 2), int), -np.ones((2, 2), int)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                audit_candidate_edges(observations, hypothesis, invalid)
        for offset in (0, -1, np.nan, np.inf, True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                audit_candidate_edges(observations, hypothesis, raster, probe_offset_px=offset)


if __name__ == "__main__":
    unittest.main()
