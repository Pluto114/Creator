"""Isolation contracts, including counterfactual heldout changes and empty slots."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.section_model_controls import generate  # noqa: E402
from creator_eval.section_model_trainonly import (  # noqa: E402
    DEFAULTS,
    diagnose,
    native_groups,
    prepare_fold,
    representation,
)


class TrainOnlySectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
        cls.cloud = np.concatenate([np.c_[1.8*np.cos(theta)+.3, 1.2*np.sin(theta)+.2,
            np.full(len(theta), z)] for z in np.arange(2.4, 58., .7)])
        cls.policy = copy.deepcopy(DEFAULTS)
        cls.policy["model"]["maximum_nfev"] = 2

    def test_fixed_native_groups_have_no_shared_cell_and_real_gap(self):
        points, cells, assignment = native_groups(self.cloud, 1.)
        a, b = np.isin(assignment, [0,2,4]), np.isin(assignment, [1,3,5])
        self.assertTrue(set(map(tuple,cells[a])).isdisjoint(map(tuple,cells[b])))
        self.assertGreaterEqual(np.min(np.abs(points[a,2,None]-points[b,2])), 4.)
        for cell in np.unique(cells,axis=0):
            self.assertEqual(len(set(assignment[(cells == cell).all(axis=1)])), 1)

    def test_native_boundary_and_outside_conventions_are_explicit(self):
        z = np.array([-1., 0., 1.9999, 2., 7.9999, 8., 12., 17., 18., 59., 60.])
        points = np.c_[np.zeros(len(z)), np.zeros(len(z)), z]
        _, _, labels = native_groups(points, 1.)
        np.testing.assert_array_equal(labels, [-2,-1,-1,0,0,-1,1,1,-1,-1,-2])

    def test_changing_heldout_and_guard_does_not_change_training_frame_weights_or_fits(self):
        points, _, assignment = native_groups(self.cloud, 1.)
        changed = points.copy()
        changed[~np.isin(assignment,[0,2,4]), :2] += [8., -5.]
        first, second = [diagnose(x,1.,self.policy) for x in (points,changed)]
        for key in ("frame","train_cell_identity","train_point_identity"):
            self.assertEqual(first["folds"][0][key], second["folds"][0][key])
        for a,b in zip(first["representations"],second["representations"]):
            for x,y in zip(a["rows"][:3],b["rows"][:3]):
                for key in ("fit","train_representation_identity","training_weights_identity"):
                    self.assertEqual(x[key],y[key])
        self.assertNotEqual(first["representations"][0]["rows"][0]["heldout"], second["representations"][0]["rows"][0]["heldout"])

    def test_removing_heldout_points_preserves_frame_but_leaves_missing_scores(self):
        points, _, assignment = native_groups(self.cloud, 1.)
        keep = np.isin(assignment,[0,2,4])
        self.assertEqual(prepare_fold(points,1.,0)[2]["frame"],prepare_fold(points[keep],1.,0)[2]["frame"])
        result = diagnose(points[keep],1.,self.policy)
        for rep in result["representations"]:
            self.assertEqual(len(rep["rows"]),6)
            for row in rep["rows"][:3]:
                self.assertTrue(all(s["state"]=="missing_test_points" for s in row["heldout"]))
            for row in rep["rows"][3:]:
                self.assertEqual(row["fit"]["state"],"unavailable")
                self.assertTrue(all(s["state"]=="not_scored_missing_fit" for s in row["heldout"]))

    def test_raw_weights_are_computed_per_side_and_sum_to_one_per_cell(self):
        points = np.array([[.1,.1,2.1],[.2,.1,2.2],[1.1,.1,2.1]])
        raw, weights, cells = representation(points,1.,"equal_voxel_raw_points")
        np.testing.assert_array_equal(raw,points)
        np.testing.assert_array_equal(weights,[.5,.5,1.])
        self.assertEqual(weights.sum(),len(cells))

    def test_frame_uses_only_training_cell_centers_and_training_radius_span(self):
        points, cells, labels = native_groups(self.cloud,1.)
        frame = prepare_fold(points,1.,0)[2]["frame"]
        centers = np.unique(cells[np.isin(labels,[0,2,4])],axis=0)+.5
        np.testing.assert_allclose(frame["origin"],centers.mean(axis=0),atol=0,rtol=0)
        axial = (centers-frame["origin"])@np.array(frame["vectors"])[:,-1]
        edges = np.quantile(axial,[.05,.95])
        np.testing.assert_allclose(frame["axial_quantiles_m"],edges,atol=0,rtol=0)
        self.assertEqual(frame["maximum_radius_voxels"],(edges[1]-edges[0])*.25)

    def test_empty_and_small_inputs_preserve_all_twelve_model_slots(self):
        for points in (np.empty((0,3)),np.array([[.1,.1,2.1]])):
            result = diagnose(points,1.,self.policy)
            rows = [r for rep in result["representations"] for r in rep["rows"]]
            self.assertEqual(len(rows),12)
            self.assertTrue(all(r["fit"]["state"]=="unavailable" for r in rows))
            self.assertTrue(all(len(r["heldout"])==3 for r in rows))
            self.assertFalse(result["emits_axis"])
            self.assertIsNone(result["qualification"])
            self.assertNotIn("segments",result)
            json.dumps(result,allow_nan=False)

    def test_fold_swap_and_input_order_do_not_change_membership(self):
        a = prepare_fold(self.cloud,1.,0)[2]
        b = prepare_fold(self.cloud[::-1],1.,1)[2]
        self.assertEqual(a["train_cell_identity"],b["heldout_cell_identity"])
        self.assertEqual(a["heldout_cell_identity"],b["train_cell_identity"])

    def test_protocol_keeps_model_parameters_and_anonymous_parent_inputs(self):
        protocol=json.loads((ROOT/"configs/section_model_trainonly_v1.json").read_text())
        self.assertEqual(protocol["method"],DEFAULTS)
        old=json.loads((ROOT/"configs/section_model_diagnostic_v1.json").read_text())
        self.assertEqual(protocol["method"]["model"],old["method"])
        # CI has only tracked files: regenerate fixtures instead of reading local experiment artifacts.
        cases, labels=generate(old)
        self.assertEqual(len(cases),64)
        self.assertEqual(len(labels),64)
        for case in cases:
            self.assertEqual(set(case),{"case_id","points","voxel_size"})

    def test_invalid_input_rejected_without_silent_zero_scores(self):
        for points,voxel in ((np.ones((3,2)),1.),(np.full((3,3),np.nan),1.),(self.cloud,0.)):
            with self.assertRaises(ValueError):
                native_groups(points,voxel)


if __name__ == "__main__":
    unittest.main()
