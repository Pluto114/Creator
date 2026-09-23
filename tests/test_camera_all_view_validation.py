"""Coverage and leakage checks for the supplemental all-view diagnostic."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.camera_all_view_validation import all_view_plan, score_all_views
from creator_eval.camera_bundle import project, validation_plan


def fixture():
    k = np.tile([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]], (5, 1, 1))
    c = np.c_[np.linspace(-1., 1., 5), np.zeros((5, 2))]
    e = np.concatenate([np.tile(np.eye(3), (5, 1, 1)), -c[:, :, None]], axis=2)
    xyz = np.array([[.2, .1, 3.], [-.2, .3, 4.]])
    xy = [project(xyz, a, b)[0] for a, b in zip(k, e)]
    tracks = [dict(track_id=i, observations=[dict(view=v, xy=xy[v][i].tolist()) for v in range(5)]) for i in range(2)]
    return k, e, tracks


class AllViewValidationTests(unittest.TestCase):
    def test_extreme_cameras_are_actually_scored_and_score_is_unused(self):
        k, e, tracks = fixture()
        old = validation_plan(tracks, e)
        self.assertEqual({o["view"] for p in old for o in p["scoring"]}, {1, 2, 3})
        plan = all_view_plan(tracks, e, training_track_ids=[10])
        result = score_all_views(k, e, plan)
        self.assertEqual([v["sample_count"] for v in result["per_view"]], [2] * 5)
        self.assertLess(result["summary"]["p95_px"], 1e-10)
        first = plan["rows"][0]
        self.assertNotIn(first["scoring"]["view"], [o["view"] for o in first["triangulation"]])
        first["scoring"]["xy"][0] += 12
        self.assertAlmostEqual(score_all_views(k, e, plan)["rows"][0]["error_px"], 12., places=8)

    def test_same_plan_for_changed_cameras_without_source_mutation(self):
        k, e, tracks = fixture()
        source = copy.deepcopy(tracks)
        plan = all_view_plan(tracks, e, training_track_ids=[10])
        saved = copy.deepcopy(plan)
        before = score_all_views(k, e, plan)
        moved = e.copy()
        moved[4, 0, 3] += .05
        after = score_all_views(k, moved, plan)
        self.assertEqual(plan, saved)
        self.assertEqual(tracks, source)
        self.assertEqual([(r["track_id"], r["view"], r["triangulation_views"]) for r in before["rows"]],
                         [(r["track_id"], r["view"], r["triangulation_views"]) for r in after["rows"]])
        self.assertGreater(after["per_view"][4]["median_px"], 3.)

    def test_fit_overlap_duplicate_and_tampered_scoring_are_rejected(self):
        k, e, tracks = fixture()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            all_view_plan(tracks, e, training_track_ids=[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            all_view_plan(tracks + tracks[:1], e, training_track_ids=[])
        plan = all_view_plan(tracks, e, training_track_ids=[10])
        plan["rows"][0]["scoring"] = copy.deepcopy(plan["rows"][0]["triangulation"][0])
        with self.assertRaisesRegex(ValueError, "must not enter"):
            score_all_views(k, e, plan)


if __name__ == "__main__":
    unittest.main()
