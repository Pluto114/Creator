"""Contracts for final-support refitting and evidence-preserving comparisons."""
import copy
import unittest

import numpy as np
from test_rod_candidate_association import canonical, settings, views_for_lines

# isort: split
from creator_eval.line_controls import fit_multiview_line
from creator_eval.rod_assignment_refit import _refit_assignment, associate_refitted_variants


def measured_views(offsets=(.1,), **kwargs):
    views = views_for_lines(offsets, **kwargs)
    for view in views:
        for i, candidate in enumerate(view["candidates"]):
            candidate["row_matches"] = [[j, i, 0.] for j in range(20 + i * 5)]
    return views


class AssignmentRefitTests(unittest.TestCase):
    def test_result_axis_is_fit_to_all_fixed_support_and_inputs_stay_intact(self):
        views = measured_views(noise_seed=31)
        before = canonical(views)
        result = associate_refitted_variants(views, settings())["refit_residual"]
        self.assertEqual(result["state"], "accepted")
        h = result["selected"]
        indices = h["supporting_views"]
        expected = fit_multiview_line([views[i]["candidates"][h["matches"][i]["candidate_index"]]["line"] for i in indices],
            [views[i]["K_index"] for i in indices], [views[i]["world_to_camera_cv"] for i in indices])
        np.testing.assert_allclose(h["model"]["anchor"], expected["anchor"], atol=1e-12)
        np.testing.assert_allclose(h["model"]["direction"], expected["direction"], atol=1e-12)
        self.assertTrue(all(h["matches"][i]["residual_px"] <= 1. for i in indices))
        self.assertEqual(canonical(views), before)

    def test_bad_final_support_is_rejected_without_deleting_the_bad_view(self):
        views = measured_views()
        views[-1]["candidates"][0]["line"][2] += 10.
        seed = dict(supporting_views=list(range(5)), matches=[dict(candidate_index=0) for _ in views])
        before = copy.deepcopy(seed)
        self.assertIsNone(_refit_assignment(seed, views, settings()))
        self.assertEqual(seed, before)

    def test_close_but_different_pixel_assignments_survive_retention_comparison(self):
        views = measured_views((.1, .12))
        variants = associate_refitted_variants(views, settings())
        kept = variants["retained_assignments"]
        assignments = {tuple(h["matches"][i]["candidate_index"] for i in h["supporting_views"])
            for h in [kept["selected"], *kept["alternatives"]]}
        self.assertIn((0, 0, 0, 0, 0), assignments)
        self.assertIn((1, 1, 1, 1, 1), assignments)
        self.assertEqual(kept["state"], "ambiguous")
        self.assertGreater(len(kept["alternatives"]), len(variants["refit_residual"]["alternatives"]))

    def test_exhausted_search_cannot_claim_unique_acceptance(self):
        for limit in ("maximum_fit_attempts", "maximum_hypotheses"):
            variants = associate_refitted_variants(measured_views((.1, .3)), settings(**{limit: 1}))
            for row in variants.values():
                self.assertFalse(row["search_complete"])
                self.assertEqual(row["state"], "ambiguous")

    def test_missing_or_duplicate_measured_rows_cannot_fake_coverage(self):
        for rows in ([], [[0, 0, 0.], [0, 1, 0.]]):
            views = measured_views()
            views[0]["candidates"][0]["row_matches"] = rows
            with self.assertRaises(ValueError):
                associate_refitted_variants(views, settings())


if __name__ == "__main__":
    unittest.main()
