"""Strict full-result equivalence for cached candidate-to-view matching."""

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_candidate_association import (
    _CachedViewMatcher,
    associate_multiview_lines_cached,
)
from creator_eval.rod_multiview_candidates import (
    _view_match,
    associate_multiview_lines,
)


def canonical(value):
    def plain(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return json.dumps(value, default=plain, sort_keys=True, separators=(",", ":"), allow_nan=False)


def settings(**changes):
    return {
        "fit_view_count": 3, "minimum_support_views": 4,
        "reprojection_threshold_px": 1.0, "deduplicate_separation_px": .6,
        "ambiguity_support_fraction": .9, "ambiguity_separation_px": 2.5,
        "maximum_hypotheses": 1000, "maximum_fit_attempts": 1000,
        **changes,
    }


def views_for_lines(offsets=(.1,), *, homogeneous=False, noise_seed=None):
    rng = np.random.default_rng(noise_seed)
    views = []
    intrinsic = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    for index, x in enumerate([-1, -.5, 0, .5, 1]):
        extrinsic = np.c_[np.eye(3), [-x, -.03 * index, 0]]
        view = {
            "view_id": str(index), "K_index": intrinsic.copy(),
            "world_to_camera_cv": np.vstack((extrinsic, [0, 0, 0, 1])) if homogeneous else extrinsic,
            "y_range": [60.0, 180.0], "candidates": [],
        }
        for offset in offsets:
            points = np.array([[offset, -1, 6, 1], [offset, 1, 6, 1]])
            pixels = points @ (intrinsic @ extrinsic).T
            line = np.cross(pixels[0], pixels[1])
            line /= np.linalg.norm(line[:2])
            if noise_seed is not None:
                line[2] += rng.normal(0, .06)
            view["candidates"].append({"line": line})
        views.append(view)
    return views


class CachedCandidateAssociationTests(unittest.TestCase):
    def assert_equivalent(self, views, config=None):
        config = settings() if config is None else config
        before = canonical(views)
        reference = associate_multiview_lines(views, config)
        cached = associate_multiview_lines_cached(views, config)
        # This comparison permits no numeric tolerance or rank reshuffling. It
        # includes source triples, selected/alternative order, and all fit data.
        self.assertEqual(canonical(cached), canonical(reference))
        self.assertEqual(hashlib.sha256(canonical(cached).encode()).hexdigest(),
                         hashlib.sha256(canonical(reference).encode()).hexdigest())
        self.assertEqual(canonical(views), before)
        return cached

    def test_unique_accepted_line_with_both_camera_shapes(self):
        for homogeneous in (False, True):
            result = self.assert_equivalent(views_for_lines(homogeneous=homogeneous))
            self.assertEqual(result["state"], "accepted")
            self.assertTrue(result["search_complete"])

    def test_ambiguous_noisy_alternatives_keep_full_order_for_multiple_seeds(self):
        for seed in (0, 3, 19092027):
            with self.subTest(seed=seed):
                views = views_for_lines((-.2, .1, .34), noise_seed=seed)
                result = self.assert_equivalent(views)
                self.assertEqual(result["state"], "ambiguous")
                self.assertGreater(len(result["alternatives"]), 0)
                for view in views:
                    view["candidates"].reverse()
                self.assert_equivalent(views)

    def test_attempt_and_hypothesis_budgets_retain_truncation_semantics(self):
        views = views_for_lines((-.15, .2))
        for key in ("maximum_fit_attempts", "maximum_hypotheses"):
            result = self.assert_equivalent(views, settings(**{key: 1}))
            self.assertEqual(result["state"], "ambiguous")
            self.assertEqual(result["reason"], "search_budget_exhausted")
            self.assertFalse(result["search_complete"])

    def test_empty_and_non_supporting_views_retain_rejection_semantics(self):
        result = self.assert_equivalent(views_for_lines(()))
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(result["attempted_combination_count"], 0)
        self.assert_equivalent([])
        views = views_for_lines()
        views[-1]["candidates"] = []
        self.assertEqual(self.assert_equivalent(views)["state"], "accepted")
        self.assertEqual(self.assert_equivalent(views, settings(minimum_support_views=5))["state"], "rejected")

    def test_first_candidate_tie_is_identical_locally_and_in_full_association(self):
        views = views_for_lines((.1, .1))
        model = {"anchor": [.1, 0, 6], "direction": [0, 1, 0]}
        for view in views:
            reference = _view_match(model, view)
            cached = _CachedViewMatcher(view).match(model)
            self.assertEqual(canonical(cached), canonical(reference))
            self.assertEqual(cached["candidate_index"], 0)
        result = self.assert_equivalent(views)
        self.assertEqual(result["state"], "accepted")
        self.assertTrue(all(match["candidate_index"] == 0 for match in result["selected"]["matches"]))

    def test_local_match_values_are_exact_for_random_finite_models(self):
        views = views_for_lines((-.3, -.1, .1, .3), noise_seed=42)
        rng = np.random.default_rng(51)
        for view in views:
            matcher = _CachedViewMatcher(view)
            for _ in range(10):
                model = {"anchor": [rng.uniform(-.4, .4), 0, rng.uniform(4, 8)],
                         "direction": [rng.uniform(-.04, .04), 1, 0]}
                self.assertEqual(canonical(matcher.match(model)), canonical(_view_match(model, view)))

    def test_invalid_camera_and_budget_errors_are_not_swallowed(self):
        invalid = []
        rotation = views_for_lines()
        rotation[0]["world_to_camera_cv"][0, 0] = 2
        invalid.append((rotation, settings()))
        intrinsic = views_for_lines()
        intrinsic[0]["K_index"][:] = 0
        invalid.append((intrinsic, settings()))
        duplicate = views_for_lines()
        duplicate[1]["view_id"] = duplicate[0]["view_id"]
        invalid.append((duplicate, settings()))
        invalid.append((views_for_lines(), settings(maximum_fit_attempts=0)))
        for views, config in invalid:
            with self.assertRaises(ValueError) as reference:
                associate_multiview_lines(copy.deepcopy(views), config)
            with self.assertRaises(ValueError) as cached:
                associate_multiview_lines_cached(copy.deepcopy(views), config)
            self.assertEqual(str(cached.exception), str(reference.exception))


if __name__ == "__main__":
    unittest.main()
