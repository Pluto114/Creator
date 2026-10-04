"""Independent shell/center and training-context contracts, without pack IO."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "tests"))
import test_section_local_evidence as local_contracts  # noqa: E402
from creator_eval import section_context_evidence as subject  # noqa: E402
from creator_eval import section_sampling_evidence as sampling  # noqa: E402


def small_shell(centers_in_training=True):
    z = np.arange(97)*.014
    angles = np.arange(40)*2*np.pi/40
    points = np.c_[np.tile(.004*np.cos(angles), len(z)), np.tile(.004*np.sin(angles), len(z)), np.repeat(z, len(angles))]
    fold = ((np.arange(40)[None]+np.arange(len(z))[:, None]) % 3 != 1).ravel()
    centers = np.c_[np.zeros((len(z), 2)), z]
    return np.r_[points, centers], np.r_[fold, np.full(len(centers), centers_in_training, dtype=bool)]


def isolated_contexts(outliers_in_training):
    """A known analytic training model isolates context evidence from grouping."""
    points, fold = small_shell()
    z = np.arange(97)*.014
    angles = np.arange(16)*2*np.pi/16
    extra = np.c_[np.tile(.006*np.cos(angles), len(z)), np.tile(.006*np.sin(angles), len(z)), np.repeat(z, len(angles))]
    points = np.r_[points, extra]
    fold = np.r_[fold, np.full(len(extra), outliers_in_training, dtype=bool)]
    p = subject.policy({"voxel_size": .01})
    radial = np.linalg.norm(points[:, :2], axis=1)
    shell = np.abs(radial-.004) <= .004*p["section_radius_relative_residual"]
    sampler_policy = {key: value for key, value in p.items() if not key.startswith(("section_context_", "section_local_", "section_supported_"))}
    _, detail = sampling.sampling_runs(points[shell, 2], points[shell & fold, 2], sampler_policy)
    return subject._contexts(points[:, 2], points[:, :2], fold, shell,
                             dict(center=np.zeros(2), radius=.004), subject._sampling_field(detail), p, 1e-12)


class SectionContextInheritedTests(local_contracts.SectionLocalEvidenceTests):
    def setUp(self):
        # Inherited functions resolve names in their defining module, not here.
        replacement = patch.multiple(local_contracts, analyze=subject.analyze, policy=subject.policy, DEFAULTS=subject.DEFAULTS)
        replacement.start()
        self.addCleanup(replacement.stop)


class SectionContextEvidenceTests(unittest.TestCase):
    def test_center_ownership_changes_do_not_turn_shell_into_outliers(self):
        for centers_in_training in (True, False):
            with self.subTest(centers_in_training=centers_in_training):
                points, fold = small_shell(centers_in_training)
                segments, consumed, detail = subject.analyze(points, fold, {"voxel_size": .01})
                self.assertEqual(len(segments), 1)
                self.assertTrue(consumed.all())
                np.testing.assert_allclose(np.sort(segments[0, :, 2]), [0., 1.344], atol=1e-8)
                contexts = detail["groups"][0]["training_contexts"]
                count = "training_center_points" if centers_in_training else "validation_center_points"
                self.assertTrue(all(context[count] > 0 for context in contexts))
                self.assertTrue(all(context["training_other_points"] == context["validation_other_points"] == 0 for context in contexts))

    def test_sparse_pitch_sets_context_before_heldout_corruption(self):
        points, fold = small_shell(False)
        _, _, first = subject.analyze(points, fold, {"voxel_size": .01})
        changed = points.copy()
        changed[~fold, :2] *= 1.25
        segments, _, second = subject.analyze(changed, fold, {"voxel_size": .01})
        self.assertEqual(len(segments), 0)
        a, b = first["groups"][0], second["groups"][0]
        for key in ("trained_center", "trained_axis", "trained_radius", "local_evidence"):
            self.assertEqual(a[key], b[key])
        self.assertTrue(all(context["minimum_context_radius_m"] >= .1119 for context in a["training_contexts"]))
        for before, after in zip(a["training_contexts"], b["training_contexts"]):
            for key in ("center", "window", "training_candidates", "selected_candidate", "training_pitch_m"):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after["heldout_evaluations"], 1)
            self.assertFalse(after["validation_prediction"]["accepted"])

    def test_noncentral_training_outliers_are_not_prefiltered_to_shell(self):
        contexts, detail = isolated_contexts(True)
        self.assertEqual(detail["state"], "complete")
        self.assertTrue(all(context["training_other_points"] > 0 for context in contexts))
        self.assertTrue(all(context["training_points"] == context["training_shell_points"]+context["training_other_points"] for context in contexts))
        self.assertTrue(all(not context["accepted"] for context in contexts))
        self.assertTrue(all(len(context["training_candidates"]) <= 3 and context["heldout_evaluations"] == 1 for context in contexts))

    def test_noncentral_validation_outliers_remain_in_prediction_denominator(self):
        contexts, _ = isolated_contexts(False)
        self.assertTrue(all(context["validation_other_points"] > 0 for context in contexts))
        self.assertTrue(all(context["training_candidates"][-1]["accepted"] for context in contexts))
        self.assertTrue(all(context["validation_prediction"]["shell_fraction"] < .9 for context in contexts))
        self.assertTrue(all(not context["accepted"] and context["heldout_evaluations"] == 1 for context in contexts))

    def test_context_policy_is_bounded_and_existing_gates_unchanged(self):
        from creator_eval import section_local_evidence
        for key, value in section_local_evidence.DEFAULTS.items():
            self.assertEqual(subject.DEFAULTS[key], value)
        for config in ({"section_context_unknown": 1}, {"section_context_maximum_expansions": 0},
                       {"section_context_maximum_expansions": 6}, {"section_context_maximum_expansions": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                subject.policy(config)


if __name__ == "__main__":
    unittest.main()
