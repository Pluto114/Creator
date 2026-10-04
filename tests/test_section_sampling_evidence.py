"""Independent sampling geometry contracts; no scored inputs or truth files."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import section_evidence as frozen  # noqa: E402
from creator_eval.section_sampling_evidence import (  # noqa: E402
    DEFAULTS,
    analyze,
    policy,
    sampling_runs,
)


def rings(z, *, half=False, angles_count=48, rotating=False, radius=.024):
    z = np.asarray(z)
    if rotating:
        angles = np.linspace(-np.pi/6, np.pi/6, angles_count)[None]+2*np.pi*(z-z.min())[:, None]/np.ptp(z)
    else:
        angles = np.broadcast_to(np.linspace(0, np.pi if half else 2*np.pi, angles_count, endpoint=half), (len(z), angles_count))
    points = np.c_[radius*np.cos(angles).ravel(), radius*np.sin(angles).ravel(), np.repeat(z, angles_count)]
    training = np.indices((len(z), angles_count)).sum(axis=0).ravel() % 2 == 0
    return points, training


class SectionSamplingEvidenceTests(unittest.TestCase):
    def test_dense_and_sparse_pitch_use_identical_unchanged_section_gates(self):
        for pitch in (.007, .019):
            for half in (False, True):
                with self.subTest(pitch=pitch, half=half):
                    points, mask = rings(np.arange(85)*pitch, half=half, angles_count=49 if half else 48)
                    segments, consumed, diagnostic = analyze(points, mask, {"voxel_size": .006})
                    self.assertEqual(len(segments), 1)
                    self.assertTrue(consumed.all())
                    np.testing.assert_allclose(segments[:, :, :2], 0., atol=1e-8)
                    evidence = diagnostic["groups"][0]["sampling_evidence"]
                    np.testing.assert_allclose(evidence["typical_intervals_m"], pitch, atol=1e-8)
        for key, value in frozen.DEFAULTS.items():
            self.assertEqual(DEFAULTS[key], value)

    def test_irregular_and_piecewise_sampling_density_are_local_not_global(self):
        irregular = np.r_[0., np.cumsum(np.tile([.007, .011, .019], 40))]
        mixed = np.r_[np.arange(91)*.004, .36+np.arange(1, 61)*.019]
        for z in (irregular, mixed):
            with self.subTest(kind="irregular" if z is irregular else "mixed"):
                points, mask = rings(z)
                segments, _, diagnostic = analyze(points, mask, {"voxel_size": .006})
                self.assertEqual(len(segments), 1)
                self.assertGreater(diagnostic["groups"][0]["sampling_evidence"]["learned_interval_count"], 20)

    def test_long_true_gap_remains_split_at_both_sampling_densities(self):
        for pitch in (.007, .019):
            with self.subTest(pitch=pitch):
                left = np.arange(35)*pitch
                z = np.r_[left, left[-1]+.23+np.arange(35)*pitch]
                points, mask = rings(z)
                segments, _, diagnostic = analyze(points, mask, {"voxel_size": .006})
                self.assertEqual(len(segments), 2)
                self.assertTrue(all(segment[:, 2].max() <= left[-1]+1e-8 or segment[:, 2].min() >= left[-1]+.23-1e-8 for segment in segments))
                sampling = diagnostic["groups"][0]["sampling_evidence"]
                self.assertEqual(len(sampling["observed_split_gaps_m"]), 1)
                self.assertAlmostEqual(sampling["observed_split_gaps_m"][0], .23, places=7)

    def test_a_gap_cannot_inflate_its_own_allowed_interval(self):
        training = np.r_[np.arange(20)*.009, .6+np.arange(20)*.009]
        runs, diagnostic = sampling_runs(training, training, {"voxel_size": .006})
        self.assertEqual(len(runs), 2)
        index = int(np.argmax(diagnostic["training_intervals_m"]))
        self.assertAlmostEqual(diagnostic["typical_intervals_m"][index], .009)
        self.assertAlmostEqual(diagnostic["maximum_allowed_gaps_m"][index], 2.5*.009)

    def test_validation_does_not_change_any_training_scale(self):
        training = np.arange(51)*.019
        _, first = sampling_runs(training, training, {"voxel_size": .006})
        observed = np.r_[training, np.linspace(-.5, 1.5, 401)]
        _, second = sampling_runs(observed, training, {"voxel_size": .006})
        for key in ("training_layer_positions_m", "training_intervals_m", "typical_intervals_m", "maximum_allowed_gaps_m", "scale_sources"):
            self.assertEqual(first[key], second[key])
        points, mask = rings(np.arange(85)*.019)
        _, _, first_model = analyze(points, mask, {"voxel_size": .006})
        changed = points.copy()
        changed[~mask, :2] += [.0002, -.0002]
        _, _, second_model = analyze(changed, mask, {"voxel_size": .006})
        self.assertEqual(first_model["groups"][0]["sampling_evidence"]["typical_intervals_m"],
                         second_model["groups"][0]["sampling_evidence"]["typical_intervals_m"])

    def test_dense_continuous_projection_does_not_invent_a_large_scale(self):
        training = np.linspace(0, 1., 10001)
        runs, diagnostic = sampling_runs(training, training, {"voxel_size": .006})
        self.assertEqual(len(runs), 1)
        self.assertEqual(diagnostic["training_layer_count"], 1)
        self.assertEqual(diagnostic["learning_state"], "voxel_rule_insufficient_layer_evidence")

    def test_sparse_arc_and_helix_do_not_gain_circle_acceptance(self):
        for rotating in (False, True):
            with self.subTest(rotating=rotating):
                points, mask = rings(np.arange(85)*.019, angles_count=4, rotating=rotating)
                segments, consumed, diagnostic = analyze(points, mask, {"voxel_size": .006})
                self.assertEqual(len(segments), 0)
                if rotating:
                    self.assertTrue(consumed.all())
                    self.assertEqual(diagnostic["unresolved_surface_groups"], 1)

    def test_order_duplicates_and_translation_do_not_change_inferred_gaps(self):
        training = np.arange(51)*.019
        first, evidence = sampling_runs(training, training, {"voxel_size": .006})
        second, duplicate = sampling_runs(np.repeat(training[::-1], 3), np.repeat(training[::-1], 2), {"voxel_size": .006})
        self.assertEqual(len(first), len(second))
        self.assertEqual(evidence["training_layer_positions_m"], duplicate["training_layer_positions_m"])
        shifted, phase = sampling_runs(training+.0037, training+.0037, {"voxel_size": .006})
        self.assertEqual(len(first), len(shifted))
        np.testing.assert_allclose(evidence["typical_intervals_m"], phase["typical_intervals_m"], atol=1e-14)

    def test_insufficient_training_layers_do_not_authorize_sparse_bridging(self):
        runs, diagnostic = sampling_runs([0., .1, .2], [0., .1, .2], {"voxel_size": .006})
        self.assertEqual(len(runs), 3)
        self.assertEqual(diagnostic["learned_interval_count"], 0)

    def test_budget_policy_and_nonfinite_projection_fail_explicitly(self):
        runs, diagnostic = sampling_runs(np.arange(100)*.02, np.arange(100)*.02,
                                        {"section_sampling_maximum_layers": 10})
        self.assertEqual(len(runs), 0)
        self.assertEqual(diagnostic["state"], "unmeasurable")
        for config in ({"section_sampling_bad": 1}, {"section_sampling_gap_factor": float("nan")},
                       {"section_sampling_gap_factor": 10}, {"section_sampling_minimum_neighbors": 2},
                       {"section_sampling_neighbor_intervals": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                policy(config)
        with self.assertRaises(ValueError):
            sampling_runs([np.nan], [0.], {})


if __name__ == "__main__":
    unittest.main()
