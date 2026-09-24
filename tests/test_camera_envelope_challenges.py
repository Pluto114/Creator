"""Geometry/data contracts for fresh analytic challenges, without fitting cameras."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/src'))
from creator_eval.camera_envelope_challenges import (  # noqa: E402
    INPUT_CASE_KEYS,
    generate_challenges,
    rod_support_digest,
)
from creator_eval.camera_training_groups import build_training_group_plan  # noqa: E402


def protocol():
    return json.loads((ROOT / 'configs/camera_envelope_challenges_v1.json').read_text(encoding='utf-8'))


def project_point(camera, point):
    transformed = np.asarray(camera['world_to_camera_cv']) @ np.r_[point, 1.]
    pixel = np.asarray(camera['K_index']) @ transformed[:3]
    return pixel[:2] / pixel[2]


class EnvelopeChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.truth = generate_challenges(protocol())
        cls.by_id = {case['case_id']: case for case in cls.truth['cases']}

    def test_reproducible_anonymous_schema_and_truth_separation(self):
        again, truth_again = generate_challenges(protocol())
        self.assertEqual(again, self.inputs)
        self.assertEqual(truth_again, self.truth)
        self.assertEqual(len(self.inputs['cases']), 16)
        self.assertEqual([case['case_id'] for case in self.inputs['cases']], [f'c{i:03d}' for i in range(1, 17)])
        for case in self.inputs['cases']:
            self.assertEqual(set(case), INPUT_CASE_KEYS)
            self.assertEqual(case['rod_support_sha256'], rod_support_digest(case['rod_tracks']))
            truth = self.by_id[case['case_id']]
            self.assertEqual(len(case['rod_tracks']), len(truth['rod_segments']))
            for key in ('family', 'expected_limit', 'cameras', 'rod_segments', 'generation_parameters', 'background_points'):
                self.assertNotIn(key, case)
        hidden = self.truth['cases'][0]['cameras'][0]['K_index']
        original = hidden[0][0]
        try:
            hidden[0][0] += 20
            self.assertEqual(self.inputs, again)
        finally:
            hidden[0][0] = original

    def test_camera_initialization_is_perturbed_rigid_and_noncollinear(self):
        ratios, normalized_layouts = [], []
        for case in self.inputs['cases']:
            truth = self.by_id[case['case_id']]
            real = np.asarray([camera['world_to_camera_cv'][:3] for camera in truth['cameras']])
            initial = np.asarray(case['initial_extrinsics'])
            real_k = np.asarray([camera['K_index'] for camera in truth['cameras']])
            initial_k = np.asarray(case['initial_intrinsics'])
            self.assertFalse(np.allclose(real, initial))
            self.assertFalse(np.allclose(real_k, initial_k))
            for value in (real, initial):
                rotations = value[:, :, :3]
                np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rotations.shape), atol=1e-12)
                np.testing.assert_allclose(np.linalg.det(rotations), 1., atol=1e-12)
                centers = -np.einsum('nji,nj->ni', rotations, value[:, :, 3])
                singular = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
                self.assertGreater(singular[1] / singular[0], .001)
            centers = -np.einsum('nji,nj->ni', real[:, :, :3], real[:, :, 3])
            span = np.linalg.norm(centers[-1] - centers[0])
            self.assertAlmostEqual(span, truth['generation_parameters']['camera_baseline_m'])
            normalized_layouts.append(centers / span)
            ratios.append(truth['initialization']['true_center_noncollinearity'])
        for layout in normalized_layouts[1:]:
            np.testing.assert_allclose(layout, normalized_layouts[0], atol=1e-12)
        self.assertGreater(min(ratios), .01)

    def test_band_split_and_whole_track_groups_do_not_use_truth_residuals(self):
        for case in self.inputs['cases']:
            training, validation = case['training'], case['validation']
            self.assertFalse({t['track_id'] for t in training} & {t['track_id'] for t in validation})
            for label, tracks in (('training', training), ('validation', validation)):
                for track in tracks:
                    xy = np.asarray([o['xy'] for o in track['observations']])
                    self.assertEqual([o['view'] for o in track['observations']], list(range(5)))
                    self.assertTrue(np.all(xy[:, 1] >= 192.) if label == 'training' else np.all(xy[:, 1] < 128.))
                    self.assertTrue(np.all(xy >= 0) and np.all(xy < [640, 480]))
            plan = build_training_group_plan(training, validation_track_ids=[t['track_id'] for t in validation], view_count=5)
            if self.by_id[case['case_id']]['family'] == 'no_observations':
                self.assertTrue(all(condition['state'] == 'skipped' for condition in plan['conditions']))
            else:
                self.assertEqual((len(training), len(validation)), (160, 64))
                self.assertTrue(all(condition['state'] == 'eligible' for condition in plan['conditions']))

    def test_endpoint_projection_and_counterexamples_follow_declared_process(self):
        # A noiseless unit fixture checks the data contract, not estimator quality.
        for family in ('volume_near_low_noise', 'coherent_wrong_rod_lateral', 'coherent_wrong_rod_depth', 'one_view_endpoint_swap', 'shared_frame_pixel_bias'):
            definition = protocol()
            specification = next(c for c in definition['cases'] if c['family'] == family)
            specification['pixel_noise_std'] = 0.
            definition['cases'], definition['expected_case_count'] = [specification], 1
            observed, hidden = generate_challenges(definition)
            case, truth = observed['cases'][0], hidden['cases'][0]
            for segment, measurements in zip(truth['measurement_rod_segments'], case['rod_tracks']):
                for endpoint, track in enumerate(measurements['endpoint_tracks']):
                    for observation in track['observations']:
                        view = observation['view']
                        index = 1 - endpoint if family == 'one_view_endpoint_swap' and view == 2 else endpoint
                        expected = project_point(truth['cameras'][view], segment[index]) + np.asarray(truth['shared_frame_pixel_bias'][view])
                        np.testing.assert_allclose(observation['xy'], expected, atol=1e-10)
            if family.startswith('coherent_wrong_rod'):
                self.assertNotEqual(truth['measurement_rod_segments'], truth['rod_segments'])
                self.assertEqual(truth['intended_role'], 'unidentifiable_boundary')

    def test_missing_observations_never_gain_fabricated_endpoint_support(self):
        for case in self.inputs['cases']:
            truth = self.by_id[case['case_id']]
            family = truth['family']
            expected = 0 if family == 'no_observations' else 1 if family == 'single_view_rod' else 5
            for segment in case['rod_tracks']:
                self.assertEqual(len(segment['endpoint_tracks']), 2)
                for endpoint in segment['endpoint_tracks']:
                    self.assertEqual(len(endpoint['observations']), expected)
                    self.assertEqual(len({o['view'] for o in endpoint['observations']}), expected)
            if family == 'no_observations':
                self.assertEqual(case['training'], [])
                self.assertEqual(case['validation'], [])
            self.assertTrue(truth['rod_segments'])


if __name__ == '__main__':
    unittest.main()
