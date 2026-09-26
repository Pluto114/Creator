"""Known-camera PnP contracts; no thresholds selected from new RGB outcomes."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/src'))
from creator_eval.heldout_view_pose import DEFAULTS, canonical_hash, propose_pose, verify_pose


def fixture(noise=0.):
    rng = np.random.default_rng(72093)
    k = np.array([[650., 0., 320.], [0., 650., 240.], [0., 0., 1.]])
    r = cv2.Rodrigues(np.array([.06, -.08, .04]))[0]
    e = np.c_[r, [.12, -.09, .25]]
    points = rng.uniform([-1.15, -.8, 4.], [1.15, .8, 7.], (100, 3))
    q = (points @ e[:, :3].T + e[:, 3]) @ k.T
    xy = q[:, :2] / q[:, 2, None]
    xy += rng.normal(0., noise, xy.shape)
    source = dict(source_kind='rgb', rgb_sha256=hashlib.sha256(b'new-rgb').hexdigest(),
        map_sha256=hashlib.sha256(b'fixed-estimated-map').hexdigest(), intrinsics_sha256=canonical_hash(k),
        size_wh=[640, 480], observation_role='background_pose_training')
    train = dict(points3d=points[:60].copy(), xy=xy[:60].copy(), K=k, track_ids=list(range(60)), source=source)
    valid = dict(points3d=points[60:].copy(), xy=xy[60:].copy(), track_ids=list(range(60, 100)),
                 source={**source, 'observation_role': 'background_pose_validation'})
    return train, valid, e


class HeldoutViewPoseTests(unittest.TestCase):
    def test_exact_camera_pose_and_independent_validation(self):
        train, valid, expected = fixture()
        original = copy.deepcopy((train, valid))
        result = propose_pose(**train)
        self.assertEqual(result['state'], 'fitted')
        np.testing.assert_allclose(result['E'], expected, atol=1e-7, rtol=0.)
        self.assertEqual(result['ransac_inlier_count'], 60)
        self.assertEqual(result['final_support_count'], 60)
        self.assertFalse(result['map_modified'])
        self.assertFalse(result['intrinsics_modified'])
        self.assertFalse(result['target_observations_used'])
        self.assertIsNone(result['refinement_converged'])
        check = verify_pose(result, **valid)
        self.assertEqual(check['state'], 'validated')
        self.assertLess(check['validation']['summary']['p95_px'], 1e-6)
        self.assertEqual(canonical_hash((train, valid)), canonical_hash(original))
        json.dumps([result, check], allow_nan=False)

    def test_moderate_noise_and_wrong_training_matches_do_not_need_truth(self):
        train, valid, expected = fixture(.2)
        train['xy'][:10] = train['xy'][10:20][::-1]
        result = propose_pose(**train)
        self.assertEqual(result['state'], 'fitted')
        self.assertGreaterEqual(result['ransac_inlier_count'], 45)
        self.assertLessEqual(result['ransac_inlier_count'], 52)
        np.testing.assert_allclose(result['E'], expected, atol=.025, rtol=0.)
        self.assertEqual(result['all_training']['summary']['sample_count'], 60)
        self.assertEqual(len(result['all_training']['rows']), 60)
        self.assertEqual(verify_pose(result, **valid)['state'], 'validated')

    def test_validation_changes_decision_but_cannot_change_numeric_pose(self):
        train, valid, _ = fixture()
        result = propose_pose(**train)
        saved = copy.deepcopy(result)
        first = verify_pose(result, **valid)
        valid['xy'][:12, 0] += 12.
        second = verify_pose(result, **valid)
        self.assertEqual(first['state'], 'validated')
        self.assertEqual(second['state'], 'unresolved')
        self.assertIn('validation_pixel_budget_exceeded', second['reasons'])
        self.assertEqual(result, saved)
        self.assertEqual(propose_pose(**train), saved)
        self.assertFalse(second['pose_modified'])

    def test_p95_budget_is_not_a_maximum_pixel_error_guarantee(self):
        train, valid, _ = fixture()
        result = propose_pose(**train)
        valid['xy'][0, 0] += 20.
        check = verify_pose(result, **valid)
        self.assertEqual(check['state'], 'validated')
        self.assertGreater(check['validation']['summary']['maximum_px'], 19.)
        self.assertEqual(check['validation']['summary']['fraction_within_2px'], 39/40)

    def test_training_and_validation_tracks_must_be_disjoint(self):
        train, valid, _ = fixture()
        result = propose_pose(**train)
        valid['track_ids'][0] = train['track_ids'][0]
        check = verify_pose(result, **valid)
        self.assertEqual(check['state'], 'unresolved')
        self.assertIn('validation_tracks_overlap_training', check['reasons'])
        self.assertIsNone(check['validation'])

    def test_minimum_training_and_validation_support_are_not_hidden(self):
        train, valid, _ = fixture()
        result = propose_pose(**train)
        for n in (7, 0):
            small = dict(valid, points3d=valid['points3d'][:n], xy=valid['xy'][:n], track_ids=valid['track_ids'][:n])
            check = verify_pose(result, **small)
            self.assertEqual(check['state'], 'unresolved')
            self.assertIn('insufficient_validation_tracks', check['reasons'])
            self.assertIsNone(check['validation'])
        for n in (23, 0):
            small = dict(train, points3d=train['points3d'][:n], xy=train['xy'][:n], track_ids=train['track_ids'][:n])
            result = propose_pose(**small)
            self.assertEqual(result['state'], 'unresolved')
            self.assertIsNone(result['E'])
            self.assertIsNone(result['all_training'])
            self.assertIn('insufficient_mapped_training_tracks', result['reasons'])

    def test_planar_and_nearly_planar_map_are_unresolved_before_pnp(self):
        for scale in (0., 1e-5):
            with self.subTest(scale=scale):
                train, _, _ = fixture()
                train['points3d'][:, 2] = 5. + scale * train['points3d'][:, 2]
                with patch.object(cv2, 'solvePnPRansac', side_effect=AssertionError('must not fit rank-deficient map')):
                    result = propose_pose(**train)
                self.assertEqual(result['state'], 'unresolved')
                self.assertIn('planar_or_weak_depth_map_support', result['reasons'])
                self.assertIsNone(result['E'])

    def test_nonplanar_full_map_does_not_rescue_planar_inlier_subset(self):
        train, _, _ = fixture()
        train['points3d'][:30, 2] = 5.
        with patch.object(cv2, 'solvePnPRansac', return_value=(True, np.zeros((3, 1)), np.zeros((3, 1)), np.arange(30).reshape(-1, 1))):
            with patch.object(cv2, 'solvePnPRefineLM', side_effect=AssertionError('must reject planar consensus')):
                result = propose_pose(**train)
        self.assertEqual(result['state'], 'unresolved')
        self.assertIn('planar_or_weak_depth_ransac_inliers', result['reasons'])
        self.assertIsNone(result['E'])

    def test_collinear_pixels_do_not_count_as_pose_support(self):
        train, _, _ = fixture()
        train['xy'][:, 1] = 240.
        result = propose_pose(**train)
        self.assertEqual(result['state'], 'unresolved')
        self.assertIn('collinear_image_support', result['reasons'])

    def test_bad_ransac_output_and_insufficient_consensus_are_explicit(self):
        answers = [(False, None, None, None),
            (True, np.zeros((3, 1)), np.zeros((3, 1)), np.arange(23).reshape(-1, 1)),
            (True, np.zeros((3, 1)), np.zeros((3, 1)), np.full((30, 1), 2))]
        for answer in answers:
            with self.subTest(inliers=answer[-1]):
                train, _, _ = fixture()
                with patch.object(cv2, 'solvePnPRansac', return_value=answer):
                    result = propose_pose(**train)
                self.assertEqual(result['state'], 'unresolved')
                self.assertIsNone(result['E'])
                json.dumps(result, allow_nan=False)

    def test_refined_pose_cheirality_is_checked_without_silent_flipping(self):
        train, _, _ = fixture()
        rvec, tvec = np.zeros((3, 1)), np.array([[0.], [0.], [-20.]])
        with patch.object(cv2, 'solvePnPRansac', return_value=(True, rvec, tvec, np.arange(60).reshape(-1, 1))):
            with patch.object(cv2, 'solvePnPRefineLM', return_value=(rvec, tvec)):
                result = propose_pose(**train)
        self.assertEqual(result['state'], 'unresolved')
        self.assertIn('refined_inlier_cheirality_or_projection_failure', result['reasons'])
        self.assertIsNone(result['E'])
        self.assertEqual(result['all_training']['summary']['finite_count'], 0)
        self.assertTrue(all(row['error_px'] is None for row in result['all_training']['rows']))

    def test_validation_behind_camera_stays_unresolved_and_in_denominator(self):
        train, valid, _ = fixture()
        result = propose_pose(**train)
        valid['points3d'][0, 2] = -10.
        check = verify_pose(result, **valid)
        self.assertEqual(check['state'], 'unresolved')
        self.assertIn('validation_cheirality_or_projection_failure', check['reasons'])
        self.assertEqual(check['validation']['summary']['sample_count'], 40)
        self.assertEqual(check['validation']['summary']['finite_count'], 39)
        self.assertIsNone(check['validation']['rows'][0]['error_px'])
        json.dumps(check, allow_nan=False)

    def test_source_roles_and_sha_contract_reject_target_pixels(self):
        for name, value in [('observation_role', 'target_anchor'), ('source_kind', 'gt_projection'),
                            ('map_sha256', 'bad'), ('rgb_sha256', None), ('intrinsics_sha256', '0'*64)]:
            with self.subTest(name=name):
                train, _, _ = fixture()
                train['source'][name] = value
                result = propose_pose(**train)
                self.assertEqual(result['state'], 'unresolved')
                self.assertIsNone(result['E'])
                json.dumps(result, allow_nan=False)

    def test_validation_must_use_same_rgb_map_and_k_receipts(self):
        for name in ('rgb_sha256', 'map_sha256', 'intrinsics_sha256'):
            with self.subTest(name=name):
                train, valid, _ = fixture()
                result = propose_pose(**train)
                valid['source'][name] = '0'*64
                check = verify_pose(result, **valid)
                self.assertEqual(check['state'], 'unresolved')
                self.assertIsNone(check['validation'])

    def test_unique_ids_shape_bounds_and_finite_contract(self):
        changes = [lambda t: t['track_ids'].__setitem__(1, t['track_ids'][0]),
            lambda t: t['track_ids'].__setitem__(1, str(t['track_ids'][0])),
            lambda t: t['track_ids'].__setitem__(1, True),
            lambda t: t.update(points3d=t['points3d'][:, :2]),
            lambda t: t.update(xy=t['xy'][:-1]),
            lambda t: t['points3d'].__setitem__((0, 0), np.nan),
            lambda t: t['xy'].__setitem__((0, 0), np.inf),
            lambda t: t['xy'].__setitem__((0, 0), 641.),
            lambda t: t['K'].__setitem__((0, 0), -650.)]
        for change in changes:
            with self.subTest(change=change):
                train, _, _ = fixture()
                change(train)
                result = propose_pose(**train)
                self.assertEqual(result['state'], 'unresolved')
                self.assertIsNone(result['E'])
                json.dumps(result, allow_nan=False)

    def test_proposal_seal_catches_pose_change_before_validation(self):
        train, valid, _ = fixture()
        result = propose_pose(**train)
        result['E'][0][3] += .1
        check = verify_pose(result, **valid)
        self.assertEqual(check['state'], 'unresolved')
        self.assertIn('proposal_receipt_mismatch', check['reasons'])
        self.assertIsNone(check['validation'])

    def test_malformed_proposal_remains_explicitly_unresolved(self):
        _, valid, _ = fixture()
        for proposal in (None, [], 7, {'proposal_sha256': None}):
            check = verify_pose(proposal, **valid)
            self.assertEqual(check['state'], 'unresolved')
            self.assertIsNone(check['validation'])
            json.dumps(check, allow_nan=False)

    def test_policy_cannot_be_changed_to_fit_new_results(self):
        train, valid, _ = fixture()
        result = propose_pose(**train)
        self.assertEqual(propose_pose(**train, config=copy.deepcopy(DEFAULTS))['state'], 'fitted')
        with self.assertRaises(ValueError):
            propose_pose(**train, config={'reprojection_threshold_px': 3.})
        with self.assertRaises(ValueError):
            verify_pose(result, **valid, config={'maximum_validation_p95_px': 3.})

    def test_cv_failure_does_not_publish_a_pose(self):
        train, _, _ = fixture()
        with patch.object(cv2, 'solvePnPRansac', side_effect=cv2.error('synthetic numerical failure')):
            result = propose_pose(**train)
        self.assertEqual(result['state'], 'unresolved')
        self.assertIsNone(result['E'])
        self.assertTrue(any(r.startswith('pose_numerical_failure:') for r in result['reasons']))


if __name__ == '__main__':
    unittest.main()
