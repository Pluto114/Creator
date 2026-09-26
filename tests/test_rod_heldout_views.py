"""Contracts for genuinely new observations and unchanged old geometry."""
import json
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_rod_heldout_views as run  # noqa: E402


class HeldoutViewRunnerTests(unittest.TestCase):
    def test_two_old_views_must_agree_and_conflicting_new_features_are_not_ranked(self):
        graph = [[[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]]]
        pairs = [[[0, 0], [1, 1]], [[0, 0], [1, 1]], []]
        rows = run.consensus_matches(graph, pairs, np.array([[100, 220], [120, 230]]))
        self.assertEqual([r['state'] for r in rows], ['matched', 'matched'])
        conflict = deepcopy(pairs)
        conflict[2] = [[1, 0]]
        rows = run.consensus_matches(graph, conflict, np.array([[100, 220], [120, 230]]))
        self.assertEqual(rows[0]['state'], 'excluded')
        # Two new feature IDs claim one old track; neither wins by residual.
        graph = [[[v, 0] for v in range(4)]]
        duplicate = [[[0, 0]], [[0, 0]], [[0, 1]], [[0, 1]]]
        rows = run.consensus_matches(graph, duplicate, np.array([[100, 220], [120, 230]]))
        self.assertEqual([r['state'] for r in rows], ['excluded', 'excluded'])
        self.assertTrue(all(r['reason'] == 'multiple_new_features_claim_one_old_track' for r in rows))

    def test_training_validation_and_guard_are_fixed_before_residuals(self):
        self.assertEqual(run.split_role('train', [100, 192]), 'train')
        self.assertEqual(run.split_role('validation', [100, 127.9]), 'validation')
        for label, y in [('train', 127), ('validation', 192), ('validation', 128), ('train', 191.9), ('excluded', 220)]:
            self.assertEqual(run.split_role(label, [100, y]), 'excluded')

    def test_new_evidence_cannot_promote_old_rejections_or_bad_localization(self):
        self.assertEqual(run.final_state('rejected', 'candidate_camera_correction', [True, True], 'supported'), 'original_identity_not_accepted')
        self.assertEqual(run.final_state('accepted', 'withhold_correction', [True, True], 'supported'), 'withheld_original_camera')
        self.assertEqual(run.final_state('accepted', 'candidate_camera_correction', [True, False], 'supported'), 'unresolved_new_pose')
        self.assertEqual(run.final_state('accepted', 'candidate_camera_correction', [True, True], 'contradicted'), 'rejected_new_target')
        self.assertEqual(run.final_state('accepted', 'candidate_camera_correction', [True, True], 'supported'), 'retained_candidate')

    def test_fresh_normal_stage_installs_gt_and_public_score_blocks_first(self):
        code = """
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import run_rod_heldout_views as run  # noqa: E402
checked = []
def hook_check():
    for name in ['data/eval_gt/no-file.json', 'data/evaluation/no-file.json', 'docs/experiments/results/no-file.json']:
        try:
            open(str(Path.cwd() / name))
        except PermissionError:
            checked.append(name)
        else:
            raise AssertionError('Missing GT tripwire')
    raise RuntimeError('tripwire-first-ok')
run.checked = hook_check
try:
    run.infer()
except RuntimeError as error:
    assert str(error) == 'tripwire-first-ok'
assert len(checked) == 3
"""
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_pose_verification_state_reaches_the_joint_decision(self):
        rng = np.random.default_rng(260926)
        xyz = rng.uniform([-1., -.6, 3.], [1., .6, 6.], (100, 3))
        k = np.array([[500., 0, 319.5], [0, 500., 239.5], [0, 0, 1.]])
        xy = (xyz @ k.T)[:, :2] / xyz[:, 2, None]
        source = dict(source_kind='rgb', rgb_sha256='a'*64, map_sha256='b'*64,
            intrinsics_sha256=run.canonical_hash(k), size_wh=[640, 480])
        proposal = run.propose_pose(xyz[:70], xy[:70], k, track_ids=list(range(70)),
            source={**source, 'observation_role': 'background_pose_training'})
        validation = run.verify_pose(proposal, xyz[70:], xy[70:], track_ids=list(range(70, 100)),
            source={**source, 'observation_role': 'background_pose_validation'})
        self.assertEqual(validation['state'], 'validated', validation)
        accepted = validation['state'] == 'validated'
        self.assertEqual(run.final_state('accepted', 'candidate_camera_correction', [accepted, accepted], 'supported'), 'retained_candidate')

    def test_pose_data_cannot_depend_on_clicks(self):
        # A small ordinary map tests the data boundary without running OpenCV.
        camera = dict(result=dict(intrinsics=np.repeat(np.eye(3)[None], 3, axis=0),
            extrinsics=np.tile(np.c_[np.eye(3), np.zeros(3)][None], (3, 1, 1)),
            kept_track_ids=[0], points=[[.1, .2, 2.]]))
        tracks = dict(graph=dict(tracks=[[[0, 0], [1, 0], [2, 0]]]),
            feature_xy=[[[.05, .1]]] * 3)
        matched = dict(matches=[dict(role='train', old_track_ids=[0], xy=[100., 220.], new_feature_id=1)])
        from unittest.mock import patch
        with patch.object(run, 'read_json', return_value=tracks):
            task = dict(tracks_path='unused', query={'anchors': []})
            before = json.dumps(run.map_observations(task, camera, matched))
            task['query']['anchors'] = [dict(xy=[999, -200])]
            after = json.dumps(run.map_observations(task, camera, matched))
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
