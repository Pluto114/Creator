"""Input linkage and conservative decision contracts for the paired evidence run."""
import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_rod_validation_evidence as runner  # noqa: E402


class EvidenceRunnerTests(unittest.TestCase):
    def test_empty_positive_output_is_zero_even_without_alignment(self):
        self.assertEqual(runner.emitted_metrics(None, False), dict(emitted_recovery=0., emitted_precision=None))
        self.assertEqual(runner.emitted_metrics(None, True), dict(emitted_recovery=None, emitted_precision=None))
        self.assertEqual(runner.emitted_metrics(dict(recovery_fraction=.7, precision_fraction=.8), True),
                         dict(emitted_recovery=.7, emitted_precision=.8))

    def test_declared_policy_must_match_the_executed_checker(self):
        policy = dict(endpoint=dict(threshold_px=2., minimum_views=4), target=dict(threshold_px=2., minimum_views=2))
        runner.validate_screen_policy(policy)
        for component, field, changed in [('endpoint', 'threshold_px', 3.), ('endpoint', 'minimum_views', 3),
                                           ('target', 'threshold_px', 1.), ('target', 'minimum_views', 1)]:
            bad = copy.deepcopy(policy)
            bad[component][field] = changed
            with self.assertRaises(AssertionError):
                runner.validate_screen_policy(bad)

    def test_conditions_cannot_promote_missing_geometry_or_withheld_camera(self):
        camera = dict(state='candidate_camera_correction')
        good = dict(state='supported')
        no = dict(state='unresolved')
        bad = dict(state='contradicted')
        identity = dict(state='accepted')
        for arm in runner.ARMS:
            self.assertEqual(runner.decision(dict(state='withhold_correction'), identity, good, good, arm)['state'], 'withheld_camera')
            self.assertEqual(runner.decision(camera, dict(state='unavailable'), good, good, arm)['state'], 'unresolved')
        self.assertEqual(runner.decision(camera, identity, good, no, 'joint')['state'], 'unresolved')
        self.assertEqual(runner.decision(camera, identity, bad, no, 'joint')['state'], 'rejected')
        self.assertEqual(runner.decision(camera, identity, bad, good, 'target_claims')['state'], 'accepted')
        self.assertEqual(runner.decision(camera, identity, good, bad, 'endpoints')['state'], 'accepted')
        with self.assertRaises(ValueError):
            runner.decision(camera, identity, good, good, 'best_after_gt')

    def test_pixel_edits_cannot_hide_behind_updated_case_or_frame_hashes(self):
        protocol = json.loads((ROOT / 'configs/camera_evidence_challenges_v1.json').read_text(encoding='utf-8'))
        inputs, _ = runner.generate_paired_challenges(protocol)
        runner.validate_inputs(inputs)
        edited = copy.deepcopy(inputs)
        edited['cases'][0]['rod_tracks'][0]['endpoint_tracks'][0]['observations'][0]['xy'][0] += 1
        edited['cases'][0]['rod_support_sha256'] = runner.canonical_hash(edited['cases'][0]['rod_tracks'])
        with self.assertRaisesRegex(AssertionError, 'Detached measurement bank'):
            runner.validate_inputs(edited)
        edited = copy.deepcopy(inputs)
        edited['camera_groups'][0]['frames'][0]['measurement_bank']['training'][0]['xy'][0] += 1
        edited['camera_groups'][0]['frames'][0]['source_sha256'] = runner.canonical_hash(edited['camera_groups'][0]['frames'][0]['measurement_bank'])
        with self.assertRaisesRegex(AssertionError, 'Detached measurement bank'):
            runner.validate_inputs(edited)
        edited = copy.deepcopy(inputs)
        edited['cases'][0]['camera_group_id'] = next(g['camera_group_id'] for g in inputs['camera_groups'] if g['camera_group_id'] != edited['cases'][0]['camera_group_id'])
        with self.assertRaises(AssertionError):
            runner.validate_inputs(edited)

    def test_normal_stages_install_truth_tripwire_before_loading_inputs(self):
        code = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import run_rod_validation_evidence as r
stage=sys.argv[1]
def blocked():
    (r.ROOT / 'data/eval_gt' / 'should_not_be_opened.json').read_text()
r.checked=blocked
if stage=='infer':
    r.infer()
else:
    r.worker({}, [], {}, {})
'''
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        for stage in ('infer', 'worker'):
            result = subprocess.run([sys.executable, '-B', '-c', code, stage], cwd=ROOT,
                env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('PermissionError', result.stderr)
            self.assertNotIn('FileNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
