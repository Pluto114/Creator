"""Known-image fixtures for finite endpoints/gaps; no private run input needed."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/src'))
from creator_eval import rod_fixture_finite as pilot  # noqa: E402

METHOD = json.loads((ROOT / 'configs/rod_fixture_finite_v1.json').read_text(encoding='utf-8'))['method']


def fixture(kind='complete'):
    k = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    frames, images, cameras = [], [], []
    for index, (cx, cy) in enumerate(zip([-1.4, -.7, 0., .7, 1.4], [0., .1, -.1, .2, -.2])):
        rgb = np.full((480, 640, 3), 180, np.uint8)
        x = int(round(320.-100.*cx))
        y0, y1 = int(round(140.-100.*cy)), int(round(340.-100.*cy))
        if kind != 'blank':
            half = 0 if kind == 'thin' else 3
            rgb[y0:y1+1, x-half:x+half+1] = 35
            if kind == 'gap':
                rgb[int(round(220.-100.*cy)):int(round(261.-100.*cy)), x-half:x+half+1] = 180
        name = 'view_' + str(index).zfill(2)
        frames.append(dict(view_id=name, rgb='fixture.png', rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            size_wh=[640, 480], guide_xyxy=[[x, 20], [x, 459]], guide_source='synthetic_test_only'))
        images.append(rgb)
        cameras.append(dict(view_id=name, K_index=k.tolist(), world_to_camera_cv=np.c_[np.eye(3), [-cx, -cy, 0.]].tolist(), state='validated'))
    return frames, images, cameras


class FixtureFiniteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cached = {}
        for kind in ('complete', 'gap', 'thin'):
            frames, images, cameras = fixture(kind)
            evidence = pilot.extract_fixture_evidence(frames, images, METHOD)
            result = pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)
            cls.cached[kind] = (evidence, cameras, result)

    def test_complete_known_image_has_finite_endpoints_not_infinite_line(self):
        evidence, cameras, result = self.cached['complete']
        self.assertTrue(result['camera_gate']['passed'])
        self.assertEqual([m['method'] for m in result['methods']], list(pilot.METHODS))
        for row in result['methods']:
            self.assertEqual(row['state'], 'accepted', row['rejection_reasons'])
            self.assertEqual(row['segment_count'], 1)
            segment = np.asarray(row['segments'])[0]
            np.testing.assert_allclose(segment[:, [0, 2]], [[0., 6.], [0., 6.]], atol=1e-8)
            self.assertTrue(np.all(np.abs(segment[:, 1]) <= 1. + 1e-8))
            self.assertGreater(row['total_length_m'], 1.94)
            self.assertFalse(row['finite']['ablations']['negative_veto'])
            self.assertGreater(min(len(s['row_matches']) for s in row['finite']['candidate_selection']), 190)
        self.assertEqual([f['raw_row_count'] for f in evidence], [440]*5)
        self.assertEqual(result['pool_budget']['candidate_cap'], 8)
        self.assertEqual(result['pool_budget']['truncated_views'], [])
        self.assertIn('not all image explanations', result['pool_budget']['search_scope'])
        json.dumps(result, allow_nan=False)
        self.assertEqual(result['camera_input_sha256'], pilot.canonical_hash(cameras))

    def test_gap_keeps_two_pieces_and_no_positive_bridge(self):
        _, _, result = self.cached['gap']
        for row in result['methods']:
            self.assertEqual(row['state'], 'accepted', row['rejection_reasons'])
            self.assertEqual(row['segment_count'], 2)
            for segment in row['segments']:
                self.assertFalse(min(p[1] for p in segment) < 0 < max(p[1] for p in segment))
            self.assertLess(row['total_length_m'], 1.60)
            self.assertFalse(np.any(row['finite']['evidence']['absent_view_counts']))

    def test_unresolved_one_pixel_rod_does_not_get_invented_center(self):
        evidence, _, result = self.cached['thin']
        self.assertTrue(all(f['full_pool_count'] == 0 for f in evidence))
        for row in result['methods']:
            self.assertEqual(row['state'], 'rejected')
            self.assertEqual(row['segments'], [])
            self.assertEqual(row['total_length_m'], 0.)
            self.assertTrue(row['rejection_reasons'])

    def test_camera_withheld_preserves_empty_method_slots_and_pixel_evidence(self):
        evidence, cameras, _ = copy.deepcopy(self.cached['complete'])
        cameras[2].update(state='unresolved', K_index=None, world_to_camera_cv=None)
        before = pilot.canonical_hash(evidence)
        with patch.object(pilot, 'associate_multiview_lines_cached', side_effect=AssertionError('must not fit')):
            result = pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)
        self.assertFalse(result['camera_gate']['passed'])
        self.assertEqual([r['state'] for r in result['methods']], ['withheld_camera']*2)
        self.assertTrue(all(r['segments'] == [] and r['finite'] is None for r in result['methods']))
        self.assertEqual(before, pilot.canonical_hash(evidence))

    def test_pixel_and_camera_inputs_are_unchanged(self):
        evidence, cameras, _ = copy.deepcopy(self.cached['complete'])
        before = pilot.canonical_hash((evidence, cameras))
        pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)
        self.assertEqual(before, pilot.canonical_hash((evidence, cameras)))

    def test_no_gt_fields_or_identity_claim_are_required(self):
        evidence, cameras, _ = self.cached['complete']
        result = pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)
        self.assertFalse(result['target_geometry_used'])
        self.assertFalse(result['camera_modified'])
        self.assertEqual(result['coordinate_system'], 'declared_metric_fixture_world')
        self.assertIn('not_automatic_foreground_identity', result['identity_scope'])
        self.assertNotIn('target', evidence[0])

    def test_changed_method_or_detached_evidence_cannot_run(self):
        evidence, cameras, _ = copy.deepcopy(self.cached['complete'])
        method = copy.deepcopy(METHOD)
        method['extent']['maximum_reprojection_error_px'] = 20.
        with self.assertRaises(ValueError):
            pilot.reconstruct_fixture_finite(evidence, cameras, method)
        evidence[0]['pool']['candidates'][0]['row_matches'][0][1] += 1
        with self.assertRaisesRegex(ValueError, 'evidence or candidate pool'):
            pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)

    def test_view_pairing_and_validated_camera_shape_are_checked(self):
        evidence, cameras, _ = copy.deepcopy(self.cached['complete'])
        with self.assertRaisesRegex(ValueError, 'view order'):
            pilot.reconstruct_fixture_finite(evidence, cameras[::-1], METHOD)
        cameras[0]['K_index'] = None
        with self.assertRaises(ValueError):
            pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)
        frames, images, _ = fixture()
        frames[1]['view_id'] = frames[0]['view_id']
        with self.assertRaisesRegex(ValueError, 'five distinct'):
            pilot.extract_fixture_evidence(frames, images, METHOD)

    def test_rejected_nonfinite_projection_diagnostic_is_null_never_zero(self):
        evidence, cameras, _ = self.cached['complete']
        failure = dict(state='rejected', segments=np.empty((0, 2, 3)),
            shadow_segments=np.empty((0, 2, 3)), rejection_reasons=['view_cheirality'],
            evidence={'projected_xy': np.array([[[np.nan, np.nan]]])})
        with patch.object(pilot, 'bound_selected_candidate', return_value=failure):
            result = pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)
        self.assertEqual(len(result['methods']), 2)
        self.assertTrue(all(row['segments'] == [] for row in result['methods']))
        self.assertEqual(result['methods'][0]['finite']['evidence']['projected_xy'], [[[None, None]]])
        self.assertEqual(len(result['nonfinite_diagnostic_fields']), 4)
        json.dumps(result, allow_nan=False)

    def test_incomplete_association_budget_never_emits_a_winner(self):
        evidence, cameras, _ = self.cached['complete']
        with patch.object(pilot, 'associate_multiview_lines_cached', return_value={'search_complete': False}):
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                pilot.reconstruct_fixture_finite(evidence, cameras, METHOD)

    def test_normal_runner_installs_truth_block_before_reading_inputs(self):
        code = """
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import run_rod_fixture_finite as run
checks = []
def probe():
    for item in ('data/eval_gt/nonexistent.json', 'data/evaluation/nonexistent.json',
                 'docs/experiments/results/nonexistent.json', '.runtime/experiments/any/protocol.json',
                 '.runtime/experiments/any/generation-checks.json'):
        try:
            open(Path.cwd() / item)
        except PermissionError:
            checks.append(item)
        else:
            raise AssertionError('Truth block missing')
    raise RuntimeError('before-inputs-proved')
run.checked = probe
try:
    run.infer()
except RuntimeError as error:
    assert str(error) == 'before-inputs-proved'
assert len(checks) == 5
"""
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        completed = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
            env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
