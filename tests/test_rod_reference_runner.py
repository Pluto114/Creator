"""Protect RGB annotation provenance and the pre-registered zoom comparison."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_rod_reference as runner  # noqa: E402


class ReferenceContracts(unittest.TestCase):
    def fixture(self, folder):
        rgb = folder / 'rgb.png'
        rgb.write_bytes(b'identity fixture only; no image decoding in annotation seal')
        frames = [dict(view_id=v, rgb=str(rgb), rgb_sha256=runner.digest(rgb), size_wh=[640, 480]) for v in ('a', 'b')]
        manifest = dict(cases=[dict(case_id='r01', frames=frames)])
        annotations = dict(ground_truth_used_for_coordinates=False, predicted_geometry_used_for_coordinates=False,
                           queries=[dict(case_id='r01', anchors=[dict(view_id=f['view_id'], rgb_sha256=f['rgb_sha256'], xy=[320, 120]) for f in frames])])
        path = folder / 'claims.json'
        path.write_text(json.dumps(annotations), encoding='utf-8')
        return path, annotations, (folder, folder, {'input_sha256': 'frozen-input'}, manifest)

    def test_seal_binds_rgb_and_pre_prediction_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path, annotations, state = self.fixture(Path(temp))
            with patch.object(runner, 'checked_run', return_value=state):
                runner.seal_annotations('test', path)
            self.assertEqual(runner.checked_annotations(state[0], state[2]), annotations)
            (state[0] / 'annotations.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'freeze changed'):
                runner.checked_annotations(state[0], state[2])

    def test_foreign_image_and_truth_derived_clicks_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, annotations, state = self.fixture(Path(temp))
            for field in ('ground_truth_used_for_coordinates', 'predicted_geometry_used_for_coordinates', 'foreign_image'):
                bad = copy.deepcopy(annotations)
                if field == 'foreign_image':
                    bad['queries'][0]['anchors'][0]['rgb_sha256'] = 'some other photo'
                else:
                    bad[field] = True
                path.write_text(json.dumps(bad), encoding='utf-8')
                with patch.object(runner, 'checked_run', return_value=state), self.assertRaises(ValueError):
                    runner.seal_annotations('test', path)

    def test_annotation_after_partial_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _, state = self.fixture(Path(temp))
            (state[0] / 'r01/prediction').mkdir(parents=True)
            with patch.object(runner, 'checked_run', return_value=state), self.assertRaisesRegex(ValueError, 'before model'):
                runner.seal_annotations('test', path)

    def test_zoom_pair_and_method_are_controlled(self):
        config = runner.read_json(ROOT / 'configs/rod_reference_v1.json')
        reference, zoom = config['cases'][1], config['cases'][3]
        for key in ('target', 'gap_segment', 'reference_seed', 'distractors', 'occluder'):
            self.assertEqual(reference[key], zoom[key])
        self.assertEqual(len(set(reference['lens_by_view_mm'])), 1)
        self.assertGreater(len(set(zoom['lens_by_view_mm'])), 1)
        old = runner.read_json(ROOT / 'configs/camera_bundle_shared_v1.json')
        self.assertEqual(config['calibration']['optimizer'], old['optimizer'])
        self.assertEqual(config['calibration']['decision'], old['decision'])
        for variant in config['calibration']['variants']:
            self.assertIn(variant, old['variants'])


if __name__ == '__main__':
    unittest.main()
