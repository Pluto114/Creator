"""The height intervention must not quietly change the rod or its algorithms."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from PIL import Image, PngImagePlugin  # noqa: E402
from report_rod_height import rgb_digest  # noqa: E402


class HeightProtocol(unittest.TestCase):
    def test_png_timestamps_do_not_change_decoded_rgb_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b, c = [Path(temp) / (name + '.png') for name in ('a', 'b', 'c')]
            pixels = Image.new('RGB', (8, 6), (10, 20, 30))
            meta = PngImagePlugin.PngInfo()
            meta.add_text('Date', 'a later render')
            pixels.save(a)
            pixels.save(b, pnginfo=meta)
            self.assertNotEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(rgb_digest(a), rgb_digest(b))
            pixels.putpixel((2, 3), (11, 20, 30))
            pixels.save(c)
            self.assertNotEqual(rgb_digest(a), rgb_digest(c))

    def test_only_height_sequence_changes_in_render_controls(self):
        def read(name):
            return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8-sig'))
        before, after = read('rod_reference_v1.json'), read('rod_height_v1.json')
        self.assertEqual(before['cases'][:3], after['cases'])
        generator = dict(after['generator'])
        heights = generator.pop('camera_heights_m')
        self.assertEqual(generator, before['generator'])
        self.assertEqual(len(heights), len(generator['angles_degrees']))
        self.assertEqual(heights[0], generator['camera_height_m'])
        self.assertGreater(max(heights) - min(heights), 1.0)
        for name in ('calibration', 'rod_method', 'tracks', 'evaluation'):
            self.assertEqual(after[name], before[name])

    def test_frozen_exporter_diff_is_only_explicit_worker_and_default_config(self):
        # Old export remains replayable. This version deliberately duplicates
        # its small orchestrator instead of modifying an already frozen run.
        old = (ROOT / 'scripts/prepare_rod_reference.py').read_text(encoding='utf-8')
        new = (ROOT / 'scripts/prepare_rod_height.py').read_text(encoding='utf-8')
        normalized = new.replace('Frozen height intervention, using the established paired export implementation.',
                                 'Freeze new Blender data; only RGB and screen guides enter inference inputs.')
        normalized = normalized.replace('configs/rod_height_v1.json', 'configs/rod_reference_v1.json')
        normalized = normalized.replace('scripts/blender_rod_height_pack.py', 'scripts/blender_rod_reference_pack.py')
        self.assertEqual(normalized, old)


if __name__ == '__main__':
    unittest.main()
