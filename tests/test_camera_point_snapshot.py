"""Depth/camera geometry identity and integer-pixel correspondence contracts."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reconstruction/src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from creator_recon.domain.camera_point_snapshot import (  # noqa: E402
    materialize,
    native_intrinsics,
    write_camera_snapshot,
)
from creator_recon.domain.point_patch import compose, write_patch  # noqa: E402
from run_rod_evidence import original_intrinsics  # noqa: E402


class CameraPointSnapshotTests(unittest.TestCase):
    def inputs(self):
        depth = np.arange(12, dtype=np.float32).reshape(1, 3, 4) / 10 + 2
        k = np.array([[[8., 0, 1.5], [0, 8., 1], [0, 0, 1]]])
        e = np.array([[[0., 0, 1, 2], [0, 1, 0, -.2], [-1, 0, 0, .8]]])
        frames = [dict(frame_id='a', image_sha256='1' * 64, prediction_size_wh=[4, 3])]
        return depth, k, e, frames

    def test_exact_native_pixel_and_depth_roundtrip_in_rotated_camera(self):
        depth, k, e, _ = self.inputs()
        depth[0, 0, 0], depth[0, 1, 1] = np.nan, -1
        points, ids = materialize(depth, k, e)
        self.assertEqual(len(points), 10)
        xyz = points @ e[0, :, :3].T + e[0, :, 3]
        uv = xyz @ k[0].T
        np.testing.assert_allclose(uv[:, :2] / uv[:, 2, None], ids[:, [2, 1]], atol=1e-12)
        np.testing.assert_allclose(xyz[:, 2], depth[ids[:, 0], ids[:, 1], ids[:, 2]], atol=1e-12)

    def test_half_pixel_resize_is_inverse_of_existing_lift(self):
        _, k, _, _ = self.inputs()
        native = native_intrinsics(k, [640, 480], [504, 378])
        np.testing.assert_allclose(original_intrinsics(native, [640, 480], [504, 378]), k, atol=1e-12)

    def test_camera_change_produces_new_base_and_rejects_old_patch(self):
        depth, k, e, frames = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = write_camera_snapshot(root / 'a', depth, k, e, frames=frames, prediction_sha256='2' * 64, camera_source_sha256='3' * 64)
            changed = e.copy()
            changed[0, 0, 3] += .1
            b = write_camera_snapshot(root / 'b', depth, k, changed, frames=frames, prediction_sha256='2' * 64, camera_source_sha256='4' * 64)
            self.assertEqual(a['depth_content_sha256'], b['depth_content_sha256'])
            self.assertNotEqual(a['snapshot_id'], b['snapshot_id'])
            self.assertNotEqual(a['geometry_source_id'], b['geometry_source_id'])
            write_patch(root / 'patch', root / 'a/base', np.empty((0, 2, 3)), np.empty((0, 3), np.uint32),
                selection_sha256='5' * 64, method=dict(id='test', version='1', config_sha256='6' * 64, seed=0), evidence=[], unresolved=['test'])
            with self.assertRaisesRegex(ValueError, 'Stale patch'):
                compose(root / 'b/base', root / 'patch')


if __name__ == '__main__':
    unittest.main()
