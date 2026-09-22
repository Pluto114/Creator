"""Camera-transfer boundaries; geometric success is measured separately on real runs."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/src"))
from audit_foreground_camera_transfer import substitute_cameras  # noqa: E402
from creator_eval.native_diagnostics import align_cameras  # noqa: E402
from run_foreground_estimated_pilot import require_camera_free  # noqa: E402


class EstimatedPipelineTests(unittest.TestCase):
    def test_nested_privileged_camera_cannot_hide_inside_a_pool(self):
        require_camera_free({"frames": [{"view_id": "a", "pool": {"candidates": [{"slope": .2}]}}]})
        for key in ("K_index", "world_to_camera_cv", "intrinsics", "gt_axis"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_camera_free({"frames": [{"pool": {"extra": {key: []}}}]})

    def test_frozen_alignment_configuration_handles_a_known_camera_transform(self):
        config = json.loads((ROOT / "configs/foreground_estimated_pilot_v1.json").read_text())["alignment"]
        cameras = np.array([np.c_[np.eye(3), -np.array(c)] for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))])
        scale, rotation, translation, report = align_cameras(cameras, cameras, config)
        self.assertAlmostEqual(scale, 1.)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, 0., atol=1e-12)
        self.assertFalse(report["poor_camera_fit"])
        self.assertEqual(len(report["leave_one_camera_out"]), 4)

    def test_new_entry_points_start_in_a_fresh_process(self):
        for name in ("run_foreground_estimated_pilot.py", "run_g1_point_patch_split.py", "run_common_readout_split.py"):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / name), "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_privileged_pose_transfer_preserves_projection_and_original_inputs(self):
        matrix = np.array([[0., -2., 0., 3.], [2., 0., 0., -1.], [0., 0., 2., 4.], [0., 0., 0., 1.]])
        intrinsic = [[80., 0., 40.], [0., 80., 30.], [0., 0., 1.]]
        camera = np.c_[np.eye(3), [0., 0., 8.]].tolist()
        frames = [dict(view_id="v", rgb_sha256="a" * 64, K_index=intrinsic, world_to_camera_cv=camera)]
        original = json.dumps(frames)
        changed, scale, _, _ = substitute_cameras(frames, frames, matrix, "oracle_intrinsics_and_pose")
        point = np.array([.3, -.2, 1., 1.])
        expected = np.array(camera) @ (matrix @ point)
        actual = changed[0]["world_to_camera_cv"] @ point
        np.testing.assert_allclose(actual * scale, expected, atol=1e-12)
        self.assertEqual(json.dumps(frames), original)


if __name__ == "__main__":
    unittest.main()
