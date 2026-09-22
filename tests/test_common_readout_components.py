"""The straight-component reader must handle surfaces, not only zero-width axes."""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from check_common_readout_ceiling import tube  # noqa: E402
from creator_eval.common_readout_components import merged_segments, readout  # noqa: E402
from creator_eval.line_controls import curve_metrics  # noqa: E402


class ComponentReadoutTests(unittest.TestCase):
    def test_noisy_finite_tube_has_high_centerline_coverage(self):
        segment = [[.013, .017, .011], [.013, .017, 1.011]]
        points = tube(segment, .02, .0025, 22092026)
        result = readout(points, np.empty((0, 2, 3)), {"voxel_size": .02})
        score = curve_metrics(result["segments"], [segment], tolerance=.025, spacing=.005)
        self.assertGreater(score["recovery_fraction"], .9)
        self.assertGreater(score["precision_fraction"], .9)

    def test_separate_parallel_members_do_not_become_a_middle_axis(self):
        first = np.array([[.013, .017, .011], [.013, .017, 1.011]])
        second = first + [.16, 0, 0]
        points = np.concatenate([tube(s, .02, .0025, 1) for s in (first, second)])
        result = readout(points, np.empty((0, 2, 3)), {"voxel_size": .02})
        self.assertEqual(result["components"], 2)
        self.assertGreater(curve_metrics(result["segments"], [first, second], tolerance=.025, spacing=.005)["recovery_fraction"], .9)

    def test_branch_is_rejected_as_unsupported_straight_geometry(self):
        points = np.concatenate([tube(np.asarray(s, float), 0., 0., 1) for s in ([[0, 0, 0], [0, 0, 1]], [[0, 0, .5], [.4, 0, .5]])])
        result = readout(points, np.empty((0, 2, 3)), {"voxel_size": .02})
        self.assertEqual(result["components"], 0)

    def test_overlap_merge_does_not_close_a_real_gap(self):
        segments = np.array([[[0, 0, 0], [0, 0, 1]], [[0, 0, .5], [0, 0, 1.2]], [[0, 0, 1.5], [0, 0, 2]]], float)
        result = merged_segments(segments, dict(merge_angle_degrees=3., merge_distance_voxels=.75, voxel_size=.02))
        self.assertEqual(len(result), 2)
        self.assertTrue(any(np.allclose(s[:, 2], [0, 1.2]) for s in result))
        self.assertTrue(any(np.allclose(s[:, 2], [1.5, 2]) for s in result))

    def test_replay_entry_point_starts_without_a_hidden_import_order(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_g1_point_patch_components.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
