"""Conditioned camera identity, half pixels, and real reversible bundle plumbing."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "reconstruction/src"))
import run_fixture_point_patch as runner  # noqa: E402
from creator_eval.fixture_depth_contract import (  # noqa: E402
    camera_inputs,
    exported_cameras,
    source_roundtrip,
)
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from creator_recon.domain.camera_point_snapshot import materialize, native_intrinsics  # noqa: E402


def fixture():
    k = [[100., 0., 39.2], [0., 100., 29.6], [0., 0., 1.]]
    cameras = [dict(view_id=f"view_{i:02d}", state="validated", K_index=k,
                    world_to_camera_cv=np.c_[np.eye(3), [-x, 0., 0.]].tolist())
               for i, x in enumerate([-1., -.5, 0., .5, 1.])]
    edge, e = camera_inputs(cameras)
    native = dict(depth=np.full((5, 6, 8), 6., np.float32), intrinsics=(np.diag([.1, .1, 1]) @ edge).astype(np.float32),
                  extrinsics=e[:, :3].astype(np.float32))
    return cameras, native


class FixtureDepthContractTests(unittest.TestCase):
    def test_half_pixel_conversion_matches_resize_ray_identity(self):
        cameras, native = fixture()
        before = copy.deepcopy(cameras)
        k, e, lifted = exported_cameras(native, cameras, [80, 60])
        expected = native_intrinsics([c["K_index"] for c in cameras], [80, 60], [8, 6])
        np.testing.assert_allclose(k, expected, atol=2e-7, rtol=0)
        np.testing.assert_array_equal(e, native["extrinsics"])
        for actual, old in zip(lifted, cameras):
            np.testing.assert_allclose(actual["K_index"], old["K_index"], atol=2e-6, rtol=0)
        self.assertEqual(before, cameras)
        # A mistaken second -0.5 shift must not pass as the expected export.
        bad = copy.deepcopy(native)
        bad["intrinsics"][:, :2, 2] -= .5
        with self.assertRaises(ValueError):
            exported_cameras(bad, cameras, [80, 60])

    def test_pose_version_mismatch_and_unvalidated_camera_are_rejected(self):
        cameras, native = fixture()
        bad = copy.deepcopy(native)
        bad["extrinsics"][2, 0, 3] += .01
        with self.assertRaises(ValueError):
            exported_cameras(bad, cameras, [80, 60])
        cameras[0]["state"] = "withheld"
        with self.assertRaises(ValueError):
            camera_inputs(cameras)

    def test_nonfinite_depth_and_reflected_camera_are_rejected(self):
        cameras, native = fixture()
        for value in (0., -1., float("nan")):
            bad = copy.deepcopy(native)
            bad["depth"][0, 0, 0] = value
            with self.assertRaises(ValueError):
                exported_cameras(bad, cameras, [80, 60])
        cameras[0]["world_to_camera_cv"][0][0] = -1
        with self.assertRaises(ValueError):
            camera_inputs(cameras)

    def test_original_pixel_ids_roundtrip_and_corruption_is_detected(self):
        cameras, native = fixture()
        k, e, _ = exported_cameras(native, cameras, [80, 60])
        points, ids = materialize(native["depth"], k, e)
        result = source_roundtrip(points, ids, native["depth"], k, e)
        self.assertEqual(result["pixel_checks"], 240)
        self.assertLess(result["maximum_pixel_error"], 1e-12)
        ids[0, 2] = 1
        with self.assertRaises(ValueError):
            source_roundtrip(points, ids, native["depth"], k, e)

    def test_full_bundle_compose_reopen_and_withdraw_use_same_source(self):
        cameras, native = fixture()
        method = json.loads((ROOT / "configs/rod_fixture_narrow_v1.json").read_text())["method"]
        config = json.loads((ROOT / "configs/fixture_point_patch_v1.json").read_text())
        case = dict(case_id="r01", cameras=cameras, parent_record="parent.json",
            frames=[dict(view_id=c["view_id"], rgb_sha256="1"*64, size_wh=[80, 60], guide_xyxy=[[40, 1], [40, 58]]) for c in cameras])
        rows = [dict(method=name, segments=[[[0., 0., 4.], [0., 0., 5.]]], rejection_reasons=[])
                for name in ("baseline", "cylinder_support")]
        calls = []
        def reader(points, segments, policy, region):
            calls.append((points.tobytes(), len(segments), canonical_hash(region), policy["voxel_size"]))
            return dict(segments=segments, state="complete", resolution_state="resolved", region_point_count=len(points), occupied_voxels=12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "r01"
            folder.mkdir()
            np.savez_compressed(folder / "prediction.npz", **native)
            np.savez_compressed(folder / "raw-before-api.npz", **native)
            runner.write(root / "parent.json", dict(frames=[]))
            runner.write(folder / "depth-report.json", dict(state="complete", gt_read=False,
                prediction_sha256=runner.digest(folder / "prediction.npz"), raw_sha256=runner.digest(folder / "raw-before-api.npz"),
                camera_input_sha256=canonical_hash(cameras)))
            with patch.object(runner, "ROOT", root), patch.object(runner, "RUN", root), patch.object(runner, "readout", reader), patch.object(runner, "reconstruct_fixture_narrow", return_value=dict(methods=rows)):
                result = runner.compose_case(case, method, config)
            self.assertEqual(result["full_point_count"], 240)
            self.assertEqual(len(result["patches"]), 2)
            self.assertEqual(len(result["graphs"]), 6)
            self.assertTrue(result["withdrawn_bytes_equal"])
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(c[0] == calls[0][0] and c[2] == calls[0][2] for c in calls))
            self.assertEqual([c[1] for c in calls], [0, 1, 1, 0, 1, 1])

    def test_normal_process_cannot_open_truth_or_results(self):
        code = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'scripts'))
import run_fixture_point_patch as r
sys.addaudithook(r.block_truth)
for path in ['data/eval_gt/never.json', 'docs/experiments/results/never.json', '.runtime/any/protocol.json']:
    try:
        Path(path).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError(path)
print('DEPTH_INPUT_GUARD_OK')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("DEPTH_INPUT_GUARD_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
