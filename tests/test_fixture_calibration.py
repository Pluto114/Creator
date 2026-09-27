"""Fixture assumptions and withheld-marker independence, without experiment GT."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.fixture_calibration import (  # noqa: E402
    POLICY,
    calibrate_case,
    detect_markers,
    marker_image,
    observations,
    validate_cad,
)


class FixtureCalibrationTests(unittest.TestCase):
    def cad(self):
        return json.loads((ROOT / "configs/rod_fixture_cad_v1.json").read_text())

    def frames(self):
        cad = self.cad()
        k = np.array([[845., 0, 313.2], [0, 845., 242.3], [0, 0, 1]])
        frames = []
        for i, angle in enumerate(np.deg2rad([-29, -12, 5, 21, 34])):
            center = np.array([5.6*np.sin(angle), -5.6*np.cos(angle), [.2, .7, -.4, .6, -.3][i]])
            forward = -center / np.linalg.norm(center)
            right = np.cross(forward, [0, 0, 1.])
            right /= np.linalg.norm(right)
            rotation = np.array([right, np.cross(forward, right), forward])
            records = []
            for m in cad["markers"]:
                xyz = (np.array(m["corners_world"])-center) @ rotation.T
                pixels = xyz @ k.T
                records.append(dict(marker_id=m["marker_id"], plane_id=m["plane_id"], role=m["role"],
                    state="detected", corners_xy=(pixels[:, :2]/pixels[:, 2, None]).tolist()))
            frames.append(dict(view_id=f"view_{i:02d}", size_wh=[640, 480], detection=dict(markers=records)))
        return frames, k

    def test_declared_roles_have_two_planes_and_exact_black_boundary(self):
        cad = self.cad()
        validate_cad(cad)
        self.assertEqual(len(cad["markers"]), 30)
        for m in cad["markers"]:
            corners = np.array(m["corners_world"])
            np.testing.assert_allclose(np.linalg.norm(np.roll(corners, -1, axis=0)-corners, axis=1), .3, atol=1e-15)
        a = cad["artwork"]
        self.assertEqual((a["black_square_pixels"]+2*a["white_margin_pixels_each_side"])/a["black_square_pixels"], a["quad_side_to_marker_side"])
        for role in ("training", "validation"):
            chosen = [m for m in cad["markers"] if m["role"] == role]
            self.assertEqual({m["plane_id"] for m in chosen}, {"plane_0", "plane_1"})

    def test_real_decoder_recovers_id_and_corner_order_without_half_pixel_shift(self):
        canvas = np.full((480, 640, 3), 255, np.uint8)
        tile = marker_image(7)
        canvas[80:360, 160:440] = tile[:, :, None]
        result = detect_markers(canvas, self.cad())
        row = next(r for r in result["markers"] if r["marker_id"] == 7)
        self.assertEqual(row["state"], "detected")
        # Boundary pixel centres, not a renderer projection passed to detection.
        np.testing.assert_allclose(row["corners_xy"], [[188, 108], [411, 108], [411, 331], [188, 331]], atol=.6)
        self.assertEqual(result["detected_count"], 1)
        self.assertEqual(sum(r["state"] == "missing" for r in result["markers"]), 29)

    def test_known_metric_points_recover_unknown_intrinsics_and_all_poses(self):
        frames, expected = self.frames()
        result = calibrate_case(frames, self.cad())
        self.assertEqual(result["state"], "validated")
        for camera in result["cameras"]:
            np.testing.assert_allclose(camera["K_index"], expected, atol=.003, rtol=0)
            self.assertLess(camera["validation_score"]["p95_px"], .001)
            self.assertEqual(camera["training"]["geometry"]["rank"], 3)
            self.assertFalse(set(camera["training"]["marker_ids"]) & set(camera["validation"]["marker_ids"]))
        self.assertNotEqual(result["initial_K"], expected.tolist())
        self.assertEqual(result["distortion"], [0.] * 5)

    def test_corrupt_validation_changes_decision_but_cannot_change_fit(self):
        frames, _ = self.frames()
        good = calibrate_case(frames, self.cad())
        poisoned = copy.deepcopy(frames)
        for f in poisoned:
            for row in f["detection"]["markers"]:
                if row["role"] == "validation":
                    row["corners_xy"] = (np.array(row["corners_xy"]) + [30., -20.]).tolist()
        bad = calibrate_case(poisoned, self.cad())
        self.assertEqual(bad["state"], "withheld")
        for a, b in zip(good["cameras"], bad["cameras"]):
            np.testing.assert_allclose(a["K_index"], b["K_index"], atol=1e-8, rtol=0)
            np.testing.assert_allclose(a["world_to_camera_cv"], b["world_to_camera_cv"], atol=1e-10, rtol=0)

    def test_validation_never_reaches_optimizer_arguments(self):
        frames, _ = self.frames()
        poisoned = copy.deepcopy(frames)
        for frame in poisoned:
            for row in frame["detection"]["markers"]:
                if row["role"] == "validation":
                    row["corners_xy"] = (np.asarray(row["corners_xy"]) + 1000).tolist()
        captured = []
        original = cv2.calibrateCamera
        def observe(*args, **kwargs):
            captured.append((copy.deepcopy(args), copy.deepcopy(kwargs)))
            return original(*args, **kwargs)
        with patch("creator_eval.fixture_calibration.cv2.calibrateCamera", side_effect=observe) as solver:
            calibrate_case(frames, self.cad())
            calibrate_case(poisoned, self.cad())
        self.assertEqual(solver.call_count, 2)
        a, b = captured
        # Some BLAS builds differ in their last optimization bits. The actual
        # training arrays, initial K and distortion are still exactly identical.
        for index in (0, 1):
            for first, second in zip(a[0][index], b[0][index]):
                np.testing.assert_array_equal(first, second)
        for index in (3, 4):
            np.testing.assert_array_equal(a[0][index], b[0][index])
        self.assertEqual(a[1], b[1])

    def test_single_plane_training_is_not_repaired_with_validation(self):
        frames, _ = self.frames()
        for row in frames[1]["detection"]["markers"]:
            if row["role"] == "training" and row["plane_id"] == "plane_1":
                row["state"], row["corners_xy"] = "missing", None
        result = calibrate_case(frames, self.cad())
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(len(result["cameras"]), 5)
        self.assertTrue(all(r["K_index"] is None for r in result["cameras"]))
        self.assertIn("training_rank", result["cameras"][1]["reasons"])

    def test_absent_validation_does_not_discard_or_refit_a_view(self):
        frames, _ = self.frames()
        good = calibrate_case(frames, self.cad())
        for row in frames[0]["detection"]["markers"]:
            if row["role"] == "validation":
                row["state"], row["corners_xy"] = "missing", None
        result = calibrate_case(frames, self.cad())
        self.assertEqual(len(result["cameras"]), 5)
        self.assertEqual(result["cameras"][0]["state"], "withheld")
        np.testing.assert_allclose(result["cameras"][0]["K_index"], good["cameras"][0]["K_index"], atol=1e-8, rtol=0)
        self.assertEqual(result["cameras"][0]["validation_score"]["count"], 0)

    def test_role_tampering_and_duplicate_ids_fail(self):
        frames, _ = self.frames()
        d = frames[0]["detection"]
        d["markers"][0]["role"] = "validation"
        with self.assertRaises(ValueError):
            observations(d, self.cad(), "training")
        cad = self.cad()
        cad["markers"][1]["marker_id"] = cad["markers"][0]["marker_id"]
        with self.assertRaises(ValueError):
            validate_cad(cad)

    def test_policy_optimizes_principal_point_without_truth(self):
        self.assertTrue(POLICY["principal_point_fitted"])
        frames, _ = self.frames()
        result = calibrate_case(frames, self.cad())
        self.assertFalse(result["flags"] & cv2.CALIB_FIX_PRINCIPAL_POINT)
        self.assertTrue(result["flags"] & cv2.CALIB_FIX_ASPECT_RATIO)

    def test_fresh_process_blocks_generation_truth_and_physical_results(self):
        code = """import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path.cwd()/'scripts'))
import run_fixture_calibration as r
sys.addaudithook(r.block_truth)
for path in ['data/eval_gt/rod-fixture-scenes-v1-20260927/manifest.json',
             '.runtime/experiments/rod-fixture-scenes-v1-20260927/protocol.json',
             'docs/experiments/results/never-read.json',
             'data/evaluation/nope.json']:
 try:
  Path(path).read_text()
 except PermissionError:
  pass
 else:
  raise AssertionError(path)
try:
 open(b'data/eval_gt/does-not-exist.json')
except PermissionError:
 pass
else:
 raise AssertionError('bytes relative GT open')
print('TRIPWIRE_OK')
"""
        p = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("TRIPWIRE_OK", p.stdout)


if __name__ == "__main__":
    unittest.main()
