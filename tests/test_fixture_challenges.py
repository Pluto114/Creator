"""Known-answer interventions and fail-closed checks; no experiment GT reads."""

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
sys.path.insert(0, str(ROOT / "experiments/src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixture_challenges as runner  # noqa: E402
import test_fixture_calibration as calibration_tests  # noqa: E402
from creator_eval.fixture_calibration import calibrate_case  # noqa: E402
from creator_eval.fixture_challenges import (  # noqa: E402
    distort_rgb,
    perturb_cad,
    perturb_detections,
    radial_forward,
    radial_source_map,
)
from creator_eval.rod_fixture_finite import canonical_hash, reconstruct_fixture_narrow  # noqa: E402


class FixtureChallengeTests(unittest.TestCase):
    def setUp(self):
        self.cad = json.loads((ROOT / "configs/rod_fixture_cad_v1.json").read_text())
        self.config = json.loads((ROOT / "configs/fixture_calibration_challenges_v1.json").read_text())
        self.scenarios = {s["id"]: s for s in self.config["scenarios"]}

    def test_cad_similarity_preserves_projection_but_moves_metric_frame(self):
        points = np.array([m["corners_world"] for m in self.cad["markers"]])
        original = copy.deepcopy(self.cad)
        for sid in ("c01_scale_plus_1pct", "c02_scale_minus_1pct", "c04_origin_x_10mm"):
            scenario = self.scenarios[sid]
            changed = perturb_cad(self.cad, scenario)
            factor = scenario.get("factor", 1.)
            shift = np.array(scenario.get("offset_m", [0., 0., 0.]))
            transformed = np.array([m["corners_world"] for m in changed["markers"]])
            np.testing.assert_allclose(transformed, factor*points+shift)
            # Camera coordinates under C'=s*C+t equal s times old coordinates.
            center = np.array([0.1, -6., 0.3])
            a, b = points-center, transformed-(factor*center+shift)
            np.testing.assert_allclose(a[..., :2]/a[..., 2:3], b[..., :2]/b[..., 2:3], atol=1e-11)
            self.assertAlmostEqual(changed["marker_side_m"], factor*.3)
        self.assertEqual(self.cad, original)

    def test_panel_shift_affects_both_roles_on_only_one_plane(self):
        changed = perturb_cad(self.cad, self.scenarios["c03_panel_depth_20mm"])
        for a, b in zip(self.cad["markers"], changed["markers"]):
            expected = np.array(a["corners_world"])+([0., .02, 0.] if a["plane_id"] == "plane_1" else [0., 0., 0.])
            np.testing.assert_allclose(b["corners_world"], expected)
            self.assertEqual(a["role"], b["role"])

    def test_exact_marker_calibration_cannot_detect_shared_scale_error(self):
        frames, _ = calibration_tests.FixtureCalibrationTests().frames()
        nominal = calibrate_case(frames, self.cad)
        changed = calibrate_case(frames, perturb_cad(self.cad, self.scenarios["c01_scale_plus_1pct"]))
        self.assertEqual(changed["state"], "validated")
        for a, b in zip(nominal["cameras"], changed["cameras"]):
            ea, eb = np.array(a["world_to_camera_cv"]), np.array(b["world_to_camera_cv"])
            ca, cb = -ea[:, :3].T @ ea[:, 3], -eb[:, :3].T @ eb[:, 3]
            np.testing.assert_allclose(cb, 1.01*ca, atol=2e-5, rtol=0)
            self.assertLess(b["validation_score"]["p95_px"], .001)

    def test_marker_loss_and_wrong_ids_leave_input_immutable(self):
        frames, _ = calibration_tests.FixtureCalibrationTests().frames()
        before = copy.deepcopy(frames)
        for sid in ("c05_missing_train_plane", "c06_missing_validation", "c07_wrong_training_ids"):
            changed = perturb_detections(frames, self.scenarios[sid])
            for old, new in zip(frames, changed):
                for a, b in zip(old["detection"]["markers"], new["detection"]["markers"]):
                    self.assertEqual((a["marker_id"], a["role"], a["plane_id"]), (b["marker_id"], b["role"], b["plane_id"]))
            self.assertEqual(frames, before)
        swapped = perturb_detections(frames, self.scenarios["c07_wrong_training_ids"])
        self.assertEqual(swapped[0]["detection"]["markers"][0]["corners_xy"], frames[0]["detection"]["markers"][12]["corners_xy"])

    def test_missing_markers_reject_without_validation_refit_or_rod_output(self):
        frames, _ = calibration_tests.FixtureCalibrationTests().frames()
        nominal = calibrate_case(frames, self.cad)
        method = json.loads((ROOT / "configs/rod_fixture_narrow_v1.json").read_text())["method"]
        evidence = [dict(view_id=f["view_id"], size_wh=f["size_wh"], rgb_sha256="0"*64,
            guide_xyxy=[[320, 20], [320, 459]], observations={}, pool={"candidates": []},
            observation_sha256=canonical_hash({}), pool_sha256=canonical_hash({"candidates": []})) for f in frames]
        for sid, state in (("c05_missing_train_plane", "unavailable"), ("c06_missing_validation", "withheld")):
            changed = calibrate_case(perturb_detections(frames, self.scenarios[sid]), self.cad)
            self.assertEqual(changed["state"], state)
            if state == "withheld":
                for a, b in zip(nominal["cameras"], changed["cameras"]):
                    np.testing.assert_allclose(a["world_to_camera_cv"], b["world_to_camera_cv"], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(a["K_index"], b["K_index"], atol=1e-8, rtol=0)
            result = reconstruct_fixture_narrow(evidence, changed["cameras"], method)
            self.assertFalse(result["camera_gate"]["passed"])
            for row in result["methods"]:
                self.assertEqual(row["state"], "withheld_camera")
                self.assertEqual(row["segments"], [])

    def test_radial_inverse_and_sign_are_known_answer(self):
        size = [161, 121]
        center = (np.array(size)-1)/2
        radius = np.linalg.norm(center)
        for k in (-.08, 0., .08):
            mapped = radial_forward(np.array([center, center+[radius, 0]]), size, k)
            np.testing.assert_allclose(mapped, [center, center+[radius*(1+k), 0]], atol=1e-12)
            source = radial_source_map(size, k)
            y, x = np.indices((size[1], size[0]))
            np.testing.assert_allclose(radial_forward(source, size, k), np.stack((x, y), -1), atol=1e-9)

    def test_zero_radial_rgb_exact_and_nonzero_moves_image(self):
        image = np.random.default_rng(77).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        before = image.copy()
        np.testing.assert_array_equal(distort_rgb(image, 0), image)
        self.assertFalse(np.array_equal(distort_rgb(image, .08), image))
        np.testing.assert_array_equal(image, before)
        for k in (float("nan"), .2):
            with self.assertRaises(ValueError):
                distort_rgb(image, k)

    def test_empty_physical_result_is_retained_as_zero_recovery(self):
        score = runner.physical_summary([], [[[0, 0, 0], [0, 0, 2]]], [])
        self.assertEqual(score["segment_count"], 0)
        self.assertEqual(score["signed_length_error_m"], -2)
        self.assertEqual(score["recovery_fraction"], 0)
        self.assertIsNone(score["precision_fraction"])
        self.assertIsNone(score["boundary_max_error_m"])

    def test_camera_metrics_accept_homogeneous_truth_and_known_translation(self):
        predicted = dict(view_id="v", state="validated", reasons=[], K_index=np.eye(3).tolist(),
                         world_to_camera_cv=np.c_[np.eye(3), [-.01, 0, 0]].tolist(),
                         training_score=None, validation_score=None)
        truth = dict(view_id="v", world_to_camera_cv=np.eye(4).tolist())
        row = runner.camera_summary(dict(cameras=[predicted]), [truth])[0]
        self.assertAlmostEqual(row["center_error_m"], .01)
        self.assertAlmostEqual(row["rotation_error_degrees"], 0)

    def test_receipt_mutation_is_rejected_and_writes_are_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "input.json"
            runner.write(path, {"a": 1})
            hashes = {}
            with patch.object(runner, "ROOT", root):
                runner.receipt(path, hashes)
                with self.assertRaises(ValueError):
                    runner.receipt(path, hashes, "0"*64)
                with self.assertRaises(FileExistsError):
                    runner.write(path, {"a": 2})

    def test_fresh_process_blocks_truth_results_and_render_metadata(self):
        code = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'scripts'))
import run_fixture_challenges as r
sys.addaudithook(r.block_truth)
for path in ['data/eval_gt/never-read.json', 'data/evaluation/never-read.json',
             'docs/experiments/results/never-read.json', '.runtime/any/protocol.json',
             '.runtime/any/x-render_request.json', '.runtime/any/rendered-rgb/x.png']:
    try:
        Path(path).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError(path)
try:
    open(b'data/eval_gt/never-read.json')
except PermissionError:
    pass
else:
    raise AssertionError('bytes path escaped guard')
print('CHALLENGE_TRIPWIRE_OK')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn("CHALLENGE_TRIPWIRE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
