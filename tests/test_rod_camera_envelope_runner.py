"""Adversarial contracts for analytic rod inference and separate envelope scoring."""
import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_rod_camera_envelope as runner  # noqa: E402

# isort: split
from creator_eval.camera_training_groups import build_training_group_plan  # noqa: E402
from creator_eval.rod_camera_envelope import DEFAULT_CONDITIONS, build_envelope  # noqa: E402


def cameras():
    """An exact little rig, deliberately not a collinear camera trajectory."""
    centers = np.array([[-1., 0., 0.], [-.5, .2, .1], [0., -.2, -.1],
                        [.5, .15, .05], [1., 0., 0.]])
    k = np.tile([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]], (5, 1, 1))
    e = np.concatenate((np.tile(np.eye(3), (5, 1, 1)), -centers[:, :, None]), axis=2)
    return k, e


def endpoint_case(segments, k, e):
    # Use the pinhole equation directly: the oracle must not call the runner's
    # project or triangulate implementation that this fixture is checking.
    tracks = []
    for index, segment in enumerate(segments):
        endpoints = []
        for endpoint_index, point in enumerate(segment):
            observations = []
            for view in range(len(k)):
                homogeneous_pixel = k[view] @ e[view] @ np.r_[point, 1.]
                observations.append(dict(view=view, xy=(homogeneous_pixel[:2] / homogeneous_pixel[2]).tolist()))
            endpoints.append(dict(track_id=f"p{index}-{endpoint_index}", observations=observations))
        tracks.append(dict(segment_id=f"s{index}", endpoint_tracks=endpoints))
    return dict(rod_tracks=tracks)


def family(segments=None):
    if segments is None:
        segments = [[[0., -.5, 4.], [0., .5, 4.]]]
    k, e = cameras()
    support = dict(assignment_sha256="a" * 64, source_rows_sha256="b" * 64,
                   pool_sha256="c" * 64, source_row_count=10)
    control = dict(condition_id="control", intrinsics=k.tolist(), extrinsics=e.tolist(),
                   camera_decision=dict(state="candidate_camera_correction"),
                   identity=dict(state="accepted", segments=segments), support=support)
    folds = [dict(copy.deepcopy(control), condition_id=cid) for cid in DEFAULT_CONDITIONS]
    return control, folds, k, e


def similarity_truth(k, e, segments):
    """Hand-defined native-to-world transform, independent of camera alignment."""
    angle = np.radians(30.)
    q = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    scale, translation = 2.5, np.array([2., -3., .7])
    true_cameras = []
    for view, (intrinsic, extrinsic) in enumerate(zip(k, e)):
        center = -extrinsic[:, :3].T @ extrinsic[:, 3]
        true_center = scale * q @ center + translation
        true_e = np.eye(4)
        true_e[:3, :3] = extrinsic[:, :3] @ q.T
        true_e[:3, 3] = -true_e[:3, :3] @ true_center
        true_cameras.append(dict(view_id=f"v{view:02d}", K_index=intrinsic.tolist(),
                                 world_to_camera_cv=true_e.tolist(), size_wh=[640, 480]))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = scale * q, translation
    true_segments = np.asarray(segments) @ matrix[:3, :3].T + translation
    return true_cameras, true_segments, matrix


class EnvelopeRunnerTests(unittest.TestCase):
    def test_supplied_endpoints_recover_finite_gap_but_missing_views_produce_no_rod(self):
        k, e = cameras()
        segments = np.array([[[0., -.8, 4.], [0., -.2, 4.]],
                             [[0., .2, 4.], [0., .8, 4.]]])
        case = endpoint_case(segments, k, e)
        result = runner.analytical_rod(case, k, e)
        self.assertEqual(result["state"], "accepted")
        np.testing.assert_allclose(result["segments"], segments, atol=1e-11)
        self.assertLess(max(result["endpoint_reprojection_errors_px"]), 1e-10)
        self.assertGreater(result["segments"][1, 0, 1] - result["segments"][0, 1, 1], .39)
        for count in (0, 1):
            with self.subTest(distinct_views=count):
                missing = copy.deepcopy(case)
                # Other endpoints remain perfectly observed. One bad endpoint
                # cannot quietly shorten the rod or leave a misleading fragment.
                missing["rod_tracks"][1]["endpoint_tracks"][1]["observations"] = (
                    missing["rod_tracks"][1]["endpoint_tracks"][1]["observations"][:count])
                unavailable = runner.analytical_rod(missing, k, e)
                self.assertEqual(unavailable["state"], "unavailable")
                self.assertEqual(unavailable["segments"], [])
                self.assertIn("fewer_than_two_endpoint_views", unavailable["reason"])
        repeated = copy.deepcopy(case)
        endpoint = repeated["rod_tracks"][0]["endpoint_tracks"][0]
        endpoint["observations"] = [endpoint["observations"][0]] * 3
        self.assertEqual(runner.analytical_rod(repeated, k, e)["state"], "unavailable")

    def test_negative_depth_is_not_accepted_even_with_exact_pixel_reprojection(self):
        k, e = cameras()
        # These pixels have an exact algebraic solution, but the rod is behind
        # every camera. A tiny pixel error is not enough to make it observable.
        segments = np.array([[[0., -.5, -4.], [0., .5, -4.]]])
        result = runner.analytical_rod(endpoint_case(segments, k, e), k, e)
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["reason"], "endpoint_behind_camera")

    def test_one_control_camera_sim3_scores_every_fold_without_refitting_to_rods(self):
        control, folds, k, e = family()
        truth_cameras, truth_segments, expected_matrix = similarity_truth(k, e, control["identity"]["segments"])
        # This fold has distinct cameras and a 4 cm native rod shift. Keeping
        # the anchor cameras fixed makes it admissible in the same native gauge.
        folds[0]["extrinsics"][2][0][3] -= .2
        folds[0]["identity"]["segments"] = (np.asarray(control["identity"]["segments"]) + [.04, 0., 0.]).tolist()
        envelope = build_envelope(control, folds, initial_extrinsics=e)
        self.assertEqual(envelope["state"], "complete_empirical_envelope")
        with patch.object(runner, "camera_score", wraps=runner.camera_score) as align:
            result = runner.score_family(envelope, [folds[2], control, folds[0], folds[3], folds[1]],
                                         truth_segments, truth_cameras, runner.EVALUATION)
        align.assert_called_once()
        np.testing.assert_array_equal(align.call_args.args[1], e)
        np.testing.assert_allclose(result["alignment"]["prediction_world_to_gt_world"], expected_matrix, atol=1e-12)
        self.assertTrue(result["alignment_reused_for_all_conditions"])
        by_id = {row["condition_id"]: row for row in result["rows"]}
        self.assertEqual(by_id["control"]["metrics"]["recovery_fraction"], 1.)
        shifted = by_id["leave_group_0"]["metrics"]
        self.assertEqual(shifted["recovery_fraction"], 0.)
        self.assertEqual(shifted["precision_fraction"], 0.)
        self.assertAlmostEqual(shifted["truth_to_prediction"]["distance_p95"], .1)
        self.assertEqual(result["coverage"]["fraction"], 1.)

    def test_constant_wrong_rod_has_zero_width_box_and_zero_truth_coverage(self):
        control, folds, k, e = family()
        truth_cameras, truth_segments, _ = similarity_truth(k, e, control["identity"]["segments"])
        for packet in [control, *folds]:
            packet["identity"]["segments"] = (np.asarray(packet["identity"]["segments"]) + [.12, 0., 0.]).tolist()
        envelope = build_envelope(control, folds, initial_extrinsics=e)
        self.assertEqual(envelope["state"], "complete_empirical_envelope")
        result = runner.score_family(envelope, [control, *folds], truth_segments, truth_cameras, runner.EVALUATION)
        self.assertEqual(result["coverage"]["fraction"], 0.)
        box = result["coverage"]["segments"][0]
        self.assertEqual(box["maximum_box_diagonal_m"], 0.)
        self.assertAlmostEqual(box["max_outside_box_m"], .3)
        self.assertEqual(box["sample_count"], 33)
        for row in result["rows"]:
            self.assertEqual(row["metrics"]["recovery_fraction"], 0.)
            self.assertAlmostEqual(row["metrics"]["truth_to_prediction"]["distance_p95"], .3)

    def test_missing_condition_and_unavailable_alignment_preserve_all_five_slots(self):
        for unavailable in ("missing_control_camera", "collinear_control"):
            with self.subTest(unavailable=unavailable):
                control, folds, k, e = family()
                true_cameras, truth, _ = similarity_truth(k, e, control["identity"]["segments"])
                if unavailable == "missing_control_camera":
                    control["extrinsics"], control["intrinsics"] = None, None
                else:
                    for packet in [control, *folds]:
                        for index, extrinsic in enumerate(packet["extrinsics"]):
                            extrinsic[1][3], extrinsic[2][3] = 0., 0.
                            extrinsic[0][3] = 1. - index * .5
                    e = np.asarray(control["extrinsics"])
                envelope = build_envelope(control, folds[:3], initial_extrinsics=e)
                result = runner.score_family(envelope, [control, *folds[:3]], truth, true_cameras, runner.EVALUATION)
                expected_state = "no_control_camera" if unavailable == "missing_control_camera" else "alignment_unavailable"
                self.assertEqual(result["state"], expected_state)
                self.assertIsNone(result["alignment"])
                self.assertIsNone(result["coverage"])
                self.assertEqual([row["condition_id"] for row in result["rows"]], ["control", *DEFAULT_CONDITIONS])
                self.assertTrue(all(row["metrics"] is None for row in result["rows"]))
                self.assertEqual(result["rows"][-1]["reason"], "missing_condition")

    def test_skipped_record_cannot_be_relabelled_or_detached_from_fit(self):
        k, e = cameras()
        method = dict(decision=dict(minimum_scored_observations=1, maximum_median_ratio=.98,
                                   maximum_p95_px=2., minimum_fraction_within_2px=.9))
        case = dict(case_id="empty-unit", initial_intrinsics=k.tolist(), initial_extrinsics=e.tolist(),
                    training=[], validation=[], original_plan=[], all_view_plan=[], rod_tracks=[])
        case["group_plan"] = build_training_group_plan([], validation_track_ids=[], view_count=5)
        condition = case["group_plan"]["conditions"][0]
        result = dict(state="skipped_group_coverage", success=False)
        before = runner.validate_heldout(k, e, [])
        packet = dict(condition_id="control", intrinsics=None, extrinsics=None,
                      camera_decision=runner.correction_decision(result, before, before, method["decision"]),
                      identity=dict(state="unavailable", reason="no_camera", segments=[]), support=None)
        receipt = dict(case_id=case["case_id"], condition="control")
        record = dict(case_id=case["case_id"], case_sha256=runner.canonical_hash(case),
                      condition=condition, packet=packet, result=result, before=before,
                      after=None, all_views=None, gt_read_during_inference=False)
        runner.validate_analytic_record(record, receipt, case, method)
        for tamper in ("relabel_packet", "detach_camera", "invent_rod", "claim_truth_read"):
            bad = copy.deepcopy(record)
            if tamper == "relabel_packet":
                bad["packet"]["condition_id"] = "leave_group_0"
            elif tamper == "detach_camera":
                bad["packet"]["extrinsics"] = e.tolist()
            elif tamper == "invent_rod":
                bad["packet"]["identity"]["segments"] = [[[0., 0., 4.], [0., 1., 4.]]]
            else:
                bad["gt_read_during_inference"] = True
            with self.subTest(tamper=tamper), self.assertRaises(AssertionError):
                runner.validate_analytic_record(bad, receipt, case, method)

    def test_each_fresh_inference_process_installs_its_own_truth_tripwire(self):
        # Audit hooks cannot be removed from this interpreter. Each entry point
        # therefore gets a clean child, just like a Windows spawned worker.
        child = """
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
import run_rod_camera_envelope as runner
probe = runner.ROOT / "data/eval_gt/__unit_tripwire_probe_never_created__.json"
def read_forbidden(mode):
    probe.read_text(encoding="utf-8")
runner.checked = read_forbidden
try:
    if sys.argv[1] == "worker":
        runner.analytical_worker({}, {}, {}, {})
    else:
        runner.infer("analytic")
except PermissionError as error:
    assert "Inference tried to open evaluation truth" in str(error), str(error)
    # This is a targeted tripwire, not a blanket prohibition on input reads.
    assert (runner.ROOT / "configs/camera_envelope_challenges_v1.json").read_text(encoding="utf-8")
    print(json.dumps({"entry": sys.argv[1], "blocked": True}))
else:
    raise AssertionError("Fresh inference process read eval_gt")
"""
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for entry in ("worker", "infer"):
            with self.subTest(entry=entry):
                process = subprocess.run([sys.executable, "-B", "-c", child, entry],
                                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=False)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(json.loads(process.stdout), dict(entry=entry, blocked=True))


if __name__ == "__main__":
    unittest.main()
