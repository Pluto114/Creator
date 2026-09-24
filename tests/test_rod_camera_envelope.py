"""Failure modes and hand-checkable ranges for the five-condition rod envelope."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.rod_camera_envelope import DEFAULT_CONDITIONS, build_envelope


def fixture(segments=None):
    if segments is None:
        segments = [[[0., 0., 3.], [1., 0., 3.]], [[1.5, 0., 3.], [2., 0., 3.]]]
    k = np.tile([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]], (3, 1, 1))
    c = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    e = np.concatenate((np.tile(np.eye(3), (3, 1, 1)), -c[:, :, None]), axis=2)
    support = {name: hashlib.sha256(name.encode()).hexdigest()
               for name in ("assignment_sha256", "source_rows_sha256", "pool_sha256")}
    support["source_row_count"] = 100
    control = dict(condition_id="control", intrinsics=k.tolist(), extrinsics=e.tolist(),
                   camera_decision=dict(state="candidate_camera_correction"),
                   identity=dict(state="accepted", segments=segments), support=support)
    folds = [dict(copy.deepcopy(control), condition_id=name) for name in DEFAULT_CONDITIONS]
    return control, folds, e


def run(control, folds, e, **kwargs):
    return build_envelope(control, folds, initial_extrinsics=e, **kwargs)


class RodCameraEnvelopeTests(unittest.TestCase):
    def test_identical_conditions_have_zero_width_without_claiming_reliability(self):
        control, folds, e = fixture()
        saved = copy.deepcopy((control, folds))
        result = run(control, folds, e)
        self.assertEqual(result["state"], "complete_empirical_envelope")
        self.assertEqual(result["expected_condition_count"], 5)
        self.assertEqual(result["comparable_condition_count"], 5)
        self.assertEqual(result["control_segments"], control["identity"]["segments"])
        self.assertFalse(result["control_modified"])
        self.assertFalse(result["alignment_performed"])
        self.assertFalse(result["perturbation_selected"])
        self.assertNotIn("reliable", result)
        self.assertEqual((control, folds), saved)
        self.assertEqual(len(result["segment_envelopes"][0]["fractions"]), 33)
        for segment in result["segment_envelopes"]:
            np.testing.assert_array_equal(segment["min_delta_local"], np.zeros((33, 3)))
            np.testing.assert_array_equal(segment["max_delta_local"], np.zeros((33, 3)))
        self.assertAlmostEqual(result["gap_envelopes"][0]["gap_along_control_axis_native_min"], .5)
        json.dumps(result, allow_nan=False)

    def test_translation_rotation_and_endpoint_ranges_are_hand_checkable(self):
        control, folds, e = fixture([[[0., 0., 3.], [1., 0., 3.]]])
        folds[0]["identity"]["segments"] = [[[0., .2, 3.], [1., .2, 3.]]]
        theta = np.radians(10.)
        folds[1]["identity"]["segments"] = [[[0., 0., 3.], [float(np.cos(theta)), float(np.sin(theta)), 3.]]]
        result = run(control, folds, e)
        segment = result["segment_envelopes"][0]
        self.assertEqual(result["state"], "complete_empirical_envelope")
        self.assertAlmostEqual(segment["angle_from_control_degrees_max"], 10.)
        self.assertAlmostEqual(segment["maximum_pairwise_direction_angle_degrees"], 10.)
        self.assertAlmostEqual(segment["maximum_fractional_displacement_over_baseline"], .1)
        frame = np.asarray(segment["local_frame"])
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-15)
        self.assertAlmostEqual(np.linalg.det(frame), 1.)
        observed = segment["observations"][1]
        np.testing.assert_allclose(observed["center_delta_local"], np.array([0., .2, 0.]) @ frame.T)
        np.testing.assert_allclose(segment["control_points"][16], [.5, 0., 3.])
        self.assertAlmostEqual(segment["length_native_min"], 1.)
        self.assertAlmostEqual(segment["length_native_max"], 1.)

    def test_missing_or_withheld_conditions_keep_five_slots_and_no_partial_envelope(self):
        control, folds, e = fixture()
        missing = run(control, folds[:3], e)
        self.assertEqual(len(missing["conditions"]), 5)
        self.assertEqual(missing["conditions"][-1]["state"], "missing")
        self.assertEqual(missing["comparable_condition_count"], 4)
        self.assertIsNone(missing["segment_envelopes"])
        folds[0]["camera_decision"]["state"] = "withhold_correction"
        withheld = run(control, folds, e)
        self.assertEqual(withheld["state"], "unresolved")
        self.assertEqual(withheld["conditions"][1]["identity_state"], "accepted")
        self.assertIn("camera_not_accepted", withheld["conditions"][1]["reasons"])
        self.assertIsNone(withheld["gap_envelopes"])

    def test_rejected_and_empty_identities_are_not_zero_width_evidence(self):
        control, folds, e = fixture()
        folds[0]["identity"] = dict(state="rejected", segments=[])
        folds[0]["support"] = None
        result = run(control, folds, e)
        self.assertEqual(result["state"], "unresolved")
        self.assertIsNone(result["conditions"][1]["support"])
        self.assertIsNone(result["segment_envelopes"])
        control["identity"] = dict(state="rejected", segments=[])
        control["support"] = None
        self.assertEqual(run(control, folds, e)["state"], "unavailable")
        control, folds, e = fixture()
        folds[0]["identity"]["segments"] = []
        self.assertIn("empty_accepted_segments", run(control, folds, e)["conditions"][1]["reasons"])

    def test_changed_support_blocks_identical_geometry(self):
        for field in ("assignment_sha256", "source_rows_sha256", "pool_sha256", "source_row_count"):
            with self.subTest(field=field):
                control, folds, e = fixture()
                folds[0]["support"][field] = 101 if field == "source_row_count" else "0" * 64
                result = run(control, folds, e)
                self.assertEqual(result["state"], "unresolved")
                self.assertIn("support_changed:" + field, result["conditions"][1]["reasons"])
                self.assertIsNone(result["segment_envelopes"])

    def test_split_merge_gap_closure_and_ambiguous_overlap_are_not_repaired(self):
        alternatives = [
            [[[0., 0., 3.], [2., 0., 3.]]],
            [[[0., 0., 3.], [.5, 0., 3.]], [[.6, 0., 3.], [1., 0., 3.]], [[1.5, 0., 3.], [2., 0., 3.]]],
            [[[0., 0., 3.], [1.25, 0., 3.]], [[1.25, 0., 3.], [2., 0., 3.]]],
            [[[0., 0., 3.], [.4, 0., 3.]], [[.6, 0., 3.], [2., 0., 3.]]],
            [[[3., 0., 3.], [4., 0., 3.]], [[4.5, 0., 3.], [5., 0., 3.]]],
        ]
        for segments in alternatives:
            with self.subTest(segments=segments):
                control, folds, e = fixture()
                folds[0]["identity"]["segments"] = segments
                result = run(control, folds, e)
                self.assertEqual(result["state"], "unresolved")
                self.assertTrue(result["conditions"][1]["reasons"])
                self.assertIsNone(result["segment_envelopes"])

    def test_segment_reordering_and_endpoint_reversal_preserve_correspondence(self):
        control, folds, e = fixture()
        control["identity"]["segments"][1].reverse()
        folds[0]["identity"]["segments"] = [segment[::-1] for segment in folds[0]["identity"]["segments"][::-1]]
        result = run(control, list(reversed(folds)), e)
        self.assertEqual(result["state"], "complete_empirical_envelope")
        self.assertEqual(result["control_segments"], control["identity"]["segments"])
        for slot in result["conditions"]:
            np.testing.assert_array_equal(slot["matched_segments"], control["identity"]["segments"])
        self.assertEqual(result["conditions"][1]["matched_source_segment_indices"], [1, 0])

    def test_gauge_change_and_non_collinear_or_degenerate_segments_are_explicit(self):
        control, folds, e = fixture()
        folds[0]["extrinsics"][2][0][3] *= 1.01
        result = run(control, folds, e)
        self.assertEqual(result["state"], "unresolved")
        self.assertTrue(any("native_gauge_changed" in r for r in result["conditions"][1]["reasons"]))
        invalid = [
            ([[[0., 0., 3.], [1., 0., 3.]], [[1.5, .1, 3.], [2., .1, 3.]]], "non_collinear"),
            ([[[0., 0., 3.], [0., 0., 3.]]], "short"),
            ([[[0., 0., 3.], [1., 0., 3.]], [[0., 0., 3.], [1., 0., 3.]]], "overlapping"),
        ]
        for segments, reason in invalid:
            with self.subTest(reason=reason):
                control, folds, e = fixture()
                folds[0]["identity"]["segments"] = segments
                result = run(control, folds, e)
                self.assertTrue(any(reason in r for r in result["conditions"][1]["reasons"]))
                self.assertIsNone(result["segment_envelopes"])

    def test_duplicate_extra_malformed_nonfinite_and_unidentified_support_fail_loudly(self):
        control, folds, e = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            run(control, [folds[0], folds[0], *folds[2:]], e)
        extra = copy.deepcopy(folds[0])
        extra["condition_id"] = "best_fold"
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            run(control, [*folds, extra], e)
        with self.assertRaisesRegex(ValueError, "four"):
            run(control, folds, e, expected_condition_ids=DEFAULT_CONDITIONS[:3])
        with self.assertRaisesRegex(ValueError, "fractional"):
            run(control, folds, e, samples_per_segment=True)
        bad = copy.deepcopy(folds)
        bad[0]["identity"]["segments"][0][0][0] = float("nan")
        with self.assertRaises(ValueError):
            run(control, bad, e)
        bad = copy.deepcopy(folds)
        bad[0]["support"] = None
        with self.assertRaisesRegex(ValueError, "support"):
            run(control, bad, e)
        bad = copy.deepcopy(folds)
        bad[0]["identity"]["segments"] = [[1., 2., 3.]]
        with self.assertRaisesRegex(ValueError, "Nx2x3"):
            run(control, bad, e)


if __name__ == "__main__":
    unittest.main()
