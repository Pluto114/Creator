"""Pairing, provenance and non-circular source identity before any fitting."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.camera_evidence_challenges import (  # noqa: E402
    ANCHOR_KEYS,
    CASE_KEYS,
    GROUP_KEYS,
    MODES,
    canonical_digest,
    generate_paired_challenges,
)
from creator_eval.camera_training_groups import build_training_group_plan  # noqa: E402


def protocol():
    return json.loads((ROOT / "configs/camera_evidence_challenges_v1.json").read_text(encoding="utf-8"))


def projected_point(camera, point):
    transformed = np.asarray(camera["world_to_camera_cv"]) @ np.r_[point, 1.]
    pixel = np.asarray(camera["K_index"]) @ transformed[:3]
    return pixel[:2] / pixel[2]


def keys_inside(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(keys_inside(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(keys_inside(child) for child in value))
    return set()


class PairedEvidenceChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.truth = generate_paired_challenges(protocol())
        cls.hidden_groups = {item["camera_group_id"]: item for item in cls.truth["camera_groups"]}
        cls.hidden_cases = {item["case_id"]: item for item in cls.truth["cases"]}

    def test_twenty_paired_groups_are_reproducible_and_truth_is_not_an_input_alias(self):
        inputs, truth = generate_paired_challenges(protocol())
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(truth, self.truth)
        self.assertEqual(len(inputs["camera_groups"]), 20)
        self.assertEqual(len(inputs["cases"]), 80)
        self.assertEqual(len({case["case_id"] for case in inputs["cases"]}), 80)
        forbidden = {"seed", "seeds", "mode", "role", "cameras", "rod_segments", "measurement_rod_segments",
                     "world_xyz", "generation_parameters", "background_points", "noise_uv_px"}
        self.assertFalse(keys_inside(inputs) & forbidden)
        for group in inputs["camera_groups"]:
            self.assertEqual(set(group), GROUP_KEYS)
            siblings = [case for case in inputs["cases"] if case["camera_group_id"] == group["camera_group_id"]]
            self.assertEqual(len(siblings), 4)
            self.assertEqual({self.hidden_cases[case["case_id"]]["mode"] for case in siblings}, set(MODES))
            for case in siblings:
                self.assertEqual(set(case), CASE_KEYS)
                self.assertEqual(len(case["evidence_sets"]), 3)
        before = canonical_digest(inputs)
        truth["camera_groups"][0]["cameras"][0]["K_index"][0][0] += 500
        truth["cases"][0]["rod_segments"][0][0][0] += 100
        self.assertEqual(canonical_digest(inputs), before)

    def test_every_mode_shares_endpoint_noise_and_swap_occurs_after_noise(self):
        for group in self.inputs["camera_groups"]:
            hidden_group = self.hidden_groups[group["camera_group_id"]]
            siblings = {self.hidden_cases[case["case_id"]]["mode"]: case for case in self.inputs["cases"]
                        if case["camera_group_id"] == group["camera_group_id"]}
            reference_noise = None
            for mode in ("ordinary", "coherent_lateral", "coherent_depth"):
                case = siblings[mode]
                latent = self.hidden_cases[case["case_id"]]["measurement_rod_segments"][0]
                noise = []
                for endpoint_index, endpoint in enumerate(case["rod_tracks"][0]["endpoint_tracks"]):
                    self.assertEqual(len(endpoint["observations"]), 5)
                    noise.append([np.asarray(observation["xy"]) - projected_point(
                        hidden_group["cameras"][observation["view"]], latent[endpoint_index])
                        for observation in endpoint["observations"]])
                noise = np.asarray(noise)
                if reference_noise is None:
                    reference_noise = noise
                else:
                    np.testing.assert_allclose(noise, reference_noise, atol=1e-12, rtol=0.)
            original = siblings["ordinary"]["rod_tracks"][0]["endpoint_tracks"]
            swapped = siblings["endpoint_swap"]["rod_tracks"][0]["endpoint_tracks"]
            for endpoint_index in range(2):
                for view in range(5):
                    donor = 1 - endpoint_index if view == 2 else endpoint_index
                    self.assertEqual(swapped[endpoint_index]["observations"][view],
                                     original[donor]["observations"][view])

    def test_target_claims_use_independent_noise_and_preserve_missing_and_equal_controls(self):
        for group in self.inputs["camera_groups"]:
            cameras = self.hidden_groups[group["camera_group_id"]]["cameras"]
            target_by_mode = []
            for case in self.inputs["cases"]:
                if case["camera_group_id"] != group["camera_group_id"]:
                    continue
                hidden = self.hidden_cases[case["case_id"]]
                labels = {item["evidence_id"]: item for item in hidden["evidence_labels"]}
                evidence = {labels[item["evidence_id"]]["role"]: item for item in case["evidence_sets"]}
                target = evidence["target"]["anchors"]
                candidate = evidence["candidate_consistent_target"]["anchors"]
                partial = evidence["single_view_target"]["anchors"]
                self.assertEqual(partial, target[:1])
                self.assertEqual([anchor["view_id"] for anchor in target], ["v00", "v04"])
                if hidden["mode"] in ("ordinary", "endpoint_swap"):
                    self.assertEqual(candidate, target)
                else:
                    self.assertNotEqual([item["xy"] for item in candidate], [item["xy"] for item in target])
                target_by_mode.append(target)
                for item in case["evidence_sets"]:
                    label = labels[item["evidence_id"]]
                    segment = np.asarray(hidden["measurement_rod_segments"] if label["physical_source"] == "measurement_rod"
                                         else hidden["rod_segments"])[0]
                    for index, anchor in enumerate(item["anchors"]):
                        self.assertEqual(set(anchor), ANCHOR_KEYS)
                        fraction = label["point_fractions"][index]
                        point = segment[0] + fraction * (segment[1] - segment[0])
                        view = group["view_ids"].index(anchor["view_id"])
                        expected = projected_point(cameras[view], point) + label["noise_uv_px"][index]
                        np.testing.assert_allclose(anchor["xy"], expected, atol=1e-12, rtol=0.)
                        self.assertEqual(anchor["uncertainty_xy_px"], [.5, .5])
            for target in target_by_mode[1:]:
                self.assertEqual(target, target_by_mode[0])
            hidden = self.hidden_groups[group["camera_group_id"]]
            expected_noise = np.random.default_rng(hidden["seed"] + 5000000).normal(0., .25, (2, 2))
            np.testing.assert_array_equal(hidden["anchor_noise_uv_px"], expected_noise)
            endpoint_noise = np.random.default_rng(hidden["seed"] + 3000000).normal(0., .25, (5, 2, 2))
            self.assertFalse(np.allclose(expected_noise, endpoint_noise[[0, 4], 0]))

    def test_every_view_hash_binds_exact_measurements_without_a_hash_cycle_or_fake_rgb(self):
        for group in self.inputs["camera_groups"]:
            cases = sorted((case for case in self.inputs["cases"] if case["camera_group_id"] == group["camera_group_id"]),
                           key=lambda case: case["case_id"])
            for view, frame in enumerate(group["frames"]):
                bank = frame["measurement_bank"]
                self.assertEqual(frame["view_id"], group["view_ids"][view])
                self.assertEqual(frame["source_kind"], "synthetic_pixel_measurements")
                self.assertEqual(frame["source_sha256"], canonical_digest(bank))
                self.assertFalse(any("sha" in key or "rgb" in key for key in keys_inside(bank)))
                for label, count in (("training", 160), ("validation", 64)):
                    expected = [dict(track_id=track["track_id"], xy=observation["xy"]) for track in group[label]
                                for observation in track["observations"] if observation["view"] == view]
                    self.assertEqual(bank[label], expected)
                    self.assertEqual(len(bank[label]), count)
                expected_rods, expected_anchors = [], []
                for case in cases:
                    self.assertEqual(case["rod_support_sha256"], canonical_digest(case["rod_tracks"]))
                    for segment in case["rod_tracks"]:
                        for endpoint in segment["endpoint_tracks"]:
                            for observation in endpoint["observations"]:
                                if observation["view"] == view:
                                    expected_rods.append(dict(case_id=case["case_id"], segment_id=segment["segment_id"],
                                                              endpoint_track_id=endpoint["track_id"], xy=observation["xy"]))
                    for evidence in case["evidence_sets"]:
                        self.assertEqual(evidence["source_kind"], "synthetic_2d_claim")
                        self.assertEqual(evidence["evidence_sha256"], canonical_digest(
                            {key: evidence[key] for key in ("source_kind", "anchors")}))
                        for anchor in evidence["anchors"]:
                            if anchor["view_id"] == frame["view_id"]:
                                self.assertEqual(anchor["source_sha256"], frame["source_sha256"])
                                expected_anchors.append(dict(case_id=case["case_id"], evidence_id=evidence["evidence_id"],
                                                             xy=anchor["xy"], uncertainty_xy_px=anchor["uncertainty_xy_px"]))
                self.assertEqual(bank["rod_observations"], expected_rods)
                self.assertEqual(bank["anchor_observations"], expected_anchors)
                self.assertEqual(len(expected_rods), 8)
                self.assertEqual(len(expected_anchors), 12 if view == 0 else 8 if view == 4 else 0)
                changed = copy.deepcopy(bank)
                changed["rod_observations"][0]["xy"][0] += .01
                self.assertNotEqual(canonical_digest(changed), frame["source_sha256"])

    def test_predeclared_training_groups_and_initialization_need_no_reroll_or_truth(self):
        for group in self.inputs["camera_groups"]:
            training, validation = group["training"], group["validation"]
            self.assertEqual((len(training), len(validation)), (160, 64))
            self.assertFalse({track["track_id"] for track in training} & {track["track_id"] for track in validation})
            plan = build_training_group_plan(training, validation_track_ids=[track["track_id"] for track in validation], view_count=5)
            self.assertEqual(len(plan["conditions"]), 5)
            self.assertTrue(all(condition["state"] == "eligible" for condition in plan["conditions"]))
            true = self.hidden_groups[group["camera_group_id"]]["cameras"]
            self.assertFalse(np.allclose(group["initial_intrinsics"], [camera["K_index"] for camera in true]))
            self.assertFalse(np.allclose(group["initial_extrinsics"], [camera["world_to_camera_cv"][:3] for camera in true]))
        definition = protocol()
        definition["seeds"][1] = definition["seeds"][0]
        with self.assertRaisesRegex(ValueError, "unique seed"):
            generate_paired_challenges(definition)


if __name__ == "__main__":
    unittest.main()
