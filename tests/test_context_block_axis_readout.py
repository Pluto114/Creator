"""Normal-input guide direction, transverse evidence and geometric fidelity."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import context_block_axis_readout as reader  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_context_block_axis_readout import ContextBlockAxisReadout  # noqa: E402
from creator_eval.rgb_single_axis_readout import SingleAxisReadout  # noqa: E402
from test_single_axis_readout import views  # noqa: E402


def yes(points):
    return np.ones(len(points), bool)


def line():
    return np.c_[np.linspace(0, 1, 101), np.zeros((101, 2))]


def guided_views():
    result = views()
    for view, angle in zip(result, (-.3, 0, .3)):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
        e = np.c_[rotation, [0., 0, 3.]]
        view["world_to_camera_cv"] = e
        projected = (np.array([[-.8, 0, 2], [.8, 0, 2]]) @ rotation.T+e[:, 3]) @ view["K_index"].T
        view["guide_xyxy"] = projected[:, :2]/projected[:, 2, None]
        view["guide_source"] = "independent_mechanism_fixture_input"
    return result


class ContextBlockContracts(unittest.TestCase):
    def test_slanted_short_branch_masks_fixed_blocks_and_refits_known_points(self):
        t = np.linspace(0, 1, 201)
        main = np.c_[t, .08*t, t*0]
        branch = main[(t >= .401) & (t <= .475)]+[0, .04, 0]
        values = np.r_[main, branch]
        result = reader.readout(values, .01, [0, 0, 0], yes, [1, 0, 0])
        np.testing.assert_array_equal(result["unknown_blocks"], [10, 11])
        self.assertEqual(result["components"], 2)
        unique = np.unique(values, axis=0)
        unknown = np.isin(np.floor(unique[:, 0]/.04).astype(int), [10, 11])
        self.assertEqual(result["unknown_original_point_count"], int(unknown.sum()))
        self.assertEqual(result["known_original_point_count"], int((~unknown).sum()))
        for segment in result["segments"]:
            sampled = sample_segments(segment[None], .0001, 2000000)
            self.assertFalse(np.isin(np.floor(sampled[:, 0]/.04).astype(int), [10, 11]).any())
            np.testing.assert_allclose(sampled[:, 1], .08*sampled[:, 0], atol=1e-12)

    def test_all_unknown_is_explicit_insufficient_support(self):
        evidence = dict(state="unresolved", reason="nonpersistent_split_transverse_evidence",
                        split_blocks=[dict(block=i) for i in range(26)])
        with patch.object(reader, "transverse_evidence", return_value=evidence):
            result = reader.readout(line(), .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertEqual(result["components"], 0)
        self.assertEqual(result["known_original_point_count"], 0)
        self.assertEqual(result["reason"], "too_few_unambiguous_occupied_voxels")

    def test_one_ambiguous_block_does_not_erase_other_observed_geometry(self):
        values = np.r_[line(), [[.412, .03, 0], [.423, .03, 0]]]
        result = reader.readout(values, .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertEqual(result["components"], 2)
        self.assertEqual(result["resolution_state"], "partially_resolved")
        np.testing.assert_array_equal(result["unknown_blocks"], [10])
        self.assertEqual(result["input_count"], len(values))
        self.assertGreater(result["unknown_original_point_count"], 2)
        self.assertGreater(result["unknown_axis_sample_count"], 0)
        for segment in result["segments"]:
            self.assertFalse(min(segment[:, 0]) <= .42 <= max(segment[:, 0]))
        np.testing.assert_allclose(result["segments"][:, :, 1:], 0., atol=1e-12)

    def test_local_unknown_follows_fixed_origin_not_output_axis(self):
        values = np.r_[line(), [[.412, .03, 0], [.423, .03, 0]]]
        shift = np.array([.017, .31, -.2])
        first = reader.readout(values, .01, [0, 0, 0], yes, [1, 0, 0])
        second = reader.readout(values+shift, .01, shift, yes, [1, 0, 0])
        np.testing.assert_array_equal(first["unknown_blocks"], second["unknown_blocks"])
        np.testing.assert_allclose(second["segments"], first["segments"]+shift, atol=.00500001)

    def test_local_unknown_never_invents_rgb_evidence(self):
        values = np.r_[line(), [[.412, .03, 0], [.423, .03, 0]]]
        result = reader.readout(values, .01, [0, 0, 0], lambda p: p[:, 0] < .7, [1, 0, 0])
        self.assertEqual(result["components"], 2)
        self.assertLess(float(result["segments"].max(axis=(0, 1))[0]), .7)

    def test_guide_planes_supply_direction_not_location(self):
        original = guided_views()
        first = reader.guide_direction(original)
        for view in original:
            view["guide_xyxy"] += [0, 3]
        second = reader.guide_direction(original)
        self.assertEqual(first["state"], "resolved")
        np.testing.assert_allclose(first["direction"], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(second["direction"], first["direction"], atol=1e-12)
        self.assertFalse(first["location_used"])

    def test_degenerate_and_invalid_guides_are_not_silently_replaced(self):
        v = guided_views()
        for view in v[1:]:
            view["world_to_camera_cv"] = v[0]["world_to_camera_cv"].copy()
            view["guide_xyxy"] = v[0]["guide_xyxy"].copy()
        self.assertEqual(reader.guide_direction(v)["reason"], "guide_direction_degenerate")
        v[0]["guide_xyxy"][:] = 0
        with self.assertRaises(ValueError):
            reader.guide_direction(v)
        with self.assertRaises(ValueError):
            reader.guide_direction(guided_views()[:2])

    def test_noisy_shifted_rod_keeps_its_actual_geometry(self):
        points = line()+[0, .06, -.04]
        points[:, 1:] += np.random.default_rng(7).uniform(-.003, .003, (101, 2))
        result = reader.readout(points, .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertEqual(result["components"], 1)
        np.testing.assert_allclose(result["segments"][0, :, 1:], [[.06, -.04]]*2, atol=.003)

    def test_short_parallel_tracks_are_not_averaged_when_persistence_is_insufficient(self):
        first = np.c_[np.linspace(0, .05, 11), np.full(11, -.02), np.zeros(11)]
        result = reader.readout(np.r_[first, first+[0, .04, 0]], .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertEqual(result["components"], 0)
        self.assertTrue(result["reason"])

    def test_weak_bridge_points_do_not_merge_two_persistent_tracks(self):
        first = line()+[0, -.01, 0]
        bridge = np.c_[np.arange(.015, 1, .04), np.zeros((25, 2))]
        result = reader.readout(np.r_[first, first+[0, .02, 0], bridge], .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertEqual(result["components"], 0)
        self.assertEqual(result["reason"], "persistent_multiple_transverse_tracks")

    def test_rotated_parallel_tracks_and_shallow_crossing_remain_unresolved(self):
        first = line()
        for angle in np.deg2rad([0, 30, 45, 70]):
            offset = np.array([0, .04*np.cos(angle), .04*np.sin(angle)])
            result = reader.readout(np.r_[first, first+offset], .01, [0, 0, 0], yes, [1, 0, 0])
            self.assertEqual(result["components"], 0)
        x = np.linspace(-.5, .5, 101)
        result = reader.readout(np.r_[np.c_[x, x*0, x*0], np.c_[x, .04*x, x*0]],
                                .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertEqual(result["components"], 0)

    def test_off_axis_gap_structure_cannot_bridge_actual_gap(self):
        ends = line()
        ends = ends[(ends[:, 0] < .3) | (ends[:, 0] > .7)]
        noise = np.c_[np.linspace(.3, .7, 21), np.full(21, .06), np.zeros(21)]
        result = reader.readout(np.r_[ends, noise], .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertTrue(all(not min(segment[:, 0]) <= .5 <= max(segment[:, 0]) for segment in result["segments"]))

    def test_sparse_long_and_dense_short_second_rod_are_not_density_selected(self):
        sparse = np.c_[np.linspace(0, 1, 61), np.zeros((61, 2))]
        dense = np.c_[np.linspace(0, .25, 1001), np.full(1001, .2), np.zeros(1001)]
        result = reader.readout(np.r_[sparse, dense], .01, [0, 0, 0], yes, [1, 0, 0])
        self.assertEqual(result["components"], 0)

    def test_point_curve_sampling_equivalence_and_unchanged_union_counts(self):
        segment = np.array([[[-.8, 0, 2], [.8, 0, 2]]])
        sampled = sample_segments(segment, .01, 2000000)
        config = dict(voxel_size=.02, origin=[0, 0, 0])
        base = ContextBlockAxisReadout(sampled, guided_views())(np.empty((0, 2, 3)), config)
        curve = ContextBlockAxisReadout(np.empty((0, 3)), guided_views())(segment, config)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        np.testing.assert_allclose(base["segments"], segment, atol=1e-12)
        old = SingleAxisReadout(sampled, guided_views())(np.empty((0, 2, 3)), config)
        for key in ("full_input_point_count", "supported_base_point_count", "full_input_segment_count",
                    "full_curve_sample_count", "supported_curve_sample_count", "base_vote_histogram", "curve_vote_histogram"):
            self.assertEqual(base[key], old[key], key)

    def test_budgets_invalid_inputs_and_runtime_errors_stay_visible(self):
        for key in ("maximum_unique_points", "maximum_voxels", "maximum_axis_samples"):
            with patch.dict(reader.DEFAULTS, {key: 5}):
                self.assertEqual(reader.readout(line(), .01, [0, 0, 0], yes, [1, 0, 0])["state"], "unmeasurable")
        for size in (0, -1, np.nan, True):
            with self.assertRaises(ValueError):
                reader.readout(line(), size, [0, 0, 0], yes, [1, 0, 0])
        with self.assertRaises(ValueError):
            reader.readout(line(), .01, [0, 0, 0], yes, [0, 0, 0])
        with self.assertRaises(ValueError):
            reader.readout(line(), .01, [0, 0, 0], lambda p: np.ones(len(p)), [1, 0, 0])
        with patch.object(reader, "readout", side_effect=RuntimeError("visible failure")):
            result = ContextBlockAxisReadout(line()+[0, 0, 2], guided_views())(
                np.empty((0, 2, 3)), dict(voxel_size=.01, origin=[0, 0, 0]))
        self.assertEqual(result["state"], "error")

    def test_empty_and_coincident_remain_explicit_unresolved(self):
        for points in (np.empty((0, 3)), np.zeros((100, 3))):
            result = reader.readout(points, .01, [0, 0, 0], yes, [1, 0, 0])
            self.assertEqual(result["state"], "complete")
            self.assertTrue(result["reason"])
            self.assertEqual(result["components"], 0)

    def test_guide_location_cannot_pull_output_and_policies_are_fixed(self):
        shifted = line()+[0, .2, 2]
        a, b = guided_views(), guided_views()
        for view in b:
            view["guide_xyxy"] += [0, 3]
        config = dict(voxel_size=.01, origin=[0, 0, 0])
        first, second = [ContextBlockAxisReadout(shifted, v)(np.empty((0, 2, 3)), config) for v in (a, b)]
        np.testing.assert_array_equal(first["segments"], second["segments"])
        self.assertEqual(first["components"], 1)
        with self.assertRaises(ValueError):
            ContextBlockAxisReadout(shifted, a, dict(minimum_views=2, horizontal_padding_px=1))
        with self.assertRaises(ValueError):
            ContextBlockAxisReadout(shifted, a)(np.empty((0, 2, 3)), {**config, "method": "candidate"})


if __name__ == "__main__":
    unittest.main()
