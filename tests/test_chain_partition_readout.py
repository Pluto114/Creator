"""Synthetic partition/alias contracts; no GT or experiment-score inputs.

The multi-branch contexts below attach explicitly labelled synthetic required-
view metadata to an all-chain fixture. They are mechanism probes, not a claim
that actual foreground annotations selected several objects.
"""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval import chain_partition_readout as partition  # noqa: E402
from creator_eval import rgb_chain_support as chain  # noqa: E402
from creator_eval import rgb_stroke_chain_support as stroke  # noqa: E402
from creator_eval.chain_segment_union import merge_collinear_segments  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.line_controls import point_to_segments_distance  # noqa: E402
from creator_eval.rgb_chain_partition_axis_readout import ChainPartitionAxisReadout  # noqa: E402
from creator_eval.rod_fixture_finite import canonical_hash  # noqa: E402
from test_rgb_chain_support import fixture  # noqa: E402
from test_rgb_stroke_chain_support import marks  # noqa: E402


def mechanism_context(xs=(-.3, .3), shift=None):
    frames, cameras = fixture(xs)
    if shift is not None:
        shift = np.asarray(shift, float)
        for camera in cameras:
            e = np.asarray(camera["world_to_camera_cv"], float)
            e[:, 3] -= e[:, :3] @ shift
            camera["world_to_camera_cv"] = e.tolist()
    context = chain.build_context(frames, cameras)
    claims = marks(frames)
    context["association"].update(anchors=claims, anchors_sha256=canonical_hash(claims),
        anchor_view_count=2, anchor_point_count=4,
        synthetic_test_metadata_only=True, anchor_selection_performed=False)
    context["sha256"] = canonical_hash(context["association"])
    return context


def line_points(x, ys=None):
    ys = np.linspace(-1., 1., 129) if ys is None else np.asarray(ys, float)
    return np.column_stack((np.full(len(ys), x), ys, np.full(len(ys), 3.)))


def alias_inputs(segments, supports=None):
    if supports is None:
        supports = [np.arange(10) for _ in segments]
    rows, groups = [], []
    for i, (values, support) in enumerate(zip(segments, supports)):
        rows.append(dict(support_set_sha256=f"synthetic-support-{i}",
            graph=dict(segments=np.asarray(values, float).reshape(-1, 2, 3))))
        groups.append(dict(indices=np.asarray(support, np.int64)))
    return rows, groups


def x_segment(a, b, y=0.):
    return [[a, y, 0.], [b, y, 0.]]


class PartitionGeometryContracts(unittest.TestCase):
    def test_real_fixture_groups_correspond_to_actual_support_sets(self):
        context = mechanism_context()
        values = np.concatenate((line_points(-.3), line_points(.3)))
        groups = partition.support_groups(values, context)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(g["indices"]) for g in groups), [129, 129])
        self.assertFalse(set(groups[0]["indices"]) & set(groups[1]["indices"]))
        self.assertEqual(set(np.concatenate([g["indices"] for g in groups])), set(range(len(values))))

    def test_two_independent_axes_are_not_averaged(self):
        context = mechanism_context()
        values = np.concatenate((line_points(-.3), line_points(.3)))
        graph = partition.readout(values, .02, [0., 0., 0.], context)
        self.assertEqual(graph["components"], 2)
        np.testing.assert_allclose(np.sort(graph["segments"][:, 0, 0]), [-.3, .3], atol=1e-12)
        np.testing.assert_allclose(graph["segments"][:, 0, 0], graph["segments"][:, 1, 0], atol=1e-12)
        self.assertFalse(graph["global_union_fit_performed"])
        self.assertFalse(graph["target_identity_confirmed"])
        self.assertEqual(graph["suppressed_aliases"], [])

    def test_geometrically_degenerate_chain_does_not_veto_good_chain(self):
        context = mechanism_context()
        values = np.concatenate((line_points(-.3), line_points(.3, [-.001, .001])))
        graph = partition.readout(values, .02, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "complete")
        self.assertEqual(graph["components"], 1)
        np.testing.assert_allclose(graph["segments"][..., 0], -.3, atol=1e-12)
        bad = [r["graph"] for r in graph["readouts"] if not len(r["graph"]["segments"])]
        self.assertEqual(len(bad), 1)
        self.assertIn("too_few_points", bad[0]["reason"])

    def test_actual_coordinates_are_not_snapped_to_image_chain(self):
        frames, cameras = fixture()
        context = stroke.build_context(frames, cameras, marks(frames))
        values = line_points(.006)
        original = values.copy()
        graph = partition.readout(values, .02, [0., 0., 0.], context)
        self.assertEqual(graph["components"], 1)
        np.testing.assert_allclose(graph["segments"][..., 0], .006, atol=1e-12)
        np.testing.assert_array_equal(values, original)

    def test_real_geometry_gap_is_not_filled(self):
        context = mechanism_context((0.,))
        values = line_points(0., np.r_[np.linspace(-1., -.3, 65), np.linspace(.3, 1., 65)])
        graph = partition.readout(values, .02, [0., 0., 0.], context)
        self.assertEqual(graph["components"], 2)
        for segment in graph["segments"]:
            self.assertTrue(segment[:, 1].max() < -.25 or segment[:, 1].min() > .25)

    def test_rgb_unknown_gap_is_not_filled(self):
        frames, cameras = fixture()
        claims = marks(frames)
        for anchor in claims:
            if anchor["xy"][1] == 100:
                anchor["xy"][1] = 80.
        for frame in frames:
            for row in frame["observations"]["rows"]:
                if 95 <= row["y"] <= 105:
                    row.update(status="unknown", candidates=[])
            frame["observation_sha256"] = canonical_hash(frame["observations"])
        context = stroke.build_context(frames, cameras, claims)
        graph = partition.readout(line_points(0.), .02, [0., 0., 0.], context)
        self.assertEqual(graph["components"], 2)
        for segment in graph["segments"]:
            self.assertTrue(segment[:, 1].max() < 0 or segment[:, 1].min() > 0)

    def test_duplicate_points_assignments_and_input_order_are_idempotent(self):
        context = mechanism_context()
        values = np.concatenate((line_points(-.3), line_points(.3)))
        before = partition.readout(values, .02, [0., 0., 0.], context)
        duplicated = copy.deepcopy(context)
        original = duplicated["association"]["assignments"]
        duplicated["association"]["assignments"] = list(reversed(original+original))
        duplicated["association"]["chain_count"] = len(original)*2
        duplicated["association"]["assignment_sha256"] = canonical_hash(duplicated["association"]["assignments"])
        duplicated["sha256"] = canonical_hash(duplicated["association"])
        after = partition.readout(np.concatenate((values[::-1], values, values)), .02, [0., 0., 0.], duplicated)
        np.testing.assert_array_equal(before["segments"], after["segments"])
        self.assertEqual(before["fitted_support_groups"], after["fitted_support_groups"])
        self.assertEqual([r["assignment_count"] for r in before["readouts"]],
                         [r["assignment_count"] for r in after["readouts"]])

    def test_identical_sample_coordinates_are_fair_between_base_and_curve(self):
        context = mechanism_context()
        segments = np.array([[[-.3, -1., 3.], [-.3, 1., 3.]],
                             [[.3, -1., 3.], [.3, 1., 3.]]])
        samples = sample_segments(segments, .01, 2000000)
        policy = dict(voxel_size=.02, origin=[0., 0., 0.])
        base = ChainPartitionAxisReadout(samples, context)(np.empty((0, 2, 3)), policy)
        curve = ChainPartitionAxisReadout(np.empty((0, 3)), context)(segments, policy)
        np.testing.assert_array_equal(base["segments"], curve["segments"])
        self.assertEqual(base["supported_base_point_count"], curve["supported_curve_sample_count"])
        self.assertEqual(base["unique_count"], curve["unique_count"])
        self.assertFalse(base["reader_algorithm_unchanged"])

    def test_joint_translation_of_points_cameras_and_origin(self):
        values = np.concatenate((line_points(-.25), line_points(.25)))
        shift = np.array([8., -4., 16.])
        before = partition.readout(values, .03125, [0., 0., 0.], mechanism_context((-.25, .25)))
        after = partition.readout(values+shift, .03125, shift, mechanism_context((-.25, .25), shift))
        np.testing.assert_allclose(after["segments"]-shift, before["segments"], atol=1e-11)
        self.assertEqual(before["components"], after["components"])

    def test_partition_budgets_fail_without_partial_positive_output(self):
        context = mechanism_context()
        values = np.concatenate((line_points(-.3), line_points(.3)))
        for field, reason in (
            ("maximum_assignments", "partition_assignment_budget_exceeded"),
            ("maximum_fit_groups", "partition_fit_group_budget_exceeded"),
            ("maximum_membership_checks", "partition_membership_budget_exceeded"),
            ("maximum_output_segments", "partition_output_budget_exceeded"),
        ):
            with self.subTest(field=field), patch.dict(partition.DEFAULTS, {field: 1}):
                graph = partition.readout(values, .02, [0., 0., 0.], context)
                self.assertEqual(graph["state"], "unmeasurable")
                self.assertEqual(graph["reason"], reason)
                self.assertEqual(len(graph["segments"]), 0)

    def test_point_budget_and_incomplete_search_have_no_partition_claim(self):
        context = mechanism_context()
        values = line_points(-.3)
        with patch.dict(partition.axis.DEFAULTS, {"maximum_axis_samples": 1}):
            graph = partition.readout(values, .02, [0., 0., 0.], context)
        self.assertEqual(graph["reason"], "partition_point_budget_exceeded")
        context["association"]["fallback_to_raw_union"] = True
        graph = partition.readout(values, .02, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "unmeasurable")
        self.assertEqual(graph["reason"], "incomplete_chain_search_no_partition_claim")
        self.assertEqual(len(graph["segments"]), 0)

    def test_one_incomplete_geometry_group_is_not_silently_ignored(self):
        context = mechanism_context()
        values = np.concatenate((line_points(-.3), line_points(.3)))
        real_readout = partition.axis.readout
        calls = []
        def incomplete_second(*args, **kwargs):
            calls.append(1)
            graph = real_readout(*args, **kwargs)
            if len(calls) == 2:
                graph.update(state="unmeasurable", reason="synthetic_budget_exhaustion")
            return graph
        with patch.object(partition.axis, "readout", side_effect=incomplete_second):
            graph = partition.readout(values, .02, [0., 0., 0.], context)
        self.assertEqual(graph["state"], "unmeasurable")
        self.assertEqual(graph["reason"], "incomplete_partition_geometry")
        self.assertEqual(len(graph["segments"]), 0)


class ResolutionAliasContracts(unittest.TestCase):
    def test_gap_sampling_counterexample_requires_continuous_certificate(self):
        kept = [x_segment(0., 10.), x_segment(12.2, 22.2)]
        candidate = [x_segment(8.85, 13.35)]
        rows, groups = alias_inputs([kept, candidate], [np.arange(20), np.arange(10)])
        samples = sample_segments(np.asarray(candidate), .5, 100)
        self.assertLess(float(point_to_segments_distance(samples, np.asarray(kept)).max()), 1.)
        self.assertGreater(float(point_to_segments_distance([[11.1, 0., 0.]], np.asarray(kept))[0]), 1.)
        selected, aliases, _ = partition.select_readouts(rows, groups, 1.)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(aliases, [])

    def test_near_alias_is_suppressed_without_moving_retained_endpoints(self):
        first, second = [x_segment(0., 10.)], [x_segment(.5, 9.5, .3)]
        rows, groups = alias_inputs([first, second], [[0, 1, 2, 3], [0, 1, 2, 4]])
        original = copy.deepcopy(rows)
        selected, aliases, _ = partition.select_readouts(rows, groups, 1.)
        self.assertEqual(selected, [0])
        self.assertEqual(aliases[0]["group"], 1)
        self.assertEqual(aliases[0]["retained_group"], 0)
        self.assertLessEqual(aliases[0]["continuous_distance_upper_bound_m"], 1.)
        actual = merge_collinear_segments([s for i in selected for s in rows[i]["graph"]["segments"]])
        np.testing.assert_array_equal(actual, np.asarray(first))
        for old, new in zip(original, rows):
            np.testing.assert_array_equal(old["graph"]["segments"], new["graph"]["segments"])

    def test_independent_support_sets_are_not_aliased(self):
        rows, groups = alias_inputs([[x_segment(0., 10.)], [x_segment(.5, 9.5, .3)]],
                                    [np.arange(10), np.arange(10, 20)])
        selected, aliases, _ = partition.select_readouts(rows, groups, 1.)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(aliases, [])

    def test_less_than_half_shared_support_keeps_competitor(self):
        rows, groups = alias_inputs([[x_segment(0., 10.)], [x_segment(.5, 9.5, .3)]],
                                    [[0, 1, 2, 3], [0, 4, 5, 6]])
        selected, aliases, _ = partition.select_readouts(rows, groups, 1.)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(aliases, [])

    def test_far_parallel_and_extending_geometry_are_retained(self):
        for candidate in ([x_segment(.5, 9.5, 2.)], [x_segment(8., 13.)]):
            with self.subTest(candidate=candidate):
                rows, groups = alias_inputs([[x_segment(0., 10.)], candidate])
                selected, aliases, _ = partition.select_readouts(rows, groups, 1.)
                self.assertEqual(selected, [0, 1])
                self.assertEqual(aliases, [])

    def test_alias_suppression_is_not_transitive(self):
        rows, groups = alias_inputs([[x_segment(0., 10., y)] for y in (0., .6, 1.2)])
        selected, aliases, _ = partition.select_readouts(rows, groups, 1.)
        self.assertEqual(selected, [0, 2])
        self.assertEqual([(a["group"], a["retained_group"]) for a in aliases], [(1, 0)])

    def test_alias_distance_work_budget_raises_not_partial_selection(self):
        rows, groups = alias_inputs([[x_segment(0., 10.)], [x_segment(.5, 9.5, .3)]])
        with patch.dict(partition.DEFAULTS, {"maximum_alias_distance_checks": 1}):
            with self.assertRaisesRegex(OverflowError, "partition_alias_work_budget_exceeded"):
                partition.select_readouts(rows, groups, 1.)

    def test_tie_breaking_is_geometry_deterministic_not_list_order(self):
        rows, groups = alias_inputs([[x_segment(0., 10., y)] for y in (0., .3)])
        first, _, _ = partition.select_readouts(rows, groups, 1.)
        reverse_rows, reverse_groups = rows[::-1], groups[::-1]
        second, _, _ = partition.select_readouts(reverse_rows, reverse_groups, 1.)
        np.testing.assert_array_equal(rows[first[0]]["graph"]["segments"],
                                      reverse_rows[second[0]]["graph"]["segments"])


class SegmentUnionContracts(unittest.TestCase):
    def test_only_machine_collinear_overlap_is_unioned(self):
        source = np.array([x_segment(0., 2.), x_segment(1., 3.), x_segment(0., 3., .2)])
        actual = merge_collinear_segments(source)
        np.testing.assert_array_equal(actual, np.array([x_segment(0., 3.), x_segment(0., 3., .2)]))

    def test_positive_gap_and_endpoint_coordinates_survive_union(self):
        source = np.array([x_segment(0., 1.), x_segment(1.+1e-8, 2.)])
        actual = merge_collinear_segments(source[::-1, ::-1])
        np.testing.assert_array_equal(actual, source)

    def test_segment_union_is_order_and_multiplicity_idempotent(self):
        source = np.array([x_segment(0., 2.), x_segment(1., 3.), x_segment(4., 5.)])
        expected = merge_collinear_segments(source)
        actual = merge_collinear_segments(np.concatenate((source[::-1, ::-1], source, source)))
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
