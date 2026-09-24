"""Stable cell groups, held-out isolation and explicit support failures."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/src"))
from creator_eval.camera_training_groups import build_training_group_plan, select_training_tracks


def fixture():
    return [dict(track_id=3 * cell + within,
                 observations=[dict(view=view, xy=[64 * cell + 8 + within + view, 210. + within]) for view in range(5)])
            for cell in range(16) for within in range(3)]


def plan_for(tracks, **kwargs):
    return build_training_group_plan(tracks, validation_track_ids=[100, 101], view_count=5,
                                     minimum_tracks=12, minimum_tracks_per_view=8, **kwargs)


class TrainingGroupTests(unittest.TestCase):
    def test_track_and_observation_order_do_not_change_plan(self):
        tracks = fixture()
        before = copy.deepcopy(tracks)
        shuffled = list(reversed(copy.deepcopy(tracks)))
        for track in shuffled:
            track["observations"].reverse()
        self.assertEqual(plan_for(tracks), plan_for(shuffled))
        self.assertEqual(tracks, before)

    def test_whole_spatial_cells_and_whole_tracks_are_removed(self):
        tracks = fixture()
        plan = plan_for(tracks)
        self.assertTrue(all(c["removed_track_count"] > 0 for c in plan["conditions"][1:]))
        all_removed = []
        for condition in plan["conditions"][1:]:
            removed = set(condition["removed_track_ids"])
            all_removed.extend(removed)
            for cell in plan["cells"]:
                self.assertIn(len(removed & set(cell["track_ids"])), (0, len(cell["track_ids"])))
            selected = select_training_tracks(tracks, plan, condition["condition_id"])
            self.assertTrue(all(len(track["observations"]) == 5 for track in selected))
            self.assertEqual([t["track_id"] for t in selected], condition["kept_track_ids"])
        self.assertEqual(sorted(all_removed), plan["training_track_ids"])

    def test_validation_identity_isolated_and_source_coordinates_bound(self):
        tracks = fixture()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_training_group_plan(tracks, validation_track_ids=[0], view_count=5)
        plan = plan_for(tracks)
        changed = copy.deepcopy(tracks)
        changed[0]["observations"][0]["xy"][0] += .01
        with self.assertRaisesRegex(ValueError, "changed"):
            select_training_tracks(changed, plan, "control")
        other_validation = build_training_group_plan(tracks, validation_track_ids=[200], view_count=5,
                                                     minimum_tracks=12, minimum_tracks_per_view=8)
        self.assertEqual(plan["assignments"], other_validation["assignments"])
        self.assertEqual(plan["conditions"], other_validation["conditions"])

    def test_frozen_selection_reused_across_camera_inputs_without_aliases(self):
        tracks = fixture()
        plan = plan_for(tracks)
        frozen = copy.deepcopy(plan)
        # 两次拟合的相机可以任意变化。计划根本不收相机参数，选出的RGB保持不变。
        selected_a = select_training_tracks(tracks, plan, "leave_group_0")
        selected_b = select_training_tracks(tracks, plan, "leave_group_0")
        self.assertEqual(selected_a, selected_b)
        selected_a[0]["observations"][0]["xy"][0] += 10
        self.assertEqual(plan, frozen)
        self.assertEqual(selected_b, select_training_tracks(tracks, plan, "leave_group_0"))
        self.assertEqual(select_training_tracks(tracks, plan, "control"), tracks)

    def test_insufficient_coverage_and_empty_groups_skip_without_rebalancing(self):
        tracks = fixture()
        initial = plan_for(tracks)
        group_zero = {a["track_id"] for a in initial["assignments"] if a["group"] == 0}
        for track in tracks:
            if track["track_id"] not in group_zero:
                track["observations"] = [o for o in track["observations"] if o["view"] != 4]
        plan = plan_for(tracks)
        self.assertEqual(plan["conditions"][0]["state"], "eligible")
        deleted = plan["conditions"][1]
        self.assertEqual(deleted["coverage"][4], 0)
        self.assertEqual(deleted["state"], "skipped")
        self.assertIn("insufficient_remaining_per_view_coverage", deleted["skip_reasons"])
        with self.assertRaisesRegex(ValueError, "skipped"):
            select_training_tracks(tracks, plan, "leave_group_0")
        tiny = build_training_group_plan(fixture()[:2], validation_track_ids=[], view_count=5,
                                         minimum_tracks=1, minimum_tracks_per_view=1)
        self.assertEqual(sum("empty_removed_group" in c["skip_reasons"] for c in tiny["conditions"][1:]), 3)
        removed = next(c for c in tiny["conditions"][1:] if c["removed_track_count"])
        self.assertIn("insufficient_remaining_tracks", removed["skip_reasons"])


if __name__ == "__main__":
    unittest.main()
