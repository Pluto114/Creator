"""Mock-only complete reader intervention and normal-before-GT contracts."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixture_robust_axis_readout as runner  # noqa: E402


class FixtureRobustAxisContracts(unittest.TestCase):
    def setUp(self):
        self.config = runner.validated_config()

    def test_exact_five_arm_sixty_row_two_repeat_inventory(self):
        rows = runner.inventory(self.config)
        self.assertEqual(len(rows), 60)
        self.assertEqual(len(set(rows)), 60)
        self.assertEqual(rows[0], (0, "r01", .0025, "base"))
        self.assertEqual(rows[-1], (1, "r03", .005, "cylinder_support"))
        self.assertEqual(rows[:30], runner.inventory(self.config, 0))

    def test_fixed_config_cannot_tune_masks_reader_or_drop_arms(self):
        for key, value in (("expected_normal_rows", 59), ("repeats", [0]), ("variants", ["base"]),
                           ("reader_overrides", {"minimum_voxels": 3}), ("support_policy", {"minimum_views": 2})):
            with self.subTest(key=key), patch.object(runner, "read", return_value={**self.config, key: value}), \
                    self.assertRaises(ValueError):
                runner.validated_config()

    def test_closed_inventory_includes_independent_deepseek_tests_without_glob(self):
        with patch.object(runner.previous, "source_inventory", return_value=["inherited.py"]):
            inventory = runner.source_inventory()
        self.assertEqual(len(inventory), 11)
        self.assertEqual(inventory, sorted(set(inventory)))
        self.assertIn("tests/test_robust_axis_readout_deepseek.py", inventory)
        self.assertIn("experiments/src/creator_eval/rgb_robust_axis_readout.py", inventory)
        self.assertIn("tests/test_fixture_robust_axis_readout.py", inventory)
        self.assertIn("experiments/src/creator_eval/single_axis_readout.py", inventory)
        self.assertIn("experiments/src/creator_eval/rgb_single_axis_readout.py", inventory)
        self.assertIn("tests/test_single_axis_readout.py", inventory)

    def test_reopens_each_original_repeat_bundle_never_algorithm_renamed_storage(self):
        for repeat in (0, 1):
            base, enabled, withdrawn = runner.candidate_paths("r03", "depth_tls_single_span", repeat)
            self.assertEqual(base, runner.previous.RUN / "r03/bundle/base")
            self.assertEqual(enabled, runner.previous.RUN / f"r03/bundle/depth_tls_single_span-r{repeat}-enabled.json")
            self.assertEqual(withdrawn, runner.previous.RUN / f"r03/bundle/depth_tls_single_span-r{repeat}-withdrawn.json")
            _, enabled, _ = runner.candidate_paths("r03", "cylinder_support", repeat)
            self.assertEqual(enabled, runner.parent.RUN / "r03/bundle/cylinder_support-enabled.json")
        self.assertEqual(runner.candidate_paths("r01", "base", 0)[1:], (None, None))
        with self.assertRaises(ValueError):
            runner.candidate_paths("r01", "invented", 0)

    def test_normal_guard_and_existing_attempt_preservation(self):
        for name in ("data/eval_gt/x.json", "data/evaluation/x.json", "docs/experiments/results/x.json",
                     ".runtime/x/protocol.json", ".runtime/x/evaluation.json"):
            with self.subTest(name=name), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(ROOT / name), "r"))
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, "RUN", Path(folder)), \
                patch.object(runner.sys, "addaudithook"), self.assertRaises(FileExistsError):
            runner.prepare()

    def fake_normal(self):
        rows, old_rows, old_graphs, files, hashes = [], [], {}, {}, {}
        call = dict(voxel_size=.02, origin=[0., 0., 0.])
        for index, (repeat, cid, fraction, variant) in enumerate(runner.inventory(self.config)):
            name = f"records-r{repeat}/{cid}-{fraction}-{variant}.json"
            old_row = dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant, path=name, sha256=f"old-{index}")
            row = {**old_row, "sha256": f"new-{index}", "previous_path": name, "previous_sha256": old_row["sha256"]}
            graph = dict(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction, variant=variant,
                state="complete", resolution_state="unresolved", reason=None, segments=[], native_segments=[],
                native_state="complete", native_metric_scope="added_segments_only_not_raw_point_quality",
                control=None, raw_row_states=[], call_policy=call, full_input_point_count=10,
                supported_base_point_count=6, full_input_segment_count=0, full_curve_sample_count=0,
                supported_curve_sample_count=0, base_vote_histogram=[4, 0, 0, 6], curve_vote_histogram=[0]*4)
            old_graphs[runner.row_key(row)] = copy.deepcopy(graph)
            graph.update(reader_only_intervention=True, reader_elapsed_seconds=.1,
                effective_policy={**runner.reader.DEFAULTS, **call})
            files[runner.RUN / name], hashes[runner.RUN / name] = graph, row["sha256"]
            rows.append(row)
            old_rows.append(old_row)
        repeats = []
        for repeat in (0, 1):
            path = runner.RUN / f"repeat-{repeat}.json"
            repeats.append(dict(repeat=repeat, path=path.relative_to(runner.RUN).as_posix(), sha256=f"repeat-{repeat}"))
            files[path] = dict(run_id=runner.RUN_ID, repeat=repeat, state="complete", gt_read=False,
                pre_sha256="pre", rows=[row for row in rows if row["repeat"] == repeat])
            hashes[path] = f"repeat-{repeat}"
        normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, pre_sha256="pre", rows=rows,
            fresh_process_repeats=True, reader_only_intervention=True, repeats=repeats, elapsed_seconds=1.)
        files[runner.RUN / "inference.json"] = normal
        hashes[runner.RUN / "pre.json"] = "pre"
        hashes[runner.RUN / "inference.json"] = "normal"
        hashes[runner.previous.RUN / "inference.json"] = "old-normal"
        return normal, ({}, dict(rows=old_rows), old_graphs), files, hashes

    def test_all_sixty_normals_have_exact_original_pair(self):
        normal, parent, files, hashes = self.fake_normal()
        with patch.object(runner, "parent_normal", return_value=parent), \
                patch.object(runner, "read", side_effect=lambda p: files[p]), \
                patch.object(runner, "digest", side_effect=lambda p: hashes[p]):
            observed, graphs = runner.verified_normal(self.config)
        self.assertIs(observed, normal)
        self.assertEqual(len(graphs), 60)

    def test_count_native_and_call_changes_are_never_reader_only(self):
        normal, parent, files, _ = self.fake_normal()
        graph = files[runner.RUN / normal["rows"][0]["path"]]
        old = parent[2][runner.row_key(normal["rows"][0])]
        for key, value in (("full_input_point_count", 9), ("supported_base_point_count", 5),
                           ("full_input_segment_count", 1), ("base_vote_histogram", [10]),
                           ("native_segments", [[[0, 0, 0], [1, 0, 0]]]), ("call_policy", {})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "pairing changed"):
                runner.validate_pair({**graph, key: value}, old)

    def test_wrong_effective_policy_rejected(self):
        normal, parent, files, _ = self.fake_normal()
        graph = files[runner.RUN / normal["rows"][0]["path"]]
        with self.assertRaisesRegex(ValueError, "effective policy"):
            runner.validate_pair({**graph, "effective_policy": {}}, parent[2][runner.row_key(normal["rows"][0])])

    def test_missing_last_row_stops_before_parent_verification_or_truth(self):
        normal, _, _, _ = self.fake_normal()
        normal["rows"].pop()
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "read", return_value=normal) as read, \
                patch.object(runner, "digest", return_value="pre"), \
                patch.object(runner, "parent_normal") as parent, self.assertRaisesRegex(ValueError, "All 60"):
            runner.evaluation_payload()
        self.assertEqual(read.call_count, 1)
        parent.assert_not_called()

    def test_last_hash_native_or_repeat_tamper_blocks_evaluation(self):
        for kind in ("last_hash", "native", "repeat"):
            normal, parent, files, hashes = self.fake_normal()
            path = runner.RUN / normal["rows"][-1]["path"]
            if kind == "last_hash":
                hashes[path] = "changed"
            elif kind == "native":
                files[path]["native_segments"] = [[[0, 0, 0], [0, 0, 1]]]
            else:
                hashes[runner.RUN / "repeat-1.json"] = "changed"
            with self.subTest(kind=kind), patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                    patch.object(runner, "parent_normal", return_value=parent), \
                    patch.object(runner, "read", side_effect=lambda p: files[p]), \
                    patch.object(runner, "digest", side_effect=lambda p: hashes[p]), \
                    patch.object(runner, "physical_summary") as score, self.assertRaises(ValueError):
                runner.evaluation_payload()
            score.assert_not_called()

    def test_complete_normals_before_truth_and_native_scores_are_unchanged(self):
        normal, parent, files, hashes = self.fake_normal()
        files[runner.RUN / normal["rows"][0]["path"]]["state"] = "error"
        truth_path = ROOT / "data/eval_gt/mock/manifest.json"
        files[runner.parent.CONFIG] = dict(scene_run_id="mock")
        files[ROOT / ".runtime/experiments/mock/prepared.json"] = dict(truth_sha256="truth")
        files[truth_path] = dict(cases=[dict(case_id=cid, declared=dict(target=dict(present=False))) for cid in self.config["case_ids"]])
        files[runner.RUN / "evaluation-freeze.json"] = dict(physical_policy={})
        hashes[truth_path] = "truth"
        loaded = []
        def read(path):
            if path == truth_path:
                self.assertEqual(len(loaded), 60)
            if path.parent.name.startswith("records-r"):
                loaded.append(path)
            return files[path]
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "parent_normal", return_value=parent), \
                patch.object(runner, "read", side_effect=read), \
                patch.object(runner, "digest", side_effect=lambda p: hashes[p]), \
                patch.object(runner, "physical_summary", return_value={"mock": True}) as score:
            result = runner.evaluation_payload()
        self.assertEqual(score.call_count, 239)
        self.assertEqual(len(result["rows"]), 60)
        self.assertEqual(len(result["base_rows"]), 12)
        self.assertIsNone(result["rows"][0]["physical"])
        self.assertEqual(result["native_unchanged_rows"], 60)
        self.assertEqual(result["physical_repeat_equal"], 29)
        self.assertFalse(result["reader_qualified"])
        self.assertFalse(result["g1_passed"])

    def test_fresh_process_replays_do_not_refit_or_rebuild_patches(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, "RUN", Path(folder)), \
                patch.object(runner.sys, "addaudithook"), \
                patch.object(runner, "checked", return_value=({}, self.config)), \
                patch.object(runner, "repeat_normal", return_value=dict(rows=[], elapsed_seconds=0.)), \
                patch.object(runner, "digest", return_value="hash"), patch.object(runner, "verified_normal"), \
                patch.object(runner.subprocess, "run") as run, \
                patch.object(runner.previous.controls, "fit_controls") as fit, \
                patch.object(runner.previous, "persist_control") as persist:
            runner.infer()
            self.assertEqual(run.call_count, 2)
            for repeat, call in enumerate(run.call_args_list):
                self.assertEqual(call.args[0][-2:], ["infer-repeat", str(repeat)])
                self.assertTrue(call.kwargs["check"])
            fit.assert_not_called()
            persist.assert_not_called()
            with self.assertRaises(FileExistsError):
                runner.infer()


if __name__ == "__main__":
    unittest.main()
