"""Mock-only complete reader intervention and normal-before-GT contracts."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_fixture_anchored_chain_readout as runner  # noqa: E402


class FixtureRGBChainContracts(unittest.TestCase):
    def setUp(self):
        self.config = runner.validated_config()
        self.control_patch = patch.object(runner, "verify_old_controls")
        self.control_patch.start()
        self.addCleanup(self.control_patch.stop)
        self.anchor_patch = patch.object(runner, "validate_anchor_input")
        self.anchor_patch.start()
        self.addCleanup(self.anchor_patch.stop)

    def test_exact_five_arm_two_reader_180_row_two_repeat_inventory(self):
        rows = runner.inventory(self.config)
        self.assertEqual(len(rows), 180)
        self.assertEqual(len(set(rows)), 180)
        self.assertEqual(rows[0], (0, "r01", .0025, "base", "single_axis"))
        self.assertEqual(rows[-1], (1, "r03", .005, "cylinder_support", "anchored_chain_axis"))
        self.assertEqual(rows[:90], runner.inventory(self.config, 0))

    def test_fixed_config_cannot_tune_masks_reader_or_drop_arms(self):
        for key, value in (("expected_normal_rows", 59), ("repeats", [0]), ("variants", ["base"]),
                           ("reader_overrides", {"minimum_voxels": 3}), ("support_policy", {"minimum_views": 2}),
                           ("development_prior_scores_seen", False), ("blind_evaluation", True)):
            with self.subTest(key=key), patch.object(runner, "read", return_value={**self.config, key: value}), \
                    self.assertRaises(ValueError):
                runner.validated_config()

    def test_closed_inventory_includes_all_new_sources_and_reused_deepseek_tests_without_glob(self):
        with patch.object(runner.previous_sources, "source_inventory", return_value=["inherited.py"]):
            inventory = runner.source_inventory()
        self.assertEqual(len(inventory), 20)
        self.assertEqual(inventory, sorted(set(inventory)))
        self.assertIn("tests/test_rgb_chain_support_deepseek.py", inventory)
        self.assertIn("experiments/src/creator_eval/rgb_chain_axis_readout.py", inventory)
        self.assertIn("tests/test_fixture_anchored_chain_readout.py", inventory)
        self.assertIn("scripts/run_g1_object_anchored_chain_readout.py", inventory)
        self.assertIn("configs/g1_object_anchored_chain_readout_v1.json", inventory)
        self.assertIn("tests/test_g1_object_anchored_chain_readout.py", inventory)
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
        for index, (repeat, cid, fraction, variant, name) in enumerate(runner.inventory(self.config)):
            old_name = f"records-r{repeat}/{cid}-{fraction}-{variant}.json"
            filename = f"records-r{repeat}/{cid}-{fraction}-{variant}-{name}.json"
            old_row = dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant, path=old_name, sha256="old-"+old_name)
            row = {**old_row, "reader": name, "path": filename, "sha256": f"new-{index}",
                "previous_path": old_name, "previous_sha256": old_row["sha256"]}
            graph = dict(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction, variant=variant,
                state="complete", resolution_state="unresolved", reason=None, segments=[], native_segments=[],
                native_state="complete", native_metric_scope="added_segments_only_not_raw_point_quality",
                control=None, raw_row_states=[], call_policy=call, full_input_point_count=10,
                supported_base_point_count=6, full_input_segment_count=0, full_curve_sample_count=0,
                supported_curve_sample_count=0, base_vote_histogram=[4, 0, 0, 6], curve_vote_histogram=[0]*4)
            if name == "single_axis":
                old_graphs[runner.row_key(row)[:-1]] = copy.deepcopy(graph)
                old_rows.append(old_row)
            graph.update(reader=name, reader_elapsed_seconds=.1, **runner.FLAGS,
                effective_policy={**runner.reader.DEFAULTS, **call})
            if name in runner.CONTEXT_READERS:
                graph.update(supported_base_point_count=4, raw_base_vote_histogram=[4, 0, 0, 6],
                    raw_curve_vote_histogram=[0]*4, base_chain_mask_subset_raw=True, curve_chain_mask_subset_raw=True,
                    chain_context_sha256=runner.canonical_hash({}), chain_association={}, target_identity_confirmed=False)
            files[runner.RUN / filename], hashes[runner.RUN / filename] = graph, row["sha256"]
            rows.append(row)
        repeats = []
        for repeat in (0, 1):
            path = runner.RUN / f"repeat-{repeat}.json"
            repeats.append(dict(repeat=repeat, path=path.relative_to(runner.RUN).as_posix(), sha256=f"repeat-{repeat}"))
            associations = []
            for cid in self.config["case_ids"]:
                for name in runner.CONTEXT_READERS:
                    assoc_path = runner.RUN / f"records-r{repeat}/{cid}-{name}-association.json"
                    files[assoc_path], hashes[assoc_path] = {}, "association"
                    associations.append(dict(case_id=cid, reader=name, path=assoc_path.relative_to(runner.RUN).as_posix(),
                        sha256="association", chain_context_sha256=runner.canonical_hash({})))
            files[path] = dict(run_id=runner.RUN_ID, repeat=repeat, state="complete", gt_read=False,
                pre_sha256="pre", rows=[row for row in rows if row["repeat"] == repeat], associations=associations, **runner.FLAGS)
            hashes[path] = f"repeat-{repeat}"
        normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, pre_sha256="pre", rows=rows,
            fresh_process_repeats=True, repeats=repeats, elapsed_seconds=1., **runner.FLAGS)
        files[runner.RUN / "inference.json"] = normal
        hashes[runner.RUN / "pre.json"] = "pre"
        hashes[runner.RUN / "inference.json"] = "normal"
        hashes[runner.previous.RUN / "inference.json"] = "old-normal"
        return normal, ({}, dict(rows=old_rows), old_graphs), files, hashes

    def test_all_180_normals_have_exact_original_pair(self):
        normal, parent, files, hashes = self.fake_normal()
        with patch.object(runner, "parent_normal", return_value=parent), \
                patch.object(runner, "read", side_effect=lambda p: files[p]), \
                patch.object(runner, "digest", side_effect=lambda p: hashes[p]):
            observed, graphs = runner.verified_normal(self.config)
        self.assertIs(observed, normal)
        self.assertEqual(len(graphs), 180)

    def test_seen_development_cannot_be_relabelled_blind_at_any_normal_level(self):
        for level in ("inference", "repeat", "graph"):
            for key, value in (("development_prior_scores_seen", False), ("blind_evaluation", True)):
                normal, parent, files, hashes = self.fake_normal()
                selected = {"inference": normal, "repeat": files[runner.RUN / "repeat-1.json"],
                            "graph": files[runner.RUN / normal["rows"][-1]["path"]]}[level]
                selected[key] = value
                with self.subTest(level=level, key=key), \
                        patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                        patch.object(runner, "parent_normal", return_value=parent), \
                        patch.object(runner, "read", side_effect=lambda p: files[p]), \
                        patch.object(runner, "digest", side_effect=lambda p: hashes[p]), \
                        patch.object(runner, "physical_summary") as score, self.assertRaises(ValueError):
                    runner.evaluation_payload()
                score.assert_not_called()

    def test_seen_development_is_checked_in_pre_and_policy_freeze(self):
        for level in ("pre", "freeze"):
            for key, value in (("development_prior_scores_seen", False), ("blind_evaluation", True)):
                before = dict(run_id=runner.RUN_ID, gt_read=False, inference_existed=False,
                    runtime={}, hashes={}, **runner.FLAGS)
                frozen = dict(config=self.config, reader_defaults=runner.reader.DEFAULTS,
                    expected_normal_rows=180, **runner.FLAGS)
                {"pre": before, "freeze": frozen}[level][key] = value
                files = {runner.RUN / "pre.json": before, runner.RUN / "evaluation-freeze.json": frozen}
                with self.subTest(level=level, key=key), \
                        patch.object(runner, "validated_config", return_value=self.config), \
                        patch.object(runner.analytic, "runtime", return_value={}), \
                        patch.object(runner, "read", side_effect=lambda p: files[p]), self.assertRaises(ValueError):
                    runner.checked()

    def test_count_native_and_call_changes_are_never_reader_only(self):
        normal, parent, files, _ = self.fake_normal()
        graph = files[runner.RUN / normal["rows"][0]["path"]]
        old = parent[2][runner.row_key(normal["rows"][0])[:-1]]
        for key, value in (("full_input_point_count", 9), ("supported_base_point_count", 5),
                           ("full_input_segment_count", 1), ("base_vote_histogram", [10]),
                           ("native_segments", [[[0, 0, 0], [1, 0, 0]]]), ("call_policy", {})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "pairing changed"):
                runner.validate_pair({**graph, key: value}, old)

    def test_wrong_effective_policy_rejected(self):
        normal, parent, files, _ = self.fake_normal()
        graph = files[runner.RUN / normal["rows"][0]["path"]]
        with self.assertRaisesRegex(ValueError, "effective policy"):
            runner.validate_pair({**graph, "effective_policy": {}}, parent[2][runner.row_key(normal["rows"][0])[:-1]])

    def test_missing_last_row_stops_before_parent_verification_or_truth(self):
        normal, _, _, _ = self.fake_normal()
        normal["rows"].pop()
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "read", return_value=normal) as read, \
                patch.object(runner, "digest", return_value="pre"), \
                patch.object(runner, "parent_normal") as parent, self.assertRaisesRegex(ValueError, "All 180"):
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
                self.assertEqual(len(loaded), 180)
            if path.parent.name.startswith("records-r") and not path.name.endswith("-association.json"):
                loaded.append(path)
            return files[path]
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "parent_normal", return_value=parent), \
                patch.object(runner, "read", side_effect=read), \
                patch.object(runner, "digest", side_effect=lambda p: hashes[p]), \
                patch.object(runner, "physical_summary", return_value={"mock": True}) as score:
            result = runner.evaluation_payload()
        self.assertEqual(score.call_count, 719)
        self.assertEqual(len(result["rows"]), 180)
        self.assertEqual(len(result["base_rows"]), 36)
        self.assertIsNone(result["rows"][0]["physical"])
        self.assertEqual(result["native_unchanged_rows"], 180)
        self.assertEqual(result["physical_repeat_equal"], 89)
        self.assertFalse(result["reader_qualified"])
        self.assertFalse(result["g1_passed"])
        self.assertTrue(result["development_prior_scores_seen"])
        self.assertFalse(result["blind_evaluation"])

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
            for name in ("inference-started.json", "inference.json"):
                record = runner.read(runner.RUN / name)
                self.assertTrue(record["development_prior_scores_seen"])
                self.assertFalse(record["blind_evaluation"])
            with self.assertRaises(FileExistsError):
                runner.infer()



    def test_association_receipt_context_sha_and_missing_case_block_truth(self):
        for kind in ("receipt", "content", "graph", "missing"):
            normal, parent, files, hashes = self.fake_normal()
            association = runner.RUN / "records-r1/r03-anchored_chain_axis-association.json"
            if kind == "receipt":
                hashes[association] = "tampered"
            elif kind == "content":
                files[association] = {"changed": True}
            elif kind == "graph":
                files[runner.RUN / normal["rows"][-1]["path"]]["chain_context_sha256"] = "wrong"
            else:
                files[runner.RUN / "repeat-1.json"]["associations"].pop()
            with self.subTest(kind=kind), patch.object(runner, "checked", return_value=({}, self.config)), \
                    patch.object(runner, "parent_normal", return_value=parent), \
                    patch.object(runner, "read", side_effect=lambda path: files[path]), \
                    patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                    patch.object(runner, "physical_summary") as score, self.assertRaises(ValueError):
                runner.evaluation_payload()
            score.assert_not_called()

    def test_flags_cannot_claim_reader_only_or_different_algorithm(self):
        for level in ("inference", "repeat", "graph"):
            for key, value in (("reader_only_intervention", True), ("reader_algorithm_unchanged", False),
                               ("input_support_intervention", False)):
                normal, parent, files, hashes = self.fake_normal()
                record = {"inference": normal, "repeat": files[runner.RUN / "repeat-1.json"],
                    "graph": files[runner.RUN / normal["rows"][-1]["path"]]}[level]
                record[key] = value
                with self.subTest(level=level, key=key), patch.object(runner, "checked", return_value=({}, self.config)), \
                        patch.object(runner, "parent_normal", return_value=parent), \
                        patch.object(runner, "read", side_effect=lambda path: files[path]), \
                        patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                        patch.object(runner, "physical_summary") as score, self.assertRaises(ValueError):
                    runner.evaluation_payload()
                score.assert_not_called()



class AnchorInputContracts(unittest.TestCase):
    def manifest(self):
        return dict(annotation_source="codex_visual_rgb_not_human", cases=[
            dict(case_id=cid, anchors=[dict(view_id="a"), dict(view_id="b")], observed_from_rgb_only=True)
            for cid in ("r01", "r02", "r03", "chair01", "aframe01")])

    def test_anchor_source_cost_and_same_arm_contract_retained(self):
        with patch.object(runner, "read", return_value=self.manifest()), patch.object(runner, "digest", return_value="sha"):
            anchors, meta = runner.anchor_input("r01")
            self.assertEqual(len(anchors), 2)
            self.assertEqual(meta["anchor_input_metadata"]["anchor_view_count"], 2)
            self.assertTrue(meta["anchor_input_metadata"]["anchors_shared_across_all_method_arms"])
            self.assertEqual(meta["anchor_input_metadata"]["manifest"]["annotation_source"], "codex_visual_rgb_not_human")
            runner.validate_anchor_input(dict(case_id="r01", **meta))
            with self.assertRaises(ValueError):
                runner.validate_anchor_input(dict(case_id="r01", **{**meta, "anchor_input_sha256": "changed"}))

    def test_missing_case_or_single_anchor_not_silently_accepted(self):
        for mode in ("case", "anchor"):
            manifest = self.manifest()
            if mode == "case":
                manifest["cases"].pop()
            else:
                manifest["cases"][0]["anchors"].pop()
            with self.subTest(mode=mode), patch.object(runner, "read", return_value=manifest), self.assertRaises(ValueError):
                runner.anchor_input("r01")

    def test_exact_old_control_projection_and_missing_old_control(self):
        rows = [dict(repeat=0, case_id="r01", fraction=.0025, variant="base", reader=name)
                for name in ("single_axis", "chain_axis", "anchored_chain_axis")]
        graphs = [dict(state="complete", segments=[], native_segments=[], reader_elapsed_seconds=1.) for _ in rows]
        old_normal = dict(rows=rows[:2])
        with patch.object(runner.control_run, "checked", return_value=({}, {})), \
                patch.object(runner.control_run, "verified_normal", return_value=(old_normal, graphs[:2])):
            runner.verify_old_controls(rows, graphs)
            changed = [dict(graphs[0], segments=[[[0, 0, 0], [0, 0, 1]]]), *graphs[1:]]
            with self.assertRaisesRegex(ValueError, "control changed"):
                runner.verify_old_controls(rows, changed)
            with self.assertRaisesRegex(ValueError, "Every October"):
                runner.verify_old_controls(rows[:1], graphs[:1])

    def test_new_input_core_and_assistant_are_explicit_freeze_sources(self):
        with patch.object(runner.previous_sources, "source_inventory", return_value=[]):
            names = runner.source_inventory()
        for name in ("experiments/src/creator_eval/rgb_anchored_chain_support.py",
                     "tests/test_rgb_anchored_chain_support.py", "tests/test_rgb_anchor_states_deepseek.py",
                     "data/inputs/rgb-foreground-anchors-v1-20261009/manifest.json"):
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()
