"""Mock-only contracts for 80 complete paired object readouts and separate native scores."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_g1_object_rgb_chain_readout as runner  # noqa: E402
from creator_eval.rgb_candidate_readout import support_views  # noqa: E402
from test_fixture_candidate_readout import fixture  # noqa: E402


class ObjectRGBChainReadoutContracts(unittest.TestCase):
    def setUp(self):
        self.config = runner.validated_config()

    def test_exact_eighty_rows_two_objects_five_arms_two_readers_two_scales_two_repeats(self):
        rows = runner.inventory(self.config)
        self.assertEqual(len(rows), 80)
        self.assertEqual(len(set(rows)), 80)
        self.assertEqual(rows[0], (0, "chair01", .0025, "base", "single_axis"))
        self.assertEqual(rows[-1], (1, "aframe01", .005, "cylinder_support", "chain_axis"))
        self.assertEqual(rows[:40], runner.inventory(self.config, 0))
        with self.assertRaises(ValueError):
            runner.inventory(self.config, 2)

    def test_frozen_counts_scales_defaults_and_no_overrides(self):
        for key, value in (("expected_normal_rows", 40), ("readers", ["single_axis"]), ("reader_overrides", {"min": 1}),
                           ("repeats", [0]), ("voxel_camera_span_fractions", [.005]), ("parent_output_receipts", 61),
                           ("development_prior_scores_seen", False), ("blind_evaluation", True)):
            with self.subTest(key=key), patch.object(runner, "read", return_value={**self.config, key: value}), \
                    self.assertRaises(ValueError):
                runner.validated_config()

    def test_source_closure_inherits_prior_and_explicit_single_and_context(self):
        prefix = runner.parent.RUN.relative_to(ROOT).as_posix() + "/source_snapshot/"
        before = dict(source_count=1, hashes={prefix+"point.py": "point", "model.data": "model"})
        with patch.object(runner.previous_sources, "source_inventory", return_value=["point.py", "metric.py"]):
            sources = runner.source_inventory(before)
        self.assertEqual(sources, sorted(set(sources)))
        self.assertTrue({"point.py", "metric.py", "scripts/run_g1_object_rgb_chain_readout.py", "tests/test_g1_object_rgb_chain_readout.py",
            "experiments/src/creator_eval/single_axis_readout.py", "experiments/src/creator_eval/rgb_single_axis_readout.py",
            "tests/test_single_axis_readout.py", "tests/test_single_axis_readout_deepseek.py",
            "experiments/src/creator_eval/rgb_chain_support.py", "experiments/src/creator_eval/rgb_chain_axis_readout.py",
            "tests/test_rgb_chain_support.py", "tests/test_rgb_chain_support_deepseek.py",
            "scripts/run_fixture_rgb_chain_readout.py", "configs/fixture_rgb_chain_readout_v1.json",
            "tests/test_fixture_rgb_chain_readout.py"}.issubset(sources))
        self.assertNotIn("model.data", sources)


    def test_actual_adapter_subset_preserves_raw_votes_sampling_and_tls_defaults(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        views = support_views(frames, cameras)
        points = np.array([[-.25, 0, 5], [.25, 0, 5], [0, 0, 5], [0, 0, -5]])
        lines = np.array([[[-.25, -.5, 5.], [-.25, .5, 5.]]])
        association = dict(fallback_to_raw_union=False, assignments=[])
        context = dict(views=views, association=association, sha256=runner.canonical_hash(association),
            row_members=[np.zeros((*view["left"].shape, 0), bool) for view in views])
        old, new = runner.SingleAxisReadout(points, views), runner.ChainAxisReadout(points, context)
        runner.validate_adapters(old, new)
        first, second = [adapter(lines, dict(voxel_size=.1, origin=[0., 0., 0.])) for adapter in (old, new)]
        self.assertEqual(second["supported_base_point_count"], 0)
        self.assertEqual(second["supported_curve_sample_count"], 0)
        self.assertGreater(first["supported_base_point_count"], 0)
        runner.validate_support(first, second)
        self.assertEqual(first["effective_policy"], second["effective_policy"])

    def test_actual_mask_expansion_or_raw_vote_change_rejected(self):
        from types import SimpleNamespace
        raw = SimpleNamespace(mask=np.array([True, False]), votes=np.array([3, 0]))
        for mask, votes in (([True, True], [3, 0]), ([True, False], [3, 1])):
            chain = SimpleNamespace(mask=np.array(mask), votes=np.array(votes),
                raw_votes=np.array(votes), points=np.zeros((sum(mask), 3)))
            with self.subTest(mask=mask, votes=votes), self.assertRaises(ValueError):
                runner.validate_adapters(raw, chain)

    def test_pair_allows_support_reduction_not_full_or_native_change(self):
        first = dict(full_input_point_count=10, supported_base_point_count=6, full_input_segment_count=0,
            full_curve_sample_count=0, supported_curve_sample_count=0, base_vote_histogram=[4, 0, 0, 6],
            curve_vote_histogram=[0]*4, call_policy={}, native_segments=[], native_source_state="rejected", native_patch_id="patch")
        second = dict(first, supported_base_point_count=4, raw_base_vote_histogram=[4, 0, 0, 6],
            raw_curve_vote_histogram=[0]*4, base_chain_mask_subset_raw=True, curve_chain_mask_subset_raw=True,
            target_identity_confirmed=False)
        runner.validate_pair(first, second)
        for key in (*runner.FULL_KEYS, "native_patch_id", "native_segments"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_pair(first, {**second, key: "changed"})
        for key, value in (("supported_base_point_count", 7), ("base_chain_mask_subset_raw", False),
                           ("raw_base_vote_histogram", [10]), ("target_identity_confirmed", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_pair(first, {**second, key: value})

    def fake_normal(self):
        files, hashes, rows, objects, cases = {}, {}, [], [], []
        for cid in self.config["case_ids"]:
            source_frames = [dict(view_id=f"view_{i:02d}", rgb_sha256="a"*64) for i in range(5)]
            cases.append(dict(case_id=cid, frames=source_frames))
            bound = dict(case_id=cid, snapshot_id=cid+"-snapshot", full_point_count=10,
                supported_base_point_count=5, support_sha256=cid+"-support", camera_span_m=1., patches=[])
            for variant in self.config["variants"][1:]:
                rejected = cid == "aframe01" and variant == "cylinder_support"
                segments = [] if rejected else [[[0., 0., 0.], [0., 0., 1.]]]
                patch_id = cid+"-"+variant
                bound["patches"].append(dict(method=variant, native_segments=segments,
                    source_state="rejected" if rejected else "accepted", patch_id=patch_id,
                    outcome="no_supported_change" if rejected else "accepted_change"))
                files[runner.parent.RUN / cid / f"bundle/patch-{variant}/manifest.json"] = dict(
                    content_id=patch_id, metadata=dict(unresolved=["mock_rejection"] if rejected else []))
            objects.append(bound)
        indexed = {row["case_id"]: row for row in objects}
        with patch.object(runner, "read", side_effect=lambda path: files[path]):
            for index, (repeat, cid, fraction, variant, name) in enumerate(runner.inventory(self.config)):
                row = dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant, reader=name,
                    path=f"records-r{repeat}/{cid}-{fraction}-{variant}-{name}.json", sha256=str(index))
                bound = indexed[cid]
                native = runner.native_metadata(bound, variant)
                call = dict(voxel_size=fraction, origin=[0., 0., 0.])
                graph = dict(**{key: row[key] for key in runner.KEYS}, gt_read=False, state="complete",
                    resolution_state="unresolved", reason=None, segments=[], call_policy=call,
                    effective_policy={**runner.defaults()[name], **call}, base_snapshot_id=bound["snapshot_id"],
                    source_support_sha256=bound["support_sha256"], native_state="complete",
                    native_metric_scope="added_segments_only_not_raw_point_quality", **native,
                    full_input_point_count=10, supported_base_point_count=5, full_input_segment_count=len(native["native_segments"]),
                    full_curve_sample_count=20 if native["native_segments"] else 0,
                    supported_curve_sample_count=10 if native["native_segments"] else 0,
                    base_vote_histogram=[5, 0, 0, 5], curve_vote_histogram=[10, 0, 0, 10],
                    raw_row_states=[], reader_elapsed_seconds=.1, **runner.FLAGS)
                if name == "chain_axis":
                    graph.update(supported_base_point_count=3, raw_base_vote_histogram=[5, 0, 0, 5],
                        raw_curve_vote_histogram=[10, 0, 0, 10], base_chain_mask_subset_raw=True, curve_chain_mask_subset_raw=True,
                        chain_context_sha256=runner.canonical_hash({}), chain_association={}, target_identity_confirmed=False)
                files[runner.RUN / row["path"]], hashes[runner.RUN / row["path"]] = graph, row["sha256"]
                rows.append(row)
        repeats = []
        for repeat in self.config["repeats"]:
            path = runner.RUN / f"repeat-{repeat}.json"
            associations = []
            for cid in self.config["case_ids"]:
                assoc_path = runner.RUN / f"records-r{repeat}/{cid}-association.json"
                files[assoc_path], hashes[assoc_path] = {}, "association"
                associations.append(dict(case_id=cid, path=assoc_path.relative_to(runner.RUN).as_posix(),
                    sha256="association", chain_context_sha256=runner.canonical_hash({})))
            files[path] = dict(run_id=runner.RUN_ID, repeat=repeat, state="complete", gt_read=False,
                pre_sha256="pre", associations=associations, rows=[row for row in rows if row["repeat"] == repeat], **runner.FLAGS)
            hashes[path] = str(repeat)
            repeats.append(dict(repeat=repeat, path=path.name, sha256=str(repeat)))
        normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, pre_sha256="pre",
            rows=rows, repeats=repeats, fresh_reader_processes=2,
            end_to_end_repeat_performed=False, elapsed_seconds=1.,
            **runner.FLAGS)
        files[runner.RUN / "inference.json"], hashes[runner.RUN / "pre.json"] = normal, "pre"
        manifest, original = dict(cases=cases), dict(rows=objects, elapsed_seconds=2., outputs={})
        return normal, manifest, original, files, hashes

    def test_all_eighty_rows_and_original_native_identity_verified(self):
        normal, manifest, original, files, hashes = self.fake_normal()
        with patch.object(runner, "parent_normal", return_value=({}, manifest, original)), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]):
            output, graphs, _, _ = runner.verified_normal(self.config)
        self.assertIs(output, normal)
        self.assertEqual(len(graphs), 80)
        self.assertEqual(graphs[-1]["native_reason"], ["mock_rejection"])

    def test_missing_row_stops_before_parent_or_truth(self):
        normal, _, _, _, _ = self.fake_normal()
        normal["rows"].pop()
        with patch.object(runner, "checked", return_value=({}, self.config)), \
                patch.object(runner, "read", return_value=normal) as read, patch.object(runner, "digest", return_value="pre"), \
                patch.object(runner, "parent_normal") as parent, self.assertRaisesRegex(ValueError, "All 80"):
            runner.evaluation_payload()
        parent.assert_not_called()
        self.assertEqual(read.call_count, 1)

    def test_seen_development_cannot_be_relabeled_blind(self):
        for key, value in (("development_prior_scores_seen", False), ("blind_evaluation", True)):
            normal, _, _, _, _ = self.fake_normal()
            normal[key] = value
            with self.subTest(key=key), patch.object(runner, "read", return_value=normal), \
                    patch.object(runner, "digest", return_value="pre"), \
                    patch.object(runner, "parent_normal") as parent, self.assertRaisesRegex(ValueError, "All 80"):
                runner.verified_normal(self.config)
            parent.assert_not_called()

    def test_last_row_hash_failure_precedes_gt(self):
        normal, manifest, original, files, hashes = self.fake_normal()
        hashes[runner.RUN / normal["rows"][-1]["path"]] = "tampered"
        with patch.object(runner, "checked", return_value=({}, self.config)), \
                patch.object(runner, "parent_normal", return_value=({}, manifest, original)), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                patch.object(runner, "physical_summary") as score, self.assertRaisesRegex(ValueError, "graph changed"):
            runner.evaluation_payload()
        score.assert_not_called()

    def test_normal_native_geometry_cannot_be_replaced_by_readout_clipping(self):
        normal, manifest, original, files, hashes = self.fake_normal()
        files[runner.RUN / normal["rows"][2]["path"]]["native_segments"] = []
        with patch.object(runner, "parent_normal", return_value=({}, manifest, original)), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                self.assertRaisesRegex(ValueError, "native identity"):
            runner.verified_normal(self.config)

    def test_complete_normals_precede_gt_common_errors_do_not_hide_native(self):
        normal, manifest, original, files, hashes = self.fake_normal()
        files[runner.RUN / normal["rows"][0]["path"]]["state"] = "error"
        scene = ROOT / ".runtime/experiments" / self.config["scene_run_id"]
        truth_path = ROOT / "data/eval_gt" / self.config["scene_run_id"] / "manifest.json"
        files[scene / "prepared.json"] = dict(truth_sha256="truth")
        files[truth_path] = dict(input_sha256="input", cases=[dict(case_id=case["case_id"], frames=case["frames"],
            declared=dict(target=dict(present=False))) for case in manifest["cases"]])
        hashes[truth_path], hashes[ROOT / "data/inputs" / self.config["scene_run_id"] / "manifest.json"] = "truth", "input"
        hashes[runner.RUN / "inference.json"] = "normal"
        loaded = []
        def read(path):
            if path == truth_path:
                self.assertEqual(len(loaded), 80)
            if path.parent.name.startswith("records-r") and not path.name.endswith("-association.json"):
                loaded.append(path)
            return files[path]
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "parent_normal", return_value=({}, manifest, original)), \
                patch.object(runner, "read", side_effect=read), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                patch.object(runner, "physical_summary", return_value={"mock": True}) as score:
            result = runner.evaluation_payload()
        self.assertEqual(score.call_count, 89)
        self.assertEqual(len(result["rows"]), 80)
        self.assertEqual(len(result["native_rows"]), 10)
        self.assertIsNone(result["rows"][0]["physical"])
        self.assertEqual(result["native_rows"][0]["physical"], {"mock": True})
        self.assertTrue(all(not row["is_point_quality_measurement"] for row in result["native_rows"]))
        self.assertEqual(result["native_rows"][-1]["reasons"], ["mock_rejection"])
        self.assertEqual(result["reader_repeat_equal"], 39)
        self.assertEqual(result["parent_integration"], original["rows"])
        self.assertFalse(result["g1_passed"])
        self.assertFalse(result["end_to_end_repeat_performed"])
        self.assertTrue(result["development_prior_scores_seen"])
        self.assertFalse(result["blind_evaluation"])

    def test_repeat_projection_ignores_timing_not_physical_output(self):
        graph = dict(state="complete", segments=[], native_segments=[], reader_elapsed_seconds=.1)
        other = {**graph, "reader_elapsed_seconds": 100.}
        self.assertEqual(runner.repeat_projection(graph), runner.repeat_projection(other))
        other["segments"] = [[[0, 0, 0], [0, 0, 1]]]
        self.assertNotEqual(runner.repeat_projection(graph), runner.repeat_projection(other))

    def test_two_fresh_reader_only_processes_and_preserve_prior_attempt(self):
        with tempfile.TemporaryDirectory() as name, patch.object(runner, "RUN", Path(name)), \
                patch.object(runner.sys, "addaudithook"), patch.object(runner, "checked", return_value=({}, self.config)), \
                patch.object(runner, "repeat_normal", return_value=dict(rows=[], elapsed_seconds=1.)), \
                patch.object(runner, "digest", return_value="sha"), patch.object(runner, "verified_normal"), \
                patch.object(runner.subprocess, "run") as process:
            runner.infer()
            self.assertEqual(process.call_count, 2)
            for repeat, call in enumerate(process.call_args_list):
                self.assertEqual(call.args[0][-2:], ["infer-repeat", str(repeat)])
                self.assertNotIn("depth", call.args[0])
                self.assertTrue(call.kwargs["check"])
            with self.assertRaises(FileExistsError):
                runner.infer()

    def test_normal_guard_and_exclusive_attempts(self):
        for path in ("data/eval_gt/new/manifest.json", "docs/experiments/results/new.json", "configs/g1_object_pilot_v1.json"):
            with self.subTest(path=path), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(ROOT / path), "r", 32896))
        with tempfile.TemporaryDirectory() as name, patch.object(runner, "RUN", Path(name)), \
                patch.object(runner.sys, "addaudithook"), self.assertRaises(FileExistsError):
            runner.prepare()

    def test_association_receipt_and_context_identity_block_truth_on_tamper(self):
        for kind in ("receipt", "content", "graph"):
            normal, manifest, original, files, hashes = self.fake_normal()
            association = runner.RUN / "records-r1/aframe01-association.json"
            if kind == "receipt":
                hashes[association] = "tampered"
            elif kind == "content":
                files[association] = {"changed": True}
            else:
                files[runner.RUN / normal["rows"][-1]["path"]]["chain_context_sha256"] = "wrong"
            with self.subTest(kind=kind), patch.object(runner, "checked", return_value=({}, self.config)), \
                    patch.object(runner, "parent_normal", return_value=({}, manifest, original)), \
                    patch.object(runner, "read", side_effect=lambda path: files[path]), \
                    patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                    patch.object(runner, "physical_summary") as score, self.assertRaises(ValueError):
                runner.evaluation_payload()
            score.assert_not_called()

    def test_flags_cannot_claim_reader_only_or_blind_at_any_normal_level(self):
        for level in ("inference", "repeat", "graph"):
            for key, value in (("reader_only_intervention", True), ("reader_algorithm_unchanged", False),
                               ("input_support_intervention", False), ("blind_evaluation", True)):
                normal, manifest, original, files, hashes = self.fake_normal()
                record = {"inference": normal, "repeat": files[runner.RUN / "repeat-1.json"],
                    "graph": files[runner.RUN / normal["rows"][-1]["path"]]}[level]
                record[key] = value
                with self.subTest(level=level, key=key), patch.object(runner, "checked", return_value=({}, self.config)), \
                        patch.object(runner, "parent_normal", return_value=({}, manifest, original)), \
                        patch.object(runner, "read", side_effect=lambda path: files[path]), \
                        patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                        patch.object(runner, "physical_summary") as score, self.assertRaises(ValueError):
                    runner.evaluation_payload()
                score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
