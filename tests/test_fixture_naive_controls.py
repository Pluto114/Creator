"""Mock fixtures only: naive-control freezes, paired rows and real patch lifecycle."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/src"))
import run_fixture_naive_controls as runner  # noqa: E402
from creator_recon.domain.point_patch import write_snapshot  # noqa: E402


class FixtureNaiveContracts(unittest.TestCase):
    def setUp(self):
        self.config = runner.validated_config()

    def test_exact_inventory_two_process_repeats_five_variants_two_scales(self):
        rows = runner.inventory(self.config)
        self.assertEqual(len(rows), 60)
        self.assertEqual(len(set(rows)), 60)
        self.assertEqual(rows[0], (0, "r01", .0025, "base"))
        self.assertEqual(rows[-1], (1, "r03", .005, "cylinder_support"))
        self.assertEqual(rows[:30], runner.inventory(self.config, 0))
        with self.assertRaises(ValueError):
            runner.inventory(self.config, 2)

    def test_config_prevents_readout_or_naive_tuning(self):
        for key, value in (("repeats", [0]), ("naive_defaults", {}), ("reader_overrides", {"angle": 1}),
                           ("support_policy", {"minimum_views": 2}), ("expected_normal_rows", 59)):
            with self.subTest(key=key), patch.object(runner, "read", return_value={**self.config, key: value}), \
                    self.assertRaises(ValueError):
                runner.validated_config()

    def test_closed_source_inventory_includes_new_algorithm_and_contracts(self):
        with patch.object(runner.previous, "source_inventory", return_value=["old.py"]):
            result = runner.source_inventory()
        self.assertEqual(result, sorted(set(result)))
        self.assertTrue({"old.py", "scripts/run_fixture_naive_controls.py", "tests/test_fixture_naive_controls.py",
            "tests/test_fixture_naive_fit.py", "tests/test_fixture_naive_fit_deepseek.py",
            "experiments/src/creator_eval/fixture_naive_controls.py",
            "experiments/src/creator_eval/line_controls.py"}.issubset(result))

    def test_normal_guard_blocks_truth_and_scores(self):
        for name in ("data/eval_gt/x.json", "data/evaluation/x.json", "docs/experiments/results/x.json",
                     ".runtime/experiments/x/evaluation.json", ".runtime/experiments/x/post.json"):
            with self.subTest(name=name), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(ROOT / name), "r"))

    def test_exclusive_writes_and_existing_attempt_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with patch.object(runner, "RUN", directory), patch.object(runner.sys, "addaudithook"), \
                    self.assertRaises(FileExistsError):
                runner.prepare()
            path = directory / "normal.json"
            runner.write(path, {"first": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write(path, {"first": False})
            self.assertEqual(before, path.read_bytes())

    def test_json_rejects_nonfinite_payloads(self):
        with tempfile.TemporaryDirectory() as name, self.assertRaises(ValueError):
            runner.write(Path(name) / "invalid.json", {"x": float("nan")})

    def test_real_patch_saved_reopened_withdrawn_and_base_unchanged(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = root / "bundle"
            points = np.array([[0., 0., 1.], [0., 0., 2.], [0., 0., 3.]])
            ids = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.uint32)
            frames = [dict(frame_id=f"v{i}", image_sha256="a"*64, prediction_size_wh=[1, 1]) for i in range(3)]
            write_snapshot(bundle / "base", points, ids, frames=frames, world_frame_id="mock", length_unit="meter",
                source_prediction_sha256="b"*64, point_policy="complete mocked snapshot")
            base = runner.compose(bundle / "base")
            fits = runner.controls.fit_controls(points, ids, 1., np.ones(3, bool))
            def receipt(path, output):
                output[path.relative_to(root).as_posix()] = runner.digest(path)
            with patch.object(runner, "ROOT", root), patch.object(runner, "receipt", side_effect=receipt):
                for fit in fits:
                    outputs = {}
                    candidate, metadata = runner.persist_control(bundle, fit, 0, base, outputs)
                    np.testing.assert_array_equal(candidate["points"], points)
                    np.testing.assert_array_equal(candidate["point_ids"], ids)
                    np.testing.assert_array_equal(candidate["segments"], fit["segments"])
                    self.assertEqual(len(outputs), 5)
                    self.assertGreater(metadata["added_storage_bytes"], 0)
                    self.assertTrue(metadata["withdrawal_verified"])
                    self.assertEqual(len(runner.open_candidate_view(root / metadata["withdrawn_path"])["segments"]), 0)

    def test_repeat_projection_ignores_time_not_geometry(self):
        first = dict(state="complete", native_segments=[[[0, 0, 0], [1, 0, 0]]], segments=[],
                     reader_elapsed_seconds=1., repeat=0, control=None)
        second = {**first, "reader_elapsed_seconds": 42., "repeat": 1}
        self.assertEqual(runner.physical_repeat_projection(first), runner.physical_repeat_projection(second))
        second["native_segments"] = []
        self.assertNotEqual(runner.physical_repeat_projection(first), runner.physical_repeat_projection(second))

    def fake_normal(self):
        rows, references, files, hashes = [], {}, {}, {}
        call = dict(voxel_size=.02, origin=[0., 0., 0.])
        outputs = {}
        for index, (repeat, cid, fraction, variant) in enumerate(runner.inventory(self.config)):
            old_variant = variant if variant not in runner.controls.METHODS else "base"
            old_name = f"{cid}-{fraction}-{old_variant}"
            old_row = dict(case_id=cid, fraction=fraction, variant=old_variant,
                           path=f"records/{old_name}.json", sha256=old_name)
            references[(cid, fraction, old_variant)] = old_row
            row = dict(repeat=repeat, case_id=cid, fraction=fraction, variant=variant, path=f"records-r{repeat}/{index}.json",
                       sha256=f"new-{index}", previous_path=old_row["path"], previous_sha256=old_row["sha256"])
            control = None
            if variant in runner.controls.METHODS:
                fit_name = f"records-r{repeat}/{cid}-{variant}-fit.json"
                fit_path = runner.RUN / fit_name
                files[fit_path] = dict(gt_read=False, method=variant, repeat=repeat, case_id=cid, segments=[],
                                      model={"point_count": 1}, input_support_sha256="support",
                                      config=runner.controls.DEFAULTS, readout_scale_used=False)
                hashes[fit_path] = fit_name
                outputs[fit_path.relative_to(ROOT).as_posix()] = fit_name
                control = dict(fit_record_path=fit_name, fit_record_sha256=fit_name, input_support_sha256="support",
                               outcome="no_supported_change", base_unchanged=True,
                               withdrawal_verified=True, saved_reopen_verified=True)
            files[runner.RUN / row["path"]] = dict(gt_read=False, repeat=repeat, case_id=cid, fraction=fraction,
                variant=variant, state="complete", resolution_state="unresolved", segments=[], native_segments=[],
                native_state="complete", native_metric_scope="added_segments_only_not_raw_point_quality", control=control,
                reader_elapsed_seconds=0., call_policy=call, effective_policy={**runner.reader.DEFAULTS, **call},
                full_input_point_count=1, supported_base_point_count=1)
            files[runner.previous.RUN / old_row["path"]] = dict(call_policy=call, state="complete", segments=[])
            hashes[runner.RUN / row["path"]] = row["sha256"]
            hashes[runner.previous.RUN / old_row["path"]] = old_row["sha256"]
            rows.append(row)
        normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, pre_sha256="pre", rows=rows,
                      outputs=outputs, elapsed_seconds=0., fresh_process_repeats=True, independent_objects=False, repeats=[])
        files[runner.RUN / "inference.json"] = normal
        hashes[runner.RUN / "pre.json"] = "pre"
        for repeat in self.config["repeats"]:
            files[runner.RUN / f"repeat-{repeat}.json"] = dict(run_id=runner.RUN_ID, repeat=repeat,
                state="complete", gt_read=False, pre_sha256="pre", rows=[row for row in rows if row["repeat"] == repeat], outputs={})
        with patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]):
            normal["fit_repeat_comparisons"] = runner.fit_repeat_comparisons(normal, self.config)
        return normal, references, files, hashes

    def test_all_sixty_rows_verified(self):
        normal, references, files, hashes = self.fake_normal()
        with patch.object(runner, "comparison_rows", return_value=references), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]):
            result, graphs = runner.verified_normal(self.config)
        self.assertIs(result, normal)
        self.assertEqual(len(graphs), 60)

    def test_missing_last_row_prevents_graphs_and_truth(self):
        normal, _, _, _ = self.fake_normal()
        normal["rows"].pop()
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "read", return_value=normal) as read, \
                patch.object(runner, "digest", return_value="pre"), self.assertRaisesRegex(ValueError, "All 60"):
            runner.evaluation_payload()
        self.assertEqual(read.call_count, 1)

    def test_last_hash_and_fit_tamper_rejected_before_truth(self):
        for mode in ("last", "fit"):
            normal, references, files, hashes = self.fake_normal()
            path = runner.RUN / normal["rows"][-1]["path"] if mode == "last" else ROOT / next(iter(normal["outputs"]))
            hashes[path] = "tampered"
            with self.subTest(mode=mode), patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                    patch.object(runner, "comparison_rows", return_value=references), \
                    patch.object(runner, "read", side_effect=lambda path: files[path]), \
                    patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                    patch.object(runner, "physical_summary") as score, self.assertRaises(ValueError):
                runner.evaluation_payload()
            score.assert_not_called()

    def test_fit_cannot_use_readout_scale(self):
        _, references, files, hashes = self.fake_normal()
        fit = next(value for path, value in files.items() if path.name.endswith("-fit.json"))
        fit["readout_scale_used"] = True
        with patch.object(runner, "comparison_rows", return_value=references), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                self.assertRaisesRegex(ValueError, "Scale-independent"):
            runner.verified_normal(self.config)

    def test_repeat_row_reference_cannot_be_repaired(self):
        _, references, files, hashes = self.fake_normal()
        files[runner.RUN / "repeat-1.json"]["rows"] = copy.deepcopy(files[runner.RUN / "repeat-1.json"]["rows"])
        files[runner.RUN / "repeat-1.json"]["rows"][-1]["sha256"] = "different"
        with patch.object(runner, "comparison_rows", return_value=references), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                self.assertRaisesRegex(ValueError, "row references"):
            runner.verified_normal(self.config)

    def test_complete_normals_precede_truth_and_all_variants_get_both_scores(self):
        normal, references, files, hashes = self.fake_normal()
        files[runner.RUN / normal["rows"][0]["path"]]["state"] = "error"
        truth_path = ROOT / "data/eval_gt/mock/manifest.json"
        files[runner.parent.CONFIG] = dict(scene_run_id="mock")
        files[ROOT / ".runtime/experiments/mock/prepared.json"] = dict(truth_sha256="truth")
        files[truth_path] = dict(cases=[dict(case_id=cid, declared=dict(target=dict(present=False))) for cid in self.config["case_ids"]])
        files[runner.RUN / "evaluation-freeze.json"] = dict(physical_policy={})
        hashes[truth_path], hashes[runner.RUN / "inference.json"] = "truth", "normal"
        loaded = []
        def read(path):
            if path == truth_path:
                self.assertEqual(len(loaded), 120)
            if path.parent.name == "records" or (path.parent.name.startswith("records-r") and not path.name.endswith("-fit.json")):
                loaded.append(path)
            return files[path]
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "comparison_rows", return_value=references), \
                patch.object(runner, "read", side_effect=read), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                patch.object(runner, "physical_summary", return_value={"mock": True}) as score:
            result = runner.evaluation_payload()
        self.assertEqual(score.call_count, 179)
        self.assertEqual(len(result["rows"]), 60)
        self.assertIsNone(result["rows"][0]["physical"])
        self.assertEqual(result["rows"][0]["native_physical"], {"mock": True})
        self.assertFalse(result["g1_passed"])
        self.assertFalse(result["independent_objects"])
        self.assertEqual(result["physical_repeat_equal"], 29)
        self.assertEqual(result["fit_repeat_pairs"], 6)
        self.assertEqual(result["fit_repeat_equal"], 6)

    def test_fit_repeat_compares_model_without_timing_and_retains_mismatch(self):
        normal, _, files, hashes = self.fake_normal()
        path = runner.RUN / "records-r1/r03-depth_ransac_single_span-fit.json"
        files[path]["elapsed_seconds"] = 99.
        with patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]):
            result = runner.fit_repeat_comparisons(normal, self.config)
            self.assertTrue(all(pair["model_repeat_equal"] for pair in result))
            files[path]["model"] = {"point_count": 2}
            result = runner.fit_repeat_comparisons(normal, self.config)
        self.assertEqual(len(result), 6)
        self.assertEqual(sum(pair["model_repeat_equal"] for pair in result), 5)
        self.assertFalse(result[-1]["model_repeat_equal"])

    def test_infer_uses_two_fresh_processes_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as name, patch.object(runner, "RUN", Path(name)), \
                patch.object(runner.sys, "addaudithook"), \
                patch.object(runner, "checked", return_value=({}, self.config)), \
                patch.object(runner, "repeat_normal", side_effect=lambda config, repeat: dict(rows=[], outputs={}, elapsed_seconds=0.)), \
                patch.object(runner, "fit_repeat_comparisons", return_value=[]), \
                patch.object(runner, "receipt"), patch.object(runner, "digest", return_value="hash"), \
                patch.object(runner, "verified_normal"), patch.object(runner.subprocess, "run") as run:
            runner.infer()
            self.assertEqual(run.call_count, 2)
            for index, call in enumerate(run.call_args_list):
                self.assertEqual(call.args[0][-2:], ["infer-repeat", str(index)])
                self.assertTrue(call.kwargs["check"])
            with self.assertRaises(FileExistsError):
                runner.infer()


if __name__ == "__main__":
    unittest.main()
