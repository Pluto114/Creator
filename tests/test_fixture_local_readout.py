"""Mock-only fixture bridge contracts; never infer or score real fixtures."""

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
import run_fixture_local_readout as runner  # noqa: E402
from creator_eval import rgb_fixture_local_readout as adapter  # noqa: E402
from creator_eval.common_readout import sample_segments  # noqa: E402
from creator_eval.rgb_candidate_readout import (  # noqa: E402
    CandidateSupportedReadout,
    support_views,
)
from test_fixture_candidate_readout import fixture  # noqa: E402


class FixtureLocalContracts(unittest.TestCase):
    def setUp(self):
        self.config = runner.validated_config()

    def test_fixed_inventory_and_no_reader_overrides(self):
        items = runner.inventory(self.config)
        self.assertEqual(len(items), 18)
        self.assertEqual(len(set(items)), 18)
        self.assertEqual(items[0], ("r01", .0025, "base"))
        self.assertEqual(items[-1], ("r03", .005, "cylinder_support"))
        changed = {**self.config, "reader_overrides": {"minimum_validation_coverage": .1}}
        with patch.object(runner, "read", return_value=changed), self.assertRaises(ValueError):
            runner.validated_config()

    def test_transitive_new_source_inventory_is_complete(self):
        with patch.object(runner.analytic, "read", return_value={"source_files": ["experiments/src/creator_eval/ridge_proposals.py"]}):
            inventory = runner.source_inventory()
        required = {"experiments/src/creator_eval/common_readout_local.py",
            "experiments/src/creator_eval/section_local_evidence.py",
            "experiments/src/creator_eval/ridge_proposals.py",
            "experiments/src/creator_eval/rgb_candidate_readout.py",
            "experiments/src/creator_eval/rgb_fixture_local_readout.py",
            "tests/test_common_readout_local.py", "tests/test_section_local_evidence.py",
            "tests/test_local_integration.py", "tests/test_fixture_local_readout.py",
            "scripts/run_mixed_readout_local.py", "scripts/run_fixture_local_readout.py"}
        self.assertTrue(required.issubset(inventory))
        self.assertEqual(inventory, sorted(set(inventory)))

    def test_same_mask_and_sampling_with_arrays_only_reader_call(self):
        frames, cameras = fixture(offsets=(-5., 5.))
        views = support_views(frames, cameras)
        points = np.array([[-.25, 0., 5.], [.25, 0., 5.], [0., 0., 5.], [0., 0., -5.]])
        lines = np.array([[[-.25, -.5, 5.], [-.25, .5, 5.]]])
        old, new = CandidateSupportedReadout(points, views), adapter.FixtureLocalReadout(points, views)
        np.testing.assert_array_equal(new.mask, old.mask)
        call = dict(voxel_size=.1, origin=[0., 0., 0.])
        fake = dict(state="complete", resolution_state="resolved", segments=[], components=0)
        with patch.object(adapter.reader, "readout", return_value=fake) as readout:
            result = new(lines, call)
        args = readout.call_args.args
        self.assertEqual(len(args), 3)
        np.testing.assert_array_equal(args[0], np.concatenate((new.points, sample_segments(lines, .05, 2000000))))
        self.assertEqual(args[1].shape, (0, 2, 3))
        self.assertEqual(args[2], call)
        self.assertEqual(result["full_input_point_count"], len(points))
        self.assertEqual(result["supported_base_point_count"], 2)
        self.assertEqual(result["supported_curve_sample_count"], result["full_curve_sample_count"])
        self.assertEqual(result["base_vote_histogram"], [2, 0, 0, 0, 2])
        self.assertIn("not target identity", result["measurement_scope"])

    def test_no_measurement_or_legacy_policy_expansion(self):
        frames, cameras = fixture()
        views = support_views(frames, cameras)
        with self.assertRaises(ValueError):
            adapter.FixtureLocalReadout(np.empty((0, 3)), views, dict(minimum_views=2, horizontal_padding_px=1.))
        readout = adapter.FixtureLocalReadout(np.empty((0, 3)), views)
        with self.assertRaises(ValueError):
            readout(np.empty((0, 2, 3)), dict(voxel_size=.1, origin=[0., 0., 0.], split_maximum_axis_angle_degrees=12))

    def test_reader_failure_remains_error_not_empty_rejection(self):
        frames, cameras = fixture()
        readout = adapter.FixtureLocalReadout(np.array([[0., 0., 5.]]), support_views(frames, cameras))
        with patch.object(adapter.reader, "readout", side_effect=RuntimeError("mock failure")):
            result = readout(np.empty((0, 2, 3)), dict(voxel_size=.1, origin=[0., 0., 0.]))
        self.assertEqual(result["state"], "error")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["supported_base_point_count"], 1)
        self.assertIn("mock failure", result["reason"])

    def test_original_camera_span_and_only_two_transferred_fields(self):
        _, cameras = fixture()
        reference = dict(config=dict(voxel_size=1.5*.0025, origin=[0., 0., 0.], split_unknown_old_key=42))
        self.assertEqual(runner.call_policy(reference, cameras, .0025),
                         dict(voxel_size=.00375, origin=[0., 0., 0.]))
        for change in (dict(voxel_size=.01), dict(origin=[0., 1., 0.])):
            bad = dict(config={**reference["config"], **change})
            with self.assertRaises(ValueError):
                runner.call_policy(bad, cameras, .0025)

    def test_complete_base_identity_and_reversible_candidate_contract(self):
        base = dict(base_snapshot_id="snapshot", world_frame_id="fixture", length_unit="meter",
            points=np.array([[1., 2., 3.]]), point_ids=np.array([[0, 1, 2]], dtype=np.uint32),
            segments=np.empty((0, 2, 3)))
        candidate = copy.deepcopy(base)
        candidate["segments"] = np.array([[[0., 0., 0.], [1., 0., 0.]]])
        runner.validate_candidate(base, candidate)
        runner.validate_candidate(base, copy.deepcopy(base), withdrawn=True)
        with self.assertRaises(ValueError):
            runner.validate_candidate(base, candidate, withdrawn=True)
        for key, value in (("world_frame_id", "changed"), ("length_unit", "mm"),
                           ("points", base["points"]+.1), ("point_ids", base["point_ids"]+1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_candidate(base, {**candidate, key: value})

    def test_normal_guard_blocks_all_truth_and_score_locations(self):
        for path in ("data/eval_gt/x/manifest.json", "data/evaluation/x.json",
                     "docs/experiments/results/x.json", ".runtime/experiments/x/evaluation.json",
                     ".runtime/experiments/x/post.json"):
            with self.subTest(path=path), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(ROOT / path), "r"))
        runner.block_truth("open", (str(ROOT / "data/inputs/mock/points.npz"), "r"))

    def fake_normal(self):
        rows, references, files, hashes = [], {}, {}, {}
        call = dict(voxel_size=.02, origin=[0., 0., 0.])
        for index, (cid, fraction, variant) in enumerate(runner.inventory(self.config)):
            row = dict(case_id=cid, fraction=fraction, variant=variant, path=f"records/new-{index}.json",
                       sha256=f"new-{index}", previous_path=f"records/old-{index}.json", previous_sha256=f"old-{index}")
            old_row = dict(case_id=cid, fraction=fraction, variant=variant, path=row["previous_path"], sha256=row["previous_sha256"])
            graph = dict(gt_read=False, case_id=cid, fraction=fraction, variant=variant,
                state="complete", resolution_state="unresolved", segments=[], call_policy=call,
                effective_policy={**runner.reader.DEFAULTS, **call}, full_input_point_count=1,
                supported_base_point_count=1, full_curve_sample_count=0, supported_curve_sample_count=0,
                base_vote_histogram=[0, 0, 0, 1], curve_vote_histogram=[0, 0, 0, 0])
            files[runner.RUN / row["path"]] = graph
            files[runner.previous.RUN / old_row["path"]] = dict(config=call, state="complete", segments=[])
            hashes[runner.RUN / row["path"]], hashes[runner.previous.RUN / old_row["path"]] = row["sha256"], old_row["sha256"]
            rows.append(row)
            references[(cid, fraction, variant)] = old_row
        normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, pre_sha256="pre", rows=rows, elapsed_seconds=0.)
        files[runner.RUN / "inference.json"] = normal
        hashes[runner.RUN / "pre.json"] = "pre"
        return normal, references, files, hashes

    def test_last_hash_failure_happens_before_any_truth_access(self):
        normal, references, files, hashes = self.fake_normal()
        hashes[runner.RUN / normal["rows"][-1]["path"]] = "tampered"
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "comparison_rows", return_value=references), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                patch.object(runner, "physical_summary") as score:
            with self.assertRaisesRegex(ValueError, "graph changed"):
                runner.evaluation_payload()
            score.assert_not_called()

    def test_missing_row_fails_before_loading_graphs_or_truth(self):
        normal, _, _, _ = self.fake_normal()
        normal["rows"].pop()
        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "read", return_value=normal) as read, \
                patch.object(runner, "digest", return_value="pre"):
            with self.assertRaisesRegex(ValueError, "All 18"):
                runner.evaluation_payload()
            self.assertEqual(read.call_count, 1)

    def test_reference_repairing_is_rejected(self):
        normal, references, files, hashes = self.fake_normal()
        normal["rows"][-1]["previous_path"] = normal["rows"][0]["previous_path"]
        with patch.object(runner, "comparison_rows", return_value=references), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]):
            with self.assertRaisesRegex(ValueError, "identity differs"):
                runner.verified_normal(self.config)

    def test_all_graphs_verified_before_evaluation_and_errors_retained(self):
        normal, references, files, hashes = self.fake_normal()
        files[runner.RUN / normal["rows"][0]["path"]]["state"] = "error"
        truth_path = ROOT / "data/eval_gt/mock-scene/manifest.json"
        files[runner.parent.CONFIG] = dict(scene_run_id="mock-scene")
        files[ROOT / ".runtime/experiments/mock-scene/prepared.json"] = dict(truth_sha256="truth")
        files[truth_path] = dict(cases=[dict(case_id=cid, declared=dict(target=dict(present=False))) for cid in self.config["case_ids"]])
        files[runner.RUN / "evaluation-freeze.json"] = dict(physical_policy={})
        hashes[truth_path], hashes[runner.RUN / "inference.json"] = "truth", "normal"
        loaded = []

        def read(path):
            if path == truth_path:
                self.assertEqual(len(loaded), 36)
            if path.parent.name == "records":
                loaded.append(path)
            return files[path]

        with patch.object(runner, "checked", return_value=({"hashes": {}}, self.config)), \
                patch.object(runner, "comparison_rows", return_value=references), \
                patch.object(runner, "read", side_effect=read), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]), \
                patch.object(runner, "physical_summary", return_value={"mock_score": True}) as score:
            result = runner.evaluation_payload()
        self.assertEqual(len(result["rows"]), 18)
        self.assertEqual(result["rows"][0]["state"], "error")
        self.assertIsNone(result["rows"][0]["physical"])
        self.assertEqual(score.call_count, 35)
        self.assertFalse(result["g1_passed"])
        self.assertFalse(result["reader_qualified"])
        self.assertTrue(result["development_prior_scores_seen"])

    def test_analytic_prerequisite_uses_normal_only_and_freezes_all_736(self):
        rows, records = [], {}
        for i in range(736):
            key = str(i), "supported", "native"
            row = dict(input_id=key[0], reader=key[1], representation=key[2], path=f"records/{i}.json", sha256=str(i))
            rows.append(row)
            records[key] = dict(parent_reference=None if i >= 304 else dict(path=f".runtime/parent/{i}.json", sha256=f"old{i}"))
        normal = dict(rows=rows, fresh_inference_rows=432, frozen_reference_rows=304)
        with patch.object(runner.analytic, "normal_records", return_value=({"hashes": {}}, {}, {}, normal, records)) as verified, \
                patch.object(runner, "receipt") as receipt, \
                patch.object(runner.analytic, "evaluation_payload", side_effect=AssertionError("Scores forbidden")):
            runner.analytic_receipts(self.config)
        verified.assert_called_once_with()
        self.assertEqual(receipt.call_count, 2+736+304)
        normal["fresh_inference_rows"] = 431
        with patch.object(runner.analytic, "normal_records", return_value=({"hashes": {}}, {}, {}, normal, records)), \
                self.assertRaisesRegex(ValueError, "736-row"):
            runner.analytic_receipts(self.config)

    def test_prior_attempts_and_exclusive_writes_are_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            with patch.object(runner, "RUN", folder), patch.object(runner.sys, "addaudithook"), \
                    patch.object(runner, "validated_config") as config, self.assertRaises(FileExistsError):
                runner.prepare()
            config.assert_not_called()
            path = folder / "existing.json"
            runner.write(path, {"keep": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write(path, {"keep": False})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
