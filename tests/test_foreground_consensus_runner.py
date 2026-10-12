"""Fast runner inventory and isolation contracts; never read actual truth."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import foreground_consensus_runner as run  # noqa: E402


class ForegroundRunnerContracts(unittest.TestCase):
    def test_exact_five_arm_two_scale_two_repeat_inventories(self):
        for kind, count in (("fixture", 180), ("g1_object", 120)):
            runner = run.Runner(kind)
            config = runner.config()
            rows = runner.inventory(config)
            self.assertEqual(len(rows), count)
            self.assertEqual(len(set(rows)), count)
            self.assertEqual(rows[:count//2], runner.inventory(config, 0))
            self.assertEqual([r[-1] for r in rows[:3]], run.READERS)
            with self.assertRaises(ValueError):
                runner.inventory(config, True)
            with self.assertRaises(ValueError):
                runner.inventory(config, 2)

    def test_dropping_cases_arms_scales_readers_or_repeats_rejected(self):
        for kind in run.CASES:
            runner = run.Runner(kind)
            original = runner.config()
            for key in ("case_ids", "variants", "readers", "repeats", "voxel_camera_span_fractions"):
                changed = copy.deepcopy(original)
                changed[key].pop()
                with self.subTest(kind=kind, key=key), patch.object(run, "read", return_value=changed), self.assertRaises(ValueError):
                    runner.config()

    def test_new_inference_cannot_be_relabeled_blind_or_reader_only(self):
        runner = run.Runner("fixture")
        original = runner.config()
        for key, value in (("blind_evaluation", True), ("development_prior_scores_seen", False),
                           ("reader_only_intervention", True), ("reader_overrides", {"threshold": 1})):
            with self.subTest(key=key), patch.object(run, "read", return_value={**original, key: value}), self.assertRaises(ValueError):
                runner.config()

    def test_every_new_graph_audit_key_in_repeat_projection(self):
        graph = dict(repeat=0, reader_elapsed_seconds=.1, state="complete", segments=[],
            nested={"candidate_graph": {"model": [1, 2]}, "final_graph": {"indices_sha256": "a"}})
        self.assertEqual(run.physical_projection(graph, cross_repeat=True),
            run.physical_projection({**graph, "repeat": 1, "reader_elapsed_seconds": 9.}, cross_repeat=True))
        other = copy.deepcopy(graph)
        other["nested"]["candidate_graph"]["model"] = [1, 3]
        self.assertNotEqual(run.physical_projection(graph), run.physical_projection(other))
        other = copy.deepcopy(graph)
        other["extra_unknown_audit"] = True
        self.assertNotEqual(run.physical_projection(graph), run.physical_projection(other))

    def test_same_repeat_control_comparison_excludes_timing_only(self):
        original = dict(repeat=0, reader_elapsed_seconds=1., control={"fit_record_sha256": "a", "patch_id": "p"})
        changed = dict(original, reader_elapsed_seconds=9.)
        self.assertEqual(run.physical_projection(original), run.physical_projection(changed))
        changed = copy.deepcopy(original)
        changed["control"]["fit_record_sha256"] = "b"
        self.assertNotEqual(run.physical_projection(original), run.physical_projection(changed))
        self.assertEqual(run.physical_projection(original, cross_repeat=True), run.physical_projection(changed, cross_repeat=True))
        changed["control"]["patch_id"] = "different"
        self.assertNotEqual(run.physical_projection(original, cross_repeat=True), run.physical_projection(changed, cross_repeat=True))

    def test_new_support_counts_must_equal_stroke_not_required(self):
        graph = {key: 3 for key in (*run.fixture.COUNT_KEYS, "raw_base_vote_histogram", "raw_curve_vote_histogram",
            "call_policy", "native_segments", "chain_context_sha256")}
        graph["target_identity_confirmed"] = False
        run.Runner.validate_new_pair(graph, graph)
        for key in run.fixture.COUNT_KEYS:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "pairing changed"):
                run.Runner.validate_new_pair(graph, {**graph, key: 2})
        with self.assertRaisesRegex(ValueError, "identity"):
            run.Runner.validate_new_pair(graph, {**graph, "target_identity_confirmed": True})

    def test_immutable_original_candidate_paths_are_retained(self):
        fixture = run.Runner("fixture")
        objects = run.Runner("g1_object")
        self.assertEqual(fixture.candidates("r01", "depth_tls_single_span", 1)[1],
            run.fixture.previous.RUN / "r01/bundle/depth_tls_single_span-r1-enabled.json")
        self.assertEqual(objects.candidates("chair01", "baseline", 0)[1],
            run.objects.parent.RUN / "chair01/bundle/baseline-enabled.json")
        self.assertEqual(objects.candidates("chair01", "base", 1)[1:], (None, None))

    def test_closed_source_inventory_retains_inherited_names_and_new_implementation(self):
        runner = run.Runner("fixture")
        prefix = runner.old.RUN.relative_to(ROOT).as_posix() + "/source_snapshot/"
        names = runner.sources({"source_count": 1, "hashes": {prefix + "old.py": "abc"}})
        self.assertEqual(names, sorted(set(names)))
        self.assertIn("old.py", names)
        for name in run.NEW_SOURCES:
            self.assertIn(name, names)
        with self.assertRaisesRegex(ValueError, "closed source"):
            runner.sources({"source_count": 2, "hashes": {prefix + "old.py": "abc"}})

    def test_evaluation_requires_independent_authorization_before_any_checks_or_gt(self):
        runner = run.Runner("fixture")
        with patch.object(runner, "require_authorization", side_effect=ValueError("not authorized")), \
                patch.object(runner, "checked") as checked, patch.object(run, "read") as read, \
                self.assertRaisesRegex(ValueError, "not authorized"):
            runner.evaluation_payload()
        checked.assert_not_called()
        read.assert_not_called()

    def test_normal_rejects_missing_last_row_before_old_graph_or_gt_reads(self):
        runner = run.Runner("fixture")
        config = runner.config()
        rows = [dict(zip(run.KEYS, key)) for key in runner.inventory(config)][:-1]
        normal = dict(run_id=runner.run_id, state="complete", gt_read=False, pre_sha256="p",
            fresh_reader_processes=2, end_to_end_repeat_performed=False, rows=rows, repeats=[], **run.FLAGS)
        with patch.object(run, "read", return_value=normal), patch.object(run, "digest", return_value="p"), \
                patch.object(runner, "old_graphs") as old, self.assertRaisesRegex(ValueError, "All normal"):
            runner.normal(config)
        old.assert_not_called()

    def authorization_fixture(self):
        current, peer = run.Runner("fixture"), run.Runner("g1_object")
        files, hashes, bindings = {}, {}, []
        for runner in (current, peer):
            for name in ("pre.json", "inference.json", "normal-barrier.json"):
                hashes[runner.run / name] = runner.kind + "-" + name
            binding = dict(run_id=runner.run_id, pre_sha256=hashes[runner.run / "pre.json"],
                inference_sha256=hashes[runner.run / "inference.json"])
            bindings.append(binding)
            files[runner.run / "normal-barrier.json"] = dict(binding, state="passed", gt_read=False)
        audit = ROOT / ".runtime/validation/mock-independent/outcome.json"
        hashes[audit] = "audit-sha"
        files[audit] = dict(state="passed", gt_read=False, runs=bindings)
        authorization = dict(bindings[0], state="passed", gt_read=False,
            barrier_sha256=hashes[current.run / "normal-barrier.json"],
            peer_barrier_sha256=hashes[peer.run / "normal-barrier.json"],
            independent_audit_path=audit.relative_to(ROOT).as_posix(), independent_audit_sha256="audit-sha")
        files[current.run / "evaluation-authorization.json"] = authorization
        return current, peer, audit, files, hashes

    def test_authorization_binds_exact_two_normals_and_pre_gt_independent_audit(self):
        current, _, _, files, hashes = self.authorization_fixture()
        with patch.object(run, "read", side_effect=lambda p: files[p]), patch.object(run, "digest", side_effect=lambda p: hashes[p]):
            current.require_authorization()

    def test_independent_audit_cannot_be_scored_unbound_or_one_suite_only(self):
        for mode in ("gt", "missing_suite", "normal", "state", "sha"):
            current, _, audit, files, hashes = self.authorization_fixture()
            record = files[audit]
            if mode == "gt":
                record["gt_read"] = True
            elif mode == "missing_suite":
                record["runs"].pop()
            elif mode == "normal":
                record["runs"][1]["inference_sha256"] = "wrong"
            elif mode == "state":
                record["state"] = "failed"
            else:
                hashes[audit] = "changed"
            with self.subTest(mode=mode), patch.object(run, "read", side_effect=lambda p: files[p]), \
                    patch.object(run, "digest", side_effect=lambda p: hashes[p]), self.assertRaises(ValueError):
                current.require_authorization()

    def test_either_local_or_peer_barrier_identity_and_sha_must_hold(self):
        for side in ("local", "peer"):
            for key, value in (("state", "failed"), ("gt_read", True), ("pre_sha256", "wrong"),
                               ("inference_sha256", "wrong"), ("run_id", "wrong")):
                current, peer, _, files, hashes = self.authorization_fixture()
                files[(current if side == "local" else peer).run / "normal-barrier.json"][key] = value
                with self.subTest(side=side, key=key), patch.object(run, "read", side_effect=lambda p: files[p]), \
                        patch.object(run, "digest", side_effect=lambda p: hashes[p]), self.assertRaises(ValueError):
                    current.require_authorization()

    def test_numpy_graph_projection_is_json_comparable_without_dropping_geometry(self):
        import numpy as np
        graph = dict(segments=np.array([[[0., 0., 0.], [1., 0., 0.]]]), count=np.int64(1))
        self.assertEqual(run.physical_projection(graph), {"segments": [[[0., 0., 0.], [1., 0., 0.]]], "count": 1})


if __name__ == "__main__":
    unittest.main()
