"""Mock-only full snapshot/patch integration; no actual model inference or GT."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_g1_object_point_patch as runner  # noqa: E402


def case(cid="chair01"):
    frames, cameras = [], []
    for index in range(5):
        view = f"view_{index:02d}"
        frames.append(dict(view_id=view, rgb=f"inputs/{cid}/{view}.png", rgb_sha256="a"*64,
                           size_wh=[8, 6], guide_xyxy=[[3.5, 0], [3.5, 5]], guide_source="mock"))
        extrinsic = np.c_[np.eye(3), [-index*.1, 0, 0]]
        cameras.append(dict(view_id=view, state="validated", K_index=[[8., 0, 3.5], [0, 8., 2.5], [0, 0, 1]],
                            world_to_camera_cv=extrinsic.tolist()))
    return dict(case_id=cid, frames=frames, cameras=cameras, parent_record=f"parent/{cid}.json", parent_sha256="unused")


class ObjectPointContracts(unittest.TestCase):
    def setUp(self):
        self.config = runner.validated_config()
        self.method = runner.read(ROOT / "configs/rod_fixture_narrow_v1.json")["method"]

    def test_frozen_two_objects_four_patches_one_depth_each(self):
        self.assertEqual(self.config["case_ids"], ["chair01", "aframe01"])
        self.assertEqual(len(self.config["patch_methods"]), 4)
        self.assertEqual(self.config["expected_patch_count"], 8)
        for key, value in (("process_res", 756), ("model", "base"), ("align_to_input_ext_scale", False),
                           ("depth_repeats_per_object", 2), ("naive_defaults", {}), ("support_policy", {})):
            with self.subTest(key=key), patch.object(runner, "read", return_value={**self.config, key: value}), \
                    self.assertRaises(ValueError):
                runner.validated_config()

    def test_closed_sources_use_only_bound_parent_sets_plus_explicit_new_files(self):
        prefix = runner.TEMPLATE.relative_to(ROOT).as_posix() + "/source_snapshot/"
        template = dict(source_count=2, hashes={prefix+"one.py": "a", prefix+"two.py": "b", "data/input.json": "c"})
        names = runner.source_inventory(template, {"source_sha256": {"rgb.py": "d"}}, ["upstream.py"])
        self.assertEqual(names, sorted(set(names)))
        self.assertTrue({"one.py", "two.py", "rgb.py", "upstream.py", "scripts/run_g1_object_point_patch.py",
            "experiments/src/creator_eval/fixture_naive_controls.py", "tests/test_g1_object_point_patch.py"}.issubset(names))
        self.assertNotIn("data/input.json", names)
        with self.assertRaises(ValueError):
            runner.source_inventory({**template, "source_count": 3}, {"source_sha256": {}}, [])

    def test_missing_installed_model_is_not_downloaded(self):
        with patch.object(runner.importlib.util, "find_spec", return_value=None), self.assertRaisesRegex(ValueError, "never auto-install"):
            runner.installed_upstream_sources()
        lock = dict(models=[dict(name="large", repo_id="local", revision="mock",
            files=[dict(name="model.safetensors", bytes=123, sha256="a"*64)])])
        with tempfile.TemporaryDirectory() as name, patch.object(runner, "ROOT", Path(name)), \
                patch.object(runner, "receipt"), patch.object(runner, "read", return_value=lock), \
                self.assertRaisesRegex(ValueError, "downloads are not permitted"):
            runner.verify_models(self.config, {})

    def test_exact_candidate_bytes_and_withdrawn_geometry(self):
        base = dict(base_snapshot_id="mock", world_frame_id="fixture", length_unit="meter",
                    points=np.array([[1., 2., 3.]]), point_ids=np.array([[0, 0, 0]], np.uint32))
        segments = np.array([[[0., 0., 0.], [0., 0., 1.]]])
        candidate = {**base, "segments": segments}
        runner.validate_candidate(base, candidate, segments, enabled=True)
        runner.validate_candidate(base, {**base, "segments": np.empty((0, 2, 3))}, segments, enabled=False)
        for update in ({"points": base["points"]+1}, {"point_ids": base["point_ids"]+1},
                       {"world_frame_id": "different"}, {"points": base["points"].astype(np.float32)}):
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                runner.validate_candidate(base, {**candidate, **update}, segments, enabled=True)
        with self.assertRaises(ValueError):
            runner.validate_candidate(base, candidate, segments, enabled=False)

    def test_normal_guard_blocks_new_object_truth_and_generation(self):
        for name in ("data/eval_gt/g1-object-scenes-v1-20261006/manifest.json", "data/evaluation/new.json",
                     "docs/experiments/results/new.json", ".runtime/new/protocol.json",
                     ".runtime/new/case-render_request.json", "configs/g1_object_pilot_v1.json"):
            with self.subTest(name=name), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(ROOT / name), "r", 32896))

    def test_real_four_patch_mock_integration_uses_actual_export_cameras(self):
        with tempfile.TemporaryDirectory() as name:
            root, run = Path(name), Path(name) / "run"
            item = case()
            folder = run / item["case_id"]
            folder.mkdir(parents=True)
            (root / "parent").mkdir()
            runner.write(root / item["parent_record"], dict(frames=item["frames"], cameras=item["cameras"], result={"old": True}))
            item["parent_sha256"] = runner.digest(root / item["parent_record"])
            edge, full_e = runner.camera_inputs(item["cameras"])
            depth = np.ones((5, 3, 4), np.float32)
            k = np.diag([.5, .5, 1.]) @ edge
            with (folder / "prediction.npz").open("xb") as stream:
                np.savez_compressed(stream, depth=depth, intrinsics=k, extrinsics=full_e[:, :3])
            segment = [[[0., 0., .5], [0., 0., 1.5]]]
            rods = dict(methods=[dict(method="baseline", state="accepted", segments=segment, rejection_reasons=[]),
                dict(method="cylinder_support", state="rejected", segments=[], rejection_reasons=["insufficient_views"])])
            fits = [dict(method=method, state="complete", gt_read=False, input_support_count=60,
                input_support_sha256="c"*64, source_view_counts={str(index): 12 for index in range(5)},
                outcome="accepted_change" if index == 0 else "no_supported_change", segments=segment if index == 0 else [],
                reason=None if index == 0 else "unsupported", elapsed_seconds=.01)
                for index, method in enumerate(runner.controls.METHODS)]
            report = dict(prediction_sha256=runner.digest(folder / "prediction.npz"), inference_seconds=2.,
                          load_seconds=1., peak_allocated_mib=123.)
            with patch.object(runner, "ROOT", root), patch.object(runner, "RUN", run), \
                    patch.object(runner, "verified_depth", return_value=report), \
                    patch.object(runner, "reconstruct_fixture_narrow", return_value=rods) as reconstruct, \
                    patch.object(runner, "support_views", return_value=[{}]*5), \
                    patch.object(runner, "support_votes", return_value=np.full(60, 3)), \
                    patch.object(runner.controls, "fit_controls", return_value=fits) as fit:
                row = runner.compose_case(item, self.method, self.config)
                self.assertEqual(row["full_point_count"], 60)
                self.assertEqual(row["points_per_view"], [12]*5)
                self.assertEqual([p["method"] for p in row["patches"]], self.config["patch_methods"])
                self.assertEqual([p["segment_count"] for p in row["patches"]], [1, 0, 1, 0])
                self.assertFalse(row["end_to_end_repeat_performed"])
                self.assertEqual(row["roundtrip"]["pixel_checks"], 60)
                actual_cameras = reconstruct.call_args.args[1]
                self.assertTrue(all(c["camera_source"] == "exact_conditioned_prediction_export_lifted_to_original_RGB" for c in actual_cameras))
                self.assertEqual(fit.call_args.args[0].shape, (60, 3))
                self.assertEqual(fit.call_args.args[1].dtype, np.dtype(np.uint32))
                self.assertTrue(fit.call_args.args[3].all())
                base = runner.compose(folder / "bundle/base")
                for stored in row["patches"]:
                    opened = runner.open_candidate_view(root / stored["withdrawn_path"])
                    np.testing.assert_array_equal(base["points"], opened["points"])
                    np.testing.assert_array_equal(base["point_ids"], opened["point_ids"])
                    self.assertEqual(len(opened["segments"]), 0)
                rejected = runner.open_candidate_view(folder / "bundle/cylinder_support-enabled.json")
                np.testing.assert_array_equal(rejected["points"], base["points"])
                self.assertEqual(len(rejected["segments"]), 0)
                with self.assertRaises(FileExistsError):
                    runner.compose_case(item, self.method, self.config)

    def test_depth_failure_or_sha_tamper_cannot_be_relabelled_complete(self):
        item = case()
        report = dict(state="complete", case_id=item["case_id"], gt_read=False, pre_sha256="pre",
            prediction_sha256="prediction", raw_sha256="raw", camera_input_sha256=runner.canonical_hash(item["cameras"]),
            process_res=504, depth_repeats_for_object=1, api_observation={"actual": True})
        def digest(path):
            return {"pre.json": "pre", "prediction.npz": "prediction", "raw-before-api.npz": "raw"}[path.name]
        with tempfile.TemporaryDirectory() as name, patch.object(runner, "RUN", Path(name)), \
                patch.object(runner, "read", return_value=report), patch.object(runner, "digest", side_effect=digest):
            self.assertIs(runner.verified_depth(item), report)
            report["prediction_sha256"] = "tampered"
            with self.assertRaises(ValueError):
                runner.verified_depth(item)
            folder = Path(name) / item["case_id"]
            folder.mkdir()
            runner.write(folder / "depth-failure.json", {"state": "error"})
            with self.assertRaisesRegex(ValueError, "failed depth attempt"):
                runner.verified_depth(item)

    def test_infer_two_fresh_depth_subprocesses_and_no_retry_overwrite(self):
        inputs = dict(cases=[case(cid) for cid in self.config["case_ids"]], method=self.method)
        rows = [dict(case_id=cid, patches=[{}]*4, full_point_count=60) for cid in self.config["case_ids"]]
        with tempfile.TemporaryDirectory() as name, patch.object(runner, "RUN", Path(name)), \
                patch.object(runner.sys, "addaudithook"), \
                patch.object(runner, "checked", return_value=({}, inputs, self.config)), \
                patch.object(runner, "verified_depth"), patch.object(runner, "compose_case", side_effect=rows), \
                patch.object(runner, "normal_records"), patch.object(runner, "digest", return_value="sha"), \
                patch.object(runner.subprocess, "run") as process:
            runner.infer()
            self.assertEqual(process.call_count, 2)
            for cid, call in zip(self.config["case_ids"], process.call_args_list):
                self.assertEqual(call.args[0][-3:], ["depth", "--case", cid])
                self.assertTrue(call.kwargs["check"])
            with self.assertRaises(FileExistsError):
                runner.infer()

    def test_explicit_depth_stage_reused_only_after_verification(self):
        inputs = dict(cases=[case(cid) for cid in self.config["case_ids"]], method=self.method)
        rows = [dict(case_id=cid, patches=[{}]*4, full_point_count=60) for cid in self.config["case_ids"]]
        with tempfile.TemporaryDirectory() as name, patch.object(runner, "RUN", Path(name)), \
                patch.object(runner.sys, "addaudithook"), \
                patch.object(runner, "checked", return_value=({}, inputs, self.config)), \
                patch.object(runner, "verified_depth") as verify, patch.object(runner, "compose_case", side_effect=rows), \
                patch.object(runner, "normal_records"), patch.object(runner, "digest", return_value="sha"), \
                patch.object(runner.subprocess, "run") as process:
            for cid in self.config["case_ids"]:
                (Path(name) / cid).mkdir()
            runner.infer()
            process.assert_not_called()
            self.assertEqual(verify.call_count, 2)

    def test_prepare_and_json_writes_never_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            with patch.object(runner, "RUN", path), patch.object(runner.sys, "addaudithook"), self.assertRaises(FileExistsError):
                runner.prepare()
            runner.write(path / "normal.json", {"keep": True})
            with self.assertRaises(FileExistsError):
                runner.write(path / "normal.json", {"keep": False})
            with self.assertRaises(ValueError):
                runner.write(path / "nonfinite.json", {"x": float("nan")})

    def test_normal_record_inventory_checked_before_output_reads(self):
        normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, pre_sha256="pre",
                      rows=[dict(case_id="chair01")], depth_repeats_per_object=1, end_to_end_repeat_performed=False)
        with patch.object(runner, "checked", return_value=({}, {}, self.config)), \
                patch.object(runner, "read", return_value=normal) as read, \
                patch.object(runner, "digest", return_value="pre"), self.assertRaisesRegex(ValueError, "two-object"):
            runner.normal_records()
        self.assertEqual(read.call_count, 1)


if __name__ == "__main__":
    unittest.main()
