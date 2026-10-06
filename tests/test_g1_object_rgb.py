"""Mock-only new-object RGB contracts: source isolation and retained failures."""

import ast
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_g1_object_rgb as runner  # noqa: E402
from run_rod_fixture_narrow import SOURCES as NARROW_SOURCES  # noqa: E402


def manifest():
    cases = []
    for cid in ("chair01", "aframe01"):
        cases.append(dict(case_id=cid, frames=[dict(view_id=f"view_{i:02d}",
            rgb=f"data/inputs/{runner.SCENE_ID}/{cid}/view_{i:02d}.png", rgb_sha256="a"*64,
            size_wh=[640, 480], guide_xyxy=[[319.5, 20], [319.5, 459]],
            guide_source="fixed_screen_coordinates_independent_of_target_geometry") for i in range(5)]))
    return dict(run_id=runner.SCENE_ID, cases=cases,
        declared_cad=dict(path=f"data/inputs/{runner.SCENE_ID}/declared_cad.json", sha256="b"*64),
        fixture_artwork_sha256={}, scope="RGB-only mock")


class ObjectRgbContracts(unittest.TestCase):
    def setUp(self):
        self.config = runner.validated_config()
        self.method_config = runner.read(ROOT / "configs/rod_fixture_narrow_v1.json")
        self.case = manifest()["cases"][0]
        self.images = [np.zeros((2, 2, 3), np.uint8) for _ in range(5)]

    def evidence(self):
        return [dict(frame, observations={"rows": []}, pool={"candidates": []},
            observation_sha256=runner.canonical_hash({"rows": []}),
            pool_sha256=runner.canonical_hash({"candidates": []})) for frame in self.case["frames"]]

    def test_exact_two_by_five_anonymous_allowlist(self):
        value = manifest()
        self.assertIs(runner.validate_manifest(value, self.config), value)
        self.assertEqual(sum(len(case["frames"]) for case in value["cases"]), 10)
        for level, key in (("root", "target"), ("case", "camera"), ("frame", "world_to_camera_cv")):
            changed = copy.deepcopy(value)
            row = changed if level == "root" else changed["cases"][0] if level == "case" else changed["cases"][0]["frames"][0]
            row[key] = []
            with self.subTest(level=level), self.assertRaises(ValueError):
                runner.validate_manifest(changed, self.config)

    def test_missing_views_reordered_objects_and_modified_guide_rejected(self):
        for kind in ("view", "order", "guide", "path"):
            value = manifest()
            if kind == "view":
                value["cases"][1]["frames"].pop()
            elif kind == "order":
                value["cases"].reverse()
            elif kind == "guide":
                value["cases"][0]["frames"][0]["guide_xyxy"][0][0] += 1
            else:
                value["cases"][0]["frames"][0]["rgb"] = "data/eval_gt/target.png"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.validate_manifest(value, self.config)

    def test_method_sha_and_no_override_protocol(self):
        self.assertEqual(runner.canonical_hash(self.method_config["method"]), self.config["method_sha256"])
        for key in ("method_sha256", "scene_input_sha256", "scene_prepared_sha256", "method_overrides", "calibration_overrides"):
            with self.subTest(key=key), patch.object(runner, "read", return_value={**self.config, key: "changed"}), \
                    self.assertRaises(ValueError):
                runner.validated_config()

    def test_explicit_sources_cover_all_local_algorithm_imports(self):
        old = dict(source_sha256=dict.fromkeys(NARROW_SOURCES))
        inventory = set(runner.source_inventory(old))
        queue = ["rod_fixture_finite", "fixture_calibration"]
        visited = set()
        while queue:
            module = queue.pop()
            if module in visited:
                continue
            visited.add(module)
            source = f"experiments/src/creator_eval/{module}.py"
            self.assertIn(source, inventory)
            tree = ast.parse((ROOT / source).read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.level == 1:
                        queue.append(node.module)
                    elif node.module.startswith("creator_eval."):
                        queue.append(node.module.split(".", 1)[1])
        self.assertIn("rod_radius_consistency", visited)
        self.assertIn("experiments/src/creator_eval/__init__.py", inventory)
        self.assertIn("scripts/environment_paths.py", inventory)
        self.assertEqual(runner.source_inventory(old), sorted(inventory))

    def test_guard_blocks_truth_scores_generation_and_staging(self):
        names = ("data/eval_gt/x.json", "data/evaluation/x.json", "docs/experiments/results/x.json",
                 ".runtime/x/protocol.json", ".runtime/x/case-render_request.json",
                 ".runtime/x/generation-checks.json", ".runtime/x/rendered-rgb/view.png",
                 ".runtime/x/staging/view.png", ".runtime/x/evaluation.json", ".runtime/x/post.json",
                 "configs/g1_object_pilot_v1.json")
        for name in names:
            with self.subTest(name=name), self.assertRaises(PermissionError):
                runner.block_truth("open", (str(ROOT / name), "r", 32896))
        runner.block_truth("open", (str(ROOT / "data/inputs/mock/declared_cad.json"), "r", 32896))

    def test_generation_hash_exception_is_exact_and_cleared(self):
        path = (ROOT / "configs/g1_object_pilot_v1.json").resolve()
        with patch.object(runner, "_HASH_ONLY_PATH", path):
            runner.block_truth("open", (str(path), "r", 32896))
            with self.assertRaises(PermissionError):
                runner.block_truth("open", (str(ROOT / "other/g1_object_pilot_v1.json"), "r", 32896))
        self.assertIsNone(runner._HASH_ONLY_PATH)

        with patch.object(runner.hashlib, "file_digest", side_effect=RuntimeError("mock digest failure")), \
                self.assertRaises(RuntimeError):
            runner.digest(path)
        self.assertIsNone(runner._HASH_ONLY_PATH)

    def test_real_audit_hook_hashes_but_cannot_parse_generation_config(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); import run_g1_object_rgb as r\n"
            "sys.addaudithook(r.block_truth)\n"
            "p = r.ROOT / 'configs/g1_object_pilot_v1.json'\n"
            "assert len(r.digest(p)) == 64\n"
            "for forbidden in (p, r.ROOT / 'data/eval_gt/mock/manifest.json'):\n"
            "    try:\n"
            "        r.read(forbidden)\n"
            "    except PermissionError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError('forbidden normal read succeeded')\n"
            "assert r._HASH_ONLY_PATH is None\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_withheld_calibration_keeps_both_old_methods_without_camera_substitution(self):
        cameras = runner.unavailable_cameras(self.case, "insufficient_rgb_markers")
        with patch.object(runner, "load_images", return_value=self.images), \
                patch.object(runner, "detect_markers", return_value={}), \
                patch.object(runner, "calibrate_case", return_value=dict(state="unavailable", cameras=cameras)), \
                patch.object(runner, "extract_fixture_narrow_evidence", return_value=self.evidence()):
            result = runner.infer_case(self.case, {}, self.method_config["method"])
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["camera_case_state"], "unavailable")
        self.assertEqual(result["cameras"], cameras)
        self.assertEqual([row["state"] for row in result["result"]["methods"]], ["withheld_camera"]*2)
        self.assertEqual(len(result["frames"]), 5)
        self.assertFalse(result["gt_read_during_inference"])

    def test_detection_error_retains_object_and_all_camera_slots(self):
        with patch.object(runner, "load_images", return_value=self.images), \
                patch.object(runner, "detect_markers", side_effect=RuntimeError("mock marker failure")), \
                patch.object(runner, "calibrate_case") as calibration, \
                patch.object(runner, "extract_fixture_narrow_evidence", return_value=self.evidence()):
            result = runner.infer_case(self.case, {}, self.method_config["method"])
        calibration.assert_not_called()
        self.assertEqual(result["state"], "error")
        self.assertEqual(len(result["errors"]), 5)
        self.assertEqual(len(result["cameras"]), 5)
        self.assertEqual(len(result["result"]["methods"]), 2)

    def test_reconstruction_exception_is_error_not_empty_acceptance(self):
        cameras = runner.unavailable_cameras(self.case, "mock")
        with patch.object(runner, "load_images", return_value=self.images), \
                patch.object(runner, "detect_markers", return_value={}), \
                patch.object(runner, "calibrate_case", return_value=dict(state="unavailable", cameras=cameras)), \
                patch.object(runner, "extract_fixture_narrow_evidence", return_value=self.evidence()), \
                patch.object(runner, "reconstruct_fixture_narrow", side_effect=RuntimeError("mock reconstruction failure")):
            result = runner.infer_case(self.case, {}, self.method_config["method"])
        self.assertEqual(result["state"], "error")
        self.assertEqual([row["state"] for row in result["result"]["methods"]], ["error", "error"])
        self.assertEqual(result["errors"][0]["stage"], "narrow_reconstruction")

    def test_rgb_failure_keeps_original_five_frames(self):
        with patch.object(runner, "load_images", side_effect=ValueError("bad RGB SHA")), \
                patch.object(runner, "detect_markers") as detect:
            result = runner.infer_case(self.case, {}, self.method_config["method"])
        detect.assert_not_called()
        self.assertEqual(result["state"], "error")
        self.assertEqual(len(result["frames"]), 5)
        self.assertEqual(len(result["cameras"]), 5)
        self.assertEqual([row["state"] for row in result["result"]["methods"]], ["error", "error"])

    def test_receipt_tamper_and_snapshot_change_fail(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "input.json"
            runner.write(path, {"original": True})
            with patch.object(runner, "ROOT", root), self.assertRaisesRegex(ValueError, "receipt changed"):
                runner.receipt(path, {}, "0"*64)

    def test_existing_attempt_and_nonfinite_writes_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with patch.object(runner, "RUN", directory), patch.object(runner.sys, "addaudithook"), \
                    self.assertRaises(FileExistsError):
                runner.prepare()
            path = directory / "existing.json"
            runner.write(path, {"keep": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write(path, {"keep": False})
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                runner.write(directory / "invalid.json", {"bad": float("nan")})

    def test_last_normal_record_hash_blocks_complete_inventory(self):
        inputs = manifest()
        files, hashes, rows = {}, {}, []
        prepared = dict(input_sha256="input", method_config_sha256="method", source_sha256={})
        for case in inputs["cases"]:
            row = dict(case_id=case["case_id"], path=f"records/{case['case_id']}.json", sha256=case["case_id"])
            files[runner.RUN / row["path"]] = dict(case_id=case["case_id"], gt_read_during_inference=False,
                input_case_sha256=runner.canonical_hash(case), method_sha256=self.method_config["method_sha256"],
                frames=case["frames"], cameras=runner.unavailable_cameras(case, "mock"), result=runner.error_result("mock"))
            hashes[runner.RUN / row["path"]] = row["sha256"]
            rows.append(row)
        normal = dict(run_id=runner.RUN_ID, state="complete", gt_read=False, prepared_sha256="pre",
                      input_sha256="input", config_sha256="method", source_sha256={}, records=rows)
        files[runner.RUN / "inference.json"] = normal
        hashes[runner.RUN / "prepared.json"] = "pre"
        with patch.object(runner, "checked", return_value=(prepared, inputs, self.method_config)), \
                patch.object(runner, "read", side_effect=lambda path: files[path]), \
                patch.object(runner, "digest", side_effect=lambda path: hashes[path]):
            self.assertEqual(len(runner.verified_inference()[1]), 2)
            hashes[runner.RUN / rows[-1]["path"]] = "tampered"
            with self.assertRaisesRegex(ValueError, "normal record changed"):
                runner.verified_inference()


if __name__ == "__main__":
    unittest.main()
