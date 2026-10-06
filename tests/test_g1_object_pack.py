"""Data-boundary mechanisms for new G1 assemblies; never render or read GT."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import blender_g1_object_pack as worker  # noqa: E402
import prepare_g1_object_pack as pilot  # noqa: E402


class G1ObjectPackTests(unittest.TestCase):
    def config(self):
        return json.loads(pilot.CONFIG.read_text(encoding="utf-8"))

    def parent(self):
        return json.loads(pilot.PARENT_CONFIG.read_text(encoding="utf-8"))

    def cases(self):
        return [dict(case_id=cid, target="must-not-leak", source_asset_id="must-not-leak",
            frames=[dict(view_id=f"view_{i:02d}", rgb=f"inputs/{cid}/{i}.png", rgb_sha256="f" * 64,
                size_wh=[640, 480], guide_xyxy=copy.deepcopy(worker.FIXED_GUIDE),
                guide_source=worker.GUIDE_SOURCE, camera="must-not-leak", endpoints="must-not-leak")
                for i in range(5)]) for cid in pilot.CASE_IDS]

    def test_two_distinct_assemblies_ten_frames_and_original_acquisition(self):
        config = self.config()
        self.assertEqual(pilot.validate_config(config, self.parent()), 10)
        chair, support = config["cases"]
        self.assertEqual(chair["target"]["endpoints"], [[0, 0, -.75], [0, 0, .8]])
        self.assertEqual(support["target"]["endpoints"], [[-.05, 0, -.8], [.05, 0, .85]])
        self.assertEqual([chair["target"]["radius"], support["target"]["radius"]], [.012, .009])
        self.assertTrue(chair["occluder"])
        self.assertFalse(support["occluder"])
        self.assertGreaterEqual(len(chair["distractors"]), 6)
        self.assertGreaterEqual(len(support["distractors"]), 8)

    def test_acquisition_drift_and_capture_mislabel_are_rejected(self):
        mutations = [lambda c: c["generator"].update(camera_radius_m=4.8),
                     lambda c: c["cases"][0].update(independent_capture=True),
                     lambda c: c["cases"][1].update(source_asset_id=c["cases"][0]["source_asset_id"]),
                     lambda c: c["cases"].reverse()]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                config = self.config()
                mutation(config)
                with self.assertRaises(ValueError):
                    pilot.validate_config(config, self.parent())

    def test_missing_or_target_projected_guide_never_reaches_legacy_worker(self):
        mutations = [lambda c: c["generator"].pop("guide_xyxy"),
                     lambda c: c["generator"].update(guide_xyxy=[[320, 40], [320, 410]]),
                     lambda c: c["generator"].update(guide_source="synthetic_world_guide_projection"),
                     lambda c: c["cases"][0].update(guide_endpoints=[[0, 0, 0], [0, 0, 1]])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                config = self.config()
                mutation(config)
                with self.assertRaises(ValueError):
                    worker.validate_request(dict(config=config))

    def test_wrapper_uses_fixture_setup_and_height_position_without_geometry_api(self):
        prior = SimpleNamespace(main=Mock())
        fixture = SimpleNamespace(setup=object(), height=SimpleNamespace(
            reference=SimpleNamespace(prior=prior), position=object()))
        request = dict(config=self.config(), declared_cad={"test_fixture": True}, marker_assets="test-assets")
        with patch.object(sys, "argv", ["worker", "--", "request.json"]), \
                patch.object(Path, "read_text", return_value=json.dumps(request)), \
                patch.dict(sys.modules, {"blender_rod_fixture_pack": fixture}):
            worker.main()
        self.assertIs(prior.setup_static_scene, fixture.setup)
        self.assertIs(prior.position_camera, fixture.height.position)
        self.assertEqual(fixture.CAD, request["declared_cad"])
        self.assertEqual(fixture.ASSETS, Path("test-assets"))
        prior.main.assert_called_once_with()

    def test_normal_manifest_is_explicit_allowlist_and_does_not_mutate_inputs(self):
        cases = self.cases()
        before = copy.deepcopy(cases)
        manifest = pilot.normal_manifest(cases, "inputs/declared_cad.json", "a" * 64, {"0.png": "b" * 64})
        self.assertEqual(cases, before)
        self.assertNotIn("must-not-leak", json.dumps(manifest))
        self.assertEqual(set(manifest), {"run_id", "cases", "declared_cad", "fixture_artwork_sha256", "scope"})
        for case in manifest["cases"]:
            self.assertEqual(set(case), {"case_id", "frames"})
            for frame in case["frames"]:
                self.assertEqual(set(frame), pilot.FRAME_FIELDS)
        manifest["cases"][0]["frames"][0]["guide_xyxy"][0][0] = 10
        self.assertEqual(cases, before)

    def test_normal_manifest_rejects_missing_reordered_or_retuned_views(self):
        mutations = [lambda c: c.pop(), lambda c: c[0]["frames"].pop(),
                     lambda c: c[0]["frames"].reverse(),
                     lambda c: c[0]["frames"][0].update(guide_source="target_projection")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                cases = self.cases()
                mutation(cases)
                with self.assertRaises(ValueError):
                    pilot.normal_manifest(cases, "cad.json", "a" * 64, {})

    def test_source_inventory_inherits_frozen_parent_and_only_explicit_new_files(self):
        parent = dict(source_sha256={"fixed/a.py": "a", "fixed/b.py": "b"})
        expected = set(parent["source_sha256"]) | pilot.OWN_SOURCES
        self.assertEqual(pilot.source_names(parent), expected)
        self.assertIn("tests/test_g1_object_pack.py", expected)
        self.assertIn("scripts/run_rod_identity_blender.py", expected)
        self.assertNotIn("experiments/src/creator_eval/fixture_naive_controls.py", expected)
        for name in pilot.OWN_SOURCES:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_parent_config_and_cad_hash_match_declared_inputs(self):
        config = self.config()
        self.assertEqual(pilot.digest(pilot.PARENT_CONFIG), config["source_parent_config_sha256"])
        self.assertEqual(pilot.digest(pilot.CAD), config["declared_cad_sha256"])
        self.assertEqual(len(config["source_parent_prepared_sha256"]), 64)

    def test_renderer_unknown_and_duplicate_case_ids_rejected(self):
        for ids in (["chair01", "chair01"], ["r01"], []):
            with self.subTest(ids=ids):
                config = self.config()
                config["cases"] = [dict(case_id=cid) for cid in ids]
                with self.assertRaises(ValueError):
                    worker.validate_request(dict(config=config))

    def test_partial_existing_attempt_is_never_overwritten(self):
        folders = [Mock(exists=Mock(return_value=False)) for _ in range(4)]
        pilot.require_new_paths(folders)
        for index in range(4):
            with self.subTest(existing=index):
                folders[index].exists.return_value = True
                with self.assertRaises(FileExistsError):
                    pilot.require_new_paths(folders)
                folders[index].exists.return_value = False

    def migration_fixture(self):
        config = self.config()
        migration = config["source_migrations"][0]
        parent = ROOT / ".runtime/experiments" / config["source_parent_run_id"]
        newer = ROOT / ".runtime/experiments" / migration["source_run_id"]
        name, old, new = migration["path"], migration["old_sha256"], migration["new_sha256"]
        frozen = dict(source_sha256={name: old, "unrelated.py": "unchanged"})
        hashes = {
            parent / "source_snapshot" / name: old,
            parent / "source_snapshot/unrelated.py": "unchanged",
            ROOT / "unrelated.py": "unchanged",
            ROOT / name: new,
            newer / "prepared.json": migration["prepared_sha256"],
            newer / "source_snapshot" / name: new,
        }
        successor = dict(source_sha256={name: new})
        return config, parent, frozen, hashes, successor

    def test_explicit_historical_migration_preserves_original_and_new_snapshot_identity(self):
        config, parent, frozen, hashes, successor = self.migration_fixture()
        with patch.object(pilot, "digest", side_effect=lambda p: hashes[Path(p)]), \
                patch.object(pilot, "read_json", return_value=successor):
            self.assertEqual(pilot.verify_inherited_sources(config, frozen, parent), config["source_migrations"])

    def test_missing_wrong_hash_or_unapproved_migration_is_rejected(self):
        mutations = [lambda c: c.pop("source_migrations"),
                     lambda c: c["source_migrations"][0].update(new_sha256="wrong"),
                     lambda c: c["source_migrations"][0].update(path="unrelated.py"),
                     lambda c: c["source_migrations"].append(copy.deepcopy(c["source_migrations"][0]))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                config, parent, frozen, hashes, successor = self.migration_fixture()
                mutation(config)
                with patch.object(pilot, "digest", side_effect=lambda p: hashes[Path(p)]), \
                        patch.object(pilot, "read_json", return_value=successor), self.assertRaises(ValueError):
                    pilot.verify_inherited_sources(config, frozen, parent)

    def test_migration_does_not_excuse_changed_old_snapshot_or_unrelated_live_source(self):
        for where in ("snapshot", "unrelated"):
            with self.subTest(where=where):
                config, parent, frozen, hashes, successor = self.migration_fixture()
                name = config["source_migrations"][0]["path"]
                hashes[parent / "source_snapshot" / name if where == "snapshot" else ROOT / "unrelated.py"] = "changed"
                with patch.object(pilot, "digest", side_effect=lambda p: hashes[Path(p)]), \
                        patch.object(pilot, "read_json", return_value=successor), self.assertRaises(ValueError):
                    pilot.verify_inherited_sources(config, frozen, parent)


if __name__ == "__main__":
    unittest.main()
