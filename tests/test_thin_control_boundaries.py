"""Small RGB fixtures catch accidental GT leakage and self-consistent wrong crops."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_thin_controls as controls
from thin_pack_gt import sha256, write_json
from thin_pack_infer_worker import input_bundle_path, rgb_input_path


class CropBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "data/inputs" / controls.SOURCE_ID
        self.bundle = self.root / "data/inputs/fixture-crops"
        self.source.mkdir(parents=True)
        self.bundle.mkdir()
        self.config = {
            "input_bundle": "fixture-crops",
            "crops": {"small": [1, 0, 5, 4]},
            "jobs": [
                {
                    "case_id": "crop-" + n,
                    "source_case": "source-" + n,
                    "crop": "small",
                    "process_res": 504,
                }
                for n in ("a", "b")
            ],
        }
        self.original = {"bundle_id": controls.SOURCE_ID, "known_cameras": [], "groups": []}
        self.manifest = {"bundle_id": "fixture-crops", "known_cameras": [], "groups": []}
        for case_index, job in enumerate(self.config["jobs"]):
            frame_order = ["view_+00", "view_+15"]
            original = {"case_id": job["source_case"], "frame_order": frame_order, "frames": []}
            group = {
                "case_id": job["case_id"],
                "source_case": job["source_case"],
                "crop_box_xyxy": [1, 0, 5, 4],
                "frame_order": frame_order.copy(),
                "frames": [],
            }
            (self.source / job["source_case"]).mkdir()
            (self.bundle / job["case_id"]).mkdir()
            for frame_index, frame_id in enumerate(frame_order):
                rgb = (
                    np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3) + case_index + frame_index
                )
                source_path = self.source / job["source_case"] / (frame_id + ".png")
                crop_path = self.bundle / job["case_id"] / (frame_id + ".png")
                Image.fromarray(rgb).save(source_path)
                Image.fromarray(rgb[:4, 1:5]).save(crop_path)
                original_frame = {
                    "frame_id": frame_id,
                    "rgb": source_path.relative_to(self.source).as_posix(),
                    "sha256": sha256(source_path),
                }
                original["frames"].append(original_frame)
                group["frames"].append(
                    {
                        "frame_id": frame_id,
                        "rgb": crop_path.relative_to(self.bundle).as_posix(),
                        "sha256": sha256(crop_path),
                        "source_rgb": original_frame["rgb"],
                        "source_sha256": original_frame["sha256"],
                        "size_wh": [4, 4],
                    }
                )
            self.original["groups"].append(original)
            self.manifest["groups"].append(group)
        write_json(self.source / "manifest.json", self.original)
        write_json(self.bundle / "selection_protocol.json", self.config)
        self.manifest["source_manifest_sha256"] = sha256(self.source / "manifest.json")
        self.manifest["selection_protocol_sha256"] = sha256(self.bundle / "selection_protocol.json")
        write_json(self.bundle / "manifest.json", self.manifest)

    def check(self, manifest=None):
        controls.validate_crop_inputs(
            self.config, self.source, self.original, self.bundle, manifest or self.manifest
        )

    def test_known_crop_pixels_and_order_pass(self):
        self.check()
        self.assertEqual(input_bundle_path(self.root, "fixture-crops"), self.bundle.resolve())

    def test_bundle_escape_paths_are_rejected(self):
        for bad in (
            "../eval_gt/answers",
            r"..\eval_gt\answers",
            r"D:\eval_gt\answers",
            r"\\server\share",
            "C:relative",
            "/tmp/answers",
            "",
            ".",
        ):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                input_bundle_path(self.root, bad)

    def test_rgb_escape_paths_are_rejected(self):
        for bad in (
            "../outside.png",
            r"..\outside.png",
            r"D:\outside.png",
            "C:relative.png",
            "/outside.png",
        ):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                rgb_input_path(self.bundle, bad)

    def test_missing_group_does_not_silently_zip_to_complete(self):
        changed = copy.deepcopy(self.manifest)
        changed["groups"].pop()
        with self.assertRaisesRegex(ValueError, "group set/order"):
            self.check(changed)

    def test_changed_box_cannot_create_self_consistent_wrong_gt(self):
        changed = copy.deepcopy(self.manifest)
        changed["groups"][0]["crop_box_xyxy"] = [2, 0, 6, 4]
        with self.assertRaisesRegex(ValueError, "source/box"):
            self.check(changed)

    def test_missing_or_reordered_frames_fail(self):
        for mode in ("missing", "reordered", "duplicated"):
            changed = copy.deepcopy(self.manifest)
            group = changed["groups"][0]
            if mode == "missing":
                group["frames"].pop()
            elif mode == "reordered":
                group["frames"].reverse()
            else:
                group["frames"][1] = copy.deepcopy(group["frames"][0])
            group["frame_order"] = [frame["frame_id"] for frame in group["frames"]]
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "[Ff]rame"):
                self.check(changed)

    def test_original_manifest_hash_is_enforced(self):
        write_json(self.source / "manifest.json", {"groups": []})
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.check()

    def test_updated_crop_hash_cannot_hide_wrong_pixels(self):
        changed = copy.deepcopy(self.manifest)
        frame = changed["groups"][0]["frames"][0]
        path = self.bundle / frame["rgb"]
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(path)
        frame["sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "pixels do not match"):
            self.check(changed)

    def test_existing_input_bundle_is_never_overwritten(self):
        before = sha256(self.bundle / "manifest.json")
        with (
            patch.object(controls, "ROOT", self.root),
            patch.object(controls, "CONFIG", self.bundle / "selection_protocol.json"),
            self.assertRaises(FileExistsError),
        ):
            controls.inputs()
        self.assertEqual(before, sha256(self.bundle / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
