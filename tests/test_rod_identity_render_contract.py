"""Incomplete Blender output must not quietly become a smaller experiment."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rod_identity_blender import validate_render_manifest  # noqa: E402


class RenderContractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.inputs, self.truth = root / "inputs", root / "truth"
        self.config = {
            "cases": [{"case_id": "s01"}, {"case_id": "s02"}],
            "generator": {"angles_degrees": [-16, 2, 19], "size_wh": [16, 12]},
        }
        self.render = {"cases": []}
        for case in self.config["cases"]:
            case_id = case["case_id"]
            (self.inputs / case_id).mkdir(parents=True)
            (self.truth / case_id).mkdir(parents=True)
            geometry = {"objects": [{"object": "Target", "surface_id": 1}]}
            (self.truth / case_id / "geometry.json").write_text(json.dumps(geometry))
            # This contract checks publication completeness, not NPZ geometry.
            # MeshRays owns the numerical validation immediately afterward.
            (self.truth / case_id / "mesh.npz").write_bytes(b"mesh fixture")
            frames = []
            for angle in self.config["generator"]["angles_degrees"]:
                frame_id = f"view_{angle:+03d}"
                rgb = f"{case_id}/{frame_id}.png"
                Image.new("RGB", (16, 12), "gray").save(self.inputs / rgb)
                camera = {"size_wh": [16, 12], "angle_fixture": angle}
                (self.truth / case_id / f"{frame_id}-camera.json").write_text(json.dumps(camera))
                frames.append({"frame_id": frame_id, "angle_degrees": angle,
                               "rgb": rgb, "camera": camera})
            self.render["cases"].append({"case_id": case_id, "geometry": geometry, "frames": frames})

    def validate(self, render=None, config=None):
        validate_render_manifest(render or self.render, config or self.config,
                                 self.inputs, self.truth)

    def test_complete_plan_passes_even_when_worker_completion_order_differs(self):
        self.validate()
        changed = copy.deepcopy(self.render)
        changed["cases"].reverse()
        for case in changed["cases"]:
            case["frames"].reverse()
        self.validate(changed)

    def test_missing_duplicate_and_extra_cases_fail(self):
        for mode in ("missing", "duplicate", "unexpected"):
            changed = copy.deepcopy(self.render)
            if mode == "missing":
                changed["cases"].pop()
            elif mode == "duplicate":
                changed["cases"].append(copy.deepcopy(changed["cases"][0]))
            else:
                changed["cases"][0]["case_id"] = "not-planned"
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Rendered cases"):
                self.validate(changed)

    def test_missing_duplicate_and_extra_frames_fail(self):
        for mode in ("missing", "duplicate", "unexpected"):
            changed = copy.deepcopy(self.render)
            frames = changed["cases"][0]["frames"]
            if mode == "missing":
                frames.pop()
            elif mode == "duplicate":
                frames.append(copy.deepcopy(frames[0]))
            else:
                frames[0]["frame_id"] = "view_+99"
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Rendered frames"):
                self.validate(changed)

    def test_wrong_angle_and_reused_rgb_fail(self):
        changed = copy.deepcopy(self.render)
        changed["cases"][0]["frames"][0]["angle_degrees"] = 2
        with self.assertRaisesRegex(ValueError, "angle"):
            self.validate(changed)
        for rgb in ("s01/view_+02.png", "../outside.png"):
            changed = copy.deepcopy(self.render)
            changed["cases"][0]["frames"][0]["rgb"] = rgb
            with self.subTest(rgb=rgb), self.assertRaisesRegex(ValueError, "RGB does not belong"):
                self.validate(changed)

    def test_missing_artifact_blocks_publication(self):
        paths = [self.inputs / "s01/view_-16.png", self.truth / "s01/view_-16-camera.json",
                 self.truth / "s01/geometry.json", self.truth / "s01/mesh.npz"]
        for path in paths:
            original = path.read_bytes()
            path.unlink()
            with self.subTest(path=path.name), self.assertRaises(FileNotFoundError):
                self.validate()
            path.write_bytes(original)

    def test_camera_geometry_and_image_mismatch_fail(self):
        for key in ("camera", "geometry"):
            changed = copy.deepcopy(self.render)
            if key == "camera":
                changed["cases"][0]["frames"][0]["camera"]["size_wh"] = [17, 12]
            else:
                changed["cases"][0]["geometry"]["objects"][0]["surface_id"] = 99
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.validate(changed)
        Image.new("RGB", (17, 12), "gray").save(self.inputs / "s01/view_-16.png")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.validate()

    def test_duplicate_protocol_case_or_angle_is_not_hidden_by_a_set(self):
        for key in ("case", "angle"):
            changed = copy.deepcopy(self.config)
            if key == "case":
                changed["cases"].append({"case_id": "s01"})
            else:
                changed["generator"]["angles_degrees"].append(2)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unique"):
                self.validate(config=changed)


if __name__ == "__main__":
    unittest.main()
