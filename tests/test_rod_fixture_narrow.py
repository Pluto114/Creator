"""Synthetic multi-view checks for the frozen unresolved-narrow fixture method."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval import rod_fixture_finite as pilot  # noqa: E402

CONFIG = json.loads((ROOT / "configs/rod_fixture_narrow_v1.json").read_text(encoding="utf-8"))
METHOD = CONFIG["method"]


def fixture(kind="complete"):
    k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    frames, images, cameras = [], [], []
    for index, (cx, cy) in enumerate(zip([-1.4, -0.7, 0.0, 0.7, 1.4], [0.0, 0.1, -0.1, 0.2, -0.2])):
        rgb = np.full((480, 640, 3), 180, np.uint8)
        x = int(round(320.0 - 100.0 * cx))
        y0, y1 = int(round(140.0 - 100.0 * cy)), int(round(340.0 - 100.0 * cy))
        rgb[y0 : y1 + 1, x] = 35
        if kind == "gap":
            rgb[int(round(236.0 - 100.0 * cy)) : int(round(245.0 - 100.0 * cy)), x] = 180
        name = f"view_{index:02d}"
        frames.append(
            {
                "view_id": name,
                "rgb": "fixture.png",
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "size_wh": [640, 480],
                "guide_xyxy": [[x, 20], [x, 459]],
                "guide_source": "synthetic_test_only",
            }
        )
        images.append(rgb)
        cameras.append(
            {
                "view_id": name,
                "K_index": k.tolist(),
                "world_to_camera_cv": np.c_[np.eye(3), [-cx, -cy, 0.0]].tolist(),
                "state": "validated",
            }
        )
    return frames, images, cameras


class FixtureNarrowTests(unittest.TestCase):
    def run_case(self, kind):
        frames, images, cameras = fixture(kind)
        evidence = pilot.extract_fixture_narrow_evidence(frames, images, METHOD)
        result = pilot.reconstruct_fixture_narrow(evidence, cameras, METHOD)
        return evidence, result

    def test_method_hash_is_frozen(self):
        self.assertEqual(CONFIG["method_sha256"], pilot.canonical_hash(METHOD))
        self.assertEqual(CONFIG["method_sha256"], pilot.NARROW_METHOD_SHA256)

    def test_one_pixel_multiview_line_is_recovered_without_width_claim(self):
        evidence, result = self.run_case("complete")
        self.assertTrue(all(frame["full_pool_count"] == 1 for frame in evidence))
        for frame in evidence:
            candidates = [
                candidate
                for row in frame["observations"]["rows"]
                for candidate in row["candidates"]
            ]
            self.assertTrue(candidates)
            self.assertTrue(all(c["physical_width_resolved"] is False for c in candidates))
        for method in result["methods"]:
            self.assertEqual(method["state"], "accepted", method["rejection_reasons"])
            self.assertEqual(method["segment_count"], 1)
            self.assertGreater(method["total_length_m"], 1.94)

    def test_small_true_gap_remains_two_segments(self):
        _, result = self.run_case("gap")
        for method in result["methods"]:
            self.assertEqual(method["state"], "accepted", method["rejection_reasons"])
            self.assertEqual(method["segment_count"], 2)
            self.assertFalse(np.any(method["finite"]["evidence"]["absent_view_counts"]))

    def test_changed_narrow_threshold_is_rejected(self):
        frames, images, cameras = fixture()
        evidence = pilot.extract_fixture_narrow_evidence(frames, images, METHOD)
        changed = json.loads(json.dumps(METHOD))
        changed["narrow_observation"]["minimum_edge_balance"] = 0.1
        with self.assertRaisesRegex(ValueError, "predeclared"):
            pilot.reconstruct_fixture_narrow(evidence, cameras, changed)


if __name__ == "__main__":
    unittest.main()
