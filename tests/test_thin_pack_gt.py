"""Independent sampling and visibility checks; run in the installed DA3 environment."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from thin_pack_gt import MeshRays, pure_resize_support, resize_plan, support_bounds


class SamplingTests(unittest.TestCase):
    def test_actual_da3_dimensions(self):
        self.assertEqual(resize_plan(1920, 1080, 504)["intermediate_size_wh"], [504, 284])
        self.assertEqual(resize_plan(1920, 1080, 756)["intermediate_size_wh"], [756, 425])
        self.assertEqual(resize_plan(1920, 1080, 504)["target_size_wh"], [504, 280])
        self.assertEqual(resize_plan(1920, 1080, 756)["target_size_wh"], [756, 420])

    def test_pixel_center_inverse(self):
        p = resize_plan(1920, 1080, 504)
        a = np.array(p["source_index_to_target_index"])
        target = np.array([100.0, 120.0, 1.0])
        original = np.linalg.inv(a) @ target
        self.assertAlmostEqual(original[0], (100.5) * 1920 / 504 - 0.5)
        self.assertAlmostEqual(original[1], (120.5) * 1080 / 280 - 0.5)

    def test_support_has_no_zero_weight_neighbor(self):
        low, high = support_bounds(8, 4)
        np.testing.assert_array_equal(low, [0, 2, 4, 6])
        np.testing.assert_array_equal(high, [1, 3, 5, 7])
        low, high = support_bounds(7, 3)
        np.testing.assert_array_equal(low, [0, 2, 4])
        np.testing.assert_array_equal(high, [2, 4, 6])

    def test_mixed_foreground_is_not_pure_background(self):
        ids = np.full((4, 8), 100, np.uint32)
        ids[:, 3] = 1
        plan = {"steps": [{"output_size_wh": [4, 2]}, {"output_size_wh": [2, 2]}]}
        pure, labels = pure_resize_support(ids, np.ones_like(ids, bool), plan)
        self.assertFalse(pure[:, 0].any())
        self.assertTrue(pure[:, 1].all())
        self.assertTrue((labels[:, 1] == 100).all())

    def test_unsafe_input_invalidates_complete_resample_support(self):
        ids = np.full((4, 8), 100, np.uint32)
        safe = np.ones_like(ids, bool)
        safe[1, 3] = False
        plan = {"steps": [{"output_size_wh": [4, 2]}, {"output_size_wh": [2, 2]}]}
        pure, _ = pure_resize_support(ids, safe, plan)
        self.assertFalse(pure[0, 0])
        self.assertTrue(pure[1, 0])


class RayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.camera = {
            "size_wh": [8, 6],
            "K_edge": [[4, 0, 4], [0, 4, 3], [0, 0, 1]],
            "world_to_camera_cv": np.eye(4).tolist(),
            "clip_start": 0.1,
            "clip_end": 10,
        }

    def tearDown(self):
        self.temp.cleanup()

    def make_scene(self, thin=False):
        vertices = []
        faces = []
        labels = []
        slabs = [(-3, 3, -2, 2, 4, 100)]
        slabs += [(0.02, 0.1, -1, 1, 2, 1)] if thin else [(-0.3, 0.3, -1, 1, 2, 1)]
        for left, right, bottom, top, z, sid in slabs:
            n = len(vertices)
            vertices.extend(
                [(left, bottom, z), (right, bottom, z), (right, top, z), (left, top, z)]
            )
            faces.extend([(n, n + 1, n + 2), (n, n + 2, n + 3)])
            labels.extend([sid, sid])
        path = Path(self.temp.name) / "mesh.npz"
        np.savez(
            path,
            vertices=np.array(vertices, np.float32),
            triangles=np.array(faces, np.uint32),
            surface_ids=np.array(labels, np.uint32),
        )
        return MeshRays(path)

    def test_occlusion_no_hit_and_range(self):
        rays = self.make_scene()
        z, ids, distance = rays.cast(self.camera, np.array([[3.5, 3.5], [6.5, 3.5], [0.5, 0.5]]))
        np.testing.assert_array_equal(ids, [1, 100, 0])
        np.testing.assert_allclose(z[:2], [2, 4], atol=1e-6)
        self.assertTrue(np.isnan(z[2]))
        self.assertGreater(distance[1], z[1] + 0.1)

    def test_subpixel_coverage_without_center_hit(self):
        arrays = self.make_scene(thin=True).raster(self.camera, 4)
        self.assertFalse((arrays["rod_id"] == 1).any())
        self.assertGreater(int(arrays["segment_coverage_counts"][0].sum()), 0)
        self.assertTrue(arrays["sample_mixed"].any())

    def test_far_clip_is_no_hit(self):
        self.camera["clip_end"] = 3
        z, ids, _ = self.make_scene().cast(self.camera, np.array([[6.5, 3.5]]))
        self.assertEqual(ids[0], 0)
        self.assertTrue(np.isnan(z[0]))


if __name__ == "__main__":
    unittest.main()
