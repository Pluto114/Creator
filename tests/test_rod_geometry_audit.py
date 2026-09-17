"""Synthetic known answers for geometry-only scanlines, without image/model packages."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_rod_geometry import (
    audit_segment_axis,
    axis_at_row,
    mesh_scanline_intervals,
    native_pixel_runs,
)


class GeometryAuditTests(unittest.TestCase):
    camera = {
        "K_index": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
        "world_to_camera_cv": np.eye(4).tolist(),
        "clip_start": 0.1,
        "clip_end": 100,
    }

    def test_finite_axis_and_perspective_fraction(self):
        endpoints = [[1, -1, 2], [1, 1, 4]]
        result = axis_at_row(endpoints, self.camera, 5)
        self.assertAlmostEqual(result["world_fraction"], 0.5)
        self.assertAlmostEqual(result["u_index"], 5 + 10 / 3)
        self.assertIsNone(axis_at_row(endpoints, self.camera, 20))

    def test_cap_geometry_detects_shifted_axis_and_elliptic_scaling(self):
        vertices = np.array(
            [
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
                [-1, -1, 2],
                [1, -1, 2],
                [1, 1, 2],
                [-1, 1, 2],
            ],
            float,
        )
        triangles = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
        result = audit_segment_axis(vertices, triangles, [[0, 0, 0], [0, 0, 2]])
        self.assertEqual(result["cap_center_to_axis_endpoint_m"], [0, 0])
        self.assertAlmostEqual(result["radial_principal_rms_ratio"], 1)
        self.assertEqual(result["opposite_vertex_symmetry_max_residual_m"], 0)
        shifted = audit_segment_axis(vertices, triangles, [[0.25, 0, 0], [0.25, 0, 2]])
        self.assertEqual(shifted["cap_center_to_axis_endpoint_m"], [0.25, 0.25])
        vertices[:, 0] *= 2
        ellipse = audit_segment_axis(vertices, triangles, [[0, 0, 0], [0, 0, 2]])
        self.assertAlmostEqual(ellipse["radial_principal_rms_ratio"], 2)

    def test_triangle_scanline_unions_shared_diagonal(self):
        vertices = [[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]]
        triangles = [[0, 1, 2], [0, 2, 3]]
        np.testing.assert_allclose(
            mesh_scanline_intervals(vertices, triangles, self.camera, 5), [[0, 10]]
        )
        np.testing.assert_allclose(
            mesh_scanline_intervals(vertices, triangles, self.camera, 0), [[0, 10]]
        )
        self.assertEqual(mesh_scanline_intervals(vertices, triangles, self.camera, 11), [])

    def test_disconnected_mesh_pieces_do_not_fill_gap(self):
        vertices = [
            [-2, -1, 2],
            [-1, -1, 2],
            [-1, 1, 2],
            [-2, 1, 2],
            [1, -1, 2],
            [2, -1, 2],
            [2, 1, 2],
            [1, 1, 2],
        ]
        triangles = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
        np.testing.assert_allclose(
            mesh_scanline_intervals(vertices, triangles, self.camera, 5), [[-5, 0], [10, 15]]
        )

    def test_native_pixel_ranges_are_quantized_and_keep_occlusion_breaks(self):
        result = native_pixel_runs([0, 5, 5, 0, 5, 0], 5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["center_midpoint_index"], 1.5)
        self.assertEqual(result[0]["pixel_cell_footprint"], [0.5, 2.5])
        self.assertEqual(result[1]["pixel_count"], 1)

    def test_projected_axis_and_silhouette_midpoint_need_not_be_equal(self):
        # Perspective alone changes symmetric +/- world offsets at different depths.
        vertices = [[-1, -1, 2], [1, -1, 4], [1, 1, 4], [-1, 1, 2]]
        intervals = mesh_scanline_intervals(vertices, [[0, 1, 2], [0, 2, 3]], self.camera, 5)
        midpoint = np.mean(intervals[0])
        axis = axis_at_row([[0, -1, 3], [0, 1, 3]], self.camera, 5)
        self.assertAlmostEqual(midpoint, 3.75)
        self.assertAlmostEqual(axis["u_index"], 5)

    def test_half_pixel_is_not_added_twice(self):
        camera = dict(self.camera, K_index=[[10, 0, 4.5], [0, 10, 4.5], [0, 0, 1]])
        axis = axis_at_row([[0, -1, 2], [0, 1, 2]], camera, 4.5)
        self.assertEqual(axis["u_index"], 4.5)

    def test_near_plane_crossing_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "clip"):
            mesh_scanline_intervals([[0, 0, 0], [1, 0, 1], [0, 1, 1]], [[0, 1, 2]], self.camera, 5)


if __name__ == "__main__":
    unittest.main()
